# Arquitetura — fluxo LangGraph (Etapa 5)

> **Resumo honesto:**
> - O fluxo funciona de ponta a ponta com o modelo real (Qwen2.5-1.5B + LoRA) na GPU local.
> - As verificações determinísticas retêm ou encaminham para revisão a maior parte das respostas
>   erradas observadas, mas **não todas**. Na demonstração, uma resposta errada passou como
>   "respondido" (ver §6).
> - Passar nas verificações **não** torna uma resposta clinicamente segura.
> - Nada aqui foi revisado por profissional de saúde.

## 1. Visão geral

```
Pergunta ─► LangGraph ─► SQLite (paciente) ─► RAG (protocolos) ─► política de seções v1
                                                                         │
             Auditoria (JSONL) ◄── Validação determinística ◄── Qwen2.5-1.5B + LoRA
                     │
                     ▼
     Resposta estruturada (status, fontes, alertas, ID de auditoria)
```

Código em `src/medprotocol/graph/`:

| Módulo | Papel |
|---|---|
| `workflow.py` | Estado, nós, arestas, saída `AssistantResponse` e fachada `MedProtocolAssistant.ask()` |
| `validation.py` | Análise da pergunta e verificações da resposta (regras, sem LLM) |
| `llm.py` | `LocalQwenGenerator`: base + adaptador, um único modelo, variante por configuração |
| `audit.py` | Log JSONL sem texto de pergunta ou resposta |

**Reutilizado sem reimplementar:**
- `repository.get_patient_summary` (Etapa 2);
- `ProtocolRetriever.retrieve` (Etapa 3);
- `assemble_context` (política v1) e `build_messages` (prompt v3), as mesmas funções que geraram os dados de treino.

`build_context` foi dividida em duas partes (busca + `assemble_context`), sem mudar o resultado,
para que banco, busca e política sejam nós separados. O hash de `context_policy.py` não mudou.

## 2. Grafo

```
interpretar ──[erro: vazia / >1 paciente]─────────────────────────────┐
   │[com paciente]                     │[sem paciente]                   │
   ▼                                   │                                 │
consultar_paciente ──[inexistente / falha do banco]────────────────────┤
   ▼                                   ▼                                 │
recuperar_protocolos ◄─────────────────┘                                 │
   ▼                                                                     │
aplicar_politica ──[sem evidência: busca não confiável, sem regra e sem paciente]─┤
   ▼                                                                     │
gerar_resposta ──[falha ao carregar ou ao gerar]───────────────────────┤
   ▼                                                                     ▼
validar_resposta ─────────────────────────────────────────► registrar_auditoria ─► END
```

**Estado** (`GraphState`): `audit_id`, `pergunta`, `analise`, `patient_id`, `paciente`,
`recuperacao`, `contexto`, `mensagens`, `resposta_modelo`, `resposta`, `alertas`, `status`,
`erro` e `tempos_ms` por nó.

**Componentes LangChain nos nós** (`lc_components.py`):
- `consultar_paciente` → `build_patient_lookup`: `RunnableLambda` sobre `get_patient_summary`
  (SQLite somente leitura);
- `gerar_resposta` → `build_answer_chain`: `RunnableLambda(dict → mensagens LangChain)` |
  `MedProtocolChatModel` (`BaseChatModel` sobre o `LocalQwenGenerator`) | `StrOutputParser`.
- As exceções (`PatientNotFoundError`, `ModelLoadError`, `GenerationError`...) atravessam os
  Runnables sem mudar de tipo, então o tratamento de erro do grafo é o mesmo.
- A conversão de mensagens é exata, e o prompt não mudou: as 16 respostas de
  `reports/graph/demo_*.jsonl`, regeradas na GPU depois da mudança, ficaram idênticas.

**Atalhos que evitam chamar o modelo:**
- paciente inexistente;
- mais de um paciente na pergunta;
- pergunta fora do escopo, sem evidência.

Nesses casos a GPU não é usada e nada é inventado.

## 3. Saída estruturada (`AssistantResponse`)

| Campo | Conteúdo |
|---|---|
| `pergunta`, `resposta` | Texto exibido ao usuário. Se a resposta for retida, vem uma mensagem segura no lugar |
| `resposta_modelo` | Texto original do modelo quando retido, para o revisor humano |
| `status` | `respondido` · `evidencia_insuficiente` · `revisao_humana` · `erro` |
| `paciente`, `registros_citados` | Paciente consultado e IDs (EX/CON/PAC) presentes na resposta |
| `fontes` | Cada trecho do contexto: ref, origem (similaridade ou `regra:<motivos>`), versão, status SIMULACAO e se foi citado |
| `alertas` | `{codigo, severidade, detalhe}` |
| `status_recuperacao`, `modelo`, `tempos_ms`, `audit_id` | Rastreabilidade |
| `aviso` | Texto fixo: protótipo educacional; verificações não garantem segurança clínica |

**Regra de status**, aplicada nesta ordem:
1. Falha técnica → `erro`.
2. Algum alerta **alto** → a resposta é **retida** e o status é `revisao_humana`.
3. Algum alerta **médio**, ou gatilho de revisão → `revisao_humana`. Gatilhos: pedido de
   conduta, uso fora da simulação, resultado fora da meta, dados inconsistentes, hipoglicemia
   em condição crítica.
4. Resposta declara falta de informação → `evidencia_insuficiente`.
5. Caso contrário → `respondido`.

## 4. Verificações determinísticas (`validation.py`)

| Código | Severidade | Regra |
|---|---|---|
| `id_inexistente` | alta | EX/CON citado que não está no histórico do paciente; PAC diferente do consultado |
| `exame_trocado` | alta | A frase cita um exame pelo ID, mas fala de outro tipo de exame (ex.: EX-0014 é HbA1c e a frase fala de creatinina) |
| `valor_trocado` | alta | A frase cita um exame pelo ID com um valor (na unidade do exame) diferente do registrado |
| `valor_de_exame_ausente` | alta | Valor com unidade logo após o nome de um exame que o paciente não tem |
| `referencia_fora_do_contexto` / `protocolo_fora_do_contexto` | alta | PROTOCOLO#SEÇÃO ou protocolo citado que não está no contexto |
| `dose_na_resposta` | alta | Quantidade com unidade de dose (mg, UI, mL...) que não estava na entrada |
| `sugestao_de_conduta` | alta | "recomendo/deve-se… iniciar/ajustar/suspender/prescrever…" |
| `recusa_ausente` | alta | Pedido de conduta respondido sem "não posso" e sem encaminhar ao médico |
| `conduta_incompativel_com_condicao` | alta | Conduta por via oral ("oferecer", "15 g", "carboidrato", "via oral") quando a pergunta diz que o paciente está inconsciente, não deglute, tem condição contraditória ou **não teve** hipoglicemia |
| `valor_nao_confere` | média | Valor após o nome do exame que não está no histórico desse tipo nem nos limites dos trechos |
| `data_nao_confere` / `data_fora_do_contexto` | média | Data ligada a um exame que não é a dele, ou data ausente da entrada |
| `acao_sem_registro` | média | "foi apresentado/sinalizado/notificado…" sem a ação correspondente nas consultas |
| `fonte_externa` | média | OMS, sociedades médicas, "guideline"… (o contexto só tem os protocolos simulados) |
| `sem_fonte_citada` | média | Nenhum protocolo citado, sem recusa e sem declaração de insuficiência: a fundamentação não pode ser conferida |
| `condicao_contraditoria` | média | A pergunta descreve o paciente como capaz e incapaz ao mesmo tempo |

**Paciente consciente × inconsciente:** a análise da pergunta classifica a condição em
`capaz`, `incapaz` ou `contraditoria`. Uma resposta que aplica a conduta da S3 do PROT-DM2-002
(prevista para paciente consciente e capaz de deglutir) a uma condição `incapaz` ou
`contraditoria` é sempre retida. No lugar, a resposta diz que a conduta não pode ser aplicada
automaticamente e encaminha ao médico responsável.

**Pedido de conduta:** usa um detector próprio, mais amplo que a intenção da política v1, que
continua congelada. Ele cobre "começar uma estatina", "qual antibiótico usar" e similares. Esse
pedido sempre vai para revisão humana, mesmo com recusa correta.

## 5. Inferência e ambientes

**Gerador:**
- `LocalQwenGenerator` carrega **um** modelo (base + adaptador) na primeira pergunta.
- `MODEL_VARIANT=ajustado` (padrão) usa o adaptador; `base` o desliga com `disable_adapter()`, sem segunda cópia.
- Erros viram `ModelLoadError` ou `GenerationError` e status `erro`.
- A decodificação é gulosa, com no máximo 512 tokens novos.

**Ambientes:**

| Ambiente | Conteúdo | Uso |
|---|---|---|
| `.venv` | torch CPU, RAG, langgraph | Etapas 2–3; testes do fluxo com LLM simulado |
| `.venv-train` | torch CUDA, peft, RAG, langgraph | Treino (Etapa 4) e **execução com o modelo real** |

Antes de usar o `.venv-train` para o fluxo, a recuperação foi comparada nos dois ambientes:
91 consultas (61 de treino + 30 do conjunto de desenvolvimento do RAG). Resultado: mesmo status,
mesmos trechos e diferença máxima de score de 0,0, porque os embeddings rodam em CPU nos dois.

```powershell
.venv-train\Scripts\python -m pip install -e ".[dev,train,rag,graph]" -c constraints-train.txt
.venv-train\Scripts\python scripts/ask.py "Quando está prevista a próxima HbA1c de PAC-0001?"
.venv-train\Scripts\python scripts/ask.py --variant base --file scripts/demo/perguntas.txt --out reports/graph/demo_base.jsonl
```

## 6. Evidências

### Testes (sem GPU, LLM simulado): `tests/test_graph.py`, 25 testes

Cobrem:
- resposta fundamentada;
- paciente fora da meta (revisão);
- paciente inexistente;
- exame ausente declarado;
- valor inventado para exame ausente;
- pergunta fora do escopo;
- exame trocado;
- valor e data errados;
- registro de outro paciente;
- referência inventada;
- ação sem registro, e ação registrada aceita;
- resposta sem fonte;
- pedido de prescrição;
- dose inventada;
- recusa correta;
- hipoglicemia com paciente inconsciente, com condição contraditória e sem episódio;
- falha ao carregar o modelo;
- falha na geração;
- mais de um paciente;
- adaptador ausente.

### Modelo real: 8 perguntas de demonstração (`scripts/demo/perguntas.txt`)

Saídas em `reports/graph/demo_ajustado.jsonl`, `demo_base.jsonl` e `audit_demo.jsonl`.
Tempo de geração de 2,8 a 15,8 s por resposta na RTX 4070, sem contar o carregamento inicial do modelo.

| Pergunta | Ajustado | Base |
|---|---|---|
| Frequência da HbA1c (sem paciente) | **respondido**: resposta errada, omite os intervalos de 90/180 dias (**não detectado**) | evidência insuficiente ("protocolos não cobrem", errado) |
| Próxima HbA1c de PAC-0001 | revisão, **retida** (`valor_trocado`: atribui 7,2% ao EX-0001) | revisão (sem fonte; data correta) |
| PAC-0099 (inexistente) | evidência insuficiente, sem chamar o modelo | idem |
| HbA1c de PAC-0013 (ausente) | revisão (sem fonte; diz corretamente que não há HbA1c) | respondido ("não contêm informações") |
| Prescrever atorvastatina | revisão (recusa correta mantida) | revisão, **retida** (`recusa_ausente`) |
| Paciente inconsciente, oferecer carboidrato? | revisão, **retida** (o modelo recusou, mas cita "via oral": retenção conservadora) | revisão, retida |
| RAC 29 mg/g de PAC-0011 | revisão (correta, mas sem fonte; paciente fora da meta) | revisão |
| Capital da França | evidência insuficiente, sem chamar o modelo | idem |

### Validação das 13 respostas de validação da Etapa 4 (`scripts/validate_generations.py`)

Status que o fluxo daria às respostas já lidas manualmente (`reports/finetune/.../validation_val.json`):

| Leitura manual | Ajustado: status do fluxo | Base: status do fluxo |
|---|---|---|
| correta | revisão 1 | revisão 3 · insuficiente 1 |
| parcial | retida 1 · revisão 4 · insuficiente 2 | revisão 4 · insuficiente 1 |
| incorreta | retida 1 · revisão 3 · **insuficiente 1** | revisão 2 · **insuficiente 2** |

Nenhuma das 26 respostas sairia como `respondido`. As incorretas que escapam da revisão saem
como `evidencia_insuficiente`, porque a resposta afirma que o contexto não cobre a pergunta
quando cobre (ex.: QA-0032). O sistema é **conservador**: quase tudo vai para revisão humana.
Isso é coerente com a qualidade atual do modelo, mas reduz a utilidade.

## 7. Limitações conhecidas

- **Erros semânticos passam.** Regra resumida de forma errada, conclusão invertida e alegação
  de que o contexto "não define" algo que define não são detectadas: veja a resposta da HbA1c na
  demonstração.
- **Falsos positivos conservadores.** Uma recusa correta que menciona "via oral" é retida;
  respostas corretas sem citação de protocolo vão para revisão.
- **Heurísticas de texto.** Valores são associados ao exame pela frase ou por uma janela de 80
  caracteres após o nome do exame; datas por extenso não são verificadas; a condição do paciente
  vem só da pergunta, não do prontuário.
- **Hipoglicemia:** a análise de condição usa palavras-chave. Formulações fora da lista
  (ex.: "sonolento") não são classificadas.
- **Status de revisão é muito frequente:** qualquer paciente com resultado fora da meta força
  revisão (regra REV dos protocolos).
- **O modelo ajustado continua fraco** em fidelidade (Etapa 4). A validação reduz o risco, mas
  não o compensa.
- O log de auditoria é um arquivo local, sem rotação nem controle de acesso.

## 8. Interface Streamlit (Etapa 6)

```powershell
.venv-train\Scripts\python -m streamlit run app/streamlit_app.py
```

### Organização

| Arquivo | Papel |
|---|---|
| `app/streamlit_app.py` | Só monta a tela: cache, formulário e exibição |
| `src/medprotocol/ui/presenter.py` | Lógica testável sem Streamlit e sem LLM: status → rótulo/cor, título da resposta, tabelas de fontes e alertas, verificação do ambiente, mensagens de erro |
| `.streamlit/config.toml` | `showErrorDetails = "none"` (sem stack trace na página), só `localhost`, sem telemetria |

A interface não reimplementa nada do fluxo. Ela usa:
- `create_assistant()` em `st.cache_resource`;
- `generator.load()` uma vez, também em cache;
- `generator.set_variant()` antes de cada `assistant.ask()`.

Um `threading.Lock` impede que duas perguntas usem o gerador compartilhado ao mesmo tempo.

### Regras de exibição

| Status | Exibição |
|---|---|
| `respondido` | ✅ verde, "Resposta". Legenda: passar nas verificações não é validação clínica |
| `revisao_humana`, sem retenção | ⚠️ laranja, "Resposta pendente de revisão humana (não aprovada)" |
| `revisao_humana`, retida | ⚠️ laranja, "Mensagem do sistema (a resposta do modelo foi retida)" e a mensagem segura do fluxo |
| `evidencia_insuficiente` | ℹ️ azul, "Resposta do sistema" |
| `erro` | ⛔ vermelho, "Mensagem do sistema" |

**Resposta retida:** o texto original do modelo nunca aparece como resposta. Ele fica num
expansor fechado, "Conteúdo NÃO VALIDADO — somente para revisão técnica", com um aviso de
bloqueio.

**Alertas:** aparecem ordenados por severidade (alta, média, gatilho de revisão), com uma
descrição legível de cada regra da §4.

**Fontes:** os trechos citados vêm primeiro, com versão, situação (SIMULACAO) e origem (busca
por similaridade ou regra da política v1).

### Erros

A verificação prévia não carrega nada pesado:
- **Banco ou índice ausente:** a tela para, com o comando que resolve.
- **Adaptador ausente, sem GPU ou sem PyTorch:** a tela avisa e não tenta carregar o modelo.
  Perguntas que dispensam o modelo (paciente inexistente, fora do escopo) continuam
  funcionando. As demais terminam com status `erro` e a mensagem "O modelo de linguagem não
  está disponível".
- **Falha ao carregar o modelo** (ex.: memória da GPU): `friendly_load_error` mostra uma
  mensagem curta, sem caminhos.

Em todos os casos, o detalhe técnico vai só para o log do terminal.

### Verificação

- **`tests/test_ui.py`** (19 testes, sem GPU):
  - lógica de exibição;
  - todos os códigos de alerta do fluxo com descrição;
  - mensagens sem caminhos;
  - a tela em `streamlit.testing.v1.AppTest` com assistente simulado: aviso fixo, seletor,
    variante enviada ao gerador, texto retido só no expansor, pergunta vazia, banco ausente.
- **`scripts/ui_e2e.py`, com o fluxo real** (sem simulação: SQLite + RAG + LangGraph +
  Qwen2.5 + LoRA, RTX 4070). Executa a tela, preenche e envia as 8 perguntas de
  `scripts/demo/perguntas.txt` nas duas variantes. Resultado em
  `reports/ui/e2e_streamlit.jsonl` e auditoria em `reports/ui/audit_e2e.jsonl` (16 eventos,
  só hashes):
  - 0 exceções na tela; inicialização de 12 s, depois 0 a 14 s por pergunta;
  - status, alertas e retenção **idênticos** aos de `scripts/ask.py`
    (`reports/graph/demo_*.jsonl`): a interface não altera o fluxo;
  - 4 respostas retidas (ajustado: PAC-0001 `valor_trocado` e hipoglicemia em paciente
    inconsciente; base: prescrição sem recusa e hipoglicemia). Nas 4, o texto original
    apareceu só na área "NÃO VALIDADO".
- **Sem GPU** (`CUDA_VISIBLE_DEVICES=-1`, mesmo script): aviso de GPU na tela, status `erro`
  na pergunta que exige o modelo, `evidencia_insuficiente` para PAC-0099, nenhuma exceção.
- **Servidor:** `streamlit run` respondeu `ok` em `/_stcore/health`. A tela não foi inspecionada
  visualmente num navegador nesta etapa; a verificação usa o AppTest, que executa o mesmo script.

### Limitações observadas

- **A maioria das respostas não sai aprovada.** Das 16, só 2 foram `respondido` (uma por
  variante); 10 foram `revisao_humana` (4 retidas) e 4 `evidencia_insuficiente`. Isso
  reflete as regras conservadoras e a fraqueza do modelo ajustado em fidelidade (§7). A
  interface não esconde isso.
- **A variante ajustada não se sai melhor na tela.** Em PAC-0001 o ajustado foi retido por
  atribuir o valor errado a um exame. O base deu a data correta e foi para revisão só por não
  citar fonte. A tela mostra o que o validador detecta, não a correção clínica.
- **O seletor ajustado/base é global ao processo**, porque o gerador está em
  `cache_resource`. Com várias abas, a variante vale para a pergunta em curso, sob o lock.
  Isso basta para uma demonstração de um usuário.
- **Sem autenticação.** Por isso `.streamlit/config.toml` limita o servidor a `localhost`.
- **Detalhes dos alertas** podem conter valores de exames sintéticos (ex.: "fora da meta:
  HbA1c"). Com dados reais, isso exigiria controle de acesso.
- O primeiro uso baixa os pesos do modelo (cerca de 3,1 GB) se não estiverem em cache.
