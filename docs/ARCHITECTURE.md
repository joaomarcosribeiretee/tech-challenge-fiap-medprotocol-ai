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

## 8. Próximo passo: interface Streamlit

A interface só precisa chamar `create_assistant()` uma vez (cache do Streamlit) e
`assistant.ask(pergunta)`. Ela deve exibir:
- status (cor por status);
- resposta e, quando retida, o texto original do modelo num bloco "para o revisor";
- fontes com origem e marca de citada;
- alertas;
- ID de auditoria;
- o aviso fixo;
- um seletor ajustado/base que chama `generator.set_variant()`, sem recarregar o modelo.

Executar no `.venv-train`.
