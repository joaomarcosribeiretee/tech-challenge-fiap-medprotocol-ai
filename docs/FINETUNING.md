# Etapa 4 — Fine-tuning (LoRA) e comparação base × ajustado

> **Resumo honesto:** o fine-tuning foi executado de verdade na GPU local, e a perda nas
> respostas de validação caiu de 1,299 para 0,882. Na leitura das 13 respostas de validação,
> porém, o modelo ajustado **não ficou mais correto** que o base: ele aprendeu o formato e o tom
> (respostas curtas, recusa explícita, encaminhamento ao médico), mas continua errando fatos e
> passou a afirmar ações que não estão registradas. Perda menor **não** é melhoria clínica.
> Nada aqui foi revisado por profissional de saúde.

## 1. Hardware e ambiente

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, 12 GB (≈ 11,6 GB livres; driver 610.88, CUDA 13.3) |
| RAM | 31 GB |
| Disco livre | ≈ 157 GB |
| SO | Windows 11 Pro |
| Python | 3.14.4 (único instalado) |

**Dois ambientes:**
- **`.venv` (RAG, Etapas 2–3):** inalterado, com torch 2.15.0 **CPU**. Os limiares do retriever
  foram calibrados nele.
- **`.venv-train` (Etapa 4):** torch 2.14.1+cu130, transformers 5.21.1, peft 0.21.3. Ver
  `constraints-train.txt`.

**Não foi preciso Python 3.11/3.12:** o PyTorch publica rodas CUDA para Python 3.14 no Windows.
Um ambiente separado basta para não trocar o torch do RAG.

```powershell
py -3.14 -m venv .venv-train
.venv-train\Scripts\python -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-train\Scripts\python -m pip install -e ".[dev,train]" -c constraints-train.txt
```

## 2. Escolha do modelo

Licença, acesso e tamanho conferidos no Hugging Face Hub em 14/11/2026:

| Modelo | Licença | Acesso | Pesos | Avaliação |
|---|---|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** | Apache-2.0 | livre | 3,1 GB | **Escolhido.** Multilíngue com bom português, template de chat estável, contexto de 32k |
| Qwen/Qwen3-1.7B | Apache-2.0 | livre | 4,1 GB | Alternativa viável; o modo de "raciocínio" do template acrescenta complexidade sem necessidade aqui |
| Qwen/Qwen2.5-3B-Instruct | "other" (Qwen Research) | livre | 6,2 GB | Descartado: licença restritiva |
| meta-llama/Llama-3.2-1B-Instruct | Llama 3.2 Community | **gated** | 2,5 GB | Descartado: exige aceite e autenticação |
| google/gemma-2-2b-it | Gemma Terms | **gated** | 5,2 GB | Descartado: exige aceite e autenticação |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | Apache-2.0 | livre | 3,4 GB | Descartado: treinado sobretudo em inglês |
| ibm-granite/granite-3.3-2b-instruct | Apache-2.0 | livre | 5,1 GB | Alternativa; maior e sem vantagem clara em português |

- **Revisão fixada:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- **LoRA em bf16, não QLoRA:** o modelo cabe com folga (pico de 4,9 GB). QLoRA exigiria
  `bitsandbytes`, que é frágil no Windows, sem ganho real.

## 3. Dados: split treino/validação

`scripts/split_sft_dataset.py` gera `data/finetune/split_v1.json`. O arquivo é versionado e
contém só IDs. Módulo: `src/medprotocol/finetuning/split.py`.

**Prioridades, em ordem:**
1. Sem paciente nos dois lados.
2. Sem quase-duplicatas entre os lados: similaridade de pergunta ou resposta ≥ 0,6, com a função do validador.
3. Categorias distribuídas.

| | Treino | Validação |
|---|---|---|
| Exemplos | **48** | **13** |
| Pacientes | 11 | 3 (PAC-0002, PAC-0003, PAC-0011) |
| consulta_protocolo | 9 | 1 |
| dados_ausentes | 6 | 3 |
| contextualizada | 8 | 2 |
| revisao_humana | 9 | 1 |
| recusa | 8 | 2 |
| contexto_insuficiente | 4 | 2 |
| selecao_de_trecho | 4 | 2 |

**Como a separação é feita:**
- Busca exaustiva das combinações de 3 pacientes.
- Um exemplo sem paciente por categoria, sorteado com a semente 20260630.
- Maior similaridade entre os lados: 0,562 (QA-0014 × QA-0055, ambos pedidos de estimar valor). Nenhum par ficou acima do limiar.
- O `qa_eval.jsonl`, PAC-0009/PAC-0010 e as seções reservadas não entram (`sources.py` rejeita).

**Limitações do split:**
- 13 exemplos de validação dão uma medida muito ruidosa: 1 exemplo vale 7,7 pontos percentuais.
- A similaridade é lexical. Sobreposições de tema passam, como hipoglicemia em QA-0071 (validação) e QA-0031 (treino).
- A validação foi usada para escolher a época, então não é independente para esse hiperparâmetro.

### Correção encontrada antes do treino: exames fora da entrada

Ao calibrar as métricas nas respostas de referência, **10 de 61 respostas citavam exames que o
modelo não via**. Exemplo: a TFG anterior EX-0023 em QA-0009. A causa é que o JSON do paciente
trazia só o último exame de cada tipo. Treinar assim ensinaria o modelo a inventar registros.

**Correções:**
- `get_patient_summary` passou a incluir `historico_exames`: todos os exames até a data de referência, em ordem de coleta.
- `PROMPT_VERSION` passou de "2" para "3".
- `build_sft_dataset.py --strict` agora também falha se uma resposta citar exame, consulta ou paciente ausente da entrada. Resultado: **0/61**.

A política de seções (v1) não mudou: os 41 contextos do piloto continuam idênticos aos pré-registrados (0 divergências).

## 4. Treino

`scripts/train_lora.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json`

Módulos:
- `chat_format.py`: template do próprio tokenizer e máscara da perda.
- `lora_trainer.py`: laço explícito.
- `train_config.py`: configuração JSON.

| Hiperparâmetro | Valor |
|---|---|
| LoRA | r = 16, alpha = 32, dropout = 0,05; q/k/v/o + gate/up/down |
| Parâmetros treináveis | 18,5 M de 1,56 B (1,18%) |
| Épocas | 3 (18 passos de otimização) |
| Lote efetivo | 1 × acumulação 8 |
| Taxa de aprendizado | 2e-4, aquecimento 10% e decaimento linear |
| Otimizador | AdamW, sem weight decay, recorte de gradiente 1,0 |
| Precisão | bf16, gradient checkpointing |
| Comprimento | 1.151 a 4.602 tokens por exemplo (limite 6.144; nada é truncado) |
| Semente | 42 |

**Como o treino trata os dados:**
- **Perda só na resposta do assistente.** O teste `test_loss_with_logits_to_keep_equals_masked_full_loss` confere que ela é igual à perda completa com rótulos −100.
- **Prompt como prefixo.** Os tokens do prompt de inferência precisam ser exatamente o prefixo da conversa de treino; caso contrário, o exemplo é rejeitado.
- **Melhor época.** O adaptador salvo é o da época com menor perda de validação.

**Execução real (14/11/2026):**

| Medida | Valor |
|---|---|
| Smoke test | 2 passos com os 4 exemplos mais longos: ok, 19,5 s, pico de 4,91 GB |
| Treino completo | 214 s |
| Pico de memória da GPU | 4,91 GB |
| Perda de validação | 1,299 (antes) → 0,977 → 0,901 → **0,882** (época 3, a escolhida) |
| Perda de treino | 1,38 (passo 1) → 0,66 (passo 18) |

Saídas:
- `models/adapters/qwen2.5-1.5b-lora-v1/`: 71 MB, **não versionado**.
- `reports/finetune/qwen2.5-1.5b-lora-v1/train_report.json`: versionado.

## 5. Comparação base × ajustado

`scripts/compare_models.py --config ...`. `--rescore` recalcula as métricas sem GPU.

**Condições justas:**
- Mesmo modelo carregado: o adaptador é desligado para o "base" e ligado para o "ajustado".
- Mesmas mensagens system + user, com o mesmo contexto RAG + SQLite do arquivo SFT.
- Mesmo template e decodificação gulosa, com limite de 512 tokens.
- Os 13 exemplos de validação, nunca vistos no treino.

### Métricas heurísticas (`answer_metrics.py`)

| Métrica (13 exemplos) | Base | Ajustado | Referência |
|---|---|---|---|
| Perda nas respostas de referência | 1,299 | 0,882 | — |
| Referência PROTOCOLO#SEÇÃO fora do contexto | 0 | 0 | 0 |
| ID de exame/consulta inventado | 0 | 0 | 0 |
| Data fora da entrada | 1 | **2** | 0 |
| Dose inventada | 0 | 0 | 0 |
| Recusa quando devida | 1/2 | 2/2 | 2/2 |
| Recusa indevida ("não posso" sem pedido proibido) | 3 | 0 | 0 |
| Encaminhamento quando devido | 1/4 | 2/4 | 4/4 |
| Insuficiência declarada (`contexto_insuficiente`) | 0/2 | 1/2 | 2/2 |
| Recall de protocolos esperados | 0,08 | 0,28 | 0,75 |
| Respostas cortadas em 512 tokens | 1 | 0 | — |

A coluna "Referência" calibra as heurísticas: as respostas de referência dos 61 exemplos não
violam nenhuma delas (teste `test_reference_answers_never_cite_outside_their_input`).

### Leitura das respostas (`manual_review_val.json`)

Revisão técnica feita com assistência de IA. **Não é revisão clínica.**

| | Correta | Parcial | Incorreta |
|---|---|---|---|
| Base | 4 | 5 | 4 |
| Ajustado | 1 | 7 | 5 |

**O que o ajustado aprendeu:**
- Respostas curtas e em português direto.
- Recusar com "Não posso…" e encaminhar ao médico responsável.
- Não repetir o "os protocolos não cobrem completamente" do base, que aparecia até quando a seção estava no contexto.

**Onde piorou ou não melhorou:**
- Troca dados entre exames: QA-0006 trata a RAC como creatinina; QA-0014 atribui à creatinina o EX-0014, que é uma HbA1c.
- Confunde trechos parecidos: QA-0063 aplica a conduta de hipoglicemia a paciente sem hipoglicemia.
- Afirma ações não registradas, como "o resultado foi apresentado ao médico" em QA-0074 e QA-0076.
- Cita fontes externas: OMS e uma sociedade médica em QA-0056.
- Inventa datas: QA-0006 e QA-0014.

**O base também erra muito:** inventa um intervalo de 3 anos (QA-0045), diz que a HbA1c de
8,4% está "dentro da meta" (QA-0074) e sugere aceitar 7,1% como dentro da meta (QA-0076).

**Conclusão:** com 48 exemplos e 3 épocas, o LoRA moldou **estilo e comportamento de
superfície**, não a fidelidade aos dados. Nenhum dos dois modelos é confiável sem a camada de
validação prevista para o LangGraph.

## 6. Limitações

- **Dados:** 48 exemplos de treino, autoria única e estilo homogêneo.
  - 14 das 61 respostas de referência não citam nenhum protocolo e 35 não citam seção de forma reconhecível, embora o prompt de sistema peça citação PROTOCOLO#SEÇÃO.
  - O modelo aprende esse estilo: o recall de seções das próprias referências é 0,37 (0,40 na validação).
- **Validação pequena (13)** e usada para escolher a época.
- **Métricas heurísticas:** detectam citação, ID, data e dose fora da entrada, além de marcadores de recusa e encaminhamento. **Não detectam** troca de valores entre exames nem conclusões erradas; para isso foi necessária a leitura manual.
- **Uma única execução** (semente 42), sem variância entre sementes nem busca de hiperparâmetros.
- **A avaliação final (`qa_eval.jsonl`) não foi usada** nesta etapa, nem para escolher modelo nem hiperparâmetros.

## 7. Próximos passos sugeridos

1. **Mais dados de treino** em lote pré-registrado (ver `docs/FINETUNING_DATA.md`), cobrindo os erros vistos: troca de exames, conduta aplicada fora da condição e afirmação de ações não registradas.
2. **Padronizar as respostas de referência** para citar PROTOCOLO#SEÇÃO, como o prompt de sistema pede.
3. **Variações de treino comparadas pela perda e pela leitura da validação:** 1–2 épocas, taxa de aprendizado menor e r = 8. Escolher uma só, para não superajustar a validação de 13 exemplos.
4. **No LangGraph:** um nó de validação que confira IDs, datas e citações contra o contexto (as mesmas funções de `answer_metrics.py`) e encaminhe para revisão humana.
5. **Avaliação final** no `qa_eval.jsonl`, uma única vez, com o modelo já escolhido.
