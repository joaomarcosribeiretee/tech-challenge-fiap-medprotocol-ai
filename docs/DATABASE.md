# Banco de dados estruturado (SQLite)

Os pacientes, consultas e exames sintéticos de `data/synthetic/` são importados para
`data/processed/medprotocol.db`. Esse arquivo é **gerado e não versionado**: está ignorado
em `.gitignore` por `data/processed/` e `*.db`.

## Criar o banco

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/build_database.py                 # usa DATA_DIR e SQLITE_PATH do ambiente
python scripts/build_database.py --db-path outro.db
```

**O que o script faz, na ordem:**
1. Executa o validador do dataset completo (`validate_dataset`). Se houver qualquer erro, interrompe sem gerar nem substituir o banco.
2. Cria o banco num arquivo `*.tmp`, numa única transação, com as chaves estrangeiras ativas.
3. Substitui o banco de destino de forma atômica (`os.replace`).

**Garantias:**
- Rodar de novo recria o banco do zero com o mesmo conteúdo. Não há timestamps.
- A tabela `build_metadata` guarda `schema_version`, `data_referencia` e o SHA-256 de cada CSV de origem, para saber exatamente de quais arquivos o banco foi gerado.

## Schema (versão 1)

Definido em `src/medprotocol/db/schema.py`.

| Tabela | Chave primária | Chaves estrangeiras | Índices |
|---|---|---|---|
| `patients` | `patient_id` | — | — |
| `consultations` | `consultation_id` | `patient_id → patients` | `(patient_id, data)` |
| `exams` | `exam_id` | `patient_id → patients` | `(patient_id, tipo, data_coleta)` |
| `build_metadata` | `key` | — | — |

As tabelas usam `STRICT`, o que rejeita, por exemplo, texto numa coluna `REAL`. Além disso, há `CHECK` para:
- formato dos IDs;
- datas ISO válidas (`2026-02-30` é rejeitada);
- enumerações (`sexo`, `tipo` de consulta);
- `ano_diagnostico_dm2 >= ano_nascimento`.

Esse conjunto funciona como segunda barreira, independente do validador.

Campos de lista (`comorbidades`, `medicamentos_em_uso`, `alergias`) ficam como texto `a;b`,
igual ao CSV. A camada de leitura devolve tuplas, e `nenhuma` vira uma tupla vazia.

## Consultar

Linha de comando: resumo em JSON. O script aceita apenas o ID do paciente, nunca SQL.

```powershell
python scripts/query_patient.py PAC-0001
```

Em Python (`src/medprotocol/db/repository.py`):

```python
from medprotocol.db.repository import (
    connect_readonly, get_patient, list_consultations, list_exams,
    get_latest_exam, get_patient_summary,
)

with connect_readonly("data/processed/medprotocol.db") as conn:
    patient = get_patient(conn, "PAC-0003")                # Patient
    consultas = list_consultations(conn, "PAC-0003")       # [Consultation], por data
    exames = list_exams(conn, "PAC-0003", tipo="HBA1C")    # [Exam], por data; tipo é opcional
    ultimo = get_latest_exam(conn, "PAC-0003", "LDL")      # Exam ou None
    resumo = get_patient_summary(conn, "PAC-0003")         # dict serializável em JSON
```

**Chaves de `get_patient_summary`:**
- `paciente`, `data_referencia`, `consultas`, `ultimos_exames`.
- `verificacoes`: mesmo formato de `verificacoes_esperadas` do dataset, calculado por `medprotocol.data.schedule.verification`. Nenhuma regra de prazo é duplicada aqui.
- `pendencias`: exames vencidos ou ausentes.
- `fora_do_limite`: exames acima ou abaixo do limite simulado.
- `inconsistencias`: avisos sobre os registros (ver a tabela abaixo).

## Erros e casos especiais

| Situação | Comportamento |
|---|---|
| Banco inexistente | `DatabaseNotFoundError` (o modo somente leitura não cria um arquivo vazio) |
| Arquivo que não é do projeto, ou de outra versão do schema | `SchemaVersionError` |
| `patient_id` malformado (ex.: `"PAC-0001' OR '1'='1"`) | `ValueError` **antes** de qualquer SQL |
| `tipo` de exame fora da lista conhecida | `ValueError` |
| Paciente inexistente | `PatientNotFoundError` em **todas** as funções. Lista vazia significa "paciente sem registros" |
| Exame ausente | `get_latest_exam` devolve `None`; no resumo, status `ausente` |
| Exame com data posterior à data de referência | Ignorado nas verificações e listado em `inconsistencias` |
| Dois resultados do mesmo exame monitorado no mesmo dia | Prevalece o de maior `exam_id` (mesma regra do `schedule`), listado em `inconsistencias` |
| Tentativa de escrita pela conexão de leitura | `sqlite3.OperationalError` (banco aberto com `mode=ro`) |
| CSV inconsistente na importação | `DatasetImportError` com a lista de erros do validador; banco anterior preservado |

## Segurança

- Todas as consultas são constantes do módulo, com parâmetros vinculados (`?`). Nenhuma função recebe SQL nem nomes de tabela ou coluna vindos de fora.
- A leitura (`repository.py`) e a escrita (`importer.py`) ficam em módulos separados. A leitura usa conexão somente leitura.

## Limitações

- Um único usuário local, sem controle de acesso. O banco contém só dados sintéticos.
- As listas não são normalizadas. Consultas do tipo "todos os pacientes que usam X" exigiriam tabelas próprias.
- Os exames têm só data, sem hora. Duas medidas no mesmo dia, como as glicemias capilares de PAC-0006, ficam ordenadas por `exam_id`.
- Os protocolos não estão no banco. São documentos para o RAG, e `protocol_id` é verificado apenas por formato, aqui, e por existência, pelo validador.
- No Windows, `os.replace` falha se o banco de destino estiver aberto por outro processo (ex.: um visualizador de SQLite).
