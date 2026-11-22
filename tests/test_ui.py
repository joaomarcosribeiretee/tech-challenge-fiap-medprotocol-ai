"""Interface (Etapa 6): lógica de exibição e tela Streamlit com assistente simulado (sem LLM, sem GPU)."""

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from medprotocol.graph.workflow import (  # noqa: E402
    ERRO,
    EVIDENCIA_INSUFICIENTE,
    MSG_RETIDA,
    RESPONDIDO,
    REVISAO_HUMANA,
    AssistantResponse,
)
from medprotocol.ui.presenter import (  # noqa: E402
    AVISO,
    BLOQUEANTE,
    alert_rows,
    build_view,
    check_environment,
    friendly_load_error,
    source_rows,
)

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def make_response(status=RESPONDIDO, resposta="HbA1c a cada 3 meses (PROT-DM2-001#S2).", resposta_modelo=None,
                  alertas=None, fontes=None, variante="ajustado") -> AssistantResponse:
    return AssistantResponse(
        audit_id="0000-test", pergunta="p", resposta=resposta, status=status, paciente="PAC-0001",
        fontes=fontes if fontes is not None else [
            {"ref": "PROT-DM2-001#S5", "origem": "regra:paciente", "versao": "1.0", "status_protocolo": "SIMULACAO",
             "citada": False, "protocolo_citado": True},
            {"ref": "PROT-DM2-001#S2", "origem": "similaridade", "versao": "1.0", "status_protocolo": "SIMULACAO",
             "citada": True, "protocolo_citado": True},
        ],
        registros_citados=[], alertas=alertas or [], status_recuperacao="ok",
        modelo={"variante": variante}, resposta_modelo=resposta_modelo, tempos_ms={"a": 1500.0, "b": 500.0},
    )


# --- presenter --------------------------------------------------------------------------------

def test_respondido_is_presented_as_answer():
    view = build_view(make_response())
    assert view.status.rotulo == "Respondido"
    assert view.titulo_resposta == "Resposta"
    assert not view.retida and view.texto_nao_validado is None
    assert view.tempo_total_s == 2.0
    assert view.modelo.startswith("Ajustado")


def test_withheld_answer_is_never_shown_as_answer():
    raw = "O EX-0001 tem HbA1c de 7,2%."
    view = build_view(make_response(REVISAO_HUMANA, MSG_RETIDA, resposta_modelo=raw,
                                    alertas=[{"codigo": "valor_trocado", "severidade": "alta", "detalhe": "x"}]))
    assert view.retida
    assert view.resposta == MSG_RETIDA and raw not in view.resposta
    assert view.texto_nao_validado == raw
    assert "retida" in view.titulo_resposta
    assert view.status.rotulo == "Requer revisão humana"


def test_review_without_withholding_is_marked_not_approved():
    view = build_view(make_response(REVISAO_HUMANA, alertas=[{"codigo": "sem_fonte_citada", "severidade": "media", "detalhe": ""}]))
    assert not view.retida
    assert "não aprovada" in view.titulo_resposta
    assert view.observacoes


@pytest.mark.parametrize("status,label", [(EVIDENCIA_INSUFICIENTE, "Evidência insuficiente"), (ERRO, "Erro"), ("xyz", "Status desconhecido")])
def test_status_labels(status, label):
    assert build_view(make_response(status)).status.rotulo == label


def test_sources_cited_first_with_readable_origin():
    rows = source_rows(make_response().fontes)
    assert [r["Trecho"] for r in rows] == ["PROT-DM2-001#S2", "PROT-DM2-001#S5"]
    assert rows[0]["Citado na resposta"] == "sim" and rows[1]["Citado na resposta"] == "protocolo"
    assert rows[0]["Como entrou no contexto"] == "Busca por similaridade"
    assert rows[1]["Como entrou no contexto"].startswith("Regra da política v1 (paciente")


def test_alerts_sorted_by_severity_with_description():
    rows = alert_rows([
        {"codigo": "resultado_fora_do_limite", "severidade": "baixa", "detalhe": "HbA1c"},
        {"codigo": "sem_fonte_citada", "severidade": "media", "detalhe": ""},
        {"codigo": "dose_na_resposta", "severidade": "alta", "detalhe": "20 mg"},
    ])
    assert [r["Código"] for r in rows] == ["dose_na_resposta", "sem_fonte_citada", "resultado_fora_do_limite"]
    assert rows[0]["Severidade"].startswith("Alta")
    assert rows[2]["Severidade"] == "Gatilho de revisão"
    assert all(r["Verificação"] != "—" for r in rows)


def test_every_validation_code_has_description():
    import re

    from medprotocol.graph import validation, workflow
    from medprotocol.ui.presenter import ALERT_DESCRIPTIONS

    codes = set()
    for module in (validation, workflow):
        codes |= set(re.findall(r'Alert\(\s*"(\w+)"', Path(module.__file__).read_text(encoding="utf-8")))
    assert codes and codes <= set(ALERT_DESCRIPTIONS), codes - set(ALERT_DESCRIPTIONS)


def test_check_environment(tmp_path):
    db, store, adapter = tmp_path / "x.db", tmp_path / "vs", tmp_path / "ad"
    problems = check_environment(db, store, adapter, cuda_available=False)
    assert sum(p.severidade == BLOQUEANTE for p in problems) == 2
    assert sum(p.severidade == AVISO for p in problems) == 2
    assert all(str(tmp_path) not in p.mensagem for p in problems)  # sem caminhos internos

    db.write_bytes(b""); store.mkdir(); (store / "index.json").write_text("{}"); adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    assert check_environment(db, store, adapter, cuda_available=True) == []
    assert [p.severidade for p in check_environment(db, store, None, None)] == [AVISO, AVISO]


@pytest.mark.parametrize("message,expected", [
    ("CUDA indisponível; o modelo local requer GPU (ver docs/ARCHITECTURE.md)", "GPU CUDA"),
    ("adaptador não encontrado em C:/x/models/adapters/run; rode scripts/train_lora.py", "Adaptador LoRA"),
    ("falha ao carregar o modelo: OutOfMemoryError: CUDA out of memory", "Memória da GPU"),
    ("falha ao carregar o modelo: OSError: C:/secret/path", "Não foi possível carregar"),
])
def test_friendly_load_error_hides_internals(message, expected):
    text = friendly_load_error(message)
    assert expected in text
    assert "C:/" not in text


# --- tela Streamlit (AppTest) com assistente simulado ------------------------------------------

class FakeGenerator:
    def __init__(self, adapter_dir):
        self.adapter_dir = adapter_dir
        self.variant = "ajustado"
        self.loads = 0

    def load(self):
        self.loads += 1

    def set_variant(self, variant):
        self.variant = variant


class FakeAssistant:
    def __init__(self, adapter_dir, response_factory):
        self.deps = type("Deps", (), {"generator": FakeGenerator(adapter_dir)})()
        self.response_factory = response_factory
        self.questions = []

    def ask(self, pergunta):
        self.questions.append((pergunta, self.deps.generator.variant))
        return self.response_factory(self.deps.generator.variant)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st

    import medprotocol.graph.workflow as workflow

    db, store, adapter = tmp_path / "x.db", tmp_path / "vs", tmp_path / "ad"
    db.write_bytes(b""); store.mkdir(); (store / "index.json").write_text("{}"); adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("SQLITE_PATH", str(db))
    monkeypatch.setenv("VECTORSTORE_DIR", str(store))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    holder = {}

    def run(response_factory):
        assistant = FakeAssistant(adapter, response_factory)
        holder["assistant"] = assistant
        monkeypatch.setattr(workflow, "create_assistant", lambda settings=None: assistant)
        st.cache_resource.clear()
        return testing.AppTest.from_file(str(APP), default_timeout=60), assistant

    yield run
    st.cache_resource.clear()


def all_text(at) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.warning]
    parts += [e.value for e in at.error] + [e.value for e in at.code]
    parts += [e.label for e in at.expander]
    return "\n".join(str(p) for p in parts)


def test_app_initial_screen_has_disclaimer_and_controls(app_env):
    at, _ = app_env(lambda v: make_response())
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("MedProtocol AI")
    assert any("sintéticos" in w.value and "não substitui" in w.value for w in at.warning)
    assert at.radio[0].options == [
        "Ajustado (Qwen2.5-1.5B + LoRA)", "Base (Qwen2.5-1.5B, sem adaptador)",
    ]
    assert len(at.text_area) == 1


def test_app_answers_with_selected_variant(app_env):
    at, assistant = app_env(lambda v: make_response(variante=v))
    at.run()
    at.radio[0].set_value("base")
    at.text_area[0].input("Com que frequência repetir a HbA1c?")
    at.button[0].click().run()
    assert not at.exception
    assert assistant.questions == [("Com que frequência repetir a HbA1c?", "base")]
    text = all_text(at)
    assert "Respondido" in text and "0000-test" in text and "Base (Qwen2.5-1.5B" in text
    assert len(at.dataframe) == 1  # fontes (sem alertas)


def test_app_withheld_answer_only_in_unvalidated_area(app_env):
    raw = "TEXTO-ORIGINAL-ERRADO"
    at, _ = app_env(lambda v: make_response(
        REVISAO_HUMANA, MSG_RETIDA, resposta_modelo=raw,
        alertas=[{"codigo": "valor_trocado", "severidade": "alta", "detalhe": "x"}]))
    at.run()
    at.text_area[0].input("Qual a HbA1c de PAC-0001?")
    at.button[0].click().run()
    assert not at.exception
    assert any("NÃO VALIDADO" in e.label for e in at.expander)
    assert [c.value for c in at.code] == [raw]
    assert all(raw not in m.value for m in at.markdown)  # nunca aparece como resposta
    assert any(MSG_RETIDA in m.value for m in at.markdown)
    assert len(at.dataframe) == 2  # alertas + fontes


def test_app_empty_question_does_not_call_assistant(app_env):
    at, assistant = app_env(lambda v: make_response())
    at.run()
    at.button[0].click().run()
    assert assistant.questions == []
    assert any("Digite uma pergunta" in i.value for i in at.info)


def test_app_blocks_when_database_missing(app_env, monkeypatch, tmp_path):
    at, assistant = app_env(lambda v: make_response())
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "nao_existe.db"))
    at.run()
    assert not at.exception
    assert any("Banco SQLite não encontrado" in e.value for e in at.error)
    assert len(at.text_area) == 0
