"""Fluxo LangGraph com LLM simulado (sem GPU): rotas, validação, status e auditoria.

A recuperação é roteirizada (trechos escolhidos por pergunta) para que cada teste controle o
contexto; o índice, a política de seções, o banco e a montagem das mensagens são os reais.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from medprotocol.db.importer import build_database  # noqa: E402
from medprotocol.graph.audit import AuditLogger  # noqa: E402
from medprotocol.graph.llm import GenerationError, LocalQwenGenerator, ModelLoadError  # noqa: E402
from medprotocol.graph.validation import analyze_question  # noqa: E402
from medprotocol.graph.workflow import (  # noqa: E402
    ERRO,
    EVIDENCIA_INSUFICIENTE,
    MSG_CONDICAO,
    MSG_CONDUTA,
    MSG_RETIDA,
    RESPONDIDO,
    REVISAO_HUMANA,
    Dependencies,
    MedProtocolAssistant,
)
from medprotocol.rag.index import build_index  # noqa: E402
from medprotocol.rag.retriever import (  # noqa: E402
    STATUS_NO_RELIABLE,
    STATUS_OK,
    ProtocolRetriever,
    RetrievalHit,
    RetrievalResult,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG = ROOT / "configs" / "finetune" / "qwen2.5-1.5b-lora-v1.json"


class ScriptedRetriever(ProtocolRetriever):
    """Índice real (embeddings falsos), mas trechos devolvidos escolhidos pelo teste."""

    def __init__(self, store, script: dict[str, tuple[str, list[str]]]):
        super().__init__(store)
        self.script = script

    def retrieve(self, pergunta, k=4):
        status, refs = self.script.get(pergunta, (STATUS_NO_RELIABLE, []))
        hits = []
        for position, ref in enumerate(refs):
            entry = next(e for e in self.store.store.values() if e["metadata"]["ref"] == ref)
            m = entry["metadata"]
            hits.append(RetrievalHit(ref, m["protocol_id"], m["section_id"], m["section_title"], m["versao"],
                                     m["status"], m["source"], m["chunk_index"], entry["text"], 0.9 - position / 100))
        return RetrievalResult(pergunta, status, hits[:k], self.lower_threshold, self.upper_threshold)


class FakeLLM:
    def __init__(self, answer: str = "", error: Exception | None = None):
        self.answer, self.error, self.calls = answer, error, []

    def generate(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.answer

    def describe(self):
        return {"variante": "simulado"}


@pytest.fixture(scope="module")
def store():
    return build_index(DATA_DIR / "protocols", DeterministicFakeEmbedding(size=32), lambda t: len(t.split()))


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "medprotocol.db"
    build_database(DATA_DIR, path)
    return path


@pytest.fixture
def ask(store, db_path, tmp_path):
    def run(pergunta, answer="", refs=("PROT-DM2-001#S3",), status=STATUS_OK, llm=None):
        llm = llm or FakeLLM(answer)
        retriever = ScriptedRetriever(store, {pergunta: (status, list(refs))})
        audit_path = tmp_path / "audit.jsonl"
        assistant = MedProtocolAssistant(Dependencies(retriever, db_path, llm, AuditLogger(audit_path)))
        result = assistant.ask(pergunta)
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        return result, llm, events

    return run


def codes(result):
    return {a["codigo"] for a in result.alertas}


# --- Respostas fundamentadas ---------------------------------------------------------------------

def test_grounded_answer_is_returned_with_sources_and_audit(ask):
    pergunta = "Quando está prevista a próxima HbA1c de PAC-0001?"
    answer = (
        "Pelo PROT-DM2-001 (seção S3), HbA1c menor ou igual a 7,0% repete em 180 dias. A HbA1c de "
        "PAC-0001 foi 6,5% em 10/03/2026 (EX-0001), e a próxima verificação está prevista para 06/09/2026."
    )
    result, llm, events = ask(pergunta, answer)
    assert result.status == RESPONDIDO, result.alertas
    assert result.resposta == answer and result.resposta_modelo is None
    assert result.paciente == "PAC-0001"
    assert result.registros_citados == ["EX-0001", "PAC-0001"]
    assert {"ref": "PROT-DM2-001#S3", "citada": True} .items() <= next(f for f in result.fontes if f["ref"] == "PROT-DM2-001#S3").items()
    assert any(f["origem"].startswith("regra:") for f in result.fontes)  # política v1 aplicada
    assert len(llm.calls) == 1 and [m["role"] for m in llm.calls[0]] == ["system", "user"]
    assert '"historico_exames"' in llm.calls[0][1]["content"]
    (event,) = events
    assert event["audit_id"] == result.audit_id and event["status"] == RESPONDIDO
    assert pergunta not in json.dumps(event, ensure_ascii=False) and answer not in json.dumps(event, ensure_ascii=False)
    assert event["pergunta"]["caracteres"] == len(pergunta)
    assert list(event["tempos_ms"]) == [
        "interpretar", "consultar_paciente", "recuperar_protocolos", "aplicar_politica", "gerar_resposta", "validar_resposta",
    ]


def test_out_of_limit_patient_always_goes_to_human_review(ask):
    answer = "Pelo PROT-DM2-001 (seção S3), a meta é 7,0% ou menos. A HbA1c de PAC-0011 foi 7,1% (EX-0057)."
    result, _, _ = ask("A HbA1c de PAC-0011 está na meta?", answer)
    assert result.status == REVISAO_HUMANA
    assert "resultado_fora_do_limite" in codes(result)
    assert result.resposta == answer  # nada bloqueante: a resposta é mantida, mas vai para revisão


# --- Paciente e dados ausentes --------------------------------------------------------------------

def test_unknown_patient_stops_before_generation(ask):
    result, llm, events = ask("Quais exames de PAC-0099 estão pendentes?")
    assert result.status == EVIDENCIA_INSUFICIENTE
    assert "paciente_inexistente" in codes(result)
    assert llm.calls == [] and events[0]["status"] == EVIDENCIA_INSUFICIENTE


def test_missing_exam_declared_as_missing_is_insufficient_evidence(ask):
    answer = (
        "Não há registro de HbA1c para PAC-0013, então não é possível calcular a próxima verificação "
        "(PROT-DM2-001, seção S4). O exame deve ser sinalizado como pendente."
    )
    result, _, _ = ask("Qual a HbA1c de PAC-0013?", answer, refs=("PROT-DM2-001#S3", "PROT-DM2-001#S4"))
    assert result.status == EVIDENCIA_INSUFICIENTE, result.alertas


def test_invented_value_for_missing_exam_is_blocked(ask):
    answer = "Pelo PROT-DM2-001 (seção S3), a HbA1c de PAC-0013 é 7,8%, acima da meta."
    result, _, _ = ask("Qual a HbA1c de PAC-0013?", answer)
    assert "valor_de_exame_ausente" in codes(result)
    assert result.status == REVISAO_HUMANA and result.resposta == MSG_RETIDA and result.resposta_modelo == answer


def test_no_reliable_evidence_without_patient_skips_the_model(ask):
    result, llm, _ = ask("Qual a capital da França?", refs=(), status=STATUS_NO_RELIABLE)
    assert result.status == EVIDENCIA_INSUFICIENTE and llm.calls == []


# --- Valor, exame, data e referência ------------------------------------------------------------

def test_swapped_exam_is_blocked(ask):
    answer = "Pelo PROT-DM2-001 (seção S3), para a creatinina, PAC-0003 tem EX-0014 com valor 8,4%."
    result, _, _ = ask("Qual a creatinina de PAC-0003?", answer)
    assert "exame_trocado" in codes(result) and result.resposta == MSG_RETIDA


def test_wrong_value_and_date_for_a_cited_exam_go_to_review(ask):
    answer = "Pelo PROT-DM2-001 (seção S3), a HbA1c de PAC-0001 foi 9,9% em 15/10/2025 (EX-0001)."
    result, _, _ = ask("Qual a última HbA1c de PAC-0001?", answer)
    assert {"valor_trocado", "data_nao_confere"} <= codes(result)
    assert result.status == REVISAO_HUMANA and result.resposta == MSG_RETIDA


def test_record_of_another_patient_is_blocked(ask):
    answer = "Pelo PROT-DM2-001 (seção S3), a última HbA1c foi 6,6% (EX-0008)."
    result, _, _ = ask("Qual a última HbA1c de PAC-0001?", answer)
    assert "id_inexistente" in codes(result) and result.resposta == MSG_RETIDA


def test_invented_reference_is_blocked(ask):
    answer = "Pelo PROT-DM2-003 (seção S5), o fundo de olho deve ser registrado a cada 365 dias."
    result, _, _ = ask("Com que frequência registrar o fundo de olho?", answer, refs=("PROT-DM2-001#S3",))
    assert "protocolo_fora_do_contexto" in codes(result) or "referencia_fora_do_contexto" in codes(result)
    assert result.resposta == MSG_RETIDA


def test_unregistered_action_claim_goes_to_review(ask):
    answer = "Pelo PROT-DM2-001 (seção REV), a HbA1c de 7,1% (EX-0057) já foi apresentada ao médico responsável."
    result, _, _ = ask("A HbA1c de PAC-0011 já foi levada ao médico?", answer, refs=("PROT-DM2-001#REV",))
    assert "acao_sem_registro" in codes(result) and result.status == REVISAO_HUMANA


def test_registered_action_is_accepted(ask):
    answer = "A consulta CON-0015 registra que a RAC (EX-0080) foi sinalizada ao médico responsável (PROT-DM2-003, seção S3)."
    result, _, _ = ask("A RAC de PAC-0015 já foi sinalizada?", answer, refs=("PROT-DM2-003#S3",))
    assert "acao_sem_registro" not in codes(result)


def test_answer_without_any_source_cannot_be_verified(ask):
    result, _, _ = ask("Qual a periodicidade do LDL?", "O LDL deve ser repetido a cada 365 dias.")
    assert "sem_fonte_citada" in codes(result) and result.status == REVISAO_HUMANA


# --- Conduta e prescrição -----------------------------------------------------------------------

def test_prescription_request_answered_with_conduct_is_replaced_by_refusal(ask):
    result, _, _ = ask("Prescreva atorvastatina 20 mg para PAC-0016.", "Inicie atorvastatina 20 mg ao dia para PAC-0016.",
                       refs=("PROT-DM2-001#REV",))
    assert "recusa_ausente" in codes(result) and "pedido_de_conduta" in codes(result)
    assert result.resposta == MSG_CONDUTA and result.status == REVISAO_HUMANA


def test_invented_dose_is_blocked_even_without_explicit_request(ask):
    answer = "Pelo PROT-DM2-002 (seção REV), use insulina NPH 10 UI e revise com a equipe médica."
    result, _, _ = ask("O que fazer depois da hipoglicemia de PAC-0014?", answer, refs=("PROT-DM2-002#REV",))
    assert "dose_na_resposta" in codes(result) and result.resposta == MSG_CONDUTA


def test_correct_refusal_is_kept_but_still_requires_review(ask):
    answer = "Não posso prescrever medicamentos. Pelo PROT-DM2-001 (seção REV), a decisão cabe ao médico responsável."
    result, _, _ = ask("Qual dose de insulina devo prescrever para PAC-0005?", answer, refs=("PROT-DM2-001#REV",))
    assert result.resposta == answer and result.status == REVISAO_HUMANA
    assert not {a["codigo"] for a in result.alertas if a["severidade"] == "alta"}


# --- Hipoglicemia e condição do paciente --------------------------------------------------------

def test_question_analysis_detects_conscious_state():
    assert analyze_question("paciente inconsciente com hipoglicemia").condicao == "incapaz"
    assert analyze_question("paciente consciente e capaz de deglutir").condicao == "capaz"
    assert analyze_question("está inconsciente mas consegue engolir").condicao == "contraditoria"
    assert analyze_question("não consegue deglutir, glicemia 50 mg/dL").condicao == "incapaz"


@pytest.mark.parametrize("pergunta", [
    "Paciente internado inconsciente com hipoglicemia de 50 mg/dL: posso oferecer carboidrato?",
    "Paciente internado está inconsciente mas consegue engolir, glicemia de 50 mg/dL. Qual a conduta?",
])
def test_conscious_patient_conduct_is_never_applied_to_incapable_or_contradictory_condition(ask, pergunta):
    answer = "Pelo PROT-DM2-002 (seção S3), ofereça 15 g de carboidrato de absorção rápida por via oral."
    result, _, _ = ask(pergunta, answer, refs=("PROT-DM2-002#S3", "PROT-DM2-002#S2"))
    assert "conduta_incompativel_com_condicao" in codes(result)
    assert "hipoglicemia_condicao_critica" in codes(result)
    assert result.resposta == MSG_CONDICAO and result.status == REVISAO_HUMANA


def test_conscious_patient_conduct_is_allowed_when_condition_matches(ask):
    answer = "Pelo PROT-DM2-002 (seção S3), para paciente consciente e capaz de deglutir, ofereça 15 g de carboidrato por via oral."
    result, _, _ = ask("Paciente internado consciente e capaz de deglutir com hipoglicemia: qual a conduta inicial?",
                       answer, refs=("PROT-DM2-002#S3",))
    assert "conduta_incompativel_com_condicao" not in codes(result)


# --- Falhas -------------------------------------------------------------------------------------

def test_model_load_failure_returns_error_and_is_audited(ask):
    llm = FakeLLM(error=ModelLoadError("adaptador não encontrado"))
    result, _, events = ask("Qual a periodicidade do LDL?", llm=llm)
    assert result.status == ERRO and "não está disponível" in result.resposta
    assert events[0]["status"] == ERRO and "adaptador" in events[0]["erro"]


def test_generation_failure_returns_error(ask):
    result, _, _ = ask("Qual a periodicidade do LDL?", llm=FakeLLM(error=GenerationError("CUDA OOM")))
    assert result.status == ERRO


def test_more_than_one_patient_is_rejected(ask):
    result, llm, _ = ask("Compare PAC-0001 e PAC-0002.")
    assert result.status == ERRO and llm.calls == []


def test_local_generator_reports_missing_adapter_as_load_error(tmp_path):
    generator = LocalQwenGenerator(CONFIG, adapter_dir=tmp_path / "inexistente")
    with pytest.raises(ModelLoadError):
        generator.generate([{"role": "user", "content": "x"}])
    with pytest.raises(ValueError):
        generator.set_variant("outro")


def test_hypoglycemia_conduct_is_blocked_when_question_says_there_was_no_episode(ask):
    pergunta = "Com que frequência devo medir a glicemia capilar de um diabético internado que não teve hipoglicemia?"
    answer = "Pelo PROT-DM2-002 (seção S3), meça após 15 minutos da oferta de 15 g de carboidrato."
    result, _, _ = ask(pergunta, answer, refs=("PROT-DM2-002#S3",))
    assert "conduta_incompativel_com_condicao" in codes(result) and result.status == REVISAO_HUMANA
