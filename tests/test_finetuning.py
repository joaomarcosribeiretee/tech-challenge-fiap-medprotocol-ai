"""Formato contextualizado para fine-tuning (embeddings falsos determinísticos; sem modelo real).

Nada aqui abre o qa_eval.jsonl: os exemplos vêm só de finetune/qa_train.jsonl.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="extra [rag] não instalado")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from medprotocol.data.validation import EVAL_ONLY_PATIENTS  # noqa: E402
from medprotocol.db.importer import build_database  # noqa: E402
from medprotocol.db.repository import PatientNotFoundError, connect_readonly  # noqa: E402
from medprotocol.finetuning.context import build_context, retrieve_excluding  # noqa: E402
from medprotocol.finetuning.messages import SYSTEM_PROMPT, build_messages, render_user_message  # noqa: E402
from medprotocol.finetuning.sft import build_sft_record, build_sft_records, coverage  # noqa: E402
from medprotocol.finetuning.sources import (  # noqa: E402
    RESERVED_SECTIONS,
    TRAIN_FILE,
    ReservedDataError,
    load_training_examples,
    to_example,
)
from medprotocol.rag.index import build_index  # noqa: E402
from medprotocol.rag.retriever import MAX_K, STATUS_NO_RELIABLE, ProtocolRetriever  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABEL_KEYS = ("categoria", "protocolos_citados", "evidencias", "requer_revisao_humana", "deve_recusar", "verificacoes_esperadas")


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def store():
    return build_index(DATA_DIR / "protocols", DeterministicFakeEmbedding(size=32), words)


@pytest.fixture(scope="module")
def retriever(store):
    return ProtocolRetriever(store)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "medprotocol.db"
    build_database(DATA_DIR, path)
    with connect_readonly(path) as connection:
        yield connection


@pytest.fixture(scope="module")
def examples():
    return load_training_examples(DATA_DIR)


@pytest.fixture(scope="module")
def records(examples, retriever, conn):
    # k=MAX_K: todos os trechos não reservados entram, o pior caso para vazamento de seções.
    return build_sft_records(examples, retriever, conn, k=MAX_K)


def raw_record(qa_id: str) -> dict:
    lines = (DATA_DIR / TRAIN_FILE).read_text(encoding="utf-8").splitlines()
    return next(json.loads(line) for line in lines if f'"{qa_id}"' in line)


# --- Fonte dos exemplos ------------------------------------------------------------------------


def test_loads_only_training_file(tmp_path):
    """Funciona com um diretório que não tem eval/: o qa_eval nunca é necessário."""
    (tmp_path / "finetune").mkdir()
    shutil.copy(DATA_DIR / TRAIN_FILE, tmp_path / TRAIN_FILE)
    assert len(load_training_examples(tmp_path)) == 20
    assert not (tmp_path / "eval").exists()


def test_reserved_patient_is_rejected():
    record = raw_record("QA-0011") | {"pergunta": "Quando está prevista a próxima HbA1c de PAC-0009?", "patient_id": "PAC-0009"}
    with pytest.raises(ReservedDataError, match="PAC-0009"):
        to_example(record)


def test_reserved_section_is_rejected():
    record = raw_record("QA-0002")
    record["protocolos_citados"] = record["protocolos_citados"] + ["PROT-DM2-002#S4"]
    with pytest.raises(ReservedDataError, match="PROT-DM2-002#S4"):
        to_example(record)


def test_patient_must_be_named_in_question():
    record = raw_record("QA-0011") | {"pergunta": "Quando está prevista a próxima HbA1c?"}
    with pytest.raises(ValueError, match="não aparece na pergunta"):
        to_example(record)


def test_no_training_example_depends_on_reserved_data(examples):
    for example in examples:
        assert example.patient_id not in EVAL_ONLY_PATIENTS
        assert not set(example.rotulos["protocolos_citados"]) & RESERVED_SECTIONS


# --- Formato das mensagens ---------------------------------------------------------------------


def test_training_messages_have_system_user_assistant(records, examples):
    for record, example in zip(records, examples):
        assert [m["role"] for m in record.messages] == ["system", "user", "assistant"]
        assert record.messages[0]["content"] == SYSTEM_PROMPT
        assert record.messages[2]["content"] == example.resposta_esperada
        user = record.messages[1]["content"]
        assert user.startswith("Data de referência: 2026-06-30")
        assert example.pergunta in user
        assert f"status da recuperação: {record.metadata['status_recuperacao']}" in user
        json.dumps(record.to_json(), ensure_ascii=False)  # serializável


def test_inference_messages_have_no_assistant_turn(retriever, conn):
    context = build_context("Quando está prevista a próxima HbA1c de PAC-0001?", retriever, conn, "PAC-0001")
    assert [m["role"] for m in build_messages(context)] == ["system", "user"]


def test_empty_answer_is_rejected(retriever):
    context = build_context("Qual é a periodicidade da HbA1c?", retriever, None)
    with pytest.raises(ValueError):
        build_messages(context, "   ")


# --- Referências e rastreabilidade -------------------------------------------------------------


def test_each_retrieved_chunk_is_shown_with_its_reference_and_literal_text(records, store):
    texts = {entry["metadata"]["ref"]: entry["text"] for entry in store.store.values()}
    for record in records:
        user = record.messages[1]["content"]
        for position, ref in enumerate(record.metadata["refs_recuperadas"], start=1):
            assert f"[{position}] {ref} |" in user
            assert texts[ref] in user
            assert "SIMULACAO" in user


def test_scores_are_not_shown_to_the_model(retriever):
    context = build_context("Qual é a periodicidade da HbA1c?", retriever, None, k=3)
    user = render_user_message(context)
    for hit in context.recuperacao.hits:
        assert f"{hit.score:.4f}" not in user and "score" not in user


@pytest.mark.parametrize(
    "cited, retrieved, by_rule, level, basis, missing",
    [
        (["A", "B"], ["A"], ["B"], "completa", "similaridade+regras", []),
        (["A"], ["A"], ["A"], "completa", "regras", []),  # recuperada e também exigida por regra
        (["A"], ["A"], [], "completa", "similaridade", []),
        (["A", "B"], ["A"], [], "parcial", "similaridade", ["B"]),
        (["A"], ["X"], ["Y"], "insuficiente", "nenhuma", ["A"]),
    ],
)
def test_coverage_levels(cited, retrieved, by_rule, level, basis, missing):
    assert coverage(cited, retrieved, by_rule) == {
        "cobertura": level,
        "fundamentacao": basis,
        "refs_citadas_fora_do_contexto": missing,
    }


# --- Isolamento entre entrada e resposta -------------------------------------------------------


def test_input_never_contains_answer_or_labels(records, examples):
    for record, example in zip(records, examples):
        prompt = record.messages[0]["content"] + record.messages[1]["content"]
        assert example.resposta_esperada not in prompt
        assert example.qa_id not in prompt
        for key in LABEL_KEYS:
            assert key not in prompt, (example.qa_id, key)


def test_retrieval_query_does_not_depend_on_answer(examples, retriever, conn):
    example = next(e for e in examples if e.qa_id == "QA-0012")
    altered = type(example)(example.qa_id, example.pergunta, example.patient_id, "Outra resposta.", example.rotulos)
    a = build_sft_record(example, retriever, conn).messages[:2]
    b = build_sft_record(altered, retriever, conn).messages[:2]
    assert a == b


# --- Dados reservados --------------------------------------------------------------------------


def test_reserved_sections_never_enter_the_input_even_as_distractors(records, store):
    reserved_texts = [e["text"] for e in store.store.values() if e["metadata"]["ref"] in RESERVED_SECTIONS]
    assert len(reserved_texts) == 2
    for record in records:
        user = record.messages[1]["content"]
        assert len(record.metadata["refs_recuperadas"]) == MAX_K - 2  # 20 trechos menos os 2 reservados
        assert not set(record.metadata["refs_recuperadas"]) & RESERVED_SECTIONS
        assert not any(text in user for text in reserved_texts)


def test_reserved_patients_never_enter_the_input(records):
    for record in records:
        content = json.dumps(record.messages, ensure_ascii=False)
        assert not any(patient in content for patient in EVAL_ONLY_PATIENTS)


def test_exclusion_recomputes_status_without_reserved_chunks(retriever):
    all_refs = frozenset(e["metadata"]["ref"] for e in retriever.store.store.values())
    result = retrieve_excluding(retriever, "Qual é a periodicidade da HbA1c?", k=4, excluded_refs=all_refs)
    assert result.hits == [] and result.status == STATUS_NO_RELIABLE


# --- Falta de dados e de resultados confiáveis -------------------------------------------------


def test_absent_exams_are_explicit_and_never_filled(retriever, conn):
    context = build_context("Estime um valor de creatinina para PAC-0003.", retriever, conn, "PAC-0003")
    user = render_user_message(context)
    creatinina = next(v for v in context.paciente["verificacoes"] if v["exame"] == "CREATININA")
    assert creatinina == {
        "exame": "CREATININA",
        "status": "ausente",
        "ultimo_exame": None,
        "data_ultimo": None,
        "proxima_verificacao": None,
        "fora_do_limite": None,
    }
    assert "CREATININA" in context.paciente["pendencias"]
    assert "CREATININA" not in context.paciente["ultimos_exames"]
    assert '"status": "ausente"' in user


def test_unreliable_retrieval_is_flagged_in_the_input(store):
    strict = ProtocolRetriever(store, lower_threshold=2.0, upper_threshold=2.0)  # nenhum score alcança
    user = render_user_message(build_context("Qual é a capital da Austrália?", strict, None))
    assert "status da recuperação: sem_resultado_confiavel" in user
    assert "podem não ter relação com a pergunta" in user


def test_no_chunks_and_no_patient_are_stated_explicitly(retriever):
    all_refs = frozenset(e["metadata"]["ref"] for e in retriever.store.store.values())
    context = build_context("Qual é a periodicidade da HbA1c?", retriever, None, excluded_refs=all_refs)
    user = render_user_message(context)
    assert "Nenhum trecho recuperado." in user
    assert "Nenhum paciente citado na pergunta." in user


def test_unknown_patient_raises(retriever, conn):
    with pytest.raises(PatientNotFoundError):
        build_context("Exames de PAC-0999?", retriever, conn, "PAC-0999")


def test_patient_question_without_database_raises(retriever):
    with pytest.raises(ValueError, match="banco"):
        build_context("Exames de PAC-0001?", retriever, None, "PAC-0001")
