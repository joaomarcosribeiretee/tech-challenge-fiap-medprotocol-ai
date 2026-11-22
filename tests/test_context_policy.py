"""Política determinística de seções adicionais (embeddings falsos; sem modelo real).

Nada aqui abre o qa_eval.jsonl nem usa PAC-0009/PAC-0010.
"""

import inspect
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="extra [rag] não instalado")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from medprotocol.data import schedule  # noqa: E402
from medprotocol.data.validation import EVAL_ONLY_PATIENTS  # noqa: E402
from medprotocol.db.importer import build_database  # noqa: E402
from medprotocol.db.repository import connect_readonly  # noqa: E402
from medprotocol.finetuning import context_policy  # noqa: E402
from medprotocol.finetuning.context import build_context  # noqa: E402
from medprotocol.finetuning.context_policy import (  # noqa: E402
    LINKED_SECTIONS,
    detect_intents,
    select_rule_sections,
)
from medprotocol.finetuning.messages import build_messages, render_user_message  # noqa: E402
from medprotocol.finetuning.sft import build_sft_record  # noqa: E402
from medprotocol.finetuning.sources import RESERVED_SECTIONS, load_training_examples  # noqa: E402
from medprotocol.rag.index import build_index  # noqa: E402
from medprotocol.rag.retriever import STATUS_NO_RELIABLE, ProtocolRetriever  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


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


def summary(**overrides):
    base = {"consultas": [], "fora_do_limite": [], "pendencias": [], "inconsistencias": []}
    return base | overrides


# --- Seleção independente da resposta esperada -------------------------------------------------


def test_policy_signature_only_takes_inference_time_inputs():
    params = list(inspect.signature(select_rule_sections).parameters)
    assert params == ["pergunta", "paciente", "hits"]


def test_selection_ignores_answer_and_labels(examples, retriever, conn):
    example = next(e for e in examples if e.qa_id == "QA-0013")
    altered = type(example)(
        example.qa_id, example.pergunta, example.patient_id, "Resposta qualquer.",
        {"categoria": "consulta_protocolo", "protocolos_citados": ["PROT-DM2-002#S1"]},
    )
    a = build_sft_record(example, retriever, conn)
    b = build_sft_record(altered, retriever, conn)
    assert a.messages[:2] == b.messages[:2]
    assert a.metadata["refs_por_regra"] == b.metadata["refs_por_regra"]


def test_question_without_patient_or_intent_adds_nothing():
    assert select_rule_sections("Qual é a periodicidade da HbA1c?", None, []) == []


def test_patient_data_adds_source_sections_of_computed_verifications():
    refs = [s.ref for s in select_rule_sections("Pendências de PAC-0001?", summary(), [])]
    expected = {schedule.STATUS_SECTION, *schedule.INTERVAL_SECTIONS.values(), *schedule.LIMIT_SECTIONS.values()}
    assert set(refs) == expected  # sem consultas, limites ou pendências: só as regras das verificações


def test_out_of_limit_and_pending_add_review_of_the_protocol_that_defines_the_rule():
    selections = select_rule_sections(
        "Situação de PAC-0001?", summary(fora_do_limite=["TFG", "LDL"], pendencias=["RAC"]), []
    )
    motives = {s.ref: s.motivos for s in selections}
    assert "revisao:fora_do_limite:TFG" in motives["PROT-DM2-003#REV"]
    assert "revisao:fora_do_limite:LDL" in motives["PROT-DM2-001#REV"]
    assert "revisao:pendencias" in motives["PROT-DM2-001#REV"]


def test_linked_protocol_adds_its_operational_sections():
    paciente = summary(consultas=[{"protocol_id": "PROT-DM2-003"}])
    refs = [s.ref for s in select_rule_sections("Pendências de PAC-0001?", paciente, [])]
    assert {f"PROT-DM2-003#{s}" for s in LINKED_SECTIONS["PROT-DM2-003"]} <= set(refs)


@pytest.mark.parametrize(
    "pergunta, intents",
    [
        ("Qual dose de insulina devo prescrever?", ["pedido_de_conduta"]),
        ("Estime um valor de creatinina", ["estimativa_de_valor"]),
        ("Posso aplicar a um paciente real?", ["uso_fora_da_simulacao"]),
        ("Qual é a periodicidade da HbA1c?", []),
        ("DIAGNÓSTICO definitivo", ["pedido_de_conduta"]),  # sem acento e maiúsculas
    ],
)
def test_intent_detection(pergunta, intents):
    assert detect_intents(pergunta) == intents


def test_conduct_request_uses_mentioned_protocol_as_focus():
    refs = [s.ref for s in select_rule_sections("O PROT-DM2-002 define dose de insulina?", None, [])]
    assert refs == ["PROT-DM2-002#REV", "PROT-DM2-002#LIM"]


# --- Rastreabilidade e deduplicação ------------------------------------------------------------


def test_every_policy_section_exists_in_the_index(store):
    indexed = {e["metadata"]["ref"] for e in store.store.values()}
    policy_refs = {schedule.STATUS_SECTION, *schedule.INTERVAL_SECTIONS.values(), *schedule.LIMIT_SECTIONS.values()}
    policy_refs |= {f"{p}#{s}" for p, sections in LINKED_SECTIONS.items() for s in sections}
    policy_refs |= {f"{p}#{s}" for p in LINKED_SECTIONS for s in ("REV", "LIM")}
    assert policy_refs <= indexed


def test_rule_sections_are_separate_traceable_and_not_duplicated(retriever, conn):
    context = build_context("Qual dose de insulina devo prescrever para PAC-0005?", retriever, conn, "PAC-0005")
    retrieved = [hit.ref for hit in context.recuperacao.hits]
    by_rule = [s.ref for s in context.secoes_por_regra]
    assert by_rule and len(by_rule) == len(set(by_rule))
    assert not set(by_rule) & set(retrieved)
    user = render_user_message(context)
    head, rules_block = user.split("## Seções adicionadas por regras determinísticas", 1)
    for position, section in enumerate(context.secoes_por_regra, start=1):
        assert section.motivos
        line = f"[R{position}] {section.ref} | {section.protocol_id} v{section.versao} | SIMULACAO | {section.source} | regra: "
        assert line in rules_block and section.trecho in rules_block
        assert f"] {section.ref} |" not in head  # não aparece também no bloco por similaridade
    for ref, motivos in context.regras_em_recuperados.items():
        assert ref in retrieved and f"também exigida por regra: {', '.join(motivos)}" in head


def test_order_is_reproducible(retriever, conn):
    a = build_context("Pendências de PAC-0007?", retriever, conn, "PAC-0007")
    b = build_context("Pendências de PAC-0007?", retriever, conn, "PAC-0007")
    assert a.refs == b.refs and render_user_message(a) == render_user_message(b)


# --- Treino × inferência -----------------------------------------------------------------------


def test_training_and_inference_share_the_same_context(examples, retriever, conn):
    """O treino usa o mesmo código da inferência; a única diferença é remover as seções reservadas.

    Quando nenhuma seção reservada aparece no contexto de inferência, as entradas são idênticas.
    """
    compared = 0
    for example in examples:
        training = build_sft_record(example, retriever, conn).messages[:2]
        inference_context = build_context(example.pergunta, retriever, conn, example.patient_id)
        if set(inference_context.refs) & RESERVED_SECTIONS:
            continue  # coberto por test_reserved_sections_selected_by_rules_are_dropped_in_training
        assert training == build_messages(inference_context), example.qa_id
        compared += 1
    assert compared >= 10


def test_policy_does_not_change_retrieval(retriever, conn):
    """A política só acrescenta seções: trechos, ordem, scores e status do RAG ficam intactos."""
    pergunta = "Qual dose de insulina devo prescrever para PAC-0005?"
    context = build_context(pergunta, retriever, conn, "PAC-0005")
    assert context.recuperacao == retriever.retrieve(pergunta)


# --- Dados reservados --------------------------------------------------------------------------


def test_reserved_sections_selected_by_rules_are_dropped_in_training(retriever, conn):
    # PAC-0006 tem internação vinculada ao PROT-DM2-002, cuja lista operacional inclui a S4 reservada.
    inference = build_context("Hipoglicemia de PAC-0006?", retriever, conn, "PAC-0006")
    training = build_context("Hipoglicemia de PAC-0006?", retriever, conn, "PAC-0006", excluded_refs=RESERVED_SECTIONS)
    assert "PROT-DM2-002#S4" in inference.refs
    assert not set(training.refs) & RESERVED_SECTIONS
    assert not any(patient in render_user_message(training) for patient in EVAL_ONLY_PATIENTS)


# --- Sem evidências ----------------------------------------------------------------------------


def test_no_evidence_keeps_status_and_states_absence(store):
    strict = ProtocolRetriever(store, lower_threshold=2.0, upper_threshold=2.0)
    all_refs = frozenset(e["metadata"]["ref"] for e in store.store.values())
    context = build_context("Qual é a capital da Austrália?", strict, None, excluded_refs=all_refs)
    user = render_user_message(context)
    assert context.recuperacao.status == STATUS_NO_RELIABLE
    assert "Nenhum trecho recuperado." in user
    assert "Nenhuma seção adicionada por regra." in user


def test_rule_sections_never_upgrade_the_retrieval_status(store, conn):
    strict = ProtocolRetriever(store, lower_threshold=2.0, upper_threshold=2.0)
    context = build_context("Pendências de PAC-0007?", strict, conn, "PAC-0007")
    assert context.secoes_por_regra  # há seções por regra...
    assert context.recuperacao.status == STATUS_NO_RELIABLE  # ...e o status continua o do retriever
    assert "status da recuperação: sem_resultado_confiavel" in render_user_message(context)


def test_missing_policy_section_is_an_error(retriever, conn, monkeypatch):
    """Se a política apontar para uma seção que não existe (ex.: protocolo mudou), falha alto."""
    monkeypatch.setitem(context_policy.LINKED_SECTIONS, "PROT-DM2-001", ("S99",))
    with pytest.raises(KeyError, match="PROT-DM2-001#S99"):
        build_context("Pendências de PAC-0001?", retriever, conn, "PAC-0001")
