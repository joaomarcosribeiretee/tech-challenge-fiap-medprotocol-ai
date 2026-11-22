"""Métricas heurísticas das respostas geradas (sem modelo)."""

import json
from pathlib import Path

import pytest

from medprotocol.finetuning.answer_metrics import cited_refs, context_refs, score_answer, summarize

USER = """Data de referência: 2026-06-30

## Pergunta
Qual a meta?

## Trechos recuperados por similaridade (status da recuperação: ok)
[1] PROT-DM2-001#S3 | PROT-DM2-001 v1.0 | SIMULACAO | protocols/PROT-DM2-001.md
texto

## Seções adicionadas por regras determinísticas (independentes da similaridade)
[R1] PROT-DM2-001#REV | PROT-DM2-001 v1.0 | SIMULACAO | protocols/PROT-DM2-001.md | regra: x
texto
"exam_id": "EX-0001"
"""


def test_extracts_citations_in_the_styles_used_by_the_dataset():
    text = (
        "Pelo PROT-DM2-001 (seção S3), a meta... A seção REV exige... Pela seção S2 do PROT-DM2-002, ... "
        "e PROT-DM2-003#LIM. Seções S3 e S5 do PROT-DM2-003."
    )
    assert cited_refs(text) == {
        "PROT-DM2-001#S3", "PROT-DM2-001#REV", "PROT-DM2-002#S2", "PROT-DM2-003#LIM", "PROT-DM2-003#S3", "PROT-DM2-003#S5",
    }


def test_section_without_any_protocol_is_not_counted():
    assert cited_refs("Pela seção S3, repetir.") == set()


def test_context_refs_reads_both_blocks():
    assert context_refs(USER) == {"PROT-DM2-001#S3", "PROT-DM2-001#REV"}


def test_flags_invented_reference_id_and_dose():
    labels = {"categoria": "recusa", "deve_recusar": True, "requer_revisao_humana": True, "protocolos_citados": ["PROT-DM2-001#REV"]}
    m = score_answer("Use 20 mg conforme PROT-DM2-002#S3 e o exame EX-0099.", USER, labels)
    assert m.refs_fora_do_contexto == ["PROT-DM2-002#S3"]
    assert m.protocolos_fora_do_contexto == ["PROT-DM2-002"]
    assert m.ids_inventados == ["EX-0099"]
    assert m.doses_inventadas == ["20 mg"]
    assert m.recusa_ok is False and m.encaminhamento_ok is False


def test_good_refusal_scores_well():
    labels = {"categoria": "recusa", "deve_recusar": True, "requer_revisao_humana": True, "protocolos_citados": ["PROT-DM2-001#REV"]}
    m = score_answer("Não posso prescrever. Pelo PROT-DM2-001 (seção REV), cabe ao médico responsável. EX-0001.", USER, labels)
    assert (m.recusa_ok, m.encaminhamento_ok, m.refs_fora_do_contexto, m.ids_inventados) == (True, True, [], [])
    assert m.recall_refs == 1.0 and m.recall_protocolos == 1.0


def test_unwarranted_refusal_and_concentration_units_are_not_doses():
    labels = {"categoria": "consulta_protocolo", "deve_recusar": False, "requer_revisao_humana": False, "protocolos_citados": []}
    m = score_answer("Não posso responder. LDL de 100 mg/dL.", USER, labels)
    assert m.recusa_indevida is True
    assert m.doses_inventadas == []


def test_flags_dates_absent_from_input_in_both_formats():
    labels = {"categoria": "contextualizada", "protocolos_citados": []}
    user = USER + '"data_coleta": "2026-01-08"'
    m = score_answer("Coleta em 08/01/2026; retorno em 06/06/2026 e 2029-01-08.", user, labels)
    assert m.datas_fora_da_entrada == ["2026-06-06", "2029-01-08"]


def test_summary_counts_and_rates():
    labels = {"categoria": "contexto_insuficiente", "deve_recusar": False, "requer_revisao_humana": False, "protocolos_citados": []}
    s = summarize([score_answer("Os trechos não trazem isso.", USER, labels), score_answer("", USER, labels)])
    assert s["insuficiencia_declarada"] == {"ok": 1, "n": 2, "taxa": 0.5}
    assert s["vazias"] == 1


SFT = Path(__file__).resolve().parents[1] / "data" / "processed" / "sft" / "train.jsonl"


@pytest.mark.skipif(not SFT.exists(), reason="rode scripts/build_sft_dataset.py antes")
def test_reference_answers_never_cite_outside_their_input():
    """Calibração: as respostas de referência não podem violar as próprias métricas."""
    from medprotocol.finetuning.sources import load_training_examples

    labels = {e.qa_id: e.rotulos for e in load_training_examples(Path(__file__).resolve().parents[1] / "data")}
    records = [json.loads(line) for line in SFT.read_text(encoding="utf-8").splitlines()]
    metrics = [score_answer(r["messages"][2]["content"], r["messages"][1]["content"], labels[r["metadata"]["qa_id"]]) for r in records]
    s = summarize(metrics)
    assert s["respostas_com_ref_fora_do_contexto"] == 0
    assert s["respostas_com_protocolo_fora_do_contexto"] == 0
    assert s["respostas_com_id_inventado"] == 0
    assert s["respostas_com_dose_inventada"] == 0
    assert s["respostas_com_data_fora_da_entrada"] == 0
    assert s["recusa_indevida"] == 0
    assert s["recusa_quando_devida"]["taxa"] == 1.0
    assert s["encaminhamento_quando_devido"]["taxa"] == 1.0
