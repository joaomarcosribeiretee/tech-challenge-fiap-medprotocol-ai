"""Métricas da avaliação final (sem GPU). Usam textos inventados para o teste, nunca o qa_eval."""

import json
from pathlib import Path

from medprotocol.evaluation.final import (
    SystemChecks,
    context_recall,
    exam_status_checks,
    expected_records,
    mentions_status,
    record_recall,
    summarize_model_level,
    summarize_system_level,
    system_checks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_record_recall():
    record = {"evidencias": [{"tipo": "exame", "ref": "EX-0001"}, {"tipo": "consulta", "ref": "CON-0001"},
                             {"tipo": "protocolo", "ref": "PROT-DM2-001#S3"}]}
    assert expected_records(record) == {"EX-0001", "CON-0001"}
    assert record_recall("Ver EX-0001.", {"EX-0001", "CON-0001"}) == 0.5
    assert record_recall("nada", set()) is None


def test_mentions_status_same_or_next_sentence():
    assert mentions_status("A HbA1c está vencida.", "HBA1C", "vencido")
    assert mentions_status("A HbA1c foi coletada em 2025. O exame está vencido.", "HBA1C", "vencido")
    assert not mentions_status("A HbA1c foi coletada. Algo. O exame está vencido.", "HBA1C", "vencido")
    assert mentions_status("Não há HbA1c registrada.", "HBA1C", "ausente")
    assert mentions_status("O LDL está em dia.", "LDL", "em_dia")


def test_negated_overdue_counts_as_on_time_not_overdue():
    text = "A TFG não está vencida."
    assert mentions_status(text, "TFG", "em_dia")
    assert not mentions_status(text, "TFG", "vencido")


def test_exam_status_checks_only_counts_what_reference_expresses():
    ver = [{"exame": "HBA1C", "status": "vencido"}, {"exame": "RAC", "status": "em_dia"}]
    reference = "A HbA1c está vencida."  # não fala do prazo da RAC
    checks = exam_status_checks("A HbA1c está em dia.", ver, reference)
    assert checks == [{"exame": "HBA1C", "status_esperado": "vencido", "ok": False}]
    assert len(exam_status_checks("x", ver)) == 2  # sem referência, conta tudo


def test_system_checks_and_summary():
    response = {
        "status": "revisao_humana", "resposta": "Não posso prescrever.", "resposta_modelo": "dose 5 mg",
        "alertas": [{"codigo": "dose_na_resposta", "severidade": "alta"}, {"codigo": "pedido_de_conduta", "severidade": "baixa"}],
        "fontes": [{"ref": "PROT-DM2-001#REV"}], "_modelo_chamado": True,
    }
    labels = {"requer_revisao_humana": True, "deve_recusar": True, "protocolos_citados": ["PROT-DM2-001#REV", "PROT-DM2-003#S3"]}
    c = system_checks(response, labels)
    assert c.retida and c.encaminhamento_ok and c.recusa_exibida_ok and not c.revisao_sem_rotulo
    assert c.alertas_altos == ["dose_na_resposta"]
    assert c.recall_contexto == 0.5
    other = system_checks({"status": "revisao_humana", "resposta": "ok", "alertas": [], "fontes": []}, {"requer_revisao_humana": False})
    assert other.revisao_sem_rotulo and other.encaminhamento_ok is None and other.recall_contexto is None
    s = summarize_system_level([c, other])
    assert s["status"] == {"revisao_humana": 2}
    assert s["encaminhamento_quando_devido"] == {"ok": 1, "n": 1}
    assert s["respostas_retidas"] == 1 and s["revisao_sem_rotulo"] == 1
    assert isinstance(SystemChecks(**c.__dict__), SystemChecks)


def test_model_level_summary():
    rows = [{"recall_registros": 1.0, "status_exames": [{"ok": True}, {"ok": False}]},
            {"recall_registros": None, "status_exames": []}]
    assert summarize_model_level(rows) == {"recall_medio_registros_esperados": 1.0, "status_de_exame_correto": {"ok": 1, "n": 2}}


def test_context_recall():
    assert context_recall(["A", "B"], ["A", "C"]) == 0.5
    assert context_recall([], []) is None


def test_heuristic_calibration_on_training_references():
    """Nas respostas de referência do TREINO, a heurística reconhece o status certo onde conta."""
    rows = [json.loads(l) for l in (ROOT / "data/finetune/qa_train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    counted = [c for r in rows for c in exam_status_checks(r["resposta_esperada"], r.get("verificacoes_esperadas"), r["resposta_esperada"])]
    assert counted and all(c["ok"] for c in counted)
    assert len(counted) == 52
