"""Regras temporais com datas calculadas à mão, independentes do dataset."""

from datetime import date

import pytest

from medprotocol.data.schedule import verification

REF = date(2026, 6, 30)


def exam(exam_id, tipo, valor, data):
    return {"exam_id": exam_id, "patient_id": "PAC-9999", "tipo": tipo, "valor": str(valor), "data_coleta": data}


def test_hba1c_above_target_uses_90_days_and_is_out_of_limit():
    result = verification("HBA1C", [exam("EX-1", "HBA1C", 8.4, "2026-05-12")], REF)
    assert result["proxima_verificacao"] == "2026-08-10"
    assert result["status"] == "em_dia"
    assert result["fora_do_limite"] is True


def test_hba1c_exactly_at_target_uses_180_days():
    result = verification("HBA1C", [exam("EX-1", "HBA1C", 7.0, "2026-04-25")], REF)
    assert result["proxima_verificacao"] == "2026-10-22"
    assert result["fora_do_limite"] is False


def test_next_check_on_reference_date_is_still_em_dia():
    # 2026-04-01 + 90 dias = 2026-06-30
    result = verification("HBA1C", [exam("EX-1", "HBA1C", 7.5, "2026-04-01")], REF)
    assert result["proxima_verificacao"] == "2026-06-30"
    assert result["status"] == "em_dia"


def test_next_check_before_reference_date_is_vencido():
    result = verification("HBA1C", [exam("EX-1", "HBA1C", 6.6, "2025-11-20")], REF)
    assert result["proxima_verificacao"] == "2026-05-19"
    assert result["status"] == "vencido"
    assert result["fora_do_limite"] is False


def test_missing_exam_is_ausente_with_no_dates():
    result = verification("RAC", [exam("EX-1", "HBA1C", 6.0, "2026-01-01")], REF)
    assert result == {
        "exame": "RAC",
        "status": "ausente",
        "ultimo_exame": None,
        "data_ultimo": None,
        "proxima_verificacao": None,
        "fora_do_limite": None,
    }


def test_latest_result_defines_interval_not_older_ones():
    exams = [exam("EX-1", "HBA1C", 7.2, "2025-09-12"), exam("EX-2", "HBA1C", 6.5, "2026-03-10")]
    result = verification("HBA1C", exams, REF)
    assert result["ultimo_exame"] == "EX-2"
    assert result["proxima_verificacao"] == "2026-09-06"


def test_exam_after_reference_date_is_ignored():
    result = verification("LDL", [exam("EX-1", "LDL", 90, "2026-07-15")], REF)
    assert result["status"] == "ausente"


@pytest.mark.parametrize(
    "tipo, valor, esperado",
    [("TFG", 59, True), ("TFG", 60, False), ("RAC", 30, True), ("RAC", 29, False),
     ("LDL", 100, True), ("LDL", 99, False), ("CREATININA", 3.0, None)],
)
def test_limits_at_boundaries(tipo, valor, esperado):
    result = verification(tipo, [exam("EX-1", tipo, valor, "2026-06-01")], REF)
    assert result["fora_do_limite"] is esperado


def test_unknown_exam_type_raises():
    with pytest.raises(ValueError, match="GLICEMIA_JEJUM"):
        verification("GLICEMIA_JEJUM", [], REF)
