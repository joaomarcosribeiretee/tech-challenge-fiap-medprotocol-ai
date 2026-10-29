"""Regras temporais simuladas de verificação de exames.

Espelham PROT-DM2-001 [S3]/[S4] e PROT-DM2-003 [S3] na versão indicada em
`SOURCE_PROTOCOL_VERSIONS`. O validador confere essas versões: se um protocolo
mudar de versão, este módulo precisa ser revisado junto.

Definições (em relação à data de referência):
- próxima verificação prevista = data da coleta do último resultado + intervalo do exame;
- em_dia: próxima verificação >= data de referência;
- vencido: há resultado, mas a próxima verificação < data de referência;
- ausente: nenhum resultado registrado até a data de referência;
- fora_do_limite: avaliado independentemente do prazo (None quando o exame não tem limite).
"""

from __future__ import annotations

from datetime import date, timedelta

REFERENCE_DATE = date(2026, 6, 30)
SOURCE_PROTOCOL_VERSIONS = {"PROT-DM2-001": "1.0", "PROT-DM2-003": "1.0"}

HBA1C_TARGET = 7.0
MONITORED_EXAMS = ("HBA1C", "CREATININA", "TFG", "RAC", "LDL")
_FIXED_INTERVAL_DAYS = {"CREATININA": 365, "TFG": 365, "RAC": 365, "LDL": 365}


def interval_days(exam_type: str, value: float) -> int:
    if exam_type == "HBA1C":
        return 90 if value > HBA1C_TARGET else 180
    return _FIXED_INTERVAL_DAYS[exam_type]


def out_of_limit(exam_type: str, value: float) -> bool | None:
    if exam_type == "HBA1C":
        return value > HBA1C_TARGET
    if exam_type == "TFG":
        return value < 60
    if exam_type == "RAC":
        return value >= 30
    if exam_type == "LDL":
        return value >= 100
    return None


def verification(
    exam_type: str,
    patient_exams: list[dict[str, str]],
    reference_date: date = REFERENCE_DATE,
) -> dict:
    """Situação de um exame monitorado para um paciente, no formato de `verificacoes_esperadas`."""
    if exam_type not in MONITORED_EXAMS:
        raise ValueError(f"exame {exam_type!r} não tem regra de periodicidade")
    candidates = [
        exam
        for exam in patient_exams
        if exam["tipo"] == exam_type and date.fromisoformat(exam["data_coleta"]) <= reference_date
    ]
    if not candidates:
        return {
            "exame": exam_type,
            "status": "ausente",
            "ultimo_exame": None,
            "data_ultimo": None,
            "proxima_verificacao": None,
            "fora_do_limite": None,
        }
    latest = max(candidates, key=lambda exam: (exam["data_coleta"], exam["exam_id"]))
    value = float(latest["valor"])
    collected = date.fromisoformat(latest["data_coleta"])
    next_check = collected + timedelta(days=interval_days(exam_type, value))
    return {
        "exame": exam_type,
        "status": "em_dia" if next_check >= reference_date else "vencido",
        "ultimo_exame": latest["exam_id"],
        "data_ultimo": latest["data_coleta"],
        "proxima_verificacao": next_check.isoformat(),
        "fora_do_limite": out_of_limit(exam_type, value),
    }
