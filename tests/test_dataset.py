"""Validação do dataset real e testes negativos (cada defeito inserido deve ser detectado)."""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from medprotocol.data.loaders import QA_FILES, load_dataset
from medprotocol.data.validation import EVAL_ONLY_PATIENTS, validate_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def data_copy(tmp_path):
    target = tmp_path / "data"
    shutil.copytree(DATA_DIR, target)
    return target


def edit_qa(data_dir, split, qa_id, change):
    path = data_dir / QA_FILES[split]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = next(r for r in records if r["qa_id"] == qa_id)
    change(record)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def replace_in(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, f"trecho de teste não encontrado em {path.name}: {old!r}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def errors_for(data_dir):
    report = validate_dataset(data_dir)
    assert not report.ok, "o defeito inserido deveria ter sido detectado"
    return "\n".join(report.errors)


# --- Dataset real -------------------------------------------------------------------------------


def test_real_dataset_is_valid():
    report = validate_dataset(DATA_DIR)
    assert report.errors == []


def test_real_dataset_sizes():
    dataset = load_dataset(DATA_DIR)
    assert len(dataset.protocols) == 3
    assert len(dataset.patients) == 10
    assert len(dataset.qa["train"]) == 20
    assert len(dataset.qa["eval"]) == 10


def test_every_category_has_four_train_and_two_eval_cases():
    dataset = load_dataset(DATA_DIR)
    assert set(Counter(r["categoria"] for r in dataset.qa["train"]).values()) == {4}
    assert set(Counter(r["categoria"] for r in dataset.qa["eval"]).values()) == {2}


def test_recusa_and_revisao_are_distinct_but_can_coexist():
    records = [r for split in load_dataset(DATA_DIR).qa.values() for r in split]
    combos = {(r["deve_recusar"], r["requer_revisao_humana"]) for r in records}
    assert (True, True) in combos  # recusa com encaminhamento
    assert (True, False) in combos  # recusa sem necessidade de revisão
    assert (False, True) in combos  # revisão sem recusa


def test_eval_only_patients_absent_from_training():
    train = load_dataset(DATA_DIR).qa["train"]
    text = json.dumps(train, ensure_ascii=False)
    for patient_id in EVAL_ONLY_PATIENTS:
        assert patient_id not in text


# --- Defeitos que o validador deve detectar -----------------------------------------------------


def test_detects_protocol_excerpt_not_in_source(data_copy):
    def change(r):
        r["evidencias"][0]["trecho"] = "HbA1c com último resultado acima de 8,0%: repetir em 60 dias."

    edit_qa(data_copy, "train", "QA-0001", change)
    assert "não encontrado literalmente" in errors_for(data_copy)


def test_detects_unknown_protocol_section(data_copy):
    def change(r):
        r["evidencias"][0]["ref"] = "PROT-DM2-001#S9"
        r["protocolos_citados"] = ["PROT-DM2-001#S9"]

    edit_qa(data_copy, "train", "QA-0001", change)
    assert "inexistente" in errors_for(data_copy)


def test_detects_exam_from_another_patient(data_copy):
    # EX-0001 pertence a PAC-0001; QA-0008 é de PAC-0005.
    def change(r):
        r["evidencias"][0] = {"tipo": "exame", "ref": "EX-0001", "trecho": "HBA1C 6.5 % (2026-03-10)"}

    edit_qa(data_copy, "train", "QA-0008", change)
    assert "pertence a PAC-0001" in errors_for(data_copy)


def test_detects_consultation_from_another_patient(data_copy):
    def change(r):
        r["evidencias"].append({"tipo": "consulta", "ref": "CON-0001", "trecho": "Consulta de rotina."})

    edit_qa(data_copy, "train", "QA-0005", change)
    assert "consulta CON-0001 pertence a PAC-0001" in errors_for(data_copy)


def test_detects_consultation_excerpt_not_in_summary(data_copy):
    def change(r):
        r["evidencias"][2]["trecho"] = "Função renal normal."

    edit_qa(data_copy, "train", "QA-0005", change)
    assert "resumo de CON-0003" in errors_for(data_copy)


def test_detects_wrong_temporal_expectation(data_copy):
    def change(r):
        r["verificacoes_esperadas"][0]["status"] = "em_dia"  # HbA1c de PAC-0005 está vencida

    edit_qa(data_copy, "train", "QA-0008", change)
    assert "diverge do calculado" in errors_for(data_copy)


def test_detects_exam_change_that_breaks_expectation(data_copy):
    # Alterar a data de coleta muda a situação calculada e invalida o caso.
    replace_in(data_copy / "synthetic/exams.csv", "EX-0024,PAC-0005,HBA1C,9.1,%,2026-02-15",
               "EX-0024,PAC-0005,HBA1C,9.1,%,2026-06-01")
    assert "QA-0008" in errors_for(data_copy)


def test_detects_recusa_without_refusal_flag(data_copy):
    edit_qa(data_copy, "train", "QA-0013", lambda r: r.update(deve_recusar=False))
    assert "deve_recusar=true" in errors_for(data_copy)


def test_detects_revisao_humana_marked_as_refusal(data_copy):
    edit_qa(data_copy, "train", "QA-0017", lambda r: r.update(deve_recusar=True))
    assert "revisao_humana não é recusa" in errors_for(data_copy)


def test_detects_eval_patient_in_training(data_copy):
    def change(r):
        r["resposta_esperada"] += " Compare com PAC-0009."

    edit_qa(data_copy, "train", "QA-0001", change)
    assert "menciona ['PAC-0009']" in errors_for(data_copy)


def test_detects_eval_exam_id_in_training(data_copy):
    edit_qa(data_copy, "train", "QA-0001", lambda r: r.update(pergunta=r["pergunta"] + " (ver EX-0047)"))
    assert "EX-0047" in errors_for(data_copy)


def test_detects_question_duplicated_across_splits(data_copy):
    question = "Com que frequência a HbA1c deve ser repetida segundo o protocolo de acompanhamento ambulatorial?"
    edit_qa(data_copy, "eval", "QA-0022", lambda r: r.update(pergunta=question.upper()))
    assert "pergunta repetida" in errors_for(data_copy)


def test_detects_protocol_without_simulation_status(data_copy):
    replace_in(data_copy / "protocols/PROT-DM2-002.md", "status: SIMULACAO", "status: OFICIAL")
    assert "status='OFICIAL'" in errors_for(data_copy)


def test_detects_clinical_validation_claim(data_copy):
    replace_in(data_copy / "protocols/PROT-DM2-003.md", "validacao_clinica: nao_realizada",
               "validacao_clinica: realizada")
    assert "validacao_clinica" in errors_for(data_copy)


def test_detects_simulated_rules_with_external_reference(data_copy):
    replace_in(data_copy / "protocols/PROT-DM2-001.md", "referencias_externas_verificadas: nenhuma",
               "referencias_externas_verificadas: Diretriz X 2025")
    assert "natureza_regras=simulada" in errors_for(data_copy)


def test_detects_personal_data_pattern(data_copy):
    replace_in(data_copy / "synthetic/consultations.csv", "Retorno sem queixas.",
               "Retorno sem queixas. Contato 123.456.789-09.")
    assert "possível CPF" in errors_for(data_copy)


def test_detects_name_column_in_patients(data_copy):
    replace_in(data_copy / "synthetic/patients.csv", "patient_id,sexo", "patient_id,nome,sexo")
    assert "difere do schema" in errors_for(data_copy)
