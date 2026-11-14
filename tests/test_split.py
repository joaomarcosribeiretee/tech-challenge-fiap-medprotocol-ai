"""Separação treino/validação: isolamento por paciente, quase-duplicatas e reprodutibilidade."""

import json
from collections import Counter
from pathlib import Path

from medprotocol.data.validation import EVAL_ONLY_PATIENTS
from medprotocol.finetuning.sources import RESERVED_SECTIONS, TrainingExample, load_training_examples
from medprotocol.finetuning.split import SplitParams, split_examples

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST = DATA_DIR / "finetune" / "split_v1.json"


def examples():
    return load_training_examples(DATA_DIR)


def test_no_patient_on_both_sides_and_full_partition():
    exs = examples()
    result = split_examples(exs)
    by_id = {e.qa_id: e for e in exs}
    assert set(result.train) | set(result.val) == set(by_id)
    assert not set(result.train) & set(result.val)
    train_patients = {by_id[i].patient_id for i in result.train} - {None}
    val_patients = {by_id[i].patient_id for i in result.val} - {None}
    assert not train_patients & val_patients
    assert val_patients == set(result.val_patients)


def test_split_is_deterministic_and_matches_versioned_manifest():
    exs = examples()
    first, second = split_examples(exs), split_examples(exs)
    assert (first.train, first.val) == (second.train, second.val)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["train"] == first.train and manifest["val"] == first.val


def test_no_near_duplicates_across_sides_and_every_category_kept_in_train():
    exs = examples()
    result = split_examples(exs)
    assert result.unresolved_pairs == []
    assert result.max_cross_similarity < SplitParams().similarity_threshold
    by_id = {e.qa_id: e for e in exs}
    train_counts = Counter(by_id[i].rotulos["categoria"] for i in result.train)
    assert min(train_counts[c] for c in {e.rotulos["categoria"] for e in exs}) >= SplitParams().min_train_per_category


def test_no_reserved_data_in_either_side():
    for e in examples():
        text = json.dumps([e.pergunta, e.resposta_esperada, e.rotulos], ensure_ascii=False)
        assert not any(p in text for p in EVAL_ONLY_PATIENTS)
        assert not set(e.rotulos["protocolos_citados"]) & RESERVED_SECTIONS


def _ex(qa_id, pergunta, patient=None, categoria="consulta_protocolo", resposta="Resposta própria"):
    return TrainingExample(qa_id, pergunta, patient, f"{resposta} {qa_id}", {"categoria": categoria})


def test_near_duplicate_without_patient_is_moved_out_of_validation():
    exs = [
        _ex("QA-0001", "Qual a periodicidade da HbA1c no protocolo?", resposta="a"),
        _ex("QA-0002", "Qual a periodicidade da HbA1c no protocolo ambulatorial?", resposta="a"),
        _ex("QA-0003", "Pergunta diferente sobre hipoglicemia grave", resposta="b c d e f g"),
        _ex("QA-0004", "PAC-0001 está em dia?", "PAC-0001", "dados_ausentes"),
        _ex("QA-0005", "PAC-0002 está em dia?", "PAC-0002", "dados_ausentes"),
        _ex("QA-0006", "PAC-0003 está em dia?", "PAC-0003", "dados_ausentes"),
    ]
    params = SplitParams(n_val_patients=1, target_patient_examples=1, min_no_patient_for_val=1, min_train_per_category=1, similarity_threshold=0.6)
    result = split_examples(exs, params)
    # Os dois quase iguais nunca ficam separados.
    assert ("QA-0001" in result.val) == ("QA-0002" in result.val)
