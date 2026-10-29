"""Índice, cache e retriever com embeddings falsos determinísticos (rápido, sem modelo)."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="extra [rag] não instalado")

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from medprotocol.data.loaders import load_dataset  # noqa: E402
from medprotocol.data.validation import text_similarity  # noqa: E402
from medprotocol.rag.evaluation import calibrate, load_dev_queries, metrics, QueryOutcome  # noqa: E402
from medprotocol.rag.index import MANIFEST_FILE, build_index, compute_manifest, load_or_build_index  # noqa: E402
from medprotocol.rag.retriever import (  # noqa: E402
    STATUS_NO_RELIABLE,
    STATUS_OK,
    STATUS_UNCERTAIN,
    ProtocolRetriever,
    classify,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROTOCOLS_DIR = DATA_DIR / "protocols"


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def fake():
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def protocols_copy(tmp_path):
    target = tmp_path / "protocols"
    shutil.copytree(PROTOCOLS_DIR, target)
    return target


# --- Cache ------------------------------------------------------------------------------------


def test_cache_is_reused_only_when_manifest_matches(protocols_copy, tmp_path, fake):
    cache = tmp_path / "vectorstore"
    _, rebuilt = load_or_build_index(protocols_copy, cache, fake, words)
    assert rebuilt and (cache / "index.json").is_file()
    _, rebuilt = load_or_build_index(protocols_copy, cache, fake, words)
    assert not rebuilt

    path = protocols_copy / "PROT-DM2-002.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _, rebuilt = load_or_build_index(protocols_copy, cache, fake, words)
    assert rebuilt  # hash do protocolo mudou


def test_cache_rebuilt_when_model_revision_changes(protocols_copy, tmp_path, fake):
    cache = tmp_path / "vectorstore"
    load_or_build_index(protocols_copy, cache, fake, words)
    other = compute_manifest(protocols_copy, model_revision="outra-revisao")
    _, rebuilt = load_or_build_index(protocols_copy, cache, fake, words, manifest=other)
    assert rebuilt
    assert json.loads((cache / MANIFEST_FILE).read_text(encoding="utf-8"))["model_revision"] == "outra-revisao"


def test_corrupted_cache_is_rebuilt(protocols_copy, tmp_path, fake):
    cache = tmp_path / "vectorstore"
    load_or_build_index(protocols_copy, cache, fake, words)
    (cache / "index.json").write_text("{corrompido", encoding="utf-8")
    store, rebuilt = load_or_build_index(protocols_copy, cache, fake, words)
    assert rebuilt and len(store.store) == 20


def test_tampered_cache_content_is_rebuilt(protocols_copy, tmp_path, fake):
    cache = tmp_path / "vectorstore"
    load_or_build_index(protocols_copy, cache, fake, words)
    data = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    first = next(iter(data))
    data[first]["text"] = "Texto que não existe no protocolo."
    (cache / "index.json").write_text(json.dumps(data), encoding="utf-8")
    _, rebuilt = load_or_build_index(protocols_copy, cache, fake, words)
    assert rebuilt


def test_build_without_cache_dir_works_in_memory(fake):
    store, rebuilt = load_or_build_index(PROTOCOLS_DIR, None, fake, words)
    assert rebuilt and len(store.store) == 20


# --- Retriever --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [(None, STATUS_NO_RELIABLE), (0.10, STATUS_NO_RELIABLE), (0.50, STATUS_UNCERTAIN), (0.80, STATUS_OK), (0.95, STATUS_OK)],
)
def test_classify_bands(score, expected):
    assert classify(score, lower=0.5, upper=0.8) == expected


def test_retrieve_returns_traceable_hits_sorted_by_score(fake):
    retriever = ProtocolRetriever(build_index(PROTOCOLS_DIR, fake, words), 0.0, 1.0)
    result = retriever.retrieve("periodicidade da HbA1c", k=5)
    assert len(result.hits) == 5
    assert [h.score for h in result.hits] == sorted((h.score for h in result.hits), reverse=True)
    for hit in result.hits:
        assert hit.ref == f"{hit.protocol_id}#{hit.section_id}"
        assert hit.status_protocolo == "SIMULACAO"


@pytest.mark.parametrize("bad", ["", "   ", None, "x" * 1001])
def test_retrieve_rejects_invalid_question(fake, bad):
    retriever = ProtocolRetriever(build_index(PROTOCOLS_DIR, fake, words))
    with pytest.raises(ValueError):
        retriever.retrieve(bad)


@pytest.mark.parametrize("k", [0, 21, 2.5])
def test_retrieve_rejects_invalid_k(fake, k):
    retriever = ProtocolRetriever(build_index(PROTOCOLS_DIR, fake, words))
    with pytest.raises(ValueError):
        retriever.retrieve("HbA1c", k=k)


def test_inverted_thresholds_rejected(fake):
    with pytest.raises(ValueError):
        ProtocolRetriever(build_index(PROTOCOLS_DIR, fake, words), lower_threshold=0.9, upper_threshold=0.5)


# --- Calibração e conjunto de desenvolvimento ---------------------------------------------------


def outcome(query_id, escopo, top, rank=1):
    return QueryOutcome(query_id, "p", escopo, ["X#S1"] if escopo == "dentro" else [], [("X#S1", top)], rank)


def test_calibration_reports_overlap_instead_of_inventing_separation():
    cal = calibrate([outcome("a", "dentro", 0.80), outcome("b", "dentro", 0.90), outcome("c", "fora", 0.85)])
    assert not cal.separable
    assert (cal.lower, cal.upper) == (0.8, 0.8501)


def test_calibration_rounding_keeps_defining_queries_on_correct_side():
    # Regressão: arredondar o menor score do escopo para cima o deixava abaixo do próprio limiar.
    cal = calibrate([outcome("a", "dentro", 0.843390), outcome("c", "fora", 0.870870)])
    assert classify(0.843390, cal.lower, cal.upper) != STATUS_NO_RELIABLE
    assert classify(0.870870, cal.lower, cal.upper) != STATUS_OK


def test_calibration_uses_midpoint_when_separable():
    cal = calibrate([outcome("a", "dentro", 0.90), outcome("c", "fora", 0.70)])
    assert cal.separable and cal.lower == cal.upper == 0.8


def test_metrics_hit_and_mrr():
    data = [outcome("a", "dentro", 0.9, rank=1), outcome("b", "dentro", 0.9, rank=3), outcome("c", "fora", 0.5)]
    result = metrics(data, lower=0.6, upper=0.8)
    assert result["hit@1"] == 0.5 and result["hit@3"] == 1.0
    assert result["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert result["rejeicao_fora"] == 1.0


def test_dev_queries_reference_real_sections():
    chunk_refs = {f"{p.stem}#{s}" for p in PROTOCOLS_DIR.glob("*.md") for s in __import__("medprotocol.data.loaders", fromlist=["parse_protocol"]).parse_protocol(p).sections}
    for query in load_dev_queries(DATA_DIR):
        assert set(query["refs_esperadas"]) <= chunk_refs, query["query_id"]


def test_dev_queries_are_distant_from_reserved_questions():
    """Proteção contra vazamento: só compara o TEXTO das perguntas; nenhuma busca é feita
    com o qa_eval e nada dele é usado para ajustar o retriever."""
    dataset = load_dataset(DATA_DIR)
    reserved = [r["pergunta"] for split in ("train", "eval") for r in dataset.qa[split]]
    for query in load_dev_queries(DATA_DIR):
        closest = max(text_similarity(query["pergunta"], other) for other in reserved)
        assert closest < 0.6, query["query_id"]


def test_dev_set_excludes_sections_reserved_for_evaluation():
    refs = {ref for q in load_dev_queries(DATA_DIR) for ref in q["refs_esperadas"]}
    assert not refs & {"PROT-DM2-002#S4", "PROT-DM2-003#S4"}
