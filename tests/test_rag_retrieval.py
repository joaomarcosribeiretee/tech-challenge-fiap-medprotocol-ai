"""Recuperação com o modelo real (intfloat/multilingual-e5-small, revisão fixada).

Pulado se o extra [rag] não estiver instalado. Na primeira execução, baixa o modelo.
"""

from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers", reason="extra [rag] não instalado")

import numpy as np  # noqa: E402

from medprotocol.rag.embeddings import (  # noqa: E402
    MAX_TOKENS,
    MODEL_REVISION,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    make_embeddings,
    make_token_counter,
)
from medprotocol.rag.chunking import build_chunks  # noqa: E402
from medprotocol.rag.index import build_index  # noqa: E402
from medprotocol.rag.retriever import (  # noqa: E402
    STATUS_NO_RELIABLE,
    STATUS_OK,
    ProtocolRetriever,
)

PROTOCOLS_DIR = Path(__file__).resolve().parents[1] / "data" / "protocols"


@pytest.fixture(scope="module")
def embeddings():
    return make_embeddings()


@pytest.fixture(scope="module")
def store(embeddings):
    return build_index(PROTOCOLS_DIR, embeddings, make_token_counter(embeddings))


@pytest.fixture(scope="module")
def retriever(store):
    return ProtocolRetriever(store)


def raw_encode(embeddings, text):
    """Codifica sem nenhum prompt (prompt="" desativa prefixos automáticos)."""
    return embeddings._client.encode(text.replace("\n", " "), prompt="", normalize_embeddings=True)


# --- Modelo e prefixos E5 ---------------------------------------------------------------------


def test_model_revision_and_limits(embeddings):
    client = embeddings._client
    assert client.max_seq_length == MAX_TOKENS
    assert not client.default_prompt_name  # nenhum prefixo automático além do nosso
    assert len(MODEL_REVISION) == 40


def test_query_prefix_applied_exactly_once(embeddings):
    question = "Qual é a periodicidade da HbA1c?"
    vector = np.array(embeddings.embed_query(question))
    assert np.allclose(vector, raw_encode(embeddings, QUERY_PREFIX + question), atol=1e-4)
    assert not np.allclose(vector, raw_encode(embeddings, QUERY_PREFIX * 2 + question), atol=1e-3)
    assert not np.allclose(vector, raw_encode(embeddings, question), atol=1e-3)
    assert not np.allclose(vector, raw_encode(embeddings, PASSAGE_PREFIX + question), atol=1e-3)


def test_passage_prefix_applied_exactly_once_at_indexing(embeddings, store):
    for chunk_id, entry in list(store.store.items())[:5]:
        stored = np.array(entry["vector"])
        assert np.allclose(stored, raw_encode(embeddings, PASSAGE_PREFIX + entry["text"]), atol=1e-4), chunk_id
        assert not np.allclose(stored, raw_encode(embeddings, PASSAGE_PREFIX * 2 + entry["text"]), atol=1e-3)
        assert not np.allclose(stored, raw_encode(embeddings, QUERY_PREFIX + entry["text"]), atol=1e-3)


def test_real_chunks_fit_model_limit_without_truncation(embeddings):
    count = make_token_counter(embeddings)
    chunks = build_chunks(PROTOCOLS_DIR, count, MAX_TOKENS)
    assert all(c.metadata["chunk_count"] == 1 for c in chunks)
    assert max(count(c.page_content) for c in chunks) < MAX_TOKENS


# --- Recuperação --------------------------------------------------------------------------------


def test_hits_reference_real_sections_and_scores_are_cosine(retriever, store):
    known = {entry["metadata"]["ref"] for entry in store.store.values()}
    result = retriever.retrieve("Como registrar um episódio de hipoglicemia?", k=20)
    assert len(result.hits) == 20
    for hit in result.hits:
        assert hit.ref in known
        assert -1.0 <= hit.score <= 1.0 + 1e-6
    assert [h.score for h in result.hits] == sorted((h.score for h in result.hits), reverse=True)


@pytest.mark.parametrize(
    "question, expected_ref",
    [
        ("Qual é a periodicidade da HbA1c?", "PROT-DM2-001#S3"),
        ("Quando deve ser registrada a avaliação dos pés?", "PROT-DM2-003#S4"),
        ("O que fazer quando o paciente não consegue deglutir?", "PROT-DM2-002#S4"),
    ],
)
def test_acceptance_demos_section_in_top3(retriever, question, expected_ref):
    """Demonstrações pedidas na especificação da etapa. Não são usadas para calibrar nada."""
    refs = [hit.ref for hit in retriever.retrieve(question, k=3).hits]
    assert expected_ref in refs


def test_unrelated_question_is_not_reported_as_reliable(retriever):
    result = retriever.retrieve("Qual é a cotação do dólar hoje?")
    assert result.status != STATUS_OK


def test_clearly_unrelated_question_has_no_reliable_result(retriever):
    result = retriever.retrieve("Receita de bolo de chocolate com cobertura")
    assert result.status == STATUS_NO_RELIABLE
    assert result.hits  # hits continuam disponíveis para depuração, com status explícito
