"""Montagem do retriever completo: modelo local + índice (do cache válido ou reconstruído)."""

from __future__ import annotations

from pathlib import Path

from medprotocol.rag.embeddings import make_embeddings, make_token_counter
from medprotocol.rag.index import load_or_build_index
from medprotocol.rag.retriever import ProtocolRetriever


def load_protocol_retriever(
    data_dir: Path, cache_dir: Path | None, *, rebuild: bool = False
) -> tuple[ProtocolRetriever, bool]:
    """Devolve (retriever, índice reconstruído?). Protocolos lidos de <data_dir>/protocols."""
    embeddings = make_embeddings()
    store, rebuilt = load_or_build_index(
        Path(data_dir) / "protocols", cache_dir, embeddings, make_token_counter(embeddings), rebuild=rebuild
    )
    return ProtocolRetriever(store), rebuilt
