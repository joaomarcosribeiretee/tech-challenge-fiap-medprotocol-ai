"""Construção do índice vetorial (InMemoryVectorStore) e cache reconstruível.

A fonte de verdade são sempre os protocolos versionados. O cache em disco
(`index.json` + `manifest.json`) é apenas uma otimização: só é reutilizado quando o
manifesto (hashes dos protocolos, modelo, revisão, prefixos e versão do chunking) é
idêntico ao atual E o conteúdo dos chunks bate com o chunking atual. Caso contrário,
o índice é reconstruído.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from medprotocol.rag import embeddings as emb
from medprotocol.rag.chunking import CHUNKING_VERSION, TokenCounter, build_chunks, chunk_id

INDEX_FILE = "index.json"
MANIFEST_FILE = "manifest.json"


def compute_manifest(protocols_dir: Path, model_name: str = emb.MODEL_NAME, model_revision: str = emb.MODEL_REVISION) -> dict:
    protocols = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(protocols_dir).glob("*.md"))
    }
    return {
        "chunking_version": CHUNKING_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "query_prefix": emb.QUERY_PREFIX,
        "passage_prefix": emb.PASSAGE_PREFIX,
        "max_tokens": emb.MAX_TOKENS,
        "normalize_embeddings": True,
        "protocols": protocols,
    }


def build_index(
    protocols_dir: Path, embeddings: Embeddings, count_tokens: TokenCounter, max_tokens: int = emb.MAX_TOKENS
) -> InMemoryVectorStore:
    documents = build_chunks(protocols_dir, count_tokens, max_tokens)
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents, ids=[chunk_id(doc) for doc in documents])
    return store


def _cache_matches_chunks(store: InMemoryVectorStore, protocols_dir: Path, count_tokens: TokenCounter, max_tokens: int) -> bool:
    expected = {chunk_id(doc): (doc.page_content, doc.metadata) for doc in build_chunks(protocols_dir, count_tokens, max_tokens)}
    cached = {key: (value["text"], value["metadata"]) for key, value in store.store.items()}
    return cached == expected


def load_or_build_index(
    protocols_dir: Path,
    cache_dir: Path | None,
    embeddings: Embeddings,
    count_tokens: TokenCounter,
    *,
    rebuild: bool = False,
    manifest: dict | None = None,
    max_tokens: int = emb.MAX_TOKENS,
) -> tuple[InMemoryVectorStore, bool]:
    """Devolve (índice, reconstruído?). Sem cache_dir, sempre constrói em memória."""
    manifest = manifest or compute_manifest(protocols_dir)
    if cache_dir is None:
        return build_index(protocols_dir, embeddings, count_tokens, max_tokens), True

    cache_dir = Path(cache_dir)
    index_path, manifest_path = cache_dir / INDEX_FILE, cache_dir / MANIFEST_FILE
    if not rebuild and index_path.is_file() and manifest_path.is_file():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest == manifest:
                store = InMemoryVectorStore.load(str(index_path), embedding=embeddings)
                if _cache_matches_chunks(store, protocols_dir, count_tokens, max_tokens):
                    return store, False
        except (OSError, ValueError, KeyError):
            pass  # cache corrompido ou incompatível: reconstruir

    store = build_index(protocols_dir, embeddings, count_tokens, max_tokens)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_index = index_path.with_name(INDEX_FILE + ".tmp")
    store.dump(str(tmp_index))
    os.replace(tmp_index, index_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return store, True
