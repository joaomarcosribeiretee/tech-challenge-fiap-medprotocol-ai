"""Constrói o índice vetorial dos protocolos a partir dos arquivos versionados.

Uso:
    python scripts/build_index.py [--data-dir data] [--cache-dir data/vectorstore]

Sempre reconstrói (ignora cache existente) e grava index.json + manifest.json no cache.
O primeiro uso baixa o modelo de embeddings (~470 MB) para o cache do Hugging Face.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.rag.pipeline import load_protocol_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="padrão: DATA_DIR do ambiente")
    parser.add_argument("--cache-dir", type=Path, default=None, help="padrão: VECTORSTORE_DIR do ambiente")
    args = parser.parse_args(argv)

    settings = load_settings()
    cache_dir = args.cache_dir or settings.vectorstore_dir
    started = time.perf_counter()
    retriever, _ = load_protocol_retriever(args.data_dir or settings.data_dir, cache_dir, rebuild=True)
    elapsed = time.perf_counter() - started

    chunks = list(retriever.store.store.values())
    per_protocol = Counter(chunk["metadata"]["protocol_id"] for chunk in chunks)
    print(f"Índice reconstruído em {elapsed:.1f}s: {len(chunks)} chunks -> {cache_dir.resolve()}")
    for protocol_id, count in sorted(per_protocol.items()):
        print(f"  {protocol_id}: {count} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
