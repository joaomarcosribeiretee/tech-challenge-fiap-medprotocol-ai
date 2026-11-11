"""Gera o dataset de fine-tuning supervisionado (mensagens com contexto RAG + SQLite).

Uso:
    python scripts/build_sft_dataset.py [--data-dir data] [--db-path ...] [--out ...] [-k 4] [--strict]

Lê somente data/finetune/qa_train.jsonl (nunca o qa_eval). Requer o extra [rag] e o banco
gerado por scripts/build_database.py. Saída padrão: data/processed/sft/train.jsonl (não
versionada), um objeto {"messages": [...], "metadata": {...}} por linha.

Imprime a cobertura: exemplos cuja resposta cita seções que não estão nos trechos recuperados.
Com --strict, o script falha (código 1) se houver algum.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.db.repository import connect_readonly
from medprotocol.finetuning.sft import build_sft_records
from medprotocol.finetuning.sources import load_training_examples
from medprotocol.rag.pipeline import load_protocol_retriever
from medprotocol.rag.retriever import DEFAULT_K


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=None, help="padrão: SQLITE_PATH do ambiente")
    parser.add_argument("--out", type=Path, default=Path("data/processed/sft/train.jsonl"))
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"trechos por pergunta (padrão {DEFAULT_K})")
    parser.add_argument("--strict", action="store_true", help="falha se alguma referência citada estiver fora do contexto")
    args = parser.parse_args(argv)

    settings = load_settings()
    data_dir = args.data_dir or settings.data_dir
    examples = load_training_examples(data_dir)
    retriever, _ = load_protocol_retriever(data_dir, settings.vectorstore_dir)
    with connect_readonly(args.db_path or settings.sqlite_path) as conn:
        records = build_sft_records(examples, retriever, conn, k=args.k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    statuses = Counter(r.metadata["status_recuperacao"] for r in records)
    gaps = [r.metadata for r in records if r.metadata["refs_citadas_fora_do_contexto"]]
    print(f"{len(records)} exemplos -> {args.out.resolve()} (k={args.k})")
    print("Status da recuperação: " + " · ".join(f"{s}={n}" for s, n in sorted(statuses.items())))
    print(f"Exemplos que citam seções fora do contexto: {len(gaps)}/{len(records)}")
    for meta in gaps:
        print(f"  {meta['qa_id']} ({meta['status_recuperacao']}): {', '.join(meta['refs_citadas_fora_do_contexto'])}")
    return 1 if args.strict and gaps else 0


if __name__ == "__main__":
    sys.exit(main())
