"""Gera a separação treino/validação dos exemplos de fine-tuning (manifesto versionado).

Uso:
    python scripts/split_sft_dataset.py [--data-dir data] [--out data/finetune/split_v1.json]

Lê somente data/finetune/qa_train.jsonl. Grava só IDs e contagens: o conteúdo continua em
qa_train.jsonl e em data/processed/sft/train.jsonl. Determinístico: rodar de novo produz o
mesmo arquivo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.finetuning.sources import load_training_examples
from medprotocol.finetuning.split import split_examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/finetune/split_v1.json"))
    args = parser.parse_args(argv)

    examples = load_training_examples(args.data_dir or load_settings().data_dir)
    result = split_examples(examples)
    categories = {e.qa_id: e.rotulos["categoria"] for e in examples}
    manifest = result.to_json(categories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    counts = manifest["counts"]
    print(f"treino={counts['train']} validação={counts['val']} -> {args.out}")
    print(f"pacientes da validação: {', '.join(result.val_patients)}")
    print(f"validação por categoria: {counts['val_por_categoria']}")
    print(f"maior similaridade entre lados: {manifest['max_similaridade_entre_lados']}")
    for move in result.moves:
        print(f"  ajuste: {move}")
    for pair in result.unresolved_pairs:
        print(f"  NÃO RESOLVIDO: {pair}")
    return 1 if result.unresolved_pairs else 0


if __name__ == "__main__":
    sys.exit(main())
