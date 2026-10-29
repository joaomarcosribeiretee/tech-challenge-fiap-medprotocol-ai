"""Valida o dataset sintético do MedProtocol AI.

Uso (a partir da raiz do repositório, com o ambiente virtual ativo):
    python scripts/validate_dataset.py [--data-dir data] [--top-similar 5]

Sai com código 0 se não houver erros e 1 caso contrário. Avisos não falham a validação,
mas indicam pontos para revisão manual (ex.: possíveis paráfrases entre treino e avaliação).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.data.validation import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="padrão: DATA_DIR do ambiente")
    parser.add_argument("--top-similar", type=int, default=5, help="pares treino x avaliação a listar")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or load_settings().data_dir
    report = validate_dataset(data_dir)

    print(f"Dataset: {data_dir.resolve()}")
    for name, count in report.counts.items():
        print(f"  {name}: {count}")

    if report.similar_pairs:
        print(f"\nPares treino x avaliação mais parecidos (revisão manual de paráfrases):")
        for pair in report.similar_pairs[: args.top_similar]:
            print(f"  {pair.score:.3f}  {pair.field:<17} {pair.train_id} x {pair.eval_id}")

    for warning in report.warnings:
        print(f"AVISO: {warning}")
    for error in report.errors:
        print(f"ERRO: {error}")

    print(f"\nResultado: {'OK' if report.ok else 'FALHOU'} "
          f"({len(report.errors)} erro(s), {len(report.warnings)} aviso(s))")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
