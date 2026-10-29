"""Gera o banco SQLite a partir dos CSVs sintéticos.

Uso (a partir da raiz do repositório, com o ambiente virtual ativo):
    python scripts/build_database.py [--data-dir data] [--db-path data/processed/medprotocol.db]

O dataset é validado antes da importação; se houver erro, o banco não é gerado nem
substituído. Rodar novamente recria o banco do zero com o mesmo conteúdo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.db.importer import DatasetImportError, build_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="padrão: DATA_DIR do ambiente")
    parser.add_argument("--db-path", type=Path, default=None, help="padrão: SQLITE_PATH do ambiente")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        result = build_database(args.data_dir or settings.data_dir, args.db_path or settings.sqlite_path)
    except DatasetImportError as exc:
        print(f"ERRO: {exc}")
        for error in exc.errors:
            print(f"  - {error}")
        return 1

    print(f"Banco gerado: {result.db_path.resolve()}")
    for table, count in result.counts.items():
        print(f"  {table}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
