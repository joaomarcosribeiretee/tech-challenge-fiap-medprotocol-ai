"""Mostra o resumo estruturado de um paciente sintético em JSON.

Uso:
    python scripts/query_patient.py PAC-0001 [--db-path data/processed/medprotocol.db]

Aceita apenas o ID do paciente; não executa SQL fornecido pelo usuário.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.db.repository import RepositoryError, connect_readonly, get_patient_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("patient_id", help="ID no formato PAC-NNNN")
    parser.add_argument("--db-path", type=Path, default=None, help="padrão: SQLITE_PATH do ambiente")
    args = parser.parse_args(argv)

    try:
        with connect_readonly(args.db_path or load_settings().sqlite_path) as conn:
            summary = get_patient_summary(conn, args.patient_id)
    except (RepositoryError, ValueError) as exc:
        print(f"ERRO: {exc}")
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
