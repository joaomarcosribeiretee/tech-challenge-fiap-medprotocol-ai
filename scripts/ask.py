"""Pergunta ao assistente pelo fluxo LangGraph completo e imprime a saída estruturada (JSON).

Uso (ambiente .venv-train, que tem RAG + torch CUDA + peft):
    python scripts/ask.py "Quais verificações estão pendentes para PAC-0007?"
    python scripts/ask.py --variant base "..."        # modelo-base (adaptador desligado)
    python scripts/ask.py --file perguntas.txt         # uma pergunta por linha, um único carregamento do modelo

Requer o banco (scripts/build_database.py) e o adaptador (scripts/train_lora.py).
A auditoria vai para AUDIT_LOG_PATH (padrão logs/audit.jsonl).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.graph.workflow import create_assistant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pergunta", nargs="?")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--variant", choices=("ajustado", "base"))
    parser.add_argument("--out", type=Path, help="grava as respostas em JSONL")
    args = parser.parse_args(argv)
    if not args.pergunta and not args.file:
        parser.error("informe a pergunta ou --file")

    settings = load_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    if args.variant:
        settings = replace(settings, model_variant=args.variant)
    questions = [args.pergunta] if args.pergunta else [
        line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")
    ]
    assistant = create_assistant(settings)
    results = [assistant.ask(q).to_json() for q in questions]
    text = "\n".join(json.dumps(r, ensure_ascii=False, indent=None if args.out else 2) for r in results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text if not args.out else f"{len(results)} respostas -> {args.out}")
    return 1 if any(r["status"] == "erro" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
