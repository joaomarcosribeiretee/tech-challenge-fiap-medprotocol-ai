"""Busca trechos dos protocolos simulados para uma pergunta (sem LLM, sem gerar respostas).

Uso:
    python scripts/search_protocols.py "Qual é a periodicidade da HbA1c?" [-k 4] [--rebuild]

Mostra os trechos literais recuperados, a referência (PROTOCOLO#SEÇÃO) e o score de
similaridade de cosseno (proximidade vetorial, não probabilidade de acerto).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.rag.pipeline import load_protocol_retriever
from medprotocol.rag.retriever import DEFAULT_K

STATUS_TEXT = {
    "ok": "ok (melhor score acima de todas as perguntas fora do escopo do conjunto de desenvolvimento)",
    "incerto": "INCERTO (faixa de score em que perguntas dentro e fora do escopo se sobrepõem)",
    "sem_resultado_confiavel": "SEM RESULTADO CONFIÁVEL (score abaixo de todas as perguntas do escopo)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pergunta")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"número de trechos (padrão {DEFAULT_K})")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true", help="reconstrói o índice antes de buscar")
    args = parser.parse_args(argv)

    settings = load_settings()
    retriever, rebuilt = load_protocol_retriever(
        args.data_dir or settings.data_dir, args.cache_dir or settings.vectorstore_dir, rebuild=args.rebuild
    )
    try:
        result = retriever.retrieve(args.pergunta, k=args.k)
    except ValueError as exc:
        print(f"ERRO: {exc}")
        return 1

    print(f"Pergunta: {result.pergunta}")
    print(f"Índice: {'reconstruído' if rebuilt else 'carregado do cache válido'}")
    print(f"Status: {STATUS_TEXT[result.status]}")
    print(f"Limiares (dev): lower={result.lower_threshold} upper={result.upper_threshold}\n")
    for position, hit in enumerate(result.hits, start=1):
        print(f"#{position}  {hit.ref}  score={hit.score:.4f}  [{hit.status_protocolo}, v{hit.versao}, {hit.source}]")
        print(textwrap.indent(hit.trecho, "    "))
        print()
    print("Protocolos SIMULADOS para fins educacionais. Trechos exibidos sem interpretação clínica.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
