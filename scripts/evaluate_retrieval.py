"""Avalia o retriever no conjunto de DESENVOLVIMENTO e sugere limiares de confiança.

Uso:
    python scripts/evaluate_retrieval.py [--data-dir data]

Nunca usa data/eval/qa_eval.jsonl. Métricas: Hit@1, Hit@3, MRR (escopo) e rejeição
(fora do escopo), calculadas com os limiares configurados em retriever.py; a calibração
sugerida é impressa para revisão, não aplicada automaticamente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.rag.evaluation import calibrate, load_dev_queries, metrics, rank_queries
from medprotocol.rag.pipeline import load_protocol_retriever
from medprotocol.rag.retriever import classify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or load_settings().data_dir
    retriever, _ = load_protocol_retriever(data_dir, cache_dir=None)
    outcomes = rank_queries(retriever.store, load_dev_queries(data_dir))
    lower, upper = retriever.lower_threshold, retriever.upper_threshold

    print(f"{'id':<8} {'escopo':<6} {'rank':>4} {'top1':>7} {'top1_ref':<18} {'status':<24} esperado")
    for o in outcomes:
        rank = "-" if o.rank is None else str(o.rank)
        print(
            f"{o.query_id:<8} {o.escopo:<6} {rank:>4} {o.top_score:>7.4f} {o.ranking[0][0]:<18} "
            f"{classify(o.top_score, lower, upper):<24} {','.join(o.refs_esperadas) or '-'}"
        )

    print(f"\nMétricas (limiares configurados lower={lower}, upper={upper}):")
    for name, value in metrics(outcomes, lower, upper).items():
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")

    cal = calibrate(outcomes)
    print("\nCalibração sugerida (somente conjunto de desenvolvimento):")
    print(f"  menor melhor-score dentro do escopo: {cal.min_in_scope:.4f}")
    print(f"  maior melhor-score fora do escopo:   {cal.max_out_of_scope:.4f}")
    if cal.separable:
        print(f"  separação observada -> limiar único {cal.lower}")
    else:
        print(f"  SOBREPOSIÇÃO -> faixa 'incerto' [{cal.lower}, {cal.upper})")
    print("\nMétricas com a calibração sugerida:")
    for name, value in metrics(outcomes, cal.lower, cal.upper).items():
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
