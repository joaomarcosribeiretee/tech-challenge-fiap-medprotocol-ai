"""Avaliação do retriever no conjunto de desenvolvimento (nunca no qa_eval.jsonl).

Métricas (perguntas do escopo): Hit@1, Hit@3 e MRR, considerando acerto quando qualquer
uma das `refs_esperadas` aparece. Perguntas fora do escopo: taxa de rejeição (status
diferente de "ok").

Calibração dos limiares (apenas com o conjunto de desenvolvimento):
- lower = menor "melhor score" entre perguntas do escopo;
- upper = maior "melhor score" entre perguntas fora do escopo;
- se lower > upper, há separação: um único limiar no ponto médio (sem faixa "incerto");
- se lower <= upper, há sobreposição: a faixa [lower, upper) vira "incerto".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

DEV_FILE = Path("rag_dev") / "retrieval_dev.jsonl"
SCOPES = ("dentro", "fora")


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    pergunta: str
    escopo: str
    refs_esperadas: list[str]
    ranking: list[tuple[str, float]]  # (ref, score) em ordem decrescente
    rank: int | None  # posição (1-based) da primeira ref esperada

    @property
    def top_score(self) -> float:
        return self.ranking[0][1]


@dataclass(frozen=True)
class Calibration:
    lower: float
    upper: float
    separable: bool
    min_in_scope: float
    max_out_of_scope: float


def load_dev_queries(data_dir: Path) -> list[dict]:
    path = Path(data_dir) / DEV_FILE
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["query_id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: query_id duplicado")
    for q in queries:
        if q["escopo"] not in SCOPES:
            raise ValueError(f"{q['query_id']}: escopo inválido {q['escopo']!r}")
        if (q["escopo"] == "dentro") != bool(q["refs_esperadas"]):
            raise ValueError(f"{q['query_id']}: refs_esperadas deve existir só para perguntas do escopo")
    return queries


def rank_queries(store: InMemoryVectorStore, queries: list[dict]) -> list[QueryOutcome]:
    total = len(store.store)
    outcomes = []
    for q in queries:
        ranking = [(doc.metadata["ref"], float(score)) for doc, score in store.similarity_search_with_score(q["pergunta"], k=total)]
        rank = next((i for i, (ref, _) in enumerate(ranking, start=1) if ref in q["refs_esperadas"]), None)
        outcomes.append(QueryOutcome(q["query_id"], q["pergunta"], q["escopo"], q["refs_esperadas"], ranking, rank))
    return outcomes


def _round_down(value: float) -> float:
    """4 casas, nunca acima do valor: a pergunta do escopo de menor score fica >= lower."""
    rounded = round(value, 4)
    return rounded if rounded <= value else round(rounded - 1e-4, 4)


def _round_above(value: float) -> float:
    """4 casas, estritamente acima do valor: a pergunta fora do escopo de maior score fica < upper."""
    rounded = round(value, 4)
    return rounded if rounded > value else round(rounded + 1e-4, 4)


def calibrate(outcomes: list[QueryOutcome]) -> Calibration:
    inside = [o.top_score for o in outcomes if o.escopo == "dentro"]
    outside = [o.top_score for o in outcomes if o.escopo == "fora"]
    if not inside or not outside:
        raise ValueError("calibração exige perguntas dentro e fora do escopo")
    low, high = min(inside), max(outside)
    if low > high:
        middle = round((low + high) / 2, 4)
        return Calibration(middle, middle, True, low, high)
    return Calibration(_round_down(low), _round_above(high), False, low, high)


def metrics(outcomes: list[QueryOutcome], lower: float, upper: float) -> dict[str, float]:
    from medprotocol.rag.retriever import STATUS_OK, classify

    inside = [o for o in outcomes if o.escopo == "dentro"]
    outside = [o for o in outcomes if o.escopo == "fora"]
    return {
        "perguntas_escopo": len(inside),
        "perguntas_fora": len(outside),
        "hit@1": sum(1 for o in inside if o.rank == 1) / len(inside),
        "hit@3": sum(1 for o in inside if o.rank is not None and o.rank <= 3) / len(inside),
        "mrr": sum(1 / o.rank for o in inside if o.rank) / len(inside),
        "rejeicao_fora": sum(1 for o in outside if classify(o.top_score, lower, upper) != STATUS_OK) / len(outside),
        "escopo_com_status_ok": sum(1 for o in inside if classify(o.top_score, lower, upper) == STATUS_OK) / len(inside),
    }
