"""Separação reproduzível dos exemplos de treino em treino e validação.

Prioridades, nesta ordem:
1. **Sem vazamento por paciente:** todos os exemplos de um paciente ficam no mesmo lado.
2. **Sem quase-duplicatas entre lados:** pares com similaridade de pergunta ou de resposta
   acima do limiar ficam do mesmo lado.
3. **Distribuição razoável de categorias** na validação, sem esvaziar nenhuma categoria do treino.

O procedimento é determinístico:
- Pacientes da validação: busca exaustiva entre as combinações de `n_val_patients` pacientes
  (em ordem lexicográfica). Vence a que cobre mais categorias, depois a que fica mais perto de
  `target_patient_examples` exemplos; empate fica com a primeira combinação.
- Exemplos sem paciente: um por categoria, sorteado com `random.Random(seed)`, apenas nas
  categorias com pelo menos `min_no_patient_for_val` exemplos sem paciente.
- Quase-duplicatas: um exemplo de validação sem paciente que se pareça com algum do treino
  volta para o treino. Se o exemplo de validação tiver paciente, o par do treino (sem paciente)
  vai para a validação; se os dois tiverem pacientes diferentes, o par é só reportado.

Usa apenas `qa_train.jsonl` (via `sources`); a avaliação final nunca entra aqui.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from medprotocol.data.validation import text_similarity
from medprotocol.finetuning.sources import TrainingExample

SPLIT_VERSION = "1"


@dataclass(frozen=True)
class SplitParams:
    seed: int = 20260630
    n_val_patients: int = 3
    target_patient_examples: int = 9
    min_no_patient_for_val: int = 3
    min_train_per_category: int = 2
    similarity_threshold: float = 0.6


@dataclass
class SplitResult:
    train: list[str]
    val: list[str]
    val_patients: list[str]
    params: SplitParams
    moves: list[str] = field(default_factory=list)
    unresolved_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    max_cross_similarity: float = 0.0

    def to_json(self, categories: dict[str, str]) -> dict:
        return {
            "split_version": SPLIT_VERSION,
            "params": self.params.__dict__,
            "val_patients": self.val_patients,
            "counts": {
                "train": len(self.train),
                "val": len(self.val),
                "train_por_categoria": dict(sorted(Counter(categories[i] for i in self.train).items())),
                "val_por_categoria": dict(sorted(Counter(categories[i] for i in self.val).items())),
            },
            "max_similaridade_entre_lados": round(self.max_cross_similarity, 3),
            "ajustes_por_similaridade": self.moves,
            "pares_nao_resolvidos": [list(p) for p in self.unresolved_pairs],
            "train": self.train,
            "val": self.val,
        }


def _category(example: TrainingExample) -> str:
    return example.rotulos["categoria"]


def _similarity(a: TrainingExample, b: TrainingExample) -> float:
    return max(text_similarity(a.pergunta, b.pergunta), text_similarity(a.resposta_esperada, b.resposta_esperada))


def _choose_val_patients(examples: list[TrainingExample], params: SplitParams) -> list[str]:
    patients = sorted({e.patient_id for e in examples if e.patient_id})
    all_counts = Counter(_category(e) for e in examples)
    best: tuple | None = None
    for combo in combinations(patients, params.n_val_patients):
        chosen = [e for e in examples if e.patient_id in combo]
        val_counts = Counter(_category(e) for e in chosen)
        if any(all_counts[c] - n < params.min_train_per_category for c, n in val_counts.items()):
            continue
        score = (len(val_counts), -abs(len(chosen) - params.target_patient_examples))
        if best is None or score > best[0]:
            best = (score, list(combo))
    if best is None:
        raise ValueError("nenhuma combinação de pacientes preserva o mínimo de exemplos por categoria no treino")
    return best[1]


def split_examples(examples: list[TrainingExample], params: SplitParams = SplitParams()) -> SplitResult:
    by_id = {e.qa_id: e for e in examples}
    if len(by_id) != len(examples):
        raise ValueError("qa_id repetido")
    val_patients = _choose_val_patients(examples, params)
    val = {e.qa_id for e in examples if e.patient_id in val_patients}

    rng = random.Random(params.seed)
    no_patient: dict[str, list[str]] = {}
    for e in sorted(examples, key=lambda x: x.qa_id):
        if e.patient_id is None:
            no_patient.setdefault(_category(e), []).append(e.qa_id)
    for category in sorted(no_patient):
        ids = no_patient[category]
        if len(ids) >= params.min_no_patient_for_val:
            val.add(rng.choice(ids))

    moves: list[str] = []
    pinned: set[str] = set()
    changed = True
    while changed:
        changed = False
        for v in sorted(val):
            for t in sorted(set(by_id) - val):
                score = _similarity(by_id[v], by_id[t])
                if score < params.similarity_threshold:
                    continue
                if by_id[v].patient_id is None and v not in pinned:
                    val.discard(v)
                    moves.append(f"{v} -> treino (similar a {t}: {score:.3f})")
                elif by_id[t].patient_id is None:
                    # Fixado na validação para o laço terminar (não volta ao treino).
                    val.add(t)
                    pinned.add(t)
                    moves.append(f"{t} -> validação (similar a {v}: {score:.3f})")
                else:
                    continue  # pacientes diferentes nos dois lados: reportado abaixo
                changed = True
                break
            if changed:
                break

    train = sorted(set(by_id) - val)
    val_sorted = sorted(val)
    train_patients = {by_id[i].patient_id for i in train} - {None}
    leaked = train_patients & {by_id[i].patient_id for i in val_sorted}
    if leaked:
        raise AssertionError(f"pacientes nos dois lados: {sorted(leaked)}")
    scores = [(v, t, _similarity(by_id[v], by_id[t])) for v in val_sorted for t in train]
    unresolved = [(v, t, round(s, 3)) for v, t, s in scores if s >= params.similarity_threshold]
    max_sim = max((s for _, _, s in scores), default=0.0)
    return SplitResult(train, val_sorted, val_patients, params, moves, unresolved, max_sim)
