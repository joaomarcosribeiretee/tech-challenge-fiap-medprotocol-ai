"""Pré-registra o contexto selecionado para um lote de perguntas, ANTES de existirem respostas.

Uso:
    python scripts/preregister_context.py data/finetune/piloto1/perguntas.jsonl \
        --out data/finetune/piloto1/contexto_preregistrado.json

Lê de cada linha apenas `qa_id`, `pergunta` e `patient_id` (outros campos são ignorados) e roda
a política de contexto congelada no modo de treino (seções reservadas excluídas). Grava, para
auditoria: versões da política e do prompt, SHA-256 de context_policy.py, limiares do RAG, k e,
por pergunta, status do retriever, trechos recuperados (com score) e seções por regra (com motivos).
Assim a cobertura medida depois não pode ter influenciado a seleção.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.db.repository import connect_readonly
from medprotocol.finetuning import context_policy
from medprotocol.finetuning.context import build_context
from medprotocol.finetuning.messages import PROMPT_VERSION
from medprotocol.finetuning.sources import RESERVED_SECTIONS, ReservedDataError
from medprotocol.data.validation import EVAL_ONLY_PATIENTS
from medprotocol.rag.pipeline import load_protocol_retriever
from medprotocol.rag.retriever import DEFAULT_K


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    settings = load_settings()
    rows = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever, _ = load_protocol_retriever(settings.data_dir, settings.vectorstore_dir)

    results = []
    with connect_readonly(settings.sqlite_path) as conn:
        for row in rows:
            qa_id, pergunta, patient_id = row["qa_id"], row["pergunta"], row["patient_id"]
            if patient_id in EVAL_ONLY_PATIENTS or any(p in pergunta for p in EVAL_ONLY_PATIENTS):
                raise ReservedDataError(f"{qa_id}: paciente reservado para a avaliação")
            if patient_id is not None and patient_id not in pergunta:
                raise ValueError(f"{qa_id}: patient_id {patient_id} não aparece na pergunta")
            context = build_context(pergunta, retriever, conn, patient_id, k=args.k, excluded_refs=RESERVED_SECTIONS)
            results.append(
                {
                    "qa_id": qa_id,
                    "pergunta": pergunta,
                    "patient_id": patient_id,
                    "status_recuperacao": context.recuperacao.status,
                    "refs_recuperadas": [{"ref": h.ref, "score": round(h.score, 4)} for h in context.recuperacao.hits],
                    "refs_por_regra": {s.ref: list(s.motivos) for s in context.secoes_por_regra},
                    "regras_em_recuperados": {r: list(m) for r, m in context.regras_em_recuperados.items()},
                }
            )

    policy_file = Path(context_policy.__file__)
    report = {
        "policy_version": context_policy.POLICY_VERSION,
        "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "k": args.k,
        "limiares": {"lower": retriever.lower_threshold, "upper": retriever.upper_threshold},
        "secoes_excluidas": sorted(RESERVED_SECTIONS),
        "perguntas": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(results)} perguntas pré-registradas -> {args.out} (política v{report['policy_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
