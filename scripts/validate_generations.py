"""Aplica a validação determinística do fluxo às respostas salvas pela comparação da Etapa 4.

Uso:
    python scripts/validate_generations.py [--run qwen2.5-1.5b-lora-v1] [--split val]

Reconstrói o mesmo contexto que os modelos viram (treino: seções reservadas excluídas), roda
`validate_answer` + `review_triggers` em cada resposta (base e ajustado) e compara com a leitura
manual (`manual_review_val.json`). Mede quantos erros conhecidos as regras capturam, sem GPU.
Grava reports/finetune/<run>/validation_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.db.repository import connect_readonly
from medprotocol.finetuning.context import build_context
from medprotocol.finetuning.messages import build_messages
from medprotocol.finetuning.sources import RESERVED_SECTIONS, load_training_examples
from medprotocol.graph.validation import ALTA, MEDIA, analyze_question, declares_insufficiency, review_triggers, validate_answer
from medprotocol.rag.pipeline import load_protocol_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="qwen2.5-1.5b-lora-v1")
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)

    settings = load_settings()
    report_dir = Path("reports/finetune") / args.run
    rows = [json.loads(l) for l in (report_dir / f"generations_{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manual = {i["qa_id"]: i for i in json.loads((report_dir / f"manual_review_{args.split}.json").read_text(encoding="utf-8"))["itens"]}
    examples = {e.qa_id: e for e in load_training_examples(settings.data_dir)}
    retriever, _ = load_protocol_retriever(settings.data_dir, settings.vectorstore_dir)

    out, table = [], Counter()
    with connect_readonly(settings.sqlite_path) as conn:
        for row in rows:
            example = examples[row["qa_id"]]
            context = build_context(example.pergunta, retriever, conn, example.patient_id, excluded_refs=RESERVED_SECTIONS)
            user = build_messages(context)[1]["content"]
            analysis = analyze_question(example.pergunta)
            item = {"qa_id": row["qa_id"]}
            for model in ("base", "ajustado"):
                alerts = validate_answer(row[model]["texto"], context, analysis, user)
                triggers = review_triggers(context, analysis)
                flagged = any(a.severidade in (ALTA, MEDIA) for a in alerts)
                verdict = manual[row["qa_id"]][model]
                table[(model, verdict, "sinalizada" if flagged else "não sinalizada")] += 1
                if any(a.severidade == ALTA for a in alerts):
                    status = "revisao_humana (retida)"
                elif flagged or triggers:
                    status = "revisao_humana"
                elif declares_insufficiency(row[model]["texto"]):
                    status = "evidencia_insuficiente"
                else:
                    status = "respondido"
                table[(model, verdict, status)] += 1
                item[model] = {
                    "leitura_manual": verdict,
                    "alertas": [f"{a.severidade}:{a.codigo}" for a in alerts],
                    "gatilhos_de_revisao": [a.codigo for a in triggers],
                    "sinalizada": flagged,
                    "status_do_fluxo": status,
                }
            out.append(item)

    statuses = ("respondido", "evidencia_insuficiente", "revisao_humana", "revisao_humana (retida)")
    summary = {
        f"{model}/{verdict}": {
            "com_alerta": table[(model, verdict, "sinalizada")],
            "sem_alerta": table[(model, verdict, "não sinalizada")],
            **{f"status:{s}": table[(model, verdict, s)] for s in statuses},
        }
        for model in ("base", "ajustado") for verdict in ("correta", "parcial", "incorreta")
    }
    (report_dir / f"validation_{args.split}.json").write_text(
        json.dumps({"resumo": summary, "itens": out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in out:
        print(item["qa_id"], "| ajustado:", item["ajustado"]["leitura_manual"], "->", item["ajustado"]["status_do_fluxo"], item["ajustado"]["alertas"], item["ajustado"]["gatilhos_de_revisao"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
