"""Compara modelo-base e modelo ajustado nas mesmas perguntas, com o mesmo contexto.

Uso (ambiente de treino):
    python scripts/compare_models.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json [--split val] [--limit N]

Para cada exemplo do split (padrão: validação interna, nunca o qa_eval):
1. usa as mensagens system + user do arquivo SFT (o mesmo contexto RAG + SQLite do treino);
2. gera com o adaptador desligado (base) e ligado (ajustado), decodificação gulosa;
3. calcula as métricas heurísticas de `answer_metrics` para as duas respostas e para a
   resposta de referência (calibração das métricas);
4. mede a perda nas respostas de referência com e sem o adaptador.

Saídas em reports/finetune/<run_name>/: comparison_<split>.json (agregados) e
generations_<split>.jsonl (respostas completas, para leitura humana).
A perda e as heurísticas não são avaliação clínica.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medprotocol.config import load_settings
from medprotocol.finetuning.answer_metrics import score_answer, summarize
from medprotocol.finetuning.chat_format import load_split_records, tokenize_example
from medprotocol.finetuning.generation import generate
from medprotocol.finetuning.lora_trainer import evaluate_loss, load_base_model, set_seed
from medprotocol.finetuning.sources import load_training_examples
from medprotocol.finetuning.train_config import load_train_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter", type=Path, default=None, help="padrão: models/adapters/<run_name>")
    parser.add_argument("--rescore", action="store_true", help="recalcula as métricas das gerações salvas (sem GPU)")
    args = parser.parse_args(argv)
    if args.rescore:
        return rescore(load_train_config(args.config), args.split)

    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        print("CUDA indisponível.", file=sys.stderr)
        return 2
    config = load_train_config(args.config)
    adapter_dir = args.adapter or Path(config.output_dir) / config.run_name
    report_dir = Path(config.report_dir) / config.run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    records = load_split_records(Path(config.sft_file), Path(config.split_file))[args.split][: args.limit]
    labels = {e.qa_id: e.rotulos for e in load_training_examples(load_settings().data_dir)}
    tokenizer, base = load_base_model(config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    tokenized = [tokenize_example(tokenizer, r["messages"], config.max_length, r["metadata"]["qa_id"]) for r in records]
    with model.disable_adapter():
        loss_base = evaluate_loss(model, tokenized)
    loss_tuned = evaluate_loss(model, tokenized)

    rows, scores = [], {"base": [], "ajustado": [], "referencia": []}
    for record in records:
        qa_id = record["metadata"]["qa_id"]
        prompt = record["messages"][:2]
        user = prompt[1]["content"]
        reference = record["messages"][2]["content"]
        with model.disable_adapter():
            out_base = generate(model, tokenizer, prompt, config.max_new_tokens)
        out_tuned = generate(model, tokenizer, prompt, config.max_new_tokens)
        row = {"qa_id": qa_id, "categoria": labels[qa_id]["categoria"], "pergunta": user.split("## Pergunta\n", 1)[1].split("\n", 1)[0]}
        for name, text, gen in (("base", out_base.texto, out_base), ("ajustado", out_tuned.texto, out_tuned), ("referencia", reference, None)):
            metrics = score_answer(text, user, labels[qa_id], qa_id)
            scores[name].append(metrics)
            row[name] = {"texto": text, "metricas": metrics.to_json()}
            if gen is not None:
                row[name].update(tokens=gen.tokens_gerados, cortada=gen.atingiu_limite, segundos=gen.segundos)
        rows.append(row)
        print(f"{qa_id}: base {out_base.tokens_gerados} tok / ajustado {out_tuned.tokens_gerados} tok")

    summary = {
        "run_name": config.run_name,
        "split": args.split,
        "n": len(records),
        "decodificacao": {"do_sample": False, "max_new_tokens": config.max_new_tokens},
        "perda_respostas_referencia": {"base": round(loss_base, 4), "ajustado": round(loss_tuned, 4)},
        "respostas_cortadas": {
            "base": sum(1 for r in rows if r["base"]["cortada"]),
            "ajustado": sum(1 for r in rows if r["ajustado"]["cortada"]),
        },
        "metricas": {name: summarize(values) for name, values in scores.items()},
        "aviso": "Heurísticas automáticas e perda não são avaliação clínica; leia as respostas em generations_*.jsonl.",
    }
    (report_dir / f"comparison_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with (report_dir / f"generations_{args.split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def rescore(config, split: str) -> int:
    """Recalcula as métricas a partir de generations_<split>.jsonl, sem gerar de novo."""
    report_dir = Path(config.report_dir) / config.run_name
    gen_file = report_dir / f"generations_{split}.jsonl"
    summary_file = report_dir / f"comparison_{split}.json"
    rows = [json.loads(line) for line in gen_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    users = {r["metadata"]["qa_id"]: r["messages"][1]["content"] for r in load_split_records(Path(config.sft_file), Path(config.split_file))[split]}
    labels = {e.qa_id: e.rotulos for e in load_training_examples(load_settings().data_dir)}
    scores = {"base": [], "ajustado": [], "referencia": []}
    for row in rows:
        for name in scores:
            metrics = score_answer(row[name]["texto"], users[row["qa_id"]], labels[row["qa_id"]], row["qa_id"])
            row[name]["metricas"] = metrics.to_json()
            scores[name].append(metrics)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["metricas"] = {name: summarize(values) for name, values in scores.items()}
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with gen_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metricas"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
