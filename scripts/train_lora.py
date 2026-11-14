"""Treina um adaptador LoRA sobre o modelo-base (fine-tuning supervisionado).

Uso (ambiente de treino, ver docs/FINETUNING.md):
    python scripts/train_lora.py --config configs/finetune/qwen2.5-1.5b-lora-v1.json
    python scripts/train_lora.py --config ... --smoke     # 2 passos, confere o pipeline e a memória

Entradas: data/processed/sft/train.jsonl (gerado por build_sft_dataset.py) e o manifesto de
split. Saídas:
- models/adapters/<run_name>/            adaptador da melhor época (não versionado)
- reports/finetune/<run_name>/train_report.json   parâmetros, perdas, tempo e memória
Com --smoke, as saídas ficam em <run_name>-smoke.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from medprotocol.finetuning.chat_format import load_split_records, tokenize_example
from medprotocol.finetuning.lora_trainer import add_lora, count_parameters, load_base_model, set_seed, train_lora
from medprotocol.finetuning.train_config import load_train_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="2 passos com 4 exemplos de treino e 2 de validação")
    args = parser.parse_args(argv)

    import torch
    import transformers
    import peft

    if not torch.cuda.is_available():
        print("CUDA indisponível: o treino requer GPU (veja docs/FINETUNING.md).", file=sys.stderr)
        return 2

    config = load_train_config(args.config)
    run_name = config.run_name + ("-smoke" if args.smoke else "")
    adapter_dir = Path(config.output_dir) / run_name
    report_dir = Path(config.report_dir) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    splits = load_split_records(Path(config.sft_file), Path(config.split_file))
    tokenizer, model = load_base_model(config)
    tokenized = {
        name: [tokenize_example(tokenizer, r["messages"], config.max_length, r["metadata"]["qa_id"]) for r in records]
        for name, records in splits.items()
    }
    train_examples, val_examples = tokenized["train"], tokenized["val"]
    max_steps = None
    if args.smoke:
        train_examples = sorted(train_examples, key=lambda e: -len(e.input_ids))[:4]  # os maiores: pior caso de memória
        val_examples = val_examples[:2]
        max_steps = 2
    lengths = [len(e.input_ids) for e in tokenized["train"] + tokenized["val"]]
    print(f"tokens por exemplo: min={min(lengths)} max={max(lengths)} (max_length={config.max_length})")

    model = add_lora(model, config)
    params = count_parameters(model)
    print(f"parâmetros treináveis: {params}")
    history = train_lora(model, config, train_examples, val_examples, adapter_dir, max_steps=max_steps)

    report = {
        "run_name": run_name,
        "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_json(),
        "smoke": args.smoke,
        "ambiente": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        },
        "dados": {
            "exemplos_treino": len(train_examples),
            "exemplos_validacao": len(val_examples),
            "tokens_min": min(lengths),
            "tokens_max": max(lengths),
            "tokens_resposta_treino": sum(e.answer_length for e in train_examples),
        },
        "parametros": params,
        "perda_validacao_antes": round(history.perda_validacao_base, 4),
        "epocas": history.epocas,
        "melhor_epoca": history.melhor_epoca,
        "passos": history.passos,
        "segundos_treino": history.segundos,
        "pico_memoria_gpu_gb": history.pico_memoria_gb,
        "adaptador": str(adapter_dir),
    }
    out = report_dir / "train_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"adaptador -> {adapter_dir} | relatório -> {out}")
    print(f"tempo {history.segundos}s | pico de memória {history.pico_memoria_gb} GB | melhor época {history.melhor_epoca}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
