"""Configuração reproduzível do fine-tuning (LoRA), lida de um JSON versionado.

Nenhum valor de treino fica escondido no código: o JSON em `configs/finetune/` é a fonte de
verdade, e uma cópia dele é gravada junto do adaptador e do relatório de cada execução.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LoraParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    base_model: str
    base_model_revision: str
    sft_file: str = "data/processed/sft/train.jsonl"
    split_file: str = "data/finetune/split_v1.json"
    output_dir: str = "models/adapters"
    report_dir: str = "reports/finetune"
    seed: int = 42
    max_length: int = 6144
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    max_new_tokens: int = 512
    lora: LoraParams = field(default_factory=LoraParams)

    def to_json(self) -> dict:
        data = asdict(self)
        data["lora"]["target_modules"] = list(self.lora.target_modules)
        return data


def load_train_config(path: Path) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lora = raw.pop("lora", {})
    if "target_modules" in lora:
        lora["target_modules"] = tuple(lora["target_modules"])
    unknown = set(raw) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"campos desconhecidos na configuração: {sorted(unknown)}")
    if not raw.get("base_model_revision"):
        raise ValueError("base_model_revision é obrigatório (reprodutibilidade)")
    if raw.get("batch_size", 1) != 1:
        raise ValueError("batch_size > 1 não é suportado: os exemplos não são preenchidos (padding)")
    return TrainConfig(**raw, lora=LoraParams(**lora))
