"""Treino supervisionado com LoRA (PEFT) num laço PyTorch explícito.

Escolhas para manter o processo simples de explicar:
- **LoRA em bf16**, sem quantização: um modelo de 1,5B cabe na GPU de 12 GB, e QLoRA
  (bitsandbytes) acrescentaria uma dependência frágil no Windows sem necessidade.
- **Laço próprio** em vez de `Trainer`/TRL: cada passo (perda, acumulação, agendamento da taxa
  de aprendizado, recorte do gradiente) fica visível em poucas linhas.
- **Lote 1 sem preenchimento:** os exemplos têm tamanhos muito diferentes; acumulação de
  gradiente faz o papel do lote.
- **Perda só na resposta**, calculada apenas nas posições necessárias (`logits_to_keep`), o que
  evita materializar os logits do vocabulário inteiro para os ~3 mil tokens de contexto.
- **Melhor época pela perda de validação:** o adaptador salvo é o da época com menor perda
  na validação (split interno do treino; a avaliação final nunca é usada).

Requer o extra `[train]` (torch com CUDA, transformers, peft). Os imports pesados ficam dentro
das funções para que o restante do pacote continue utilizável sem eles.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from medprotocol.finetuning.chat_format import TokenizedExample
from medprotocol.finetuning.train_config import TrainConfig


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_model(config: TrainConfig, device: str = "cuda"):
    """Tokenizer e modelo-base na revisão fixada pela configuração."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, revision=config.base_model_revision, dtype=dtype
    ).to(device)
    model.config.use_cache = False
    return tokenizer, model


def add_lora(model, config: TrainConfig):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return get_peft_model(model, lora)


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"treinaveis": trainable, "total": total, "percentual": round(100 * trainable / total, 4)}


def example_loss(model, example: TokenizedExample, device: str = "cuda"):
    """Entropia cruzada média nos tokens da resposta (sem calcular logits do prompt)."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([example.input_ids], device=device)
    keep = example.answer_length + 1  # posições prompt_length-1 .. fim
    logits = model(input_ids=ids, logits_to_keep=keep).logits[0]  # [keep, vocab]
    targets = ids[0, example.prompt_length :]
    return F.cross_entropy(logits[:-1].float(), targets)


def evaluate_loss(model, examples: list[TokenizedExample], device: str = "cuda") -> float:
    """Perda média por token de resposta (ponderada pelo tamanho de cada resposta)."""
    import torch

    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for example in examples:
            loss = example_loss(model, example, device)
            total += loss.item() * example.answer_length
            tokens += example.answer_length
    if was_training:
        model.train()
    return total / tokens if tokens else float("nan")


@dataclass
class TrainHistory:
    passos: list[dict] = field(default_factory=list)
    epocas: list[dict] = field(default_factory=list)
    melhor_epoca: int | None = None
    perda_validacao_base: float | None = None
    segundos: float = 0.0
    pico_memoria_gb: float | None = None


def train_lora(
    model,
    config: TrainConfig,
    train_examples: list[TokenizedExample],
    val_examples: list[TokenizedExample],
    adapter_dir: Path,
    max_steps: int | None = None,
    device: str = "cuda",
    log=print,
) -> TrainHistory:
    """Treina o modelo PEFT e salva em `adapter_dir` o adaptador da melhor época."""
    import torch

    history = TrainHistory()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history.perda_validacao_base = evaluate_loss(model, val_examples, device)
    log(f"perda de validação antes do treino: {history.perda_validacao_base:.4f}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_examples) / config.grad_accum)
    total_steps = steps_per_epoch * config.epochs if max_steps is None else max_steps
    warmup = max(1, int(config.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:  # aquecimento linear, depois decaimento linear até 0
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    order_rng = random.Random(config.seed)
    best = math.inf
    step = 0
    start = time.perf_counter()
    model.train()
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train_examples)))
        order_rng.shuffle(order)
        running, micro = 0.0, 0
        for position, index in enumerate(order, start=1):
            loss = example_loss(model, train_examples[index], device) / config.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro == config.grad_accum or position == len(order):
                grad_norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {
                    "passo": step,
                    "epoca": epoch,
                    "perda_treino": round(running * config.grad_accum / micro, 4),
                    "norma_gradiente": round(grad_norm, 4),
                    "lr": scheduler.get_last_lr()[0],
                }
                history.passos.append(record)
                log(f"passo {step}/{total_steps} época {epoch} perda {record['perda_treino']:.4f}")
                running, micro = 0.0, 0
                if max_steps is not None and step >= max_steps:
                    break
        val_loss = evaluate_loss(model, val_examples, device)
        history.epocas.append({"epoca": epoch, "perda_validacao": round(val_loss, 4), "passos": step})
        log(f"época {epoch}: perda de validação {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            history.melhor_epoca = epoch
            model.save_pretrained(adapter_dir)
        if max_steps is not None and step >= max_steps:
            break
    history.segundos = round(time.perf_counter() - start, 1)
    if device == "cuda":
        history.pico_memoria_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return history
