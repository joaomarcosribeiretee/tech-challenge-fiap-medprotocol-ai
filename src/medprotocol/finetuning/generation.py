"""Geração de respostas com o modelo-base e com o adaptador, nas mesmas condições.

A comparação usa **um único** modelo carregado: o adaptador LoRA é ligado para o modelo
ajustado e desligado (`disable_adapter`) para o modelo-base. Assim os dois recebem os mesmos
tokens de entrada, o mesmo template e a mesma decodificação (gulosa, sem amostragem).
"""

from __future__ import annotations

from dataclasses import dataclass

from medprotocol.finetuning.chat_format import prompt_ids


@dataclass(frozen=True)
class Generation:
    texto: str
    tokens_gerados: int
    atingiu_limite: bool  # parou por max_new_tokens (resposta possivelmente cortada)
    segundos: float


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int, device: str = "cuda") -> Generation:
    import time

    import torch

    ids = torch.tensor([prompt_ids(tokenizer, messages)], device=device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0, ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    hit_limit = len(new_tokens) >= max_new_tokens and int(new_tokens[-1]) != tokenizer.eos_token_id
    return Generation(text, len(new_tokens), hit_limit, round(time.perf_counter() - start, 2))
