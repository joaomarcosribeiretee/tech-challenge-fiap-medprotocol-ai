"""Tokenização dos exemplos no template de chat do modelo, com perda só na resposta.

O template vem do próprio tokenizer (`apply_chat_template`), o mesmo usado na inferência.
A perda é calculada só nos tokens da mensagem do assistente: o modelo não é treinado para
reproduzir o prompt de sistema, os trechos nem os dados do paciente.

Regra de consistência: os tokens do prompt de inferência (system + user + marcador de início
da resposta) precisam ser exatamente o prefixo da conversa completa. Se o template não
garantir isso, o exemplo é rejeitado, em vez de treinar com uma fronteira errada.
Exemplos acima de `max_length` também são rejeitados: truncar cortaria a resposta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IGNORE_INDEX = -100


class ChatFormatError(ValueError):
    """O exemplo não pode ser tokenizado de forma segura."""


@dataclass(frozen=True)
class TokenizedExample:
    qa_id: str
    input_ids: list[int]
    prompt_length: int  # tokens antes da resposta; a perda começa aqui

    @property
    def labels(self) -> list[int]:
        return [IGNORE_INDEX] * self.prompt_length + self.input_ids[self.prompt_length :]

    @property
    def answer_length(self) -> int:
        return len(self.input_ids) - self.prompt_length


def _ids(result) -> list[int]:
    # transformers 5 pode devolver BatchEncoding/dict em vez de lista.
    if isinstance(result, dict) or hasattr(result, "keys"):
        result = result["input_ids"]
    return list(result)


def prompt_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Tokens de inferência: mensagens sem a resposta + marcador de início do assistente."""
    prompt = [m for m in messages if m["role"] != "assistant"]
    return _ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))


def tokenize_example(tokenizer, messages: list[dict[str, str]], max_length: int, qa_id: str = "?") -> TokenizedExample:
    if not messages or messages[-1]["role"] != "assistant":
        raise ChatFormatError(f"{qa_id}: a última mensagem precisa ser do assistente")
    prompt = prompt_ids(tokenizer, messages)
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if full[: len(prompt)] != prompt:
        raise ChatFormatError(f"{qa_id}: o prompt de inferência não é prefixo da conversa completa")
    if len(full) <= len(prompt):
        raise ChatFormatError(f"{qa_id}: resposta vazia após a tokenização")
    if len(full) > max_length:
        raise ChatFormatError(f"{qa_id}: {len(full)} tokens > max_length={max_length} (não truncamos)")
    return TokenizedExample(qa_id, full, len(prompt))


def load_split_records(sft_file: Path, split_file: Path) -> dict[str, list[dict]]:
    """Separa os registros SFT pelo manifesto de split. Falha se faltar ou sobrar exemplo."""
    records = [json.loads(line) for line in Path(sft_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["metadata"]["qa_id"]: r for r in records}
    manifest = json.loads(Path(split_file).read_text(encoding="utf-8"))
    train, val = manifest["train"], manifest["val"]
    if set(train) & set(val):
        raise ValueError("exemplo nos dois lados do split")
    if set(train) | set(val) != set(by_id):
        raise ValueError("o manifesto de split não corresponde ao arquivo SFT; gere os dois de novo")
    return {"train": [by_id[i] for i in train], "val": [by_id[i] for i in val]}
