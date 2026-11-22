"""Template de chat, máscara da perda, configuração e manifesto de split (sem GPU)."""

import json
from pathlib import Path

import pytest

from medprotocol.finetuning.chat_format import (
    IGNORE_INDEX,
    ChatFormatError,
    load_split_records,
    prompt_ids,
    tokenize_example,
)
from medprotocol.finetuning.train_config import load_train_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "finetune" / "qwen2.5-1.5b-lora-v1.json"


class FakeTokenizer:
    """Template no estilo ChatML, tokenizado por caractere (determinístico, sem download)."""

    def __init__(self, inject_default_system=False):
        self.inject_default_system = inject_default_system

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ""
        if self.inject_default_system and not add_generation_prompt:
            text += "<|system|>padrão<|end|>"  # quebra a regra do prefixo de propósito
        for m in messages:
            text += f"<|{m['role']}|>{m['content']}<|end|>"
        if add_generation_prompt:
            text += "<|assistant|>"
        return [ord(c) for c in text]


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "pergunta"},
    {"role": "assistant", "content": "resposta"},
]


def test_loss_only_on_assistant_tokens():
    ex = tokenize_example(FakeTokenizer(), MESSAGES, max_length=1000, qa_id="QA-1")
    assert ex.input_ids[: ex.prompt_length] == prompt_ids(FakeTokenizer(), MESSAGES)
    trained = "".join(chr(t) for t in ex.labels if t != IGNORE_INDEX)
    assert trained == "resposta<|end|>"
    assert all(t == IGNORE_INDEX for t in ex.labels[: ex.prompt_length])


def test_rejects_template_whose_prompt_is_not_a_prefix():
    with pytest.raises(ChatFormatError, match="prefixo"):
        tokenize_example(FakeTokenizer(inject_default_system=True), MESSAGES, max_length=1000)


def test_rejects_examples_longer_than_max_length_instead_of_truncating():
    with pytest.raises(ChatFormatError, match="max_length"):
        tokenize_example(FakeTokenizer(), MESSAGES, max_length=10)


def test_requires_assistant_message():
    with pytest.raises(ChatFormatError):
        tokenize_example(FakeTokenizer(), MESSAGES[:2], max_length=1000)


def test_versioned_config_is_complete_and_pinned():
    config = load_train_config(CONFIG)
    assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert len(config.base_model_revision) == 40
    assert config.split_file == "data/finetune/split_v1.json"
    assert "eval" not in config.sft_file and "eval" not in config.split_file


def test_config_rejects_unknown_fields_and_missing_revision(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({**raw, "epocas": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="desconhecidos"):
        load_train_config(bad)
    bad.write_text(json.dumps({**raw, "base_model_revision": ""}), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_train_config(bad)


def test_split_records_must_match_manifest(tmp_path):
    sft = tmp_path / "sft.jsonl"
    sft.write_text("".join(json.dumps({"messages": [], "metadata": {"qa_id": i}}) + "\n" for i in ("A", "B", "C")), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["A", "B"], "val": ["C"]}), encoding="utf-8")
    parts = load_split_records(sft, split)
    assert [r["metadata"]["qa_id"] for r in parts["val"]] == ["C"]
    split.write_text(json.dumps({"train": ["A"], "val": ["C"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="não corresponde"):
        load_split_records(sft, split)
