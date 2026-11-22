"""Laço de treino LoRA de ponta a ponta num modelo minúsculo criado localmente (CPU, sem download).

Só roda no ambiente com o extra [train] (peft instalado).
"""

from pathlib import Path

import pytest

pytest.importorskip("peft")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from medprotocol.finetuning.chat_format import TokenizedExample  # noqa: E402
from medprotocol.finetuning.lora_trainer import add_lora, count_parameters, evaluate_loss, set_seed, train_lora  # noqa: E402
from medprotocol.finetuning.train_config import LoraParams, TrainConfig  # noqa: E402


def tiny_model():
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config)


def config(**overrides):
    base = dict(
        run_name="tiny", base_model="local", base_model_revision="x" * 40, epochs=2, grad_accum=2,
        learning_rate=5e-3, gradient_checkpointing=False, lora=LoraParams(r=4, alpha=8, dropout=0.0),
    )
    return TrainConfig(**{**base, **overrides})


def examples(n=4):
    return [TokenizedExample(f"QA-{i}", [1, 2, 3, 4, 5 + i, 6, 7, 8], prompt_length=4) for i in range(n)]


def test_training_reduces_loss_saves_adapter_and_only_lora_is_trainable(tmp_path):
    set_seed(0)
    cfg = config()
    model = add_lora(tiny_model(), cfg)
    params = count_parameters(model)
    assert 0 < params["treinaveis"] < params["total"]
    data = examples()
    history = train_lora(model, cfg, data, data[:2], tmp_path / "adapter", device="cpu", log=lambda *_: None)
    assert len(history.epocas) == 2
    assert history.epocas[-1]["perda_validacao"] < history.perda_validacao_base
    assert history.passos[0]["passo"] == 1 and len(history.passos) == 4  # 4 exemplos / acumulação 2 × 2 épocas
    assert (tmp_path / "adapter" / "adapter_config.json").exists()


def test_training_is_reproducible_with_same_seed(tmp_path):
    losses = []
    for run in range(2):
        set_seed(123)
        cfg = config(seed=123)
        model = add_lora(tiny_model(), cfg)
        history = train_lora(model, cfg, examples(), examples()[:2], tmp_path / f"a{run}", device="cpu", log=lambda *_: None)
        losses.append([p["perda_treino"] for p in history.passos])
    assert losses[0] == losses[1]


def test_loss_with_logits_to_keep_equals_masked_full_loss():
    """A perda calculada só nas posições da resposta é igual à perda completa com rótulos -100."""
    set_seed(0)
    model = add_lora(tiny_model(), config())
    model.eval()
    ex = TokenizedExample("a", [1, 2, 3, 4, 9, 10, 11], prompt_length=4)
    ids = torch.tensor([ex.input_ids])
    with torch.no_grad():
        reference = model(input_ids=ids, labels=torch.tensor([ex.labels])).loss.item()
    assert evaluate_loss(model, [ex], "cpu") == pytest.approx(reference, rel=1e-5)
