"""Geradores de texto usados pelo fluxo: Qwen2.5-1.5B local, com ou sem o adaptador LoRA.

- Um único modelo é carregado, sob demanda (na primeira pergunta). O adaptador é ligado para a
  variante "ajustado" e desligado (`disable_adapter`) para a variante "base": alternar não
  carrega uma segunda cópia do modelo.
- Falhas de carregamento viram `ModelLoadError`, e falhas de geração viram `GenerationError`.
  O fluxo as converte em status "erro", sem derrubar o processo.
- Os testes usam qualquer objeto com o mesmo método `generate` (LLM simulado, sem GPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from medprotocol.finetuning.train_config import TrainConfig, load_train_config

VARIANTS = ("ajustado", "base")


class ModelLoadError(RuntimeError):
    """O modelo-base ou o adaptador não pôde ser carregado."""


class GenerationError(RuntimeError):
    """O modelo foi carregado, mas a geração falhou."""


class TextGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...

    def describe(self) -> dict: ...


class LocalQwenGenerator:
    """Modelo-base + adaptador LoRA da Etapa 4, decodificação gulosa (determinística)."""

    def __init__(self, config_path: Path, variant: str = "ajustado", adapter_dir: Path | None = None, device: str = "cuda"):
        if variant not in VARIANTS:
            raise ValueError(f"variante {variant!r} inválida; use {VARIANTS}")
        self.config_path = Path(config_path)
        self.variant = variant
        self.device = device
        self._adapter_dir = adapter_dir
        self._config: TrainConfig | None = None
        self._model = None
        self._tokenizer = None

    @property
    def config(self) -> TrainConfig:
        if self._config is None:
            try:
                self._config = load_train_config(self.config_path)
            except (OSError, ValueError) as exc:
                raise ModelLoadError(f"configuração do modelo inválida: {exc}") from exc
        return self._config

    @property
    def adapter_dir(self) -> Path:
        return self._adapter_dir or Path(self.config.output_dir) / self.config.run_name

    def set_variant(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variante {variant!r} inválida; use {VARIANTS}")
        self.variant = variant

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel

            from medprotocol.finetuning.lora_trainer import load_base_model

            if self.device == "cuda" and not torch.cuda.is_available():
                raise ModelLoadError("CUDA indisponível; o modelo local requer GPU (ver docs/ARCHITECTURE.md)")
            if not (self.adapter_dir / "adapter_config.json").exists():
                raise ModelLoadError(f"adaptador não encontrado em {self.adapter_dir}; rode scripts/train_lora.py")
            tokenizer, base = load_base_model(self.config, self.device)
            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()
        except ModelLoadError:
            raise
        except Exception as exc:  # dependências ausentes, download, memória...
            raise ModelLoadError(f"falha ao carregar o modelo: {type(exc).__name__}: {exc}") from exc
        self._tokenizer, self._model = tokenizer, model

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.load()
        from medprotocol.finetuning.generation import generate

        try:
            if self.variant == "base":
                with self._model.disable_adapter():
                    out = generate(self._model, self._tokenizer, messages, self.config.max_new_tokens, self.device)
            else:
                out = generate(self._model, self._tokenizer, messages, self.config.max_new_tokens, self.device)
        except Exception as exc:
            raise GenerationError(f"falha na geração: {type(exc).__name__}: {exc}") from exc
        return out.texto

    def describe(self) -> dict:
        info = {"variante": self.variant, "config": str(self.config_path)}
        if self._config is not None:
            info.update(
                base=self._config.base_model,
                revisao=self._config.base_model_revision[:12],
                adaptador=self._config.run_name if self.variant == "ajustado" else None,
            )
        return info
