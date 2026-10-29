"""Configuração da aplicação a partir de variáveis de ambiente.

Centraliza a leitura de configurações para que nenhum outro módulo acesse
`os.environ` diretamente. O parâmetro `env` permite injetar valores nos testes
sem alterar o ambiente real do processo.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

VALID_APP_ENVS = frozenset({"dev", "test", "prod"})
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str
    data_dir: Path
    sqlite_path: Path
    vectorstore_dir: Path
    base_model_name: str
    adapter_dir: Path


def _choice(env: Mapping[str, str], key: str, default: str, valid: frozenset[str]) -> str:
    value = env.get(key, default).strip()
    if key == "LOG_LEVEL":
        value = value.upper()
    if value not in valid:
        raise ValueError(f"{key}={value!r} inválido. Valores aceitos: {sorted(valid)}")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Lê as configurações de `env` (padrão: `os.environ`), aplicando defaults."""
    env = os.environ if env is None else env
    return Settings(
        app_env=_choice(env, "APP_ENV", "dev", VALID_APP_ENVS),
        log_level=_choice(env, "LOG_LEVEL", "INFO", VALID_LOG_LEVELS),
        data_dir=Path(env.get("DATA_DIR", "data")),
        sqlite_path=Path(env.get("SQLITE_PATH", "data/processed/medprotocol.db")),
        vectorstore_dir=Path(env.get("VECTORSTORE_DIR", "data/vectorstore")),
        base_model_name=env.get("BASE_MODEL_NAME", ""),
        adapter_dir=Path(env.get("ADAPTER_DIR", "models/adapters")),
    )
