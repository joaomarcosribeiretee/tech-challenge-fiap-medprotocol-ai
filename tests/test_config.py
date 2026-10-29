from pathlib import Path

import pytest

from medprotocol.config import load_settings


def test_defaults_when_env_is_empty():
    settings = load_settings({})
    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.sqlite_path == Path("data/processed/medprotocol.db")
    assert settings.base_model_name == ""


def test_values_are_read_from_env():
    settings = load_settings(
        {"APP_ENV": "test", "LOG_LEVEL": "debug", "DATA_DIR": "/tmp/dados", "BASE_MODEL_NAME": "modelo-x"}
    )
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("/tmp/dados")
    assert isinstance(settings.adapter_dir, Path)
    assert settings.base_model_name == "modelo-x"


@pytest.mark.parametrize("key, value", [("LOG_LEVEL", "VERBOSE"), ("APP_ENV", "staging")])
def test_invalid_values_raise_clear_error(key, value):
    with pytest.raises(ValueError, match=key):
        load_settings({key: value})


def test_settings_are_immutable():
    settings = load_settings({})
    with pytest.raises(AttributeError):
        settings.app_env = "prod"
