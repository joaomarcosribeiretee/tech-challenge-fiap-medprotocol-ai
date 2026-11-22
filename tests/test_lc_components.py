"""Componentes LangChain do fluxo: chat model sobre o gerador local e consulta SQLite como Runnable."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from medprotocol.db.importer import build_database
from medprotocol.db.repository import (
    DatabaseNotFoundError,
    PatientNotFoundError,
    connect_readonly,
    get_patient_summary,
)
from medprotocol.graph.lc_components import (
    MedProtocolChatModel,
    build_answer_chain,
    build_patient_lookup,
    from_langchain_messages,
    to_langchain_messages,
)
from medprotocol.graph.llm import GenerationError, ModelLoadError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MESSAGES = [
    {"role": "system", "content": "Você é um assistente de protocolos simulados."},
    {"role": "user", "content": "Pergunta: Quando repetir a HbA1c?\n\nContexto: [PROT-DM2-001#S3] ..."},
]


class FakeGenerator:
    def __init__(self, answer: str = "ok", error: Exception | None = None):
        self.answer, self.error, self.calls = answer, error, []

    def generate(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.answer

    def describe(self):
        return {"variante": "simulado"}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "medprotocol.db"
    build_database(DATA_DIR, path)
    return path


def test_message_conversion_round_trip_is_exact():
    lc = to_langchain_messages(MESSAGES + [{"role": "assistant", "content": "resposta"}])
    assert [type(m) for m in lc] == [SystemMessage, HumanMessage, AIMessage]
    assert from_langchain_messages(lc) == MESSAGES + [{"role": "assistant", "content": "resposta"}]


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        to_langchain_messages([{"role": "tool", "content": "x"}])


def test_chat_model_is_a_langchain_chat_model_and_passes_prompt_unchanged():
    gen = FakeGenerator("Repetir conforme PROT-DM2-001#S3.")
    model = MedProtocolChatModel(generator=gen)
    assert isinstance(model, BaseChatModel)
    out = model.invoke(to_langchain_messages(MESSAGES))
    assert isinstance(out, AIMessage) and out.content == "Repetir conforme PROT-DM2-001#S3."
    assert gen.calls == [MESSAGES]  # o prompt chega ao modelo idêntico ao do treino


def test_answer_chain_returns_text_from_dict_messages():
    gen = FakeGenerator("texto gerado")
    chain = build_answer_chain(gen)
    assert isinstance(chain, Runnable)
    assert chain.invoke(MESSAGES) == "texto gerado"
    assert gen.calls == [MESSAGES]


@pytest.mark.parametrize("error", [ModelLoadError("sem GPU"), GenerationError("falhou")])
def test_answer_chain_propagates_model_errors_with_same_type(error):
    with pytest.raises(type(error)):
        build_answer_chain(FakeGenerator(error=error)).invoke(MESSAGES)


def test_stop_sequences_are_rejected():
    with pytest.raises(ValueError):
        MedProtocolChatModel(generator=FakeGenerator()).invoke(to_langchain_messages(MESSAGES), stop=["\n"])


def test_patient_lookup_runnable_reads_sqlite(db_path):
    lookup = build_patient_lookup(db_path)
    assert isinstance(lookup, Runnable)
    summary = lookup.invoke("PAC-0001")
    assert summary["paciente"]["patient_id"] == "PAC-0001"
    with connect_readonly(db_path) as conn:  # mesmo resultado da chamada direta ao repositório
        assert summary == get_patient_summary(conn, "PAC-0001")


def test_patient_lookup_propagates_not_found(db_path):
    with pytest.raises(PatientNotFoundError):
        build_patient_lookup(db_path).invoke("PAC-0099")


def test_patient_lookup_missing_database(tmp_path):
    with pytest.raises(DatabaseNotFoundError):
        build_patient_lookup(tmp_path / "nao_existe.db").invoke("PAC-0001")
