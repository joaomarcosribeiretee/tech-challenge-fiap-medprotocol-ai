"""Componentes LangChain usados pelos nós do fluxo LangGraph.

Por que este módulo existe: o enunciado pede que o LangChain seja usado para (a) integrar a LLM
customizada e (b) consultar a base estruturada. Aqui os dois viram componentes LangChain:

- `MedProtocolChatModel`: um `BaseChatModel` que envolve o gerador local (Qwen2.5-1.5B com ou sem
  o adaptador LoRA, ver `llm.py`). Ele recebe mensagens LangChain e devolve um `AIMessage`.
- `build_answer_chain`: cadeia LCEL `mensagens (dict) -> ChatModel -> StrOutputParser`, chamada
  pelo nó `gerar_resposta`.
- `build_patient_lookup`: `RunnableLambda` que consulta o SQLite em modo somente leitura e
  devolve o resumo do paciente, chamado pelo nó `consultar_paciente`.

Garantia de equivalência: a conversão dict -> mensagem LangChain -> dict preserva papel e texto
exatamente, então o prompt que chega ao modelo é o mesmo de antes (e o mesmo do treino). As
exceções do gerador (`ModelLoadError`, `GenerationError`) e do banco (`PatientNotFoundError`,
`RepositoryError`) atravessam os Runnables sem mudar de tipo; o fluxo continua a tratá-las.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ConfigDict

from medprotocol.data import schedule
from medprotocol.db.repository import connect_readonly, get_patient_summary
from medprotocol.graph.llm import TextGenerator

_TO_LC = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
_FROM_LC = {"system": "system", "human": "user", "ai": "assistant"}


def to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Converte mensagens no formato do chat template (`role`/`content`) em mensagens LangChain."""
    try:
        return [_TO_LC[m["role"]](content=m["content"]) for m in messages]
    except KeyError as exc:
        raise ValueError(f"papel de mensagem não suportado: {exc}") from exc


def from_langchain_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Converte mensagens LangChain de volta ao formato `role`/`content` usado pelo tokenizer."""
    out = []
    for m in messages:
        if m.type not in _FROM_LC or not isinstance(m.content, str):
            raise ValueError(f"mensagem não suportada: tipo {m.type!r}")
        out.append({"role": _FROM_LC[m.type], "content": m.content})
    return out


class MedProtocolChatModel(BaseChatModel):
    """Chat model LangChain sobre o gerador local (Qwen2.5 + LoRA), decodificação gulosa."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: Any  # TextGenerator: LocalQwenGenerator em produção, LLM simulado nos testes

    @property
    def _llm_type(self) -> str:
        return "medprotocol-local-qwen"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return dict(self.generator.describe())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("sequências de parada não são suportadas pelo gerador local")
        text = self.generator.generate(from_langchain_messages(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def build_answer_chain(generator: TextGenerator) -> Runnable:
    """Cadeia LCEL: lista de mensagens (dict) -> MedProtocolChatModel -> texto."""
    return (
        RunnableLambda(to_langchain_messages, name="mensagens_para_langchain")
        | MedProtocolChatModel(generator=generator)
        | StrOutputParser()
    )


def build_patient_lookup(db_path: Path, reference_date: date = schedule.REFERENCE_DATE) -> Runnable:
    """Runnable LangChain: ID do paciente -> resumo estruturado lido do SQLite (somente leitura)."""

    def lookup(patient_id: str) -> dict:
        with connect_readonly(db_path) as conn:
            return get_patient_summary(conn, patient_id, reference_date)

    return RunnableLambda(lookup, name="consultar_sqlite")
