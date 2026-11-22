"""Fluxo LangGraph do assistente: pergunta -> dados -> protocolos -> política -> LLM -> validação -> auditoria.

Grafo (arestas condicionais entre colchetes):

    interpretar ──[erro]──────────────────────────────────────────────┐
        │[com paciente]            │[sem paciente]                      │
        ▼                          │                                    │
    consultar_paciente ─[erro/inexistente]──────────────────────────────┤
        ▼                          ▼                                    │
    recuperar_protocolos ◄─────────┘                                    │
        ▼                                                               │
    aplicar_politica ──[sem evidência]──────────────────────────────────┤
        ▼                                                               │
    gerar_resposta ──[erro de modelo]───────────────────────────────────┤
        ▼                                                               │
    validar_resposta                                                    │
        ▼                                                               ▼
    registrar_auditoria ──► END

Componentes reutilizados (nada é reimplementado):
- banco: `repository.get_patient_summary` (Etapa 2), exposto como `RunnableLambda` LangChain
  (`lc_components.build_patient_lookup`);
- LLM: `LocalQwenGenerator` envolvido em `MedProtocolChatModel` (`BaseChatModel` LangChain) numa
  cadeia LCEL `mensagens -> ChatModel -> StrOutputParser` (`lc_components.build_answer_chain`);
- RAG: `ProtocolRetriever.retrieve` (Etapa 3);
- política de seções v1 e mensagens: `assemble_context` e `build_messages` (Etapas 3–4),
  os mesmos usados para gerar os dados de treino.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from medprotocol.data import schedule
from medprotocol.db.repository import PatientNotFoundError, RepositoryError
from medprotocol.finetuning.answer_metrics import cited_refs, PROTOCOL_RE
from medprotocol.finetuning.context import AssistantContext, assemble_context
from medprotocol.finetuning.messages import PROMPT_VERSION, build_messages
from medprotocol.graph.audit import AuditLogger, text_digest
from medprotocol.graph.lc_components import build_answer_chain, build_patient_lookup
from medprotocol.graph.llm import GenerationError, ModelLoadError, TextGenerator
from medprotocol.graph.validation import (
    ALTA,
    MEDIA,
    RECORD_ID_RE,
    Alert,
    QuestionAnalysis,
    analyze_question,
    declares_insufficiency,
    review_triggers,
    validate_answer,
)
from medprotocol.rag.retriever import DEFAULT_K, STATUS_NO_RELIABLE, ProtocolRetriever, RetrievalResult

logger = logging.getLogger("medprotocol.graph")

RESPONDIDO = "respondido"
EVIDENCIA_INSUFICIENTE = "evidencia_insuficiente"
REVISAO_HUMANA = "revisao_humana"
ERRO = "erro"

AVISO = (
    "Protótipo educacional com protocolos e pacientes simulados. As verificações automáticas não "
    "garantem segurança clínica; nenhuma resposta substitui a avaliação do médico responsável."
)
MSG_SEM_EVIDENCIA = (
    "Os protocolos simulados disponíveis não trazem trechos confiáveis para esta pergunta. "
    "Nenhuma resposta foi gerada."
)
MSG_RETIDA = (
    "A resposta gerada foi retida porque não passou nas verificações automáticas (veja os alertas). "
    "Encaminhe a pergunta ao médico responsável."
)
MSG_CONDUTA = (
    "Não posso prescrever, indicar medicamento ou dose, nem estabelecer diagnóstico. Essa decisão "
    "cabe ao médico responsável."
)
MSG_CONDICAO = (
    "A conduta inicial do PROT-DM2-002 para hipoglicemia vale para paciente consciente e capaz de "
    "deglutir. Com a condição informada, ela não pode ser aplicada automaticamente. Encaminhe ao "
    "médico responsável."
)


class GraphState(TypedDict, total=False):
    audit_id: str
    pergunta: str
    analise: QuestionAnalysis
    patient_id: str | None
    paciente: dict | None
    recuperacao: RetrievalResult
    contexto: AssistantContext
    mensagens: list[dict[str, str]]
    resposta_modelo: str | None
    resposta: str
    alertas: list[Alert]
    status: str
    erro: str | None
    tempos_ms: dict[str, float]


@dataclass
class AssistantResponse:
    """Saída estruturada devolvida à interface."""

    audit_id: str
    pergunta: str
    resposta: str
    status: str
    paciente: str | None
    fontes: list[dict]
    registros_citados: list[str]
    alertas: list[dict]
    status_recuperacao: str | None
    modelo: dict
    resposta_modelo: str | None = None  # texto original quando retido (para o revisor humano)
    aviso: str = AVISO
    tempos_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Dependencies:
    retriever: ProtocolRetriever
    db_path: Path
    generator: TextGenerator
    audit: AuditLogger
    k: int = DEFAULT_K
    reference_date: Any = schedule.REFERENCE_DATE


def _timed(name: str, fn):
    def node(state: GraphState) -> GraphState:
        start = time.perf_counter()
        update = fn(state)
        tempos = dict(state.get("tempos_ms", {}))
        tempos[name] = round((time.perf_counter() - start) * 1000, 1)
        return {**update, "tempos_ms": tempos}

    return node


def build_graph(deps: Dependencies):
    # Componentes LangChain: consulta estruturada (SQLite) e LLM customizada (Qwen2.5 + LoRA).
    patient_lookup = build_patient_lookup(deps.db_path, deps.reference_date)
    answer_chain = build_answer_chain(deps.generator)

    def interpretar(state: GraphState) -> GraphState:
        pergunta = state["pergunta"].strip()
        analysis = analyze_question(pergunta)
        if not pergunta:
            return {"analise": analysis, "status": ERRO, "erro": "pergunta vazia", "resposta": "Pergunta vazia."}
        if len(analysis.patient_ids) > 1:
            return {
                "analise": analysis, "status": ERRO, "erro": "mais de um paciente",
                "resposta": "Cite um paciente por pergunta (encontrados: " + ", ".join(analysis.patient_ids) + ").",
            }
        return {"analise": analysis, "patient_id": analysis.patient_ids[0] if analysis.patient_ids else None, "alertas": []}

    def consultar_paciente(state: GraphState) -> GraphState:
        patient_id = state["patient_id"]
        try:
            return {"paciente": patient_lookup.invoke(patient_id)}
        except PatientNotFoundError:
            return {
                "status": EVIDENCIA_INSUFICIENTE,
                "resposta": f"O paciente {patient_id} não existe na base sintética. Nenhum dado foi inferido.",
                "alertas": [Alert("paciente_inexistente", MEDIA, f"{patient_id} não encontrado")],
            }
        except RepositoryError as exc:
            return {"status": ERRO, "erro": f"banco de dados: {exc}", "resposta": "Falha ao consultar o banco de dados."}

    def recuperar_protocolos(state: GraphState) -> GraphState:
        return {"recuperacao": deps.retriever.retrieve(state["pergunta"], k=deps.k)}

    def aplicar_politica(state: GraphState) -> GraphState:
        context = assemble_context(
            state["pergunta"], deps.retriever, state["recuperacao"], state.get("patient_id"),
            state.get("paciente"), deps.reference_date,
        )
        update: GraphState = {"contexto": context, "mensagens": build_messages(context)}
        no_evidence = (
            context.recuperacao.status == STATUS_NO_RELIABLE and not context.secoes_por_regra and context.paciente is None
        )
        if no_evidence:
            update.update(
                status=EVIDENCIA_INSUFICIENTE, resposta=MSG_SEM_EVIDENCIA,
                alertas=[Alert("sem_evidencia", MEDIA, "recuperação sem resultado confiável e nenhuma seção por regra")],
            )
        return update

    def gerar_resposta(state: GraphState) -> GraphState:
        try:
            return {"resposta_modelo": answer_chain.invoke(state["mensagens"])}
        except (ModelLoadError, GenerationError) as exc:
            logger.error("falha do modelo: %s", exc)
            return {"status": ERRO, "erro": str(exc), "resposta": "O modelo de linguagem não está disponível. Nenhuma resposta foi gerada."}

    def validar_resposta(state: GraphState) -> GraphState:
        answer = state.get("resposta_modelo") or ""
        context, analysis = state["contexto"], state["analise"]
        alerts = validate_answer(answer, context, analysis, state["mensagens"][1]["content"])
        triggers = review_triggers(context, analysis)
        blocking = [a for a in alerts if a.severidade == ALTA]
        if blocking:
            codes = {a.codigo for a in blocking}
            if "conduta_incompativel_com_condicao" in codes:
                resposta = MSG_CONDICAO
            elif codes & {"dose_na_resposta", "recusa_ausente", "sugestao_de_conduta"}:
                resposta = MSG_CONDUTA
            else:
                resposta = MSG_RETIDA
        else:
            resposta = answer
        if blocking or any(a.severidade == MEDIA for a in alerts) or triggers:
            status = REVISAO_HUMANA
        elif declares_insufficiency(answer):
            status = EVIDENCIA_INSUFICIENTE
        else:
            status = RESPONDIDO
        return {"alertas": alerts + triggers, "resposta": resposta, "status": status}

    def registrar_auditoria(state: GraphState) -> GraphState:
        context = state.get("contexto")
        answer = state.get("resposta", "")
        deps.audit.write(
            {
                "audit_id": state["audit_id"],
                "modelo": deps.generator.describe(),
                "prompt_version": PROMPT_VERSION,
                "paciente": state.get("patient_id"),
                "pergunta": text_digest(state["pergunta"]),
                "status_recuperacao": context.recuperacao.status if context else None,
                "fontes": context.refs if context else [],
                "registros_citados": sorted(set(RECORD_ID_RE.findall(answer))),
                "alertas": [a.to_json() for a in state.get("alertas", [])],
                "status": state.get("status"),
                "resposta": text_digest(answer),
                "resposta_retida": bool(state.get("resposta_modelo")) and answer != state.get("resposta_modelo"),
                "erro": state.get("erro"),
                "tempos_ms": state.get("tempos_ms", {}),
            }
        )
        return {}

    def after_interpret(state: GraphState) -> str:
        if state.get("status") == ERRO:
            return "registrar_auditoria"
        return "consultar_paciente" if state.get("patient_id") else "recuperar_protocolos"

    def stop_if_done(next_node: str):
        return lambda state: "registrar_auditoria" if state.get("status") else next_node

    graph = StateGraph(GraphState)
    for name, fn in (
        ("interpretar", interpretar),
        ("consultar_paciente", consultar_paciente),
        ("recuperar_protocolos", recuperar_protocolos),
        ("aplicar_politica", aplicar_politica),
        ("gerar_resposta", gerar_resposta),
        ("validar_resposta", validar_resposta),
        ("registrar_auditoria", registrar_auditoria),
    ):
        graph.add_node(name, _timed(name, fn))
    graph.set_entry_point("interpretar")
    graph.add_conditional_edges("interpretar", after_interpret, ["consultar_paciente", "recuperar_protocolos", "registrar_auditoria"])
    graph.add_conditional_edges("consultar_paciente", stop_if_done("recuperar_protocolos"), ["recuperar_protocolos", "registrar_auditoria"])
    graph.add_edge("recuperar_protocolos", "aplicar_politica")
    graph.add_conditional_edges("aplicar_politica", stop_if_done("gerar_resposta"), ["gerar_resposta", "registrar_auditoria"])
    graph.add_conditional_edges("gerar_resposta", stop_if_done("validar_resposta"), ["validar_resposta", "registrar_auditoria"])
    graph.add_edge("validar_resposta", "registrar_auditoria")
    graph.add_edge("registrar_auditoria", END)
    return graph.compile()


def _sources(context: AssistantContext | None, answer: str) -> list[dict]:
    if context is None:
        return []
    cited = cited_refs(answer)
    cited_protocols = set(PROTOCOL_RE.findall(answer))
    sources = [
        {"ref": h.ref, "origem": "similaridade", "versao": h.versao, "status_protocolo": h.status_protocolo,
         "citada": h.ref in cited, "protocolo_citado": h.protocol_id in cited_protocols}
        for h in context.recuperacao.hits
    ]
    sources += [
        {"ref": s.ref, "origem": "regra:" + ",".join(s.motivos), "versao": s.versao, "status_protocolo": s.status_protocolo,
         "citada": s.ref in cited, "protocolo_citado": s.protocol_id in cited_protocols}
        for s in context.secoes_por_regra
    ]
    return sources


class MedProtocolAssistant:
    """Fachada usada por scripts e pela futura interface: `ask(pergunta) -> AssistantResponse`."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.graph = build_graph(deps)

    def ask(self, pergunta: str) -> AssistantResponse:
        audit_id = str(uuid.uuid4())
        try:
            state = self.graph.invoke({"audit_id": audit_id, "pergunta": pergunta, "tempos_ms": {}})
        except Exception as exc:  # falha inesperada: registrada e devolvida como erro, nunca silenciada
            logger.exception("falha no fluxo")
            self.deps.audit.write({"audit_id": audit_id, "status": ERRO, "erro": f"{type(exc).__name__}: {exc}", "alertas": []})
            return AssistantResponse(audit_id, pergunta, "Falha interna no fluxo.", ERRO, None, [], [], [], None,
                                     self.deps.generator.describe())
        context = state.get("contexto")
        answer = state.get("resposta", "")
        raw = state.get("resposta_modelo")
        return AssistantResponse(
            audit_id=audit_id,
            pergunta=pergunta,
            resposta=answer,
            status=state["status"],
            paciente=state.get("patient_id"),
            fontes=_sources(context, answer if raw is None or answer == raw else raw),
            registros_citados=sorted(set(RECORD_ID_RE.findall(answer))),
            alertas=[a.to_json() for a in state.get("alertas", [])],
            status_recuperacao=context.recuperacao.status if context else None,
            modelo=self.deps.generator.describe(),
            resposta_modelo=raw if raw is not None and raw != answer else None,
            tempos_ms=state.get("tempos_ms", {}),
        )


def create_assistant(settings=None, generator: TextGenerator | None = None, retriever: ProtocolRetriever | None = None) -> MedProtocolAssistant:
    """Monta o assistente a partir das configurações (.env). O modelo só é carregado na 1ª pergunta."""
    from medprotocol.config import load_settings
    from medprotocol.graph.llm import LocalQwenGenerator
    from medprotocol.rag.pipeline import load_protocol_retriever

    settings = settings or load_settings()
    if retriever is None:
        retriever, _ = load_protocol_retriever(settings.data_dir, settings.vectorstore_dir)
    if generator is None:
        generator = LocalQwenGenerator(settings.model_config, settings.model_variant)
    deps = Dependencies(retriever=retriever, db_path=settings.sqlite_path, generator=generator, audit=AuditLogger(settings.audit_log_path))
    return MedProtocolAssistant(deps)
