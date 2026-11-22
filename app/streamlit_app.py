"""Interface Streamlit do MedProtocol AI (Etapa 6).

Executar na raiz do repositório, no ambiente .venv-train (RAG + torch CUDA + peft + streamlit):
    .venv-train\\Scripts\\python -m streamlit run app/streamlit_app.py

A interface só monta a tela: o fluxo é o mesmo de scripts/ask.py (`create_assistant().ask()`),
e a lógica de exibição fica em `medprotocol.ui.presenter` (testada sem Streamlit e sem LLM).
"""

from __future__ import annotations

import logging
import threading

import streamlit as st

from medprotocol.config import load_settings
from medprotocol.graph.llm import ModelLoadError
from medprotocol.graph.workflow import AVISO, MedProtocolAssistant, create_assistant
from medprotocol.ui.presenter import (
    BLOQUEANTE,
    VARIANT_LABELS,
    build_view,
    check_environment,
    friendly_load_error,
)

logger = logging.getLogger("medprotocol.ui")

st.set_page_config(page_title="MedProtocol AI", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner=False)
def get_settings():
    settings = load_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    return settings


@st.cache_resource(show_spinner="Montando o fluxo (banco, índice de protocolos e grafo)...")
def get_assistant() -> MedProtocolAssistant:
    """Um único assistente por processo; o modelo é carregado uma vez e reutilizado."""
    return create_assistant(get_settings())


@st.cache_resource(show_spinner="Carregando Qwen2.5-1.5B + adaptador LoRA na GPU (primeira execução)...")
def load_model(_assistant: MedProtocolAssistant) -> str | None:
    """Carrega o modelo uma vez. Devolve uma mensagem amigável em caso de falha (também em cache)."""
    try:
        _assistant.deps.generator.load()
    except ModelLoadError as exc:
        logger.error("falha ao carregar o modelo: %s", exc)
        return friendly_load_error(str(exc))
    return None


@st.cache_resource(show_spinner=False)
def generation_lock() -> threading.Lock:
    # O gerador é compartilhado: a variante e a GPU não podem ser usadas por duas perguntas ao mesmo tempo.
    return threading.Lock()


def cuda_available() -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.is_available()


def adapter_dir_or_none(assistant: MedProtocolAssistant):
    try:
        return assistant.deps.generator.adapter_dir
    except (ModelLoadError, AttributeError):
        return None


def render_disclaimer() -> None:
    st.warning(
        "**Protótipo educacional.** Protocolos, pacientes e exames são **sintéticos**. "
        "O sistema não diagnostica, não prescreve e **não substitui a avaliação de um profissional de saúde**. "
        + AVISO,
        icon="⚠️",
    )


def render_response(response) -> None:
    view = build_view(response)
    s = view.status
    st.markdown(f"### {s.icone} :{s.cor}[{s.rotulo}]")
    st.caption(s.explicacao)

    st.markdown(f"**{view.titulo_resposta}**")
    with st.container(border=True):
        st.markdown(view.resposta)
    for obs in view.observacoes:
        st.caption(obs)

    if view.retida and view.texto_nao_validado:
        with st.expander("🔒 Conteúdo NÃO VALIDADO — texto original do modelo (somente para revisão técnica)"):
            st.error("Este texto foi bloqueado pelas verificações automáticas. Não use como orientação.", icon="⛔")
            st.code(view.texto_nao_validado, language=None, wrap_lines=True)

    st.markdown("#### Alertas e verificações")
    if view.alertas:
        st.dataframe(view.alertas, hide_index=True, width="stretch")
    else:
        st.caption("Nenhum alerta.")

    st.markdown("#### Fontes (trechos de protocolos usados no contexto)")
    if view.fontes:
        st.dataframe(view.fontes, hide_index=True, width="stretch")
    else:
        st.caption("Nenhum trecho de protocolo foi usado.")

    cols = st.columns(3)
    cols[0].markdown(f"**Paciente**  \n{view.paciente or '—'}")
    cols[1].markdown(f"**Modelo**  \n{view.modelo}")
    cols[2].markdown(f"**Tempo**  \n{view.tempo_total_s if view.tempo_total_s is not None else '—'} s")
    st.markdown(f"**ID de auditoria:** `{view.audit_id}`")
    st.caption("O log de auditoria guarda apenas resumos (hash) da pergunta e da resposta, nunca o texto.")


def main() -> None:
    st.title("🩺 MedProtocol AI")
    st.markdown(
        "Assistente de consulta a protocolos hospitalares **simulados** de acompanhamento de diabetes tipo 2. "
        "Cada pergunta passa por um fluxo LangGraph: dados do paciente (SQLite) → protocolos (RAG) → "
        "modelo ajustado → validação determinística → auditoria."
    )
    render_disclaimer()

    try:
        settings = get_settings()
    except ValueError as exc:  # variável de ambiente inválida
        st.error(f"Configuração inválida: {exc}")
        st.stop()

    pre = check_environment(settings.sqlite_path, settings.vectorstore_dir, None, cuda_available())
    blocking = [p for p in pre if p.severidade == BLOQUEANTE]
    if blocking:
        for p in blocking:
            st.error(p.mensagem, icon="⛔")
        st.stop()

    try:
        assistant = get_assistant()
    except Exception:
        logger.exception("falha ao montar o assistente")
        st.error("Não foi possível montar o fluxo (banco ou índice de protocolos). Detalhes no log do terminal.", icon="⛔")
        st.stop()

    problems = check_environment(settings.sqlite_path, settings.vectorstore_dir, adapter_dir_or_none(assistant), cuda_available())
    for p in problems:
        st.warning(p.mensagem, icon="⚠️")
    model_error = None if problems else load_model(assistant)
    if model_error:
        st.warning(model_error + " Perguntas que exigem o modelo terminam com status Erro.", icon="⚠️")

    with st.form("pergunta"):
        variant = st.radio(
            "Modelo", options=list(VARIANT_LABELS), format_func=VARIANT_LABELS.get, horizontal=True,
            index=list(VARIANT_LABELS).index(settings.model_variant),
            help="Os dois usam o mesmo modelo carregado; a variante base apenas desliga o adaptador LoRA.",
        )
        pergunta = st.text_area(
            "Pergunta", max_chars=1000, height=100,
            placeholder="Ex.: Quando está prevista a próxima HbA1c de PAC-0001?",
            help="Cite no máximo um paciente (PAC-XXXX) por pergunta.",
        )
        enviar = st.form_submit_button("Enviar", type="primary")

    if enviar:
        if not pergunta.strip():
            st.info("Digite uma pergunta.")
        else:
            with st.spinner("Consultando banco e protocolos, gerando e validando a resposta..."):
                with generation_lock():
                    assistant.deps.generator.set_variant(variant)
                    try:
                        st.session_state["ultima_resposta"] = assistant.ask(pergunta.strip())
                    except Exception:  # ask() já converte falhas em status "erro"; isto é só uma rede de segurança
                        logger.exception("falha inesperada na interface")
                        st.session_state.pop("ultima_resposta", None)
                        st.error("Falha inesperada ao processar a pergunta. Detalhes no log do terminal.", icon="⛔")

    if "ultima_resposta" in st.session_state:
        st.divider()
        render_response(st.session_state["ultima_resposta"])


main()
