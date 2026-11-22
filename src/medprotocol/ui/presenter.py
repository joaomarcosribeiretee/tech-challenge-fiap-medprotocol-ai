"""Lógica de apresentação da interface Streamlit, sem dependência do Streamlit nem do LLM.

Transforma a saída do fluxo (`AssistantResponse`) em dados prontos para exibir e verifica o
ambiente antes de carregar o modelo. Tudo aqui é testável com objetos simples.

Regras de exibição:
- Só o status `respondido` é apresentado como resposta que passou nas verificações automáticas.
- Resposta retida (`resposta_modelo` preenchido) nunca aparece como resposta: a interface mostra
  a mensagem segura do fluxo e, à parte, o texto original marcado como "não validado".
- Mensagens de erro são curtas e acionáveis; detalhes técnicos vão só para o log do terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medprotocol.graph.workflow import ERRO, EVIDENCIA_INSUFICIENTE, RESPONDIDO, REVISAO_HUMANA, AssistantResponse

VARIANT_LABELS = {
    "ajustado": "Ajustado (Qwen2.5-1.5B + LoRA)",
    "base": "Base (Qwen2.5-1.5B, sem adaptador)",
}


@dataclass(frozen=True)
class StatusView:
    rotulo: str
    cor: str  # nome de cor aceito pelo Markdown do Streamlit (:green[...], :orange[...] etc.)
    icone: str
    explicacao: str


STATUS_VIEWS = {
    RESPONDIDO: StatusView(
        "Respondido", "green", "✅",
        "A resposta passou nas verificações automáticas. Isso não equivale a validação clínica.",
    ),
    REVISAO_HUMANA: StatusView(
        "Requer revisão humana", "orange", "⚠️",
        "A resposta precisa ser conferida pelo médico responsável antes de qualquer uso.",
    ),
    EVIDENCIA_INSUFICIENTE: StatusView(
        "Evidência insuficiente", "blue", "ℹ️",
        "Os dados e protocolos simulados não sustentam uma resposta. Nada foi inferido.",
    ),
    ERRO: StatusView(
        "Erro", "red", "⛔",
        "A pergunta não pôde ser processada. Nenhuma resposta clínica foi gerada.",
    ),
}
UNKNOWN_STATUS = StatusView("Status desconhecido", "gray", "❔", "Status não reconhecido pela interface.")

# Descrição curta das verificações (docs/ARCHITECTURE.md §4), para quem lê os alertas na tela.
ALERT_DESCRIPTIONS = {
    "id_inexistente": "Cita exame, consulta ou paciente que não está no histórico consultado.",
    "exame_trocado": "Cita um exame pelo ID, mas descreve outro tipo de exame.",
    "valor_trocado": "Atribui a um exame um valor diferente do registrado.",
    "valor_de_exame_ausente": "Informa valor de um exame que o paciente não tem.",
    "referencia_fora_do_contexto": "Cita seção de protocolo que não estava no contexto.",
    "protocolo_fora_do_contexto": "Cita protocolo que não estava no contexto.",
    "dose_na_resposta": "Contém dose que não estava nos dados de entrada.",
    "sugestao_de_conduta": "Sugere iniciar, ajustar ou suspender tratamento.",
    "recusa_ausente": "Pedido de conduta respondido sem recusa nem encaminhamento ao médico.",
    "conduta_incompativel_com_condicao": "Conduta por via oral incompatível com a condição descrita.",
    "valor_nao_confere": "Valor que não confere com o histórico nem com os protocolos.",
    "data_nao_confere": "Data atribuída a um exame que não é a dele.",
    "data_fora_do_contexto": "Data que não aparece nos dados de entrada.",
    "acao_sem_registro": "Afirma uma ação (ex.: notificação) sem registro correspondente.",
    "fonte_externa": "Menciona fonte externa; o contexto só tem os protocolos simulados.",
    "sem_fonte_citada": "Nenhum protocolo citado: a fundamentação não pode ser conferida.",
    "condicao_contraditoria": "A pergunta descreve o paciente como capaz e incapaz ao mesmo tempo.",
    "resposta_vazia": "O modelo não gerou texto.",
    "paciente_inexistente": "Paciente não encontrado na base sintética.",
    "sem_evidencia": "Nenhum trecho confiável de protocolo para a pergunta.",
    # Gatilhos de revisão (independem do texto gerado; regras REV dos protocolos)
    "pedido_de_conduta": "Pedido de prescrição, dose, tratamento ou diagnóstico.",
    "uso_fora_da_simulacao": "Pergunta sobre uso fora da simulação.",
    "hipoglicemia_condicao_critica": "Hipoglicemia em paciente sem condição confirmada de deglutir.",
    "resultado_fora_do_limite": "Paciente com resultado fora da meta ou do limite.",
    "dados_inconsistentes": "Dados do paciente inconsistentes.",
}
SEVERITY_ORDER = {"alta": 0, "media": 1, "baixa": 2}
SEVERITY_LABELS = {"alta": "Alta (bloqueia)", "media": "Média", "baixa": "Gatilho de revisão"}


@dataclass(frozen=True)
class ResponseView:
    status: StatusView
    titulo_resposta: str
    resposta: str
    retida: bool
    texto_nao_validado: str | None
    fontes: list[dict]
    alertas: list[dict]
    audit_id: str
    paciente: str | None
    modelo: str
    tempo_total_s: float | None = None
    observacoes: list[str] = field(default_factory=list)


def answer_title(status: str, retida: bool) -> str:
    if retida:
        return "Mensagem do sistema (a resposta do modelo foi retida)"
    if status == RESPONDIDO:
        return "Resposta"
    if status == REVISAO_HUMANA:
        return "Resposta pendente de revisão humana (não aprovada)"
    if status == EVIDENCIA_INSUFICIENTE:
        return "Resposta do sistema"
    return "Mensagem do sistema"


def describe_origin(origem: str) -> str:
    if origem == "similaridade":
        return "Busca por similaridade"
    if origem.startswith("regra:"):
        return "Regra da política v1 (" + origem.removeprefix("regra:").replace(",", ", ") + ")"
    return origem


def source_rows(fontes: list[dict]) -> list[dict]:
    """Uma linha por trecho do contexto; os citados na resposta vêm primeiro."""
    rows = [
        {
            "Trecho": f["ref"],
            "Versão": f.get("versao") or "—",
            "Situação do protocolo": f.get("status_protocolo") or "—",
            "Como entrou no contexto": describe_origin(f.get("origem", "")),
            "Citado na resposta": "sim" if f.get("citada") else ("protocolo" if f.get("protocolo_citado") else "não"),
        }
        for f in fontes
    ]
    order = {"sim": 0, "protocolo": 1, "não": 2}
    return sorted(rows, key=lambda r: order[r["Citado na resposta"]])


def alert_rows(alertas: list[dict]) -> list[dict]:
    """Alertas ordenados por severidade, com descrição legível da regra."""
    rows = [
        {
            "Severidade": SEVERITY_LABELS.get(a.get("severidade", ""), a.get("severidade", "")),
            "Código": a.get("codigo", ""),
            "Verificação": ALERT_DESCRIPTIONS.get(a.get("codigo", ""), "—"),
            "Detalhe": a.get("detalhe", ""),
            "_ordem": SEVERITY_ORDER.get(a.get("severidade", ""), 9),
        }
        for a in alertas
    ]
    rows.sort(key=lambda r: r["_ordem"])
    for row in rows:
        del row["_ordem"]
    return rows


def model_label(modelo: dict) -> str:
    return VARIANT_LABELS.get(modelo.get("variante", ""), modelo.get("variante", "desconhecido"))


def build_view(response: AssistantResponse) -> ResponseView:
    retida = response.resposta_modelo is not None
    tempo = sum(response.tempos_ms.values()) / 1000 if response.tempos_ms else None
    observacoes = []
    if response.status == REVISAO_HUMANA and not retida:
        observacoes.append(
            "O texto abaixo é do modelo, mas não está aprovado: veja os alertas e encaminhe ao médico responsável."
        )
    if retida:
        observacoes.append("O texto original do modelo está disponível apenas para revisão técnica, abaixo.")
    return ResponseView(
        status=STATUS_VIEWS.get(response.status, UNKNOWN_STATUS),
        titulo_resposta=answer_title(response.status, retida),
        resposta=response.resposta,
        retida=retida,
        texto_nao_validado=response.resposta_modelo,
        fontes=source_rows(response.fontes),
        alertas=alert_rows(response.alertas),
        audit_id=response.audit_id,
        paciente=response.paciente,
        modelo=model_label(response.modelo),
        tempo_total_s=round(tempo, 1) if tempo is not None else None,
        observacoes=observacoes,
    )


# --- Verificação do ambiente e mensagens de erro -------------------------------------------------

BLOQUEANTE = "bloqueante"  # sem isso o fluxo não pode nem ser montado
AVISO = "aviso"  # o fluxo funciona, mas perguntas que exigem o modelo terminam em "erro"


@dataclass(frozen=True)
class EnvironmentProblem:
    severidade: str
    mensagem: str


def check_environment(
    sqlite_path: Path, vectorstore_dir: Path, adapter_dir: Path | None, cuda_available: bool | None
) -> list[EnvironmentProblem]:
    """Confere os pré-requisitos sem carregar nada pesado. `None` = não foi possível verificar."""
    problems = []
    if not Path(sqlite_path).exists():
        problems.append(EnvironmentProblem(BLOQUEANTE, "Banco SQLite não encontrado. Rode `python scripts/build_database.py`."))
    if not (Path(vectorstore_dir) / "index.json").exists():
        problems.append(EnvironmentProblem(BLOQUEANTE, "Índice de protocolos não encontrado. Rode `python scripts/build_index.py`."))
    if adapter_dir is None:
        problems.append(EnvironmentProblem(AVISO, "Configuração do modelo inválida; o modelo não poderá ser carregado."))
    elif not (Path(adapter_dir) / "adapter_config.json").exists():
        problems.append(EnvironmentProblem(
            AVISO,
            "Adaptador LoRA não encontrado (os pesos não são versionados). Rode `python scripts/train_lora.py`; "
            "até lá, perguntas que exigem o modelo terminam em erro.",
        ))
    if cuda_available is None:
        problems.append(EnvironmentProblem(AVISO, "PyTorch não está instalado neste ambiente. Use o `.venv-train`."))
    elif not cuda_available:
        problems.append(EnvironmentProblem(
            AVISO,
            "GPU CUDA não disponível. O modelo local requer GPU; perguntas que exigem o modelo terminam em erro.",
        ))
    return problems


def friendly_load_error(message: str) -> str:
    """Traduz a mensagem de `ModelLoadError` sem expor caminhos ou stack trace."""
    text = message.lower()
    if "cuda" in text and ("indispon" in text or "not available" in text):
        return "GPU CUDA não disponível. O modelo local requer GPU."
    if "adaptador não encontrado" in text:
        return "Adaptador LoRA não encontrado. Rode `python scripts/train_lora.py`."
    if "out of memory" in text or "outofmemory" in text:
        return "Memória da GPU insuficiente para carregar o modelo. Feche outros processos que usam a GPU."
    if "configuração do modelo" in text:
        return "Configuração do modelo inválida. Confira `MODEL_CONFIG`."
    return "Não foi possível carregar o modelo. Detalhes no log do terminal."
