"""Verificações determinísticas da pergunta e da resposta gerada (sem LLM avaliador).

Cada verificação devolve alertas com severidade:
- "alta":  a resposta contém algo comprovadamente incompatível com o contexto ou com as regras
  de segurança (registro inexistente, exame trocado, referência fora do contexto, dose, conduta
  oral para paciente que não pode deglutir...). A resposta é **retida** e vai para revisão humana.
- "media": algo que não foi possível confirmar (valor ou data sem correspondência, ação afirmada
  sem registro, nenhuma fonte citada, fonte externa). Vai para revisão humana.
- "baixa": informação para o revisor; não muda o status.

Limites (documentados em docs/ARCHITECTURE.md): as regras olham IDs, números, datas, palavras-
chave e citações. Elas **não** detectam toda interpretação clínica errada. Passar em todas não
torna a resposta clinicamente segura.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from medprotocol.data.validation import INSUFFICIENT_MARKERS, REFUSAL_MARKERS
from medprotocol.finetuning.answer_metrics import DOSE_RE, PROTOCOL_RE, cited_refs, dates_in
from medprotocol.finetuning.context import AssistantContext

ALTA, MEDIA, BAIXA = "alta", "media", "baixa"

RECORD_ID_RE = re.compile(r"\b(?:EX|CON|PAC)-\d{4}\b")
PATIENT_ID_RE = re.compile(r"\bPAC-\d{4}\b")
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ(\d])")


@dataclass(frozen=True)
class Alert:
    codigo: str
    severidade: str
    detalhe: str

    def to_json(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# --- Análise da pergunta (antes da geração) ---------------------------------------------------

# Mais amplo que a intenção `pedido_de_conduta` da política v1 (congelada): aqui o objetivo é
# segurança, não seleção de contexto. Ex.: "começar uma estatina", "qual antibiótico usar".
CONDUCT_RE = re.compile(
    r"prescrev|prescric|receit|\bdose|posolog|medicament|remedio|antibiotic|estatina|insulina|metformina|"
    r"\b(comecar|iniciar|suspender|parar|reduzir|aumentar|ajustar|trocar|intensificar)\b|"
    r"tratamento|tratar|diagnostic|estagi|classifi|devo usar|devo dar|qual .* usar"
)
REAL_USE_RE = re.compile(r"paciente real|pacientes reais|diretriz oficial|\boficial\b|minha enfermaria")
HYPO_RE = re.compile(r"hipoglic|glicemia (capilar )?(baixa|de \d)|\b[2-6]\d mg/dl")
INCAPABLE_RE = re.compile(
    r"inconscien|desacordad|desmaiad|torpor|rebaixamento|nao (responde|consegue (engolir|deglutir)|deglute|engole)|"
    r"incapaz de (deglutir|engolir)|sem conseguir (engolir|deglutir)"
)
NO_HYPO_RE = re.compile(r"(nao teve|nao tem|sem|nao apresentou|nao esta com|fora de) (episodio de )?hipoglic")
CAPABLE_RE = re.compile(r"\bconscien|\blucid|\bacordad|\bconsegue (engolir|deglutir)|\bcapaz de (deglutir|engolir)")


@dataclass(frozen=True)
class QuestionAnalysis:
    patient_ids: tuple[str, ...]
    pedido_de_conduta: bool
    uso_fora_da_simulacao: bool
    hipoglicemia: bool
    condicao: str | None  # None | "capaz" | "incapaz" | "contraditoria"
    sem_hipoglicemia: bool = False  # a pergunta nega o episódio ("não teve hipoglicemia")


def analyze_question(pergunta: str) -> QuestionAnalysis:
    text = normalize(pergunta)
    incapable = bool(INCAPABLE_RE.search(text))
    capable = bool(CAPABLE_RE.search(INCAPABLE_RE.sub(" ", text)))
    condition = "contraditoria" if incapable and capable else "incapaz" if incapable else "capaz" if capable else None
    return QuestionAnalysis(
        patient_ids=tuple(dict.fromkeys(PATIENT_ID_RE.findall(pergunta))),
        pedido_de_conduta=bool(CONDUCT_RE.search(text)),
        uso_fora_da_simulacao=bool(REAL_USE_RE.search(text)),
        hipoglicemia=bool(HYPO_RE.search(text)) and not NO_HYPO_RE.search(text),
        condicao=condition,
        sem_hipoglicemia=bool(NO_HYPO_RE.search(text)),
    )


# --- Verificações da resposta -----------------------------------------------------------------

TYPE_KEYWORDS = {
    "HBA1C": re.compile(r"hba1c|hemoglobina glicada"),
    "CREATININA": re.compile(r"(?<!albumina/)(?<!albumina / )creatinina"),
    "TFG": re.compile(r"\btfg\b|filtracao glomerular"),
    "RAC": re.compile(r"\brac\b|albumina"),
    "LDL": re.compile(r"\bldl\b|colesterol"),
    "GLICEMIA_CAPILAR": re.compile(r"glicemia capilar"),
    "GLICEMIA_JEJUM": re.compile(r"glicemia de jejum|\bjejum\b"),
}
TYPE_UNITS = {
    "HBA1C": r"%",
    "CREATININA": r"mg/dl",
    "TFG": r"ml/min",
    "RAC": r"mg/g",
    "LDL": r"mg/dl",
    "GLICEMIA_CAPILAR": r"mg/dl",
    "GLICEMIA_JEJUM": r"mg/dl",
}
ORAL_CONDUCT_RE = re.compile(r"via oral|oferec|ofereca|\b15 ?g\b|carboidrato|repetir a oferta|dar (agua|acucar|suco)")
ACTION_CLAIM_RE = re.compile(
    r"\b(?:ja )?(?:foi|foram)\s+(?:devidamente\s+|ja\s+)?"
    r"(apresentad|sinalizad|notificad|encaminhad|comunicad|avaliad|oferecid|repetid|solicitad|administrad|"
    r"prescrit|ajustad|suspens|informad|discutid|orientad)\w*"
)
SUGGESTION_RE = re.compile(
    r"\b(recomendo|sugiro|indico|deve-se|deve ser|devera|pode-se)\s+(?:\w+\s+){0,2}"
    r"(iniciar|aumentar|reduzir|suspender|ajustar|prescrever|administrar|trocar)\b"
)
EXTERNAL_SOURCE_RE = re.compile(
    r"\boms\b|organizacao mundial|sociedade brasileira|\bsbd\b|\bada\b|american diabetes|ministerio da saude|guideline"
)


def _number(token: str) -> float:
    return float(token.replace(",", "."))


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _patient_records(context: AssistantContext) -> tuple[dict[str, dict], dict[str, dict]]:
    if not context.paciente:
        return {}, {}
    exams = {e["exam_id"]: e for e in context.paciente.get("historico_exames", [])}
    consults = {c["consultation_id"]: c for c in context.paciente.get("consultas", [])}
    return exams, consults


def _context_numbers(context: AssistantContext) -> set[float]:
    """Números presentes nos trechos de protocolo (limites, metas, intervalos)."""
    texts = [h.trecho for h in context.recuperacao.hits] + [s.trecho for s in context.secoes_por_regra]
    return {_number(n) for t in texts for n in NUMBER_RE.findall(t)}


def check_record_ids(answer: str, context: AssistantContext) -> list[Alert]:
    exams, consults = _patient_records(context)
    alerts = []
    for record_id in sorted(set(RECORD_ID_RE.findall(answer))):
        if record_id.startswith("PAC-"):
            ok = record_id == context.patient_id
        else:
            ok = record_id in exams or record_id in consults
        if not ok:
            alerts.append(Alert("id_inexistente", ALTA, f"{record_id} não pertence aos dados fornecidos"))
    return alerts


def check_exam_sentences(answer: str, context: AssistantContext, user_message: str) -> list[Alert]:
    """Para cada frase que cita um exame por ID: tipo, valor e data precisam ser os daquele registro."""
    exams, consults = _patient_records(context)
    verifications = {v["exame"]: v for v in (context.paciente or {}).get("verificacoes", [])}
    alerts = []
    for sentence in _sentences(answer):
        ids = [i for i in dict.fromkeys(RECORD_ID_RE.findall(sentence)) if i in exams]
        if not ids:
            continue
        norm = normalize(sentence)
        types_in_sentence = {t for t, rx in TYPE_KEYWORDS.items() if rx.search(norm)}
        allowed_dates = {context.data_referencia.isoformat()}
        allowed_dates |= {consults[c]["data"] for c in RECORD_ID_RE.findall(sentence) if c in consults}
        for exam_id in ids:
            exam = exams[exam_id]
            tipo = exam["tipo"]
            allowed_dates.add(exam["data_coleta"])
            if verifications.get(tipo, {}).get("proxima_verificacao"):
                allowed_dates.add(verifications[tipo]["proxima_verificacao"])
            if types_in_sentence and tipo not in types_in_sentence:
                alerts.append(Alert(
                    "exame_trocado", ALTA,
                    f"{exam_id} é {tipo}, mas a frase fala de {', '.join(sorted(types_in_sentence))}",
                ))
                continue
            unit = TYPE_UNITS.get(tipo)
            values = [
                _number(m.group(1))
                for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*" + re.escape(unit), norm)
            ] if unit else []
            if len(ids) == 1 and values and not any(abs(v - float(exam["valor"])) < 1e-6 for v in values):
                # Valor ligado a um ID que tem outro valor: erro comprovado, não só "não confirmado".
                alerts.append(Alert("valor_trocado", ALTA, f"{exam_id} tem valor {exam['valor']}, a frase traz {values}"))
        for found in sorted((dates_in(sentence) & dates_in(user_message)) - allowed_dates):
            # Datas fora da entrada já são apontadas por check_dates.
            alerts.append(Alert("data_nao_confere", MEDIA, f"data {found} não corresponde a {', '.join(ids)}"))
    return alerts


def check_values_by_type(answer: str, context: AssistantContext) -> list[Alert]:
    """Valor com unidade logo após o nome de um exame precisa existir no histórico desse tipo."""
    if not context.paciente:
        return []
    exams, _ = _patient_records(context)
    by_type: dict[str, set[float]] = {}
    for exam in exams.values():
        by_type.setdefault(exam["tipo"], set()).add(float(exam["valor"]))
    thresholds = _context_numbers(context)
    norm = normalize(answer)
    keyword_spans = sorted((m.start(), m.end(), t) for t, rx in TYPE_KEYWORDS.items() for m in rx.finditer(norm))
    alerts = []
    for index, (_, end, tipo) in enumerate(keyword_spans):
        limit = keyword_spans[index + 1][0] if index + 1 < len(keyword_spans) else len(norm)
        window = norm[end : min(limit, end + 80)]
        window = re.split(r"[.;]\s", window, maxsplit=1)[0]
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*" + re.escape(TYPE_UNITS[tipo]), window)
        if not match:
            continue
        value = _number(match.group(1))
        if value in by_type.get(tipo, set()) or value in thresholds:
            continue
        if tipo not in by_type:
            alerts.append(Alert("valor_de_exame_ausente", ALTA, f"{tipo} {match.group(0)}: não há {tipo} registrado"))
        else:
            alerts.append(Alert("valor_nao_confere", MEDIA, f"{tipo} {match.group(0)} não está no histórico ({sorted(by_type[tipo])})"))
    return alerts


def check_dates(answer: str, user_message: str) -> list[Alert]:
    return [Alert("data_fora_do_contexto", MEDIA, f"data {d} não aparece na entrada") for d in sorted(dates_in(answer) - dates_in(user_message))]


def check_protocol_refs(answer: str, context: AssistantContext) -> list[Alert]:
    available = set(context.refs)
    available_protocols = {ref.split("#")[0] for ref in available}
    alerts = [Alert("referencia_fora_do_contexto", ALTA, f"{ref} não está nos trechos fornecidos") for ref in sorted(cited_refs(answer) - available)]
    alerts += [Alert("protocolo_fora_do_contexto", ALTA, f"{p} não está nos trechos fornecidos") for p in sorted(set(PROTOCOL_RE.findall(answer)) - available_protocols)]
    if EXTERNAL_SOURCE_RE.search(normalize(answer)):
        alerts.append(Alert("fonte_externa", MEDIA, "a resposta invoca fonte externa aos protocolos simulados"))
    return alerts


def check_action_claims(answer: str, context: AssistantContext) -> list[Alert]:
    """'Foi apresentado/sinalizado/notificado...' só é aceito se uma consulta registrar a ação."""
    consult_text = normalize(" ".join(c.get("resumo", "") for c in (context.paciente or {}).get("consultas", [])))
    alerts = []
    for match in ACTION_CLAIM_RE.finditer(normalize(answer)):
        stem = match.group(1)
        if stem not in consult_text:
            alerts.append(Alert("acao_sem_registro", MEDIA, f"'{match.group(0)}' não consta nas consultas registradas"))
    return alerts


def check_conduct(answer: str, analysis: QuestionAnalysis, user_message: str) -> list[Alert]:
    norm = normalize(answer)
    refuses = any(m in norm for m in (normalize(x) for x in REFUSAL_MARKERS))
    doses_in_input = {d.lower().replace(" ", "") for d in DOSE_RE.findall(user_message)}
    alerts = [
        Alert("dose_na_resposta", ALTA, f"quantidade de medicamento '{d}' não fornecida no contexto")
        for d in sorted({d for d in DOSE_RE.findall(answer) if d.lower().replace(" ", "") not in doses_in_input})
    ]
    if SUGGESTION_RE.search(norm):
        alerts.append(Alert("sugestao_de_conduta", ALTA, "a resposta sugere iniciar/ajustar/suspender conduta"))
    if analysis.pedido_de_conduta and not refuses and _looks_like_conduct_answer(norm):
        alerts.append(Alert("recusa_ausente", ALTA, "pedido de conduta respondido sem recusa explícita"))
    return alerts


def _looks_like_conduct_answer(norm_answer: str) -> bool:
    """Sem recusa explícita e sem encaminhamento ao médico: tratamos como resposta de conduta."""
    return "medico responsavel" not in norm_answer and "equipe medica" not in norm_answer


def check_hypoglycemia_condition(answer: str, analysis: QuestionAnalysis) -> list[Alert]:
    """Conduta para paciente consciente nunca é aplicada a quem não pode deglutir ou tem condição contraditória.

    Também barra a conduta de episódio de hipoglicemia quando a pergunta diz que não houve episódio.
    """
    if analysis.sem_hipoglicemia and ORAL_CONDUCT_RE.search(normalize(answer)):
        return [Alert(
            "conduta_incompativel_com_condicao", ALTA,
            "a resposta aplica a conduta de episódio de hipoglicemia, mas a pergunta diz que não houve episódio",
        )]
    if analysis.condicao not in ("incapaz", "contraditoria"):
        return []
    alerts = []
    if analysis.condicao == "contraditoria":
        alerts.append(Alert("condicao_contraditoria", MEDIA, "a pergunta descreve o paciente como capaz e incapaz de deglutir/consciente"))
    if ORAL_CONDUCT_RE.search(normalize(answer)):
        alerts.append(Alert(
            "conduta_incompativel_com_condicao", ALTA,
            "a resposta sugere conduta por via oral, prevista só para paciente consciente e capaz de deglutir",
        ))
    return alerts


def check_grounding(answer: str) -> list[Alert]:
    norm = normalize(answer)
    declares_gap = any(normalize(m) in norm for m in INSUFFICIENT_MARKERS) or "nao ha registro" in norm or "nao cobrem" in norm
    refuses = any(normalize(m) in norm for m in REFUSAL_MARKERS)
    if not PROTOCOL_RE.search(answer) and not declares_gap and not refuses:
        return [Alert("sem_fonte_citada", MEDIA, "a resposta não cita nenhum protocolo; não foi possível conferir a fundamentação")]
    return []


def declares_insufficiency(answer: str) -> bool:
    norm = normalize(answer)
    return any(normalize(m) in norm for m in INSUFFICIENT_MARKERS) or "nao cobrem" in norm


def validate_answer(answer: str, context: AssistantContext, analysis: QuestionAnalysis, user_message: str) -> list[Alert]:
    if not answer.strip():
        return [Alert("resposta_vazia", ALTA, "o modelo não gerou texto")]
    alerts = (
        check_record_ids(answer, context)
        + check_exam_sentences(answer, context, user_message)
        + check_values_by_type(answer, context)
        + check_dates(answer, user_message)
        + check_protocol_refs(answer, context)
        + check_action_claims(answer, context)
        + check_conduct(answer, analysis, user_message)
        + check_hypoglycemia_condition(answer, analysis)
        + check_grounding(answer)
    )
    unique: dict[tuple[str, str], Alert] = {}
    for alert in alerts:
        unique.setdefault((alert.codigo, alert.detalhe), alert)
    return list(unique.values())


def review_triggers(context: AssistantContext, analysis: QuestionAnalysis) -> list[Alert]:
    """Situações que exigem revisão humana independentemente do texto gerado (regras REV dos protocolos)."""
    alerts = []
    if analysis.pedido_de_conduta:
        alerts.append(Alert("pedido_de_conduta", BAIXA, "prescrição, dose, tratamento ou diagnóstico cabem ao médico responsável"))
    if analysis.uso_fora_da_simulacao:
        alerts.append(Alert("uso_fora_da_simulacao", BAIXA, "protocolos simulados não se aplicam a pacientes reais"))
    if analysis.hipoglicemia and analysis.condicao in ("incapaz", "contraditoria"):
        alerts.append(Alert("hipoglicemia_condicao_critica", BAIXA, "paciente sem condição confirmada de deglutir"))
    patient = context.paciente or {}
    if patient.get("fora_do_limite"):
        alerts.append(Alert("resultado_fora_do_limite", BAIXA, f"fora da meta/limite: {', '.join(patient['fora_do_limite'])}"))
    if patient.get("inconsistencias"):
        alerts.append(Alert("dados_inconsistentes", BAIXA, "; ".join(patient["inconsistencias"])))
    return alerts
