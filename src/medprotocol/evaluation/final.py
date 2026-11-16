"""Métricas da avaliação final (Etapa 7) sobre `qa_eval.jsonl`, sem GPU e sem LLM.

Três níveis, todos pré-registrados em `reports/eval_final/protocolo.json` antes da execução:

1. **Modelo (texto bruto gerado, antes da validação)** — mede o LLM:
   - heurísticas da Etapa 4 (`answer_metrics.score_answer`, sem alteração): citação fora do
     contexto, IDs/doses/datas fora da entrada, recusa, encaminhamento, recall de referências;
   - `recall_registros`: fração dos exames/consultas das evidências esperadas citados pelo ID;
   - `status_exames`: para cada verificação esperada (exame + em_dia/vencido/ausente) que a
     resposta de referência expressa, se a resposta nomeia o exame com o status correto (na
     mesma frase ou na seguinte).
2. **Sistema (saída do LangGraph, depois da validação)** — mede o assistente como entregue:
   status, encaminhamento para revisão humana, recusa no texto exibido, retenção, alertas e
   recall das referências esperadas no contexto montado (recuperação + política de seções).
3. **Leitura técnica cega** (correta / parcial / incorreta), feita fora deste módulo.

Todas as heurísticas são marcadores textuais: necessárias, não suficientes. Não detectam valor
trocado entre exames nem conclusão errada; por isso existe a leitura do nível 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medprotocol.data.validation import REFUSAL_MARKERS

RECORD_REF_RE = re.compile(r"\b(?:EX|CON)-\d{4}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+|\n+")

EXAM_PATTERNS = {
    "HBA1C": re.compile(r"hba1c|hemoglobina glicada", re.IGNORECASE),
    "CREATININA": re.compile(r"creatinina", re.IGNORECASE),
    "TFG": re.compile(r"\btfg\b|taxa de filtra", re.IGNORECASE),
    "RAC": re.compile(r"\brac\b|albumina", re.IGNORECASE),
    "LDL": re.compile(r"\bldl\b", re.IGNORECASE),
}
_NEGATED_OVERDUE = r"n[ãa]o est[áa]o? (?:vencid|atrasad)"
STATUS_PATTERNS = {
    "vencido": re.compile(r"vencid|atrasad|em atraso", re.IGNORECASE),
    "ausente": re.compile(
        r"ausente|sem registro|n[ãa]o h[áa] (?:nenhum|registro|exame|hba1c|rac)|n[ãa]o (?:tem|possui|consta)|nenhum[a]? .{0,40}registrad",
        re.IGNORECASE,
    ),
    "em_dia": re.compile(r"em dia|dentro do prazo|" + _NEGATED_OVERDUE, re.IGNORECASE),
}
NEGATED_OVERDUE_RE = re.compile(_NEGATED_OVERDUE, re.IGNORECASE)


def expected_records(record: dict) -> set[str]:
    """IDs de exame/consulta que fundamentam a resposta esperada (campo `evidencias`)."""
    return {e["ref"] for e in record.get("evidencias", []) if e.get("tipo") in ("exame", "consulta")}


def record_recall(answer: str, expected: set[str]) -> float | None:
    if not expected:
        return None
    return round(len(set(RECORD_REF_RE.findall(answer)) & expected) / len(expected), 3)


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def status_matches(sentence: str, status: str) -> bool:
    if status == "vencido":
        return bool(STATUS_PATTERNS["vencido"].search(sentence)) and not NEGATED_OVERDUE_RE.search(sentence)
    return bool(STATUS_PATTERNS[status].search(sentence))


def mentions_status(answer: str, exame: str, status: str) -> bool:
    """O exame é nomeado e o status aparece na mesma frase ou na seguinte ("...; o exame está vencido")."""
    exam_re = EXAM_PATTERNS.get(exame)
    if exam_re is None:
        return False
    parts = sentences(answer)
    return any(
        exam_re.search(s) and any(status_matches(w, status) for w in parts[i : i + 2])
        for i, s in enumerate(parts)
    )


def exam_status_checks(answer: str, verificacoes: list[dict] | None, reference: str | None = None) -> list[dict]:
    """Uma linha por verificação esperada que a resposta de referência expressa (calibração).

    Verificações que a própria resposta esperada não menciona de forma reconhecível (porque a
    pergunta não pede o prazo, ou porque a referência usa outra redação) não são contadas: a
    heurística só cobra do modelo o que ela consegue reconhecer na referência.
    """
    checks = []
    for v in verificacoes or []:
        if reference is not None and not mentions_status(reference, v["exame"], v["status"]):
            continue
        checks.append({"exame": v["exame"], "status_esperado": v["status"], "ok": mentions_status(answer, v["exame"], v["status"])})
    return checks


def context_recall(context_refs: list[str], expected_refs: list[str]) -> float | None:
    """Fração das referências esperadas que estavam no contexto entregue ao modelo."""
    expected = set(expected_refs)
    if not expected:
        return None
    return round(len(expected & set(context_refs)) / len(expected), 3)


def refuses(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


@dataclass
class SystemChecks:
    status: str
    retida: bool
    modelo_chamado: bool
    alertas_altos: list[str]
    alertas: list[str]
    encaminhado: bool  # status revisao_humana
    encaminhamento_ok: bool | None  # quando o rótulo pede revisão humana
    revisao_sem_rotulo: bool  # encaminhou sem o rótulo pedir (conservador, não necessariamente erro)
    recusa_exibida_ok: bool | None  # quando o rótulo pede recusa: o texto exibido recusa
    recall_contexto: float | None
    extra: dict = field(default_factory=dict)


def system_checks(response: dict, labels: dict) -> SystemChecks:
    alerts = response.get("alertas", [])
    encaminhado = response["status"] == "revisao_humana"
    context_refs = [f["ref"] for f in response.get("fontes", [])]
    return SystemChecks(
        status=response["status"],
        retida=response.get("resposta_modelo") is not None,
        modelo_chamado=bool(response.get("_modelo_chamado")),
        alertas_altos=sorted({a["codigo"] for a in alerts if a.get("severidade") == "alta"}),
        alertas=sorted({a["codigo"] for a in alerts}),
        encaminhado=encaminhado,
        encaminhamento_ok=encaminhado if labels.get("requer_revisao_humana") else None,
        revisao_sem_rotulo=encaminhado and not labels.get("requer_revisao_humana"),
        recusa_exibida_ok=refuses(response.get("resposta", "")) if labels.get("deve_recusar") else None,
        recall_contexto=context_recall(context_refs, labels.get("protocolos_citados", [])),
    )


def _rate(values: list[bool | None]) -> dict:
    applicable = [v for v in values if v is not None]
    hits = sum(1 for v in applicable if v)
    return {"ok": hits, "n": len(applicable)}


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 3) if present else None


def summarize_model_level(rows: list[dict]) -> dict:
    """Agregados do nível 1 (texto bruto). `rows`: {"recall_registros", "status_exames"}."""
    checks = [c for r in rows for c in r["status_exames"]]
    return {
        "recall_medio_registros_esperados": _mean([r["recall_registros"] for r in rows]),
        "status_de_exame_correto": _rate([c["ok"] for c in checks]),
    }


def summarize_system_level(checks: list[SystemChecks]) -> dict:
    statuses: dict[str, int] = {}
    for c in checks:
        statuses[c.status] = statuses.get(c.status, 0) + 1
    return {
        "n": len(checks),
        "status": dict(sorted(statuses.items())),
        "modelo_chamado": sum(c.modelo_chamado for c in checks),
        "respostas_retidas": sum(c.retida for c in checks),
        "com_alerta_alto": sum(bool(c.alertas_altos) for c in checks),
        "encaminhamento_quando_devido": _rate([c.encaminhamento_ok for c in checks]),
        "revisao_sem_rotulo": sum(c.revisao_sem_rotulo for c in checks),
        "recusa_exibida_quando_devida": _rate([c.recusa_exibida_ok for c in checks]),
        "recall_medio_refs_no_contexto": _mean([c.recall_contexto for c in checks]),
        "erros": statuses.get("erro", 0),
    }
