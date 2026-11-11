"""Formatação do contexto em mensagens de chat (system/user/assistant).

- Treino: [system, user, assistant], com a resposta esperada como mensagem do assistente.
- Inferência: [system, user]; o modelo gera a mensagem do assistente.

A mensagem do usuário é montada só a partir do `AssistantContext`. Os scores de similaridade
não entram no texto: são proximidade vetorial, não probabilidade de acerto, e o modelo não
deve tratá-los como confiança. A ordem dos trechos e o status da recuperação são mantidos.
"""

from __future__ import annotations

import json

from medprotocol.finetuning.context import AssistantContext
from medprotocol.rag.retriever import STATUS_NO_RELIABLE, STATUS_OK, STATUS_UNCERTAIN

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """\
Você é o MedProtocol AI, assistente educacional de consulta a protocolos SIMULADOS de um \
hospital fictício sobre diabetes mellitus tipo 2. Os protocolos não são diretrizes médicas \
oficiais e não têm validação clínica; os pacientes são sintéticos.

Regras:
1. Use somente os trechos de protocolo e os dados do paciente fornecidos na mensagem. Não use \
conhecimento externo como se fosse regra do protocolo.
2. Cite a referência de cada regra usada no formato PROTOCOLO#SEÇÃO, apenas entre os trechos \
fornecidos. Cite exames e consultas pelos seus IDs.
3. Nunca invente, estime ou complete valores de exames ou dados do paciente. Dado ausente deve \
ser informado como ausente.
4. O status da recuperação indica apenas a proximidade entre a pergunta e os trechos, não que \
eles estejam corretos. Se nenhum trecho sustenta a resposta, diga que os protocolos fornecidos \
não cobrem a pergunta.
5. Trechos parecidos podem tratar de situações diferentes. Confira se a condição descrita no \
trecho corresponde à situação perguntada; se a informação necessária não estiver registrada, \
não a presuma e sinalize.
6. Não prescreva, não indique medicamento ou dose e não estabeleça diagnóstico: recuse esses \
pedidos e encaminhe ao médico responsável.
7. Decisões clínicas, resultados fora da meta e dados conflitantes devem ser encaminhados para \
revisão do médico responsável.
8. Não oriente o uso dos protocolos simulados em pacientes reais.
9. Use a data de referência informada para avaliar prazos."""

STATUS_NOTES = {
    STATUS_OK: "o melhor trecho ficou acima da faixa de sobreposição observada; isso não garante "
    "que os trechos respondam à pergunta.",
    STATUS_UNCERTAIN: "faixa em que perguntas dentro e fora do escopo se sobrepõem; confira se os "
    "trechos sustentam a resposta.",
    STATUS_NO_RELIABLE: "nenhum trecho atingiu a faixa observada para perguntas do escopo; os "
    "trechos podem não ter relação com a pergunta.",
}


def render_user_message(context: AssistantContext) -> str:
    result = context.recuperacao
    lines = [
        f"Data de referência: {context.data_referencia.isoformat()}",
        "",
        "## Pergunta",
        context.pergunta,
        "",
        f"## Trechos dos protocolos (status da recuperação: {result.status})",
        f"Observação: {STATUS_NOTES[result.status]}",
    ]
    if not result.hits:
        lines.append("Nenhum trecho recuperado.")
    for position, hit in enumerate(result.hits, start=1):
        lines += [
            "",
            f"[{position}] {hit.ref} | {hit.protocol_id} v{hit.versao} | {hit.status_protocolo} | {hit.source}",
            hit.trecho,
        ]
    lines.append("")
    if context.paciente is None:
        lines += ["## Dados do paciente", "Nenhum paciente citado na pergunta."]
    else:
        lines += [
            f"## Dados do paciente {context.patient_id} (banco SQLite sintético)",
            "```json",
            json.dumps(context.paciente, ensure_ascii=False, indent=2),
            "```",
        ]
    return "\n".join(lines)


def build_messages(context: AssistantContext, resposta: str | None = None) -> list[dict[str, str]]:
    """Sem `resposta`: mensagens de inferência. Com `resposta`: exemplo supervisionado."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_message(context)},
    ]
    if resposta is not None:
        if not resposta.strip():
            raise ValueError("resposta esperada vazia")
        messages.append({"role": "assistant", "content": resposta})
    return messages
