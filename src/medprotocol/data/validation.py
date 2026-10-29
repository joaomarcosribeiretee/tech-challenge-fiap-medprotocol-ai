"""Validação do dataset sintético.

Grupos de verificação:
1. Estrutura: campos, tipos, formatos de ID, enumerações.
2. Protocolos: marcação de SIMULAÇÃO, natureza das regras, seções obrigatórias.
3. Integridade referencial: evidências existem, pertencem ao paciente e os trechos
   correspondem ao documento original.
4. Lógica temporal: `verificacoes_esperadas` é recalculada a partir de `exams.csv`.
5. Segurança: coerência entre categoria, recusa e revisão humana.
6. Anonimização: busca por padrões de dados pessoais (verificação adicional, não prova).
7. Vazamento treino/avaliação: IDs, pacientes exclusivos da avaliação, duplicatas e
   similaridade textual (pares mais parecidos são listados para revisão manual).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from medprotocol.data import schedule
from medprotocol.data.loaders import Dataset, DatasetFormatError, exam_text, load_dataset, patient_lines

EVAL_ONLY_PATIENTS = frozenset({"PAC-0009", "PAC-0010"})
REQUIRED_SECTIONS = ("S1", "LIM", "REV")
SIMULATION_NOTICE = "SIMULAÇÃO"

PROTOCOL_METADATA = {
    "protocol_id": None,
    "titulo": None,
    "versao": None,
    "vigencia": None,
    "status": {"SIMULACAO"},
    "natureza_regras": {"simulada", "baseada_em_referencia_verificada"},
    "referencias_externas_verificadas": None,
    # Nenhum protocolo tem validação clínica nesta versão; outro valor exige revisão deste módulo.
    "validacao_clinica": {"nao_realizada"},
    "escopo": None,
}

EXAM_TYPES = {
    # tipo: (unidade, mínimo plausível, máximo plausível)
    "HBA1C": ("%", 4, 15),
    "GLICEMIA_JEJUM": ("mg/dL", 40, 600),
    "GLICEMIA_CAPILAR": ("mg/dL", 20, 600),
    "CREATININA": ("mg/dL", 0.3, 10),
    "TFG": ("mL/min/1.73m2", 5, 150),
    "RAC": ("mg/g", 0, 3000),
    "LDL": ("mg/dL", 20, 400),
}

CATEGORIES = ("consulta_protocolo", "dados_ausentes", "contextualizada", "recusa", "revisao_humana")
EVIDENCE_TYPES = ("protocolo", "exame", "consulta", "paciente")
QA_REQUIRED = {
    "qa_id": str,
    "split": str,
    "categoria": str,
    "pergunta": str,
    "patient_id": (str, type(None)),
    "resposta_esperada": str,
    "protocolos_citados": list,
    "evidencias": list,
    "requer_revisao_humana": bool,
    "deve_recusar": bool,
}
QA_OPTIONAL = {"verificacoes_esperadas": list}
VERIFICATION_STATUSES = ("em_dia", "vencido", "ausente")

# Heurísticas textuais mínimas: a resposta precisa sinalizar a recusa ou o encaminhamento.
REVIEW_MARKERS = ("médico responsável", "equipe médica", "revisão", "validação", "validada")
REFUSAL_MARKERS = ("não posso",)

ID_PATTERNS = {
    "patient_id": re.compile(r"^PAC-\d{4}$"),
    "consultation_id": re.compile(r"^CON-\d{4}$"),
    "exam_id": re.compile(r"^EX-\d{4}$"),
    "qa_id": re.compile(r"^QA-\d{4}$"),
    "protocol_id": re.compile(r"^PROT-DM2-\d{3}$"),
}
PROTOCOL_REF_RE = re.compile(r"^(?P<protocol>PROT-DM2-\d{3})#(?P<section>[A-Z0-9]+)$")
ANY_ID_RE = re.compile(r"\b(?:PAC|CON|EX|QA)-\d{4}\b")

PII_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"),
    "CNS (cartão SUS)": re.compile(r"\b\d{15}\b"),
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "telefone": re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
}
SCANNED_SUFFIXES = (".md", ".csv", ".jsonl")

SIMILARITY_ERROR = 0.90
SIMILARITY_WARNING = 0.75


@dataclass(frozen=True)
class SimilarPair:
    score: float
    field: str
    train_id: str
    eval_id: str


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    similar_pairs: list[SimilarPair] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação e com espaços colapsados."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _mask_ids(text: str) -> str:
    return ANY_ID_RE.sub("ID", text)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def validate_dataset(data_dir: Path, reference_date: date = schedule.REFERENCE_DATE) -> ValidationReport:
    report = ValidationReport()
    try:
        dataset = load_dataset(data_dir)
    except DatasetFormatError as exc:
        report.errors.append(f"formato: {exc}")
        return report

    _check_protocols(dataset, report)
    _check_patients(dataset, report, reference_date)
    _check_consultations(dataset, report, reference_date)
    _check_exams(dataset, report, reference_date)
    for split, records in dataset.qa.items():
        for index, record in enumerate(records, start=1):
            _check_qa_record(dataset, report, split, index, record, reference_date)
    _check_splits(dataset, report)
    _check_leakage(dataset, report)
    _scan_pii(Path(data_dir), report)

    report.counts = {
        "protocolos": len(dataset.protocols),
        "pacientes": len(dataset.patients),
        "consultas": len(dataset.consultations),
        "exames": len(dataset.exams),
        "qa_treino": len(dataset.qa["train"]),
        "qa_avaliacao": len(dataset.qa["eval"]),
    }
    return report


# --- Protocolos -------------------------------------------------------------------------------


def _check_protocols(dataset: Dataset, report: ValidationReport) -> None:
    if not dataset.protocols:
        report.errors.append("protocolos: nenhum protocolo encontrado")
    for protocol_id, protocol in dataset.protocols.items():
        where = f"protocolo {protocol.path.name}"
        meta = protocol.metadata
        for key, allowed in PROTOCOL_METADATA.items():
            value = meta.get(key, "")
            if not value:
                report.errors.append(f"{where}: campo de cabeçalho '{key}' ausente ou vazio")
            elif allowed is not None and value not in allowed:
                report.errors.append(f"{where}: {key}={value!r} inválido; aceitos: {sorted(allowed)}")
        for key in meta.keys() - PROTOCOL_METADATA.keys():
            report.errors.append(f"{where}: campo de cabeçalho desconhecido '{key}'")
        if not ID_PATTERNS["protocol_id"].match(protocol_id):
            report.errors.append(f"{where}: protocol_id {protocol_id!r} fora do formato PROT-DM2-NNN")
        if protocol.path.stem != protocol_id:
            report.errors.append(f"{where}: nome do arquivo deve ser {protocol_id}.md")
        if _parse_date(meta.get("vigencia", "")) is None:
            report.errors.append(f"{where}: vigencia deve estar no formato AAAA-MM-DD")

        refs = meta.get("referencias_externas_verificadas", "")
        if meta.get("natureza_regras") == "simulada" and refs != "nenhuma":
            report.errors.append(
                f"{where}: natureza_regras=simulada exige referencias_externas_verificadas=nenhuma"
            )
        if meta.get("natureza_regras") == "baseada_em_referencia_verificada" and refs in ("", "nenhuma"):
            report.errors.append(f"{where}: regras baseadas em referência exigem a referência verificada")

        if SIMULATION_NOTICE not in protocol.body:
            report.errors.append(f"{where}: aviso '{SIMULATION_NOTICE}' ausente no corpo do documento")
        for section in REQUIRED_SECTIONS:
            if not protocol.sections.get(section):
                report.errors.append(f"{where}: seção obrigatória [{section}] ausente ou vazia")

    for protocol_id, version in schedule.SOURCE_PROTOCOL_VERSIONS.items():
        protocol = dataset.protocols.get(protocol_id)
        if protocol is None:
            report.errors.append(f"schedule: protocolo de origem {protocol_id} não encontrado")
        elif protocol.metadata.get("versao") != version:
            report.errors.append(
                f"schedule: regras temporais escritas para {protocol_id} v{version}, "
                f"mas o arquivo está na versão {protocol.metadata.get('versao')}; revise schedule.py"
            )


# --- Pacientes, consultas e exames ------------------------------------------------------------


def _check_patients(dataset: Dataset, report: ValidationReport, reference_date: date) -> None:
    for patient_id, row in dataset.patients.items():
        where = f"paciente {patient_id}"
        if not ID_PATTERNS["patient_id"].match(patient_id):
            report.errors.append(f"{where}: ID fora do formato PAC-NNNN")
        if row["sexo"] not in ("F", "M"):
            report.errors.append(f"{where}: sexo deve ser F ou M")
        try:
            birth, diagnosis = int(row["ano_nascimento"]), int(row["ano_diagnostico_dm2"])
        except ValueError:
            report.errors.append(f"{where}: anos devem ser inteiros")
            continue
        if not 1920 <= birth <= reference_date.year - 18:
            report.errors.append(f"{where}: ano_nascimento {birth} fora da faixa de adultos")
        if not birth + 18 <= diagnosis <= reference_date.year:
            report.errors.append(f"{where}: ano_diagnostico_dm2 {diagnosis} incoerente")
        for key in ("comorbidades", "medicamentos_em_uso", "alergias"):
            if not row[key]:
                report.errors.append(f"{where}: {key} vazio (use 'nenhuma')")


def _check_consultations(dataset: Dataset, report: ValidationReport, reference_date: date) -> None:
    for consultation_id, row in dataset.consultations.items():
        where = f"consulta {consultation_id}"
        if not ID_PATTERNS["consultation_id"].match(consultation_id):
            report.errors.append(f"{where}: ID fora do formato CON-NNNN")
        if row["patient_id"] not in dataset.patients:
            report.errors.append(f"{where}: paciente {row['patient_id']} inexistente")
        if row["protocol_id"] not in dataset.protocols:
            report.errors.append(f"{where}: protocolo {row['protocol_id']} inexistente")
        if row["tipo"] not in ("ambulatorial", "internacao"):
            report.errors.append(f"{where}: tipo {row['tipo']!r} inválido")
        consulted = _parse_date(row["data"])
        if consulted is None or consulted > reference_date:
            report.errors.append(f"{where}: data ausente, inválida ou posterior à data de referência")
        if not row["resumo"].strip():
            report.errors.append(f"{where}: resumo vazio")


def _check_exams(dataset: Dataset, report: ValidationReport, reference_date: date) -> None:
    for exam_id, row in dataset.exams.items():
        where = f"exame {exam_id}"
        if not ID_PATTERNS["exam_id"].match(exam_id):
            report.errors.append(f"{where}: ID fora do formato EX-NNNN")
        if row["patient_id"] not in dataset.patients:
            report.errors.append(f"{where}: paciente {row['patient_id']} inexistente")
        collected = _parse_date(row["data_coleta"])
        if collected is None or collected > reference_date:
            report.errors.append(f"{where}: data_coleta ausente, inválida ou posterior à data de referência")
        spec = EXAM_TYPES.get(row["tipo"])
        if spec is None:
            report.errors.append(f"{where}: tipo {row['tipo']!r} desconhecido")
            continue
        unit, low, high = spec
        if row["unidade"] != unit:
            report.errors.append(f"{where}: unidade {row['unidade']!r} difere de {unit!r}")
        try:
            value = float(row["valor"])
        except ValueError:
            report.errors.append(f"{where}: valor {row['valor']!r} não numérico")
            continue
        if not low <= value <= high:
            report.errors.append(f"{where}: valor {value} fora da faixa plausível [{low}, {high}]")


# --- Perguntas e respostas --------------------------------------------------------------------


def _check_qa_record(
    dataset: Dataset,
    report: ValidationReport,
    split: str,
    index: int,
    record: dict,
    reference_date: date,
) -> None:
    where = f"qa[{split}] {record.get('qa_id', f'linha {index}')}"
    for key, expected in QA_REQUIRED.items():
        if key not in record:
            report.errors.append(f"{where}: campo obrigatório '{key}' ausente")
            return
        if not isinstance(record[key], expected):
            report.errors.append(f"{where}: campo '{key}' com tipo inválido")
            return
    for key in record.keys() - QA_REQUIRED.keys() - QA_OPTIONAL.keys():
        report.errors.append(f"{where}: campo desconhecido '{key}'")

    if not ID_PATTERNS["qa_id"].match(record["qa_id"]):
        report.errors.append(f"{where}: qa_id fora do formato QA-NNNN")
    if record["split"] != split:
        report.errors.append(f"{where}: split={record['split']!r} no arquivo de {split}")
    category = record["categoria"]
    if category not in CATEGORIES:
        report.errors.append(f"{where}: categoria {category!r} inválida")
    for key in ("pergunta", "resposta_esperada"):
        if not record[key].strip():
            report.errors.append(f"{where}: {key} vazio")

    patient_id = record["patient_id"]
    if patient_id is not None and patient_id not in dataset.patients:
        report.errors.append(f"{where}: paciente {patient_id} inexistente")
        return

    protocol_refs = _check_evidences(dataset, report, where, record)
    cited = record["protocolos_citados"]
    if len(set(cited)) != len(cited):
        report.errors.append(f"{where}: protocolos_citados com itens repetidos")
    if set(cited) != protocol_refs:
        report.errors.append(
            f"{where}: protocolos_citados {sorted(cited)} difere das evidências de protocolo "
            f"{sorted(protocol_refs)}"
        )
    if not protocol_refs:
        report.errors.append(f"{where}: todo caso precisa de ao menos uma evidência de protocolo")

    _check_category_rules(report, where, record)
    if "verificacoes_esperadas" in record:
        _check_verifications(dataset, report, where, record, reference_date)


def _check_evidences(dataset: Dataset, report: ValidationReport, where: str, record: dict) -> set[str]:
    """Valida cada evidência e devolve o conjunto de referências de protocolo citadas."""
    patient_id = record["patient_id"]
    protocol_refs: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    if not record["evidencias"]:
        report.errors.append(f"{where}: lista de evidências vazia")
    for position, evidence in enumerate(record["evidencias"], start=1):
        label = f"{where} evidência {position}"
        if not isinstance(evidence, dict) or set(evidence) - {"tipo", "ref", "trecho"}:
            report.errors.append(f"{label}: deve ter apenas os campos tipo, ref e trecho")
            continue
        kind, ref, excerpt = evidence.get("tipo"), evidence.get("ref"), evidence.get("trecho")
        if kind not in EVIDENCE_TYPES or not isinstance(ref, str) or not isinstance(excerpt, str):
            report.errors.append(f"{label}: tipo inválido ou ref/trecho ausente")
            continue
        if not excerpt.strip():
            report.errors.append(f"{label}: trecho vazio")
            continue
        if (kind, ref, excerpt) in seen:
            report.errors.append(f"{label}: evidência {ref} repetida com o mesmo trecho")
        seen.add((kind, ref, excerpt))

        if kind == "protocolo":
            match = PROTOCOL_REF_RE.match(ref)
            protocol = dataset.protocols.get(match["protocol"]) if match else None
            section = protocol.sections.get(match["section"]) if protocol and match else None
            if section is None:
                report.errors.append(f"{label}: referência de protocolo {ref!r} inexistente")
            elif _collapse(excerpt) not in _collapse(section):
                report.errors.append(f"{label}: trecho não encontrado literalmente em {ref}")
            protocol_refs.add(ref)
            continue

        if patient_id is None:
            report.errors.append(f"{label}: evidência de {kind} exige patient_id no caso")
            continue
        if kind == "exame":
            exam = dataset.exams.get(ref)
            if exam is None:
                report.errors.append(f"{label}: exame {ref} inexistente")
            elif exam["patient_id"] != patient_id:
                report.errors.append(f"{label}: exame {ref} pertence a {exam['patient_id']}, não a {patient_id}")
            elif excerpt != exam_text(exam):
                report.errors.append(f"{label}: trecho deve ser exatamente {exam_text(exam)!r}")
        elif kind == "consulta":
            consultation = dataset.consultations.get(ref)
            if consultation is None:
                report.errors.append(f"{label}: consulta {ref} inexistente")
            elif consultation["patient_id"] != patient_id:
                report.errors.append(
                    f"{label}: consulta {ref} pertence a {consultation['patient_id']}, não a {patient_id}"
                )
            elif _collapse(excerpt) not in _collapse(consultation["resumo"]):
                report.errors.append(f"{label}: trecho não encontrado literalmente no resumo de {ref}")
        elif kind == "paciente":
            if ref != patient_id:
                report.errors.append(f"{label}: evidência de paciente {ref} difere do patient_id {patient_id}")
            elif excerpt not in patient_lines(dataset.patients[ref]):
                report.errors.append(f"{label}: trecho deve ser uma linha 'campo: valor' do cadastro de {ref}")
    return protocol_refs


def _check_category_rules(report: ValidationReport, where: str, record: dict) -> None:
    category = record["categoria"]
    has_patient = record["patient_id"] is not None
    patient_evidence = any(e.get("tipo") != "protocolo" for e in record["evidencias"] if isinstance(e, dict))
    answer = record["resposta_esperada"].lower()

    if category == "consulta_protocolo" and has_patient:
        report.errors.append(f"{where}: consulta_protocolo não deve ter patient_id")
    if category in ("dados_ausentes", "contextualizada") and not (has_patient and patient_evidence):
        report.errors.append(f"{where}: {category} exige patient_id e evidência de dados do paciente")
    if category == "dados_ausentes":
        statuses = {v.get("status") for v in record.get("verificacoes_esperadas", []) if isinstance(v, dict)}
        if not statuses & {"vencido", "ausente"}:
            report.errors.append(f"{where}: dados_ausentes exige verificação vencida ou ausente")
    if category == "recusa" and not record["deve_recusar"]:
        report.errors.append(f"{where}: categoria recusa exige deve_recusar=true")
    if category == "revisao_humana":
        if not record["requer_revisao_humana"]:
            report.errors.append(f"{where}: categoria revisao_humana exige requer_revisao_humana=true")
        if record["deve_recusar"]:
            report.errors.append(f"{where}: revisao_humana não é recusa; use categoria recusa")
    if record["deve_recusar"] and not any(marker in answer for marker in REFUSAL_MARKERS):
        report.errors.append(f"{where}: deve_recusar=true, mas a resposta não explicita a recusa")
    if record["requer_revisao_humana"] and not any(marker in answer for marker in REVIEW_MARKERS):
        report.errors.append(f"{where}: requer_revisao_humana=true, mas a resposta não encaminha a revisão")


def _check_verifications(
    dataset: Dataset, report: ValidationReport, where: str, record: dict, reference_date: date
) -> None:
    patient_id = record["patient_id"]
    if patient_id is None:
        report.errors.append(f"{where}: verificacoes_esperadas exige patient_id")
        return
    patient_exams = [exam for exam in dataset.exams.values() if exam["patient_id"] == patient_id]
    cited_exams = {e.get("ref") for e in record["evidencias"] if isinstance(e, dict) and e.get("tipo") == "exame"}
    seen: set[str] = set()
    for item in record["verificacoes_esperadas"]:
        if not isinstance(item, dict) or item.get("exame") not in schedule.MONITORED_EXAMS:
            report.errors.append(f"{where}: verificação inválida {item!r}")
            continue
        exam_type = item["exame"]
        if exam_type in seen:
            report.errors.append(f"{where}: verificação de {exam_type} repetida")
        seen.add(exam_type)
        if item.get("status") not in VERIFICATION_STATUSES:
            report.errors.append(f"{where}: status {item.get('status')!r} inválido para {exam_type}")
        expected = schedule.verification(exam_type, patient_exams, reference_date)
        if item != expected:
            report.errors.append(f"{where}: verificação de {exam_type} diverge do calculado: {expected}")
        if expected["ultimo_exame"] and expected["ultimo_exame"] not in cited_exams:
            report.errors.append(f"{where}: exame {expected['ultimo_exame']} usado na verificação não é citado")


# --- Divisão treino/avaliação e vazamento -----------------------------------------------------


def _check_splits(dataset: Dataset, report: ValidationReport) -> None:
    for split, records in dataset.qa.items():
        present = Counter(r.get("categoria") for r in records)
        for category in CATEGORIES:
            if not present[category]:
                report.errors.append(f"split {split}: nenhuma pergunta da categoria {category}")


def _eval_only_identifiers(dataset: Dataset) -> set[str]:
    identifiers = set(EVAL_ONLY_PATIENTS)
    identifiers |= {k for k, v in dataset.exams.items() if v["patient_id"] in EVAL_ONLY_PATIENTS}
    identifiers |= {k for k, v in dataset.consultations.items() if v["patient_id"] in EVAL_ONLY_PATIENTS}
    return identifiers


def _record_text(record: dict) -> str:
    parts = [str(record.get("pergunta", "")), str(record.get("resposta_esperada", ""))]
    for evidence in record.get("evidencias", []):
        if isinstance(evidence, dict):
            parts += [str(evidence.get("ref", "")), str(evidence.get("trecho", ""))]
    return " ".join(parts)


def _check_leakage(dataset: Dataset, report: ValidationReport) -> None:
    train, evaluation = dataset.qa["train"], dataset.qa["eval"]

    ids = Counter(r.get("qa_id") for r in train + evaluation)
    for qa_id, count in ids.items():
        if count > 1:
            report.errors.append(f"vazamento: qa_id {qa_id} aparece {count} vezes")

    forbidden = _eval_only_identifiers(dataset)
    for record in train:
        if record.get("patient_id") in EVAL_ONLY_PATIENTS:
            report.errors.append(f"vazamento: {record.get('qa_id')} (treino) usa paciente exclusivo da avaliação")
        found = sorted(set(ANY_ID_RE.findall(_record_text(record))) & forbidden)
        if found:
            report.errors.append(f"vazamento: {record.get('qa_id')} (treino) menciona {found}")

    questions = Counter(normalize_text(str(r.get("pergunta", ""))) for r in train + evaluation)
    for question, count in questions.items():
        if count > 1:
            report.errors.append(f"vazamento: pergunta repetida {count} vezes: {question!r}")

    # Similaridade por sequência de palavras, com IDs mascarados para que perguntas com o mesmo
    # molde e pacientes diferentes apareçam. autojunk=False evita distorção em textos longos.
    def words(record: dict, field_name: str) -> list[str]:
        return normalize_text(_mask_ids(str(record.get(field_name, "")))).split()

    pairs = []
    for field_name in ("pergunta", "resposta_esperada"):
        for t in train:
            for e in evaluation:
                score = SequenceMatcher(None, words(t, field_name), words(e, field_name), autojunk=False).ratio()
                pairs.append(SimilarPair(round(score, 3), field_name, t.get("qa_id", "?"), e.get("qa_id", "?")))
    pairs.sort(key=lambda pair: pair.score, reverse=True)
    report.similar_pairs = pairs
    for pair in pairs:
        message = f"{pair.field} {pair.train_id} (treino) x {pair.eval_id} (avaliação): similaridade {pair.score}"
        if pair.score >= SIMILARITY_ERROR:
            report.errors.append(f"vazamento: {message}")
        elif pair.score >= SIMILARITY_WARNING:
            report.warnings.append(f"possível paráfrase, revisar manualmente: {message}")


# --- Anonimização -----------------------------------------------------------------------------


def _scan_pii(data_dir: Path, report: ValidationReport) -> None:
    """Busca padrões de dados pessoais. Ausência de achados não prova anonimização."""
    for path in sorted(data_dir.rglob("*")):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in PII_PATTERNS.items():
                if pattern.search(line):
                    report.errors.append(f"dados pessoais: possível {label} em {path.name}:{number}")
