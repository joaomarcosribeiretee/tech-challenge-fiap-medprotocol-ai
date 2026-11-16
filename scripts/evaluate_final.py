"""Avaliação final, ÚNICA, no conjunto reservado `data/eval/qa_eval.jsonl` (Etapa 7).

Três passos, em ordem:

    python scripts/evaluate_final.py --freeze   # (.venv-train) registra o sistema congelado e as checagens de vazamento
    git commit ...                               # o protocolo é versionado ANTES da execução
    python scripts/evaluate_final.py --run      # (.venv-train, GPU) executa base e ajustado uma única vez
    python scripts/evaluate_final.py --report   # (sem GPU) junta métricas e a leitura técnica cega

`--run`:
- recusa executar se os hashes do sistema diferirem do protocolo congelado;
- recusa executar de novo se já houver resultados (avaliação única);
- passa cada pergunta pelo fluxo LangGraph completo (SQLite + RAG + política + Qwen2.5 ± LoRA +
  validação + auditoria), com as mesmas mensagens para as duas variantes (conferido por hash);
- grava o texto bruto do modelo e a saída do sistema, a perda nas respostas de referência e
  um arquivo embaralhado (A/B) para a leitura técnica cega.

Nada deste script altera o modelo, os prompts, as regras ou os dados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "eval_final"
PROTOCOL = OUT / "protocolo.json"
RESULTS = OUT / "resultados_por_exemplo.jsonl"
SUMMARY = OUT / "resumo.json"
BLIND = OUT / "leitura_cega.jsonl"
BLIND_KEY = OUT / "gabarito_cego.json"
REVIEW = OUT / "leitura_tecnica.json"
EVAL_FILE = ROOT / "data" / "eval" / "qa_eval.jsonl"
CONFIG = ROOT / "configs" / "finetune" / "qwen2.5-1.5b-lora-v1.json"
VARIANTS = ("base", "ajustado")
BLIND_SEED = 20261116

# Arquivos que definem o comportamento avaliado. Qualquer mudança invalida a execução.
FROZEN_FILES = [
    "configs/finetune/qwen2.5-1.5b-lora-v1.json",
    "models/adapters/qwen2.5-1.5b-lora-v1/adapter_config.json",
    "models/adapters/qwen2.5-1.5b-lora-v1/adapter_model.safetensors",
    "src/medprotocol/finetuning/messages.py",
    "src/medprotocol/finetuning/context.py",
    "src/medprotocol/finetuning/context_policy.py",
    "src/medprotocol/finetuning/generation.py",
    "src/medprotocol/finetuning/answer_metrics.py",
    "src/medprotocol/graph/workflow.py",
    "src/medprotocol/graph/validation.py",
    "src/medprotocol/graph/llm.py",
    "src/medprotocol/rag/retriever.py",
    "src/medprotocol/rag/pipeline.py",
    "src/medprotocol/db/repository.py",
    "src/medprotocol/evaluation/final.py",
    "scripts/evaluate_final.py",
    "data/eval/qa_eval.jsonl",
    "data/processed/medprotocol.db",
    "data/vectorstore/index.json",
]
# Arquivos do treino, registrados para a checagem de vazamento.
TRAINING_FILES = [
    "data/finetune/qa_train.jsonl",
    "data/finetune/split_v1.json",
    "data/processed/sft/train.jsonl",
    "reports/finetune/qwen2.5-1.5b-lora-v1/train_report.json",
]

METRICS = {
    "nivel_1_modelo": {
        "descricao": "Texto bruto gerado pelo LLM, antes da validação; só quando o modelo é chamado.",
        "heuristicas_etapa_4": "answer_metrics.score_answer, sem alteração: refs/protocolos fora do contexto, IDs, doses e datas fora da entrada, recusa quando devida, recusa indevida, encaminhamento quando devido (marcadores), recall de referências e de protocolos esperados.",
        "recall_registros": "Fração dos EX/CON das evidências esperadas citados pelo ID (uso dos dados do paciente).",
        "status_exames": "Para cada verificação esperada (exame + em_dia/vencido/ausente) que a resposta esperada expressa na heurística, a resposta nomeia o exame com o status correto (mesma frase ou seguinte). Calibração no treino: 52 de 70 verificações expressas pelas referências; status errado também reconhecido em 8 de 104 testes.",
        "perda_referencia": "Perda média por token nas respostas esperadas, com as mesmas mensagens, adaptador desligado × ligado.",
    },
    "nivel_2_sistema": {
        "descricao": "Saída do fluxo LangGraph completo, como a interface exibe.",
        "status": "Distribuição de respondido / revisao_humana / evidencia_insuficiente / erro.",
        "encaminhamento_quando_devido": "Rótulo requer_revisao_humana=true e status revisao_humana.",
        "revisao_sem_rotulo": "Status revisao_humana sem o rótulo pedir (comportamento conservador, contado à parte).",
        "recusa_exibida_quando_devida": "Rótulo deve_recusar=true e o texto exibido contém 'não posso'.",
        "retidas_e_alertas": "Respostas retidas (alerta alto) e códigos de alerta.",
        "recall_refs_no_contexto": "Fração das referências esperadas presentes no contexto (recuperação + política de seções).",
    },
    "nivel_3_leitura_tecnica": {
        "descricao": "Leitura cega (A/B embaralhado, semente fixa) do texto bruto do modelo contra a resposta esperada e as evidências.",
        "correta": "Responde ao essencial da resposta esperada, sem erro factual sobre dados do paciente ou protocolo e sem conduta indevida.",
        "parcial": "Sem erro grave, mas incompleta, ou com erro menor que não muda a conclusão.",
        "incorreta": "Erro factual sobre dados/protocolo, conclusão errada, conduta ou prescrição indevida, ou não responde.",
        "limite": "Revisão técnica assistida por IA (o mesmo assistente que ajudou a construir o sistema); não é revisão clínica.",
    },
    "interpretacao": "n = 10 por variante. Diferenças de 1–2 exemplos não são evidência estatística; os números descrevem estes casos.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def file_hashes(paths: list[str]) -> dict[str, str]:
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"arquivos ausentes: {missing}")
    return {p: sha256(ROOT / p) for p in paths}


def leakage_checks() -> dict:
    """Confere que nada do qa_eval entrou em treino, validação interna ou ajuste."""
    from medprotocol.data.loaders import load_dataset
    from medprotocol.data.validation import EVAL_ONLY_PATIENTS, validate_dataset

    eval_records = read_jsonl(EVAL_FILE)
    eval_ids = {r["qa_id"] for r in eval_records}
    split = json.loads((ROOT / "data/finetune/split_v1.json").read_text(encoding="utf-8"))
    sft_text = (ROOT / "data/processed/sft/train.jsonl").read_text(encoding="utf-8")
    sft_ids = {r["metadata"]["qa_id"] for r in map(json.loads, sft_text.splitlines()) if r}
    train = read_jsonl(ROOT / "data/finetune/qa_train.jsonl")
    dataset = load_dataset(ROOT / "data")
    reserved_ids = set(EVAL_ONLY_PATIENTS)
    reserved_ids |= {k for k, v in dataset.exams.items() if v["patient_id"] in EVAL_ONLY_PATIENTS}
    reserved_ids |= {k for k, v in dataset.consultations.items() if v["patient_id"] in EVAL_ONLY_PATIENTS}
    reserved_sections = ("PROT-DM2-002#S4", "PROT-DM2-003#S4")
    eval_questions = {norm(r["pergunta"]) for r in eval_records}
    demo = (ROOT / "scripts/demo/perguntas.txt").read_text(encoding="utf-8").splitlines()
    report = validate_dataset(ROOT / "data")
    checks = {
        "qa_eval_fora_do_split": not (eval_ids & (set(split["train"]) | set(split["val"]))),
        "qa_eval_fora_do_sft": not (eval_ids & sft_ids),
        "ids_reservados_fora_do_sft": sorted(i for i in reserved_ids if i in sft_text) == [],
        "secoes_reservadas_fora_do_sft": not any(s in sft_text for s in reserved_sections),
        "perguntas_eval_fora_do_treino": not (eval_questions & {norm(r["pergunta"]) for r in train}),
        "perguntas_eval_fora_da_demo": not (eval_questions & {norm(l) for l in demo if l.strip()}),
        "validador_do_dataset_sem_erros": report.ok,
    }
    return {
        "checagens": checks,
        "todas_ok": all(checks.values()),
        "ids_reservados": sorted(reserved_ids),
        "secoes_reservadas_no_treino": list(reserved_sections),
        "avisos_validador": report.warnings,
        "nota": (
            "O qa_eval não foi lido por nenhum script de treino, split, comparação ou calibração "
            "(sources.py lê só qa_train.jsonl e rejeita PAC-0009/PAC-0010 e as seções reservadas). "
            "As seções reservadas existem no índice e podem ser recuperadas na avaliação, como no uso real."
        ),
    }


def freeze() -> int:
    if PROTOCOL.exists():
        print(f"{PROTOCOL} já existe: o protocolo é congelado uma única vez.", file=sys.stderr)
        return 1
    from medprotocol.finetuning.messages import PROMPT_VERSION
    from medprotocol.finetuning.train_config import load_train_config
    from medprotocol.rag.retriever import DEFAULT_K

    system_src = [p for p in FROZEN_FILES if p.startswith(("src/", "configs/"))]
    dirty = git("status", "--porcelain", "--", *system_src)
    if dirty:
        print(f"código do sistema com alterações não commitadas:\n{dirty}", file=sys.stderr)
        return 1
    config = load_train_config(CONFIG)
    leakage = leakage_checks()
    protocol = {
        "congelado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit_base": git("rev-parse", "HEAD"),
        "modelo": {
            "base": config.base_model,
            "revisao": config.base_model_revision,
            "adaptador": f"models/adapters/{config.run_name}",
            "decodificacao": {"do_sample": False, "max_new_tokens": config.max_new_tokens},
            "dispositivo": "cuda (RTX 4070), bf16",
        },
        "fluxo": {
            "prompt_version": PROMPT_VERSION,
            "k_recuperacao": DEFAULT_K,
            "politica_de_secoes": "v1 (context_policy.py)",
            "validacao": "graph/validation.py (regras determinísticas)",
            "variantes": list(VARIANTS),
            "mesmo_contexto": "As duas variantes passam pelo mesmo fluxo; o hash das mensagens enviadas ao modelo é comparado.",
        },
        "conjunto": {"arquivo": "data/eval/qa_eval.jsonl", "n": len(read_jsonl(EVAL_FILE))},
        "hashes_sistema": file_hashes(FROZEN_FILES),
        "hashes_treino": file_hashes(TRAINING_FILES),
        "vazamento": leakage,
        "metricas": METRICS,
        "regras": [
            "Execução única; o script recusa rodar de novo se houver resultados.",
            "Os resultados não serão usados para alterar modelo, prompts, regras, dados ou hiperparâmetros.",
            "A leitura técnica é feita no arquivo cego antes de abrir o gabarito.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(leakage["checagens"], ensure_ascii=False, indent=2))
    print(f"protocolo -> {PROTOCOL.relative_to(ROOT)}; commit_base {protocol['commit_base'][:7]}")
    return 0 if leakage["todas_ok"] else 1


class RecordingGenerator:
    """Repassa ao gerador real e guarda exatamente as mensagens e o texto de cada chamada."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[list[dict], str]] = []

    def generate(self, messages):
        text = self.inner.generate(messages)
        self.calls.append((messages, text))
        return text

    def describe(self):
        return self.inner.describe()

    def set_variant(self, variant):
        self.inner.set_variant(variant)


def run(smoke_dir: Path | None = None) -> int:
    if smoke_dir is not None:
        code = execute(smoke_records(), smoke_dir)
        return code or report(smoke_dir)
    if not PROTOCOL.exists():
        print("rode --freeze (e faça commit) antes de --run", file=sys.stderr)
        return 1
    if RESULTS.exists():
        print(f"{RESULTS.relative_to(ROOT)} já existe: a avaliação final é única.", file=sys.stderr)
        return 1
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    current = file_hashes(FROZEN_FILES)
    changed = [p for p, h in protocol["hashes_sistema"].items() if current.get(p) != h]
    if changed:
        print(f"sistema diferente do congelado: {changed}", file=sys.stderr)
        return 1
    if git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))) == "":
        print("o protocolo precisa estar commitado antes da execução", file=sys.stderr)
        return 1
    return execute(read_jsonl(EVAL_FILE), OUT)


def smoke_records() -> list[dict]:
    """2 exemplos da validação INTERNA (qa_train), para testar o mecanismo sem tocar o qa_eval."""
    split = json.loads((ROOT / "data/finetune/split_v1.json").read_text(encoding="utf-8"))
    train = {r["qa_id"]: r for r in read_jsonl(ROOT / "data/finetune/qa_train.jsonl")}
    return [train[i] for i in split["val"][:2]]


def execute(records: list[dict], out: Path) -> int:
    from medprotocol.config import load_settings
    from medprotocol.finetuning.answer_metrics import score_answer
    from medprotocol.finetuning.chat_format import tokenize_example
    from medprotocol.finetuning.lora_trainer import evaluate_loss
    from medprotocol.evaluation.final import exam_status_checks, expected_records, record_recall, system_checks
    from medprotocol.graph.llm import LocalQwenGenerator
    from medprotocol.graph.workflow import create_assistant

    settings = replace(load_settings(), audit_log_path=out / "audit.jsonl")
    generator = RecordingGenerator(LocalQwenGenerator(settings.model_config, "ajustado"))
    assistant = create_assistant(settings, generator=generator)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    for variant in VARIANTS:
        generator.set_variant(variant)
        for record in records:
            labels = {k: v for k, v in record.items() if k not in ("pergunta", "resposta_esperada")}
            before = len(generator.calls)
            response = assistant.ask(record["pergunta"]).to_json()
            called = len(generator.calls) > before
            messages, raw = generator.calls[-1] if called else (None, None)
            response["_modelo_chamado"] = called
            row = {
                "qa_id": record["qa_id"], "categoria": record["categoria"], "variante": variant,
                "pergunta": record["pergunta"], "resposta_esperada": record["resposta_esperada"],
                "rotulos": {k: labels[k] for k in ("requer_revisao_humana", "deve_recusar", "protocolos_citados")},
                "mensagens_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest() if messages else None,
                "texto_modelo": raw,
                "sistema": {k: v for k, v in response.items() if k not in ("pergunta", "aviso")},
                "checagens_sistema": asdict(system_checks(response, labels)),
            }
            if called:
                user = messages[1]["content"]
                row["metricas_modelo"] = score_answer(raw, user, labels, record["qa_id"]).to_json()
                row["recall_registros"] = record_recall(raw, expected_records(record))
                row["status_exames"] = exam_status_checks(raw, record.get("verificacoes_esperadas"), record["resposta_esperada"])
                row["_mensagens"] = messages
            rows.append(row)
            print(f"{variant:8} {record['qa_id']} {response['status']:22} retida={row['checagens_sistema']['retida']}")

    # Mesmo contexto nas duas variantes
    by_id: dict[str, set] = {}
    for r in rows:
        by_id.setdefault(r["qa_id"], set()).add(r["mensagens_sha256"])
    same_context = all(len(v) == 1 for v in by_id.values())

    # Perda nas respostas esperadas, com as mesmas mensagens (só onde o modelo foi chamado)
    inner = generator.inner
    tokenized = [
        tokenize_example(inner._tokenizer, r["_mensagens"] + [{"role": "assistant", "content": r["resposta_esperada"]}],
                         inner.config.max_length, r["qa_id"])
        for r in rows if r["variante"] == "ajustado" and "_mensagens" in r
    ]
    with inner._model.disable_adapter():
        loss_base = evaluate_loss(inner._model, tokenized)
    loss_tuned = evaluate_loss(inner._model, tokenized)

    out.mkdir(parents=True, exist_ok=True)
    with (out / RESULTS.name).open("w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps({k: v for k, v in r.items() if k != "_mensagens"}, ensure_ascii=False) + "\n")
    execution = {
        "inicio_utc": started,
        "fim_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "mesmo_contexto_nas_variantes": same_context,
        "perda_respostas_esperadas": {"n": len(tokenized), "base": round(loss_base, 4), "ajustado": round(loss_tuned, 4)},
    }
    (out / "execucao.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_blind(rows, out)
    print(json.dumps(execution, ensure_ascii=False, indent=2))
    return 0 if same_context else 1


def write_blind(rows: list[dict], out: Path) -> None:
    """Arquivo para leitura cega: por exemplo, as duas respostas em ordem aleatória (A/B)."""
    rng = random.Random(BLIND_SEED)
    by_id: dict[str, dict] = {}
    for r in rows:
        by_id.setdefault(r["qa_id"], {})[r["variante"]] = r
    blind, key = [], {}
    for qa_id, pair in sorted(by_id.items(), key=lambda _: rng.random()):
        order = list(VARIANTS)
        rng.shuffle(order)
        key[qa_id] = {"A": order[0], "B": order[1]}
        ref = pair["base"]
        blind.append({
            "qa_id": qa_id, "pergunta": ref["pergunta"], "resposta_esperada": ref["resposta_esperada"],
            "A": pair[order[0]]["texto_modelo"], "B": pair[order[1]]["texto_modelo"],
        })
    with (out / BLIND.name).open("w", encoding="utf-8", newline="\n") as handle:
        for b in blind:
            handle.write(json.dumps(b, ensure_ascii=False) + "\n")
    (out / BLIND_KEY.name).write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def report(out: Path = OUT) -> int:
    from medprotocol.evaluation.final import SystemChecks, summarize_model_level, summarize_system_level
    from medprotocol.finetuning.answer_metrics import AnswerMetrics, summarize

    rows = read_jsonl(out / RESULTS.name)
    execution = json.loads((out / "execucao.json").read_text(encoding="utf-8"))
    summary = {"execucao": execution, "variantes": {}}
    review = json.loads((out / REVIEW.name).read_text(encoding="utf-8")) if (out / REVIEW.name).exists() else None
    key = json.loads((out / BLIND_KEY.name).read_text(encoding="utf-8"))
    for variant in VARIANTS:
        vr = [r for r in rows if r["variante"] == variant]
        called = [r for r in vr if r.get("metricas_modelo")]
        entry = {
            "nivel_1_modelo": {
                "n_com_modelo": len(called),
                "heuristicas_etapa_4": summarize([AnswerMetrics(**r["metricas_modelo"]) for r in called]),
                **summarize_model_level(called),
            },
            "nivel_2_sistema": summarize_system_level([SystemChecks(**r["checagens_sistema"]) for r in vr]),
        }
        if review:
            counts: dict[str, int] = {}
            for qa_id, item in review["avaliacoes"].items():
                letter = next(l for l, v in key[qa_id].items() if v == variant)
                verdict = item[letter]["veredito"]
                counts[verdict] = counts.get(verdict, 0) + 1
            entry["nivel_3_leitura_tecnica"] = counts
        summary["variantes"][variant] = entry
    (out / SUMMARY.name).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--smoke", type=Path, metavar="DIR", help="testa o mecanismo com 2 exemplos da validação interna (nunca o qa_eval)")
    args = parser.parse_args(argv)
    if args.smoke:
        return run(args.smoke)
    return freeze() if args.freeze else run() if args.run else report()


if __name__ == "__main__":
    sys.exit(main())
