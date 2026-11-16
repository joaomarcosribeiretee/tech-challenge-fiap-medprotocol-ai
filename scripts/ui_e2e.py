"""Teste de ponta a ponta da interface Streamlit com o fluxo REAL (SQLite + RAG + LangGraph + Qwen2.5 + LoRA).

Usa `streamlit.testing.v1.AppTest`: executa app/streamlit_app.py de verdade, preenche o formulário,
clica em Enviar e lê o que a tela exibiu. Nada é simulado. Requer GPU, adaptador, banco e índice.

Uso (ambiente .venv-train, na raiz do repositório):
    python scripts/ui_e2e.py                                   # perguntas de scripts/demo/perguntas.txt, variante ajustada
    python scripts/ui_e2e.py --variants ajustado base --out reports/ui/e2e_streamlit.jsonl

Cada linha da saída registra o que a TELA mostrou (status, título da resposta, se o texto original
ficou só na área "não validado", alertas, ID de auditoria). A auditoria do fluxo vai para
AUDIT_LOG_PATH (defina-o para separar a execução de demonstração do log padrão).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
LABELS = {"ajustado": "Ajustado (Qwen2.5-1.5B + LoRA)", "base": "Base (Qwen2.5-1.5B, sem adaptador)"}


def run_question(at, pergunta: str, variant: str) -> dict:
    at.radio[0].set_value(variant)
    at.text_area[0].input(pergunta)
    start = time.perf_counter()
    at.button[0].click().run()
    elapsed = round(time.perf_counter() - start, 1)
    markdown = [m.value for m in at.markdown]
    status_line = next((m for m in markdown if m.startswith("### ")), "")
    title = next((m.strip("*") for m in markdown if m.startswith("**") and m.endswith("**")), "")
    audit = next((m for m in markdown if m.startswith("**ID de auditoria:**")), "")
    expanders = [e.label for e in at.expander]
    codes = [str(c.value) for c in at.code]
    tables = [df.value for df in at.dataframe]
    alert_codes = next((list(t["Código"]) for t in tables if "Código" in t.columns), [])
    cited = next((list(t.loc[t["Citado na resposta"] == "sim", "Trecho"]) for t in tables if "Trecho" in t.columns), [])
    return {
        "pergunta": pergunta,
        "variante_selecionada": variant,
        "excecao_na_tela": [str(e.value) for e in at.exception],
        "erros_na_tela": [e.value for e in at.error],
        "status_exibido": status_line.removeprefix("### "),
        "titulo_resposta": title,
        "area_nao_validado": any("NÃO VALIDADO" in e for e in expanders),
        "texto_original_so_na_area": bool(codes) and all(all(c not in m for m in markdown) for c in codes),
        "alertas": alert_codes,
        "fontes_citadas": cited,
        "modelo_exibido": next((m for m in markdown if m.startswith("**Modelo**")), "").split("\n")[-1],
        "audit_id": audit.split("`")[1] if "`" in audit else None,
        "tempo_tela_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    from streamlit.testing.v1 import AppTest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=ROOT / "scripts" / "demo" / "perguntas.txt")
    parser.add_argument("--variants", nargs="+", choices=list(LABELS), default=["ajustado"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    questions = [l.strip() for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    at = AppTest.from_file(str(APP), default_timeout=600)
    start = time.perf_counter()
    at.run()  # monta o fluxo e carrega o modelo (cache_resource)
    startup = round(time.perf_counter() - start, 1)
    if at.exception or at.error:
        print("falha ao iniciar:", [e.value for e in at.exception] + [e.value for e in at.error])
        return 1
    warnings = [w.value for w in at.warning][1:]  # o primeiro é o aviso educacional fixo
    print(f"inicialização: {startup} s; avisos de ambiente: {warnings or 'nenhum'}")

    results = [run_question(at, q, v) for v in args.variants for q in questions]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        header = {"inicializacao_s": startup, "avisos_ambiente": warnings}
        lines = [json.dumps(header, ensure_ascii=False)] + [json.dumps(r, ensure_ascii=False) for r in results]
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"{len(results)} perguntas -> {args.out}")
    return 1 if any(r["excecao_na_tela"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
