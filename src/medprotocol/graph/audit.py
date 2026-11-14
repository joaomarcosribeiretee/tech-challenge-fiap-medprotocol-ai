"""Registro de auditoria do fluxo (JSON Lines, um evento por pergunta).

Minimização de dados: o log guarda o que é necessário para reconstruir a execução
(identificador, modelo, paciente consultado, fontes, alertas, status, tempos), mas **não** o
texto da pergunta nem o da resposta. Ambos ficam representados por hash SHA-256 e tamanho, o
que permite conferir uma resposta guardada em outro lugar sem espalhar o conteúdo pelos logs.
Os pacientes são sintéticos, e o ID é necessário para a auditoria.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("medprotocol.audit")


def text_digest(text: str | None) -> dict | None:
    if text is None:
        return None
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "caracteres": len(text)}


class AuditLogger:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        logger.info("auditoria %s status=%s alertas=%s", event.get("audit_id"), event.get("status"),
                    [a["codigo"] for a in event.get("alertas", [])])
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
