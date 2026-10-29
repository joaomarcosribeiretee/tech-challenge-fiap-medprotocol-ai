"""Divisão dos protocolos em chunks (Documents do LangChain), um por seção.

Regras:
- Uma seção nunca é misturada com outra no mesmo chunk.
- Texto do chunk: "<título do protocolo> — [<ID>] <título da seção>" + corpo literal da seção.
  Tudo vem do arquivo; nada é reescrito.
- Se a seção (com o cabeçalho) exceder o limite de tokens do modelo, é dividida entre itens
  de lista/parágrafos, mantendo o mesmo section_id e a ordem (chunk_index/chunk_count).
- Uma unidade que sozinha excede o limite gera ChunkTooLargeError: nunca truncar em silêncio.
- Registros de pacientes (PAC-/EX-/CON-) nunca podem entrar no índice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from medprotocol.data.loaders import Protocol, parse_protocol

CHUNKING_VERSION = "1"
PATIENT_RECORD_RE = re.compile(r"\b(?:PAC|EX|CON)-\d{4}\b")
METADATA_KEYS = (
    "ref",
    "protocol_id",
    "section_id",
    "section_title",
    "titulo",
    "versao",
    "status",
    "natureza_regras",
    "validacao_clinica",
    "source",
    "chunk_index",
    "chunk_count",
)

TokenCounter = Callable[[str], int]


class ChunkTooLargeError(ValueError):
    """Uma unidade de texto excede o limite de tokens e não pode ser dividida sem truncar."""


class PatientDataInIndexError(ValueError):
    """Um protocolo contém identificadores de registros de pacientes."""


def section_header(protocol: Protocol, section_id: str) -> str:
    return f"{protocol.metadata['titulo']} — [{section_id}] {protocol.section_titles[section_id]}"


def _units(text: str) -> list[str]:
    """Itens de lista e parágrafos; linhas de continuação ficam com o item anterior."""
    units: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            units.append([])
        elif line.lstrip().startswith("- ") or not units or not units[-1]:
            units.append([line])
        else:
            units[-1].append(line)
    return ["\n".join(unit) for unit in units if unit]


def _split_section(header: str, text: str, count_tokens: TokenCounter, max_tokens: int, where: str) -> list[str]:
    if count_tokens(f"{header}\n{text}") <= max_tokens:
        return [text]
    pieces: list[list[str]] = [[]]
    for unit in _units(text):
        if count_tokens(f"{header}\n{unit}") > max_tokens:
            raise ChunkTooLargeError(f"{where}: um item excede {max_tokens} tokens mesmo isolado")
        candidate = pieces[-1] + [unit]
        if pieces[-1] and count_tokens(f"{header}\n" + "\n".join(candidate)) > max_tokens:
            pieces.append([unit])
        else:
            pieces[-1] = candidate
    return ["\n".join(piece) for piece in pieces]


def build_chunks(protocols_dir: Path, count_tokens: TokenCounter, max_tokens: int) -> list[Document]:
    """Lê todos os protocolos `*.md` e devolve os chunks com metadados rastreáveis."""
    protocols_dir = Path(protocols_dir)
    paths = sorted(protocols_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"nenhum protocolo .md em {protocols_dir}")

    documents: list[Document] = []
    for path in paths:
        protocol = parse_protocol(path)
        found = PATIENT_RECORD_RE.findall(protocol.body)
        if found:
            raise PatientDataInIndexError(f"{path.name}: contém identificadores de pacientes {sorted(set(found))}")
        for section_id, text in protocol.sections.items():
            if not text:
                continue
            header = section_header(protocol, section_id)
            pieces = _split_section(header, text, count_tokens, max_tokens, f"{path.name} [{section_id}]")
            for index, piece in enumerate(pieces):
                documents.append(
                    Document(
                        page_content=f"{header}\n{piece}",
                        metadata={
                            "ref": f"{protocol.protocol_id}#{section_id}",
                            "protocol_id": protocol.protocol_id,
                            "section_id": section_id,
                            "section_title": protocol.section_titles[section_id],
                            "titulo": protocol.metadata.get("titulo", ""),
                            "versao": protocol.metadata.get("versao", ""),
                            "status": protocol.metadata.get("status", ""),
                            "natureza_regras": protocol.metadata.get("natureza_regras", ""),
                            "validacao_clinica": protocol.metadata.get("validacao_clinica", ""),
                            "source": f"{path.parent.name}/{path.name}",
                            "chunk_index": index,
                            "chunk_count": len(pieces),
                        },
                    )
                )
    return documents


def chunk_id(document: Document) -> str:
    return f"{document.metadata['ref']}:{document.metadata['chunk_index']}"
