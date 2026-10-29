"""Chunking e metadados (offline: não carrega o modelo de embeddings)."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="extra [rag] não instalado")

from medprotocol.data.loaders import parse_protocol  # noqa: E402
from medprotocol.rag.chunking import (  # noqa: E402
    METADATA_KEYS,
    ChunkTooLargeError,
    PatientDataInIndexError,
    build_chunks,
    section_header,
)

PROTOCOLS_DIR = Path(__file__).resolve().parents[1] / "data" / "protocols"


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(PROTOCOLS_DIR, words, max_tokens=10_000)


def test_one_chunk_per_section(chunks):
    protocols = [parse_protocol(p) for p in sorted(PROTOCOLS_DIR.glob("*.md"))]
    expected = {f"{p.protocol_id}#{s}" for p in protocols for s, text in p.sections.items() if text}
    refs = [c.metadata["ref"] for c in chunks]
    assert len(refs) == len(set(refs)) == len(expected) == 20
    assert set(refs) == expected


def test_metadata_complete_and_consistent(chunks):
    for chunk in chunks:
        meta = chunk.metadata
        assert set(meta) == set(METADATA_KEYS)
        assert meta["ref"] == f"{meta['protocol_id']}#{meta['section_id']}"
        assert meta["source"] == f"protocols/{meta['protocol_id']}.md"
        assert meta["status"] == "SIMULACAO"
        assert meta["natureza_regras"] == "simulada"
        assert meta["validacao_clinica"] == "nao_realizada"
        assert meta["versao"] == "1.0"
        assert (meta["chunk_index"], meta["chunk_count"]) == (0, 1)


def test_chunk_text_is_literal_section_and_never_mixes_sections(chunks):
    for chunk in chunks:
        protocol = parse_protocol(PROTOCOLS_DIR / f"{chunk.metadata['protocol_id']}.md")
        section_id = chunk.metadata["section_id"]
        header, body = chunk.page_content.split("\n", 1)
        assert header == section_header(protocol, section_id)
        assert body == protocol.sections[section_id]
        assert chunk.page_content.count("[") == 1  # apenas o ID da própria seção no cabeçalho


def test_front_matter_and_prefixes_not_in_chunks(chunks):
    for chunk in chunks:
        assert "protocol_id:" not in chunk.page_content
        assert not chunk.page_content.startswith(("query:", "passage:"))


def test_no_patient_records_in_chunks(chunks):
    for chunk in chunks:
        for marker in ("PAC-", "EX-0", "CON-0"):
            assert marker not in chunk.page_content


def test_patient_identifier_in_protocol_is_rejected(tmp_path):
    target = tmp_path / "protocols"
    shutil.copytree(PROTOCOLS_DIR, target)
    path = target / "PROT-DM2-001.md"
    path.write_text(path.read_text(encoding="utf-8").replace("Adultos com diagnóstico", "PAC-0003: adultos com diagnóstico"), encoding="utf-8")
    with pytest.raises(PatientDataInIndexError, match="PAC-0003"):
        build_chunks(target, words, max_tokens=10_000)


def test_large_section_is_split_without_losing_content_or_id(tmp_path):
    # Limite pequeno (60 "tokens" = palavras) força a divisão de PROT-DM2-001 [S4] (8 itens),
    # mas comporta cada item isolado.
    target = tmp_path / "protocols"
    target.mkdir()
    shutil.copy(PROTOCOLS_DIR / "PROT-DM2-001.md", target)
    pieces = [c for c in build_chunks(target, words, max_tokens=60) if c.metadata["ref"] == "PROT-DM2-001#S4"]
    assert len(pieces) > 1
    assert [p.metadata["chunk_index"] for p in pieces] == list(range(len(pieces)))
    assert {p.metadata["chunk_count"] for p in pieces} == {len(pieces)}
    assert all(words(p.page_content) <= 60 for p in pieces)
    header = pieces[0].page_content.split("\n", 1)[0]
    assert all(p.page_content.startswith(header + "\n") for p in pieces)
    rebuilt = "\n".join(p.page_content.split("\n", 1)[1] for p in pieces)
    original = parse_protocol(PROTOCOLS_DIR / "PROT-DM2-001.md").sections["S4"]
    assert rebuilt == original  # nada truncado, ordem preservada


def test_unit_larger_than_limit_raises_instead_of_truncating():
    with pytest.raises(ChunkTooLargeError, match="excede"):
        build_chunks(PROTOCOLS_DIR, words, max_tokens=12)


def test_empty_protocols_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path, words, max_tokens=512)
