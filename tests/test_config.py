"""Tests for mcp_zotero.config."""

import os

import pytest

from mcp_zotero.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith("ZOTERO_") or k.startswith("RAG_"):
            monkeypatch.delenv(k, raising=False)
    # Reset singleton
    import mcp_zotero.config as cfg
    cfg._settings = None
    yield
    cfg._settings = None


def test_defaults():
    s = Settings()
    assert s.zotero_library_id == ""
    assert s.zotero_local is True
    assert s.rag_chunk_size == 512
    assert s.rag_enable_ocr is False


def test_from_env(monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "999")
    monkeypatch.setenv("ZOTERO_LOCAL", "false")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "256")
    monkeypatch.setenv("RAG_ENABLE_OCR", "true")

    s = Settings.from_env()
    assert s.zotero_library_id == "999"
    assert s.zotero_local is False
    assert s.rag_chunk_size == 256
    assert s.rag_enable_ocr is True


def test_validate_missing_library_id():
    s = Settings()
    issues = s.validate_config()
    assert any("ZOTERO_LIBRARY_ID" in i for i in issues)


def test_validate_chunk_overlap_too_large():
    s = Settings(
        zotero_library_id="1",
        zotero_api_key="abc",
        rag_chunk_size=100,
        rag_chunk_overlap=200,
    )
    issues = s.validate_config()
    assert any("RAG_CHUNK_OVERLAP" in i for i in issues)


def test_read_api_key_prefers_zotero_api_key():
    s = Settings(zotero_api_key="web", zotero_local_key="local")
    assert s.read_api_key == "web"


def test_read_api_key_falls_back_to_local():
    s = Settings(zotero_local_key="local")
    assert s.read_api_key == "local"


def test_derived_paths():
    s = Settings(rag_index_dir="/tmp/test-zotero-rag")
    assert str(s.vectors_dir).endswith("vectors")
    assert str(s.sqlite_path).endswith("document_index.db")
    assert str(s.cache_dir).endswith("cache")


def test_ensure_directories(tmp_path):
    s = Settings(rag_index_dir=str(tmp_path / "rag-test"))
    s.ensure_directories()
    assert s.index_dir.exists()
    assert s.vectors_dir.exists()
    assert s.cache_dir.exists()
