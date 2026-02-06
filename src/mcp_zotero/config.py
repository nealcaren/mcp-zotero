"""Unified configuration for mcp-zotero (base + RAG)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings. Use Settings.from_env() to load."""

    # Zotero connection
    zotero_library_id: str = ""
    zotero_library_type: str = "user"
    zotero_local: bool = True
    zotero_local_key: Optional[str] = None
    zotero_api_key: Optional[str] = None
    zotero_attachments_dir: Optional[str] = None

    # RAG settings (only used when rag extra is installed)
    rag_index_dir: str = "~/.zotero-rag"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_enable_ocr: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""

        def _bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.lower() in {"1", "true", "yes", "y"}

        return cls(
            zotero_library_id=os.getenv("ZOTERO_LIBRARY_ID", ""),
            zotero_library_type=os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
            zotero_local=_bool("ZOTERO_LOCAL", True),
            zotero_local_key=os.getenv("ZOTERO_LOCAL_KEY"),
            zotero_api_key=os.getenv("ZOTERO_API_KEY"),
            zotero_attachments_dir=os.getenv("ZOTERO_ATTACHMENTS_DIR"),
            rag_index_dir=os.getenv("RAG_INDEX_DIR", "~/.zotero-rag"),
            rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
            rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
            rag_enable_ocr=_bool("RAG_ENABLE_OCR", False),
        )

    # --- Derived paths ---

    @property
    def index_dir(self) -> Path:
        return Path(self.rag_index_dir).expanduser().resolve()

    @property
    def vectors_dir(self) -> Path:
        return self.index_dir / "vectors"

    @property
    def sqlite_path(self) -> Path:
        return self.index_dir / "document_index.db"

    @property
    def cache_dir(self) -> Path:
        return self.index_dir / "cache"

    @property
    def attachments_path(self) -> Optional[Path]:
        if self.zotero_attachments_dir:
            return Path(self.zotero_attachments_dir).expanduser().resolve()
        return None

    # --- Helpers ---

    def ensure_directories(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> list[str]:
        issues: list[str] = []
        if not self.zotero_library_id:
            issues.append("ZOTERO_LIBRARY_ID is not set")
        if not self.zotero_local_key and not self.zotero_api_key:
            issues.append("ZOTERO_LOCAL_KEY or ZOTERO_API_KEY required")
        if self.rag_chunk_size < 100:
            issues.append("RAG_CHUNK_SIZE should be at least 100")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            issues.append("RAG_CHUNK_OVERLAP must be less than RAG_CHUNK_SIZE")
        return issues

    @property
    def read_api_key(self) -> Optional[str]:
        return self.zotero_api_key or self.zotero_local_key


# Lazy singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
