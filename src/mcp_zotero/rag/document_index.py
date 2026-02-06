"""SQLite-based document index for tracking processed documents."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp_zotero.models import IndexedDocument

logger = logging.getLogger(__name__)


class DocumentIndex:
    """SQLite database for tracking indexed documents and their metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _ensure_schema(self) -> None:
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    item_key TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    file_path TEXT,
                    file_type TEXT,
                    metadata TEXT
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS extracted_text (
                    item_key TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    lines TEXT NOT NULL,
                    FOREIGN KEY (item_key) REFERENCES documents(item_key)
                )
            """)
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON documents(content_hash)
            """)

    def _row_to_document(self, row) -> IndexedDocument:
        """Convert a DB row to IndexedDocument, parsing metadata JSON."""
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return IndexedDocument(
            item_key=row["item_key"],
            title=row["title"],
            content_hash=row["content_hash"],
            chunk_count=row["chunk_count"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
            file_path=row["file_path"],
            file_type=row["file_type"],
            authors=meta.get("authors"),
            year=meta.get("year"),
        )

    def get_document(self, item_key: str) -> Optional[IndexedDocument]:
        cursor = self.connection.execute(
            "SELECT * FROM documents WHERE item_key = ?", (item_key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def get_content_hash(self, item_key: str) -> Optional[str]:
        cursor = self.connection.execute(
            "SELECT content_hash FROM documents WHERE item_key = ?", (item_key,)
        )
        row = cursor.fetchone()
        return row["content_hash"] if row else None

    def needs_reindex(self, item_key: str, new_hash: str) -> bool:
        existing_hash = self.get_content_hash(item_key)
        return existing_hash is None or existing_hash != new_hash

    def add_document(
        self,
        item_key: str,
        title: str,
        content_hash: str,
        chunk_count: int,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IndexedDocument:
        indexed_at = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata) if metadata else None

        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO documents
                (item_key, title, content_hash, chunk_count, indexed_at, file_path, file_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item_key,
                    title,
                    content_hash,
                    chunk_count,
                    indexed_at.isoformat(),
                    file_path,
                    file_type,
                    metadata_json,
                ),
            )

        return IndexedDocument(
            item_key=item_key,
            title=title,
            content_hash=content_hash,
            chunk_count=chunk_count,
            indexed_at=indexed_at,
            file_path=file_path,
            file_type=file_type,
            authors=metadata.get("authors") if metadata else None,
            year=metadata.get("year") if metadata else None,
        )

    def store_extracted_text(
        self, item_key: str, content_hash: str, lines: list[str]
    ) -> None:
        lines_json = json.dumps(lines)
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO extracted_text
                (item_key, content_hash, lines)
                VALUES (?, ?, ?)
            """,
                (item_key, content_hash, lines_json),
            )

    def get_extracted_lines(self, item_key: str) -> Optional[list[str]]:
        cursor = self.connection.execute(
            "SELECT lines FROM extracted_text WHERE item_key = ?", (item_key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["lines"])

    def remove_document(self, item_key: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM documents WHERE item_key = ?", (item_key,)
            )
            self.connection.execute(
                "DELETE FROM extracted_text WHERE item_key = ?", (item_key,)
            )
        return cursor.rowcount > 0

    def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> list[IndexedDocument]:
        cursor = self.connection.execute(
            "SELECT * FROM documents ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        documents = []
        for row in cursor:
            documents.append(self._row_to_document(row))
        return documents

    def get_item_keys_by_metadata(
        self,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        authors: Optional[list[str]] = None,
    ) -> Optional[list[str]]:
        """Return item_keys matching metadata filters. Returns None if no filters."""
        if not any([year_from, year_to, authors]):
            return None

        cursor = self.connection.execute(
            "SELECT item_key, metadata FROM documents"
        )
        matching = []
        for row in cursor:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            year = meta.get("year")
            authors_list = meta.get("authors", [])

            if year_from and (not year or year < year_from):
                continue
            if year_to and (not year or year > year_to):
                continue
            if authors:
                authors_lower = [q.lower() for q in authors]
                found = any(
                    any(q in a.lower() for a in authors_list)
                    for q in authors_lower
                )
                if not found:
                    continue

            matching.append(row["item_key"])

        return matching

    def get_stats(self) -> dict:
        cursor = self.connection.execute("""
            SELECT
                COUNT(*) as total_documents,
                SUM(chunk_count) as total_chunks,
                MAX(indexed_at) as last_updated
            FROM documents
        """)
        row = cursor.fetchone()
        return {
            "total_documents": row["total_documents"] or 0,
            "total_chunks": row["total_chunks"] or 0,
            "last_updated": row["last_updated"],
        }

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
