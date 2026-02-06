"""Numpy-based vector store — lightweight replacement for ChromaDB.

Stores pre-normalized embeddings as a numpy matrix. Cosine similarity
reduces to a single matrix-vector dot product. Suitable for personal
libraries (<100K chunks).
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from mcp_zotero.models import ChunkType, DocumentChunk

logger = logging.getLogger(__name__)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores, sorted descending."""
    if k >= len(scores):
        return np.argsort(-scores)
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


class NumpyVectorStore:
    """Vector store backed by numpy arrays + pickle metadata."""

    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self._embeddings_path = persist_directory / "embeddings.npy"
        self._metadata_path = persist_directory / "metadata.pkl"

        # In-memory state
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32, L2-normalized
        self._chunks: list[DocumentChunk] = []
        self._id_to_row: dict[str, int] = {}
        self._item_key_to_rows: dict[str, list[int]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if self._embeddings_path.exists() and self._metadata_path.exists():
            self._embeddings = np.load(str(self._embeddings_path))
            with open(self._metadata_path, "rb") as f:
                self._chunks = pickle.load(f)
            self._rebuild_indices()
        else:
            self._embeddings = None
            self._chunks = []

        self._loaded = True

    def _rebuild_indices(self) -> None:
        self._id_to_row = {}
        self._item_key_to_rows = {}
        for i, chunk in enumerate(self._chunks):
            self._id_to_row[chunk.chunk_id] = i
            self._item_key_to_rows.setdefault(chunk.item_key, []).append(i)

    def add_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        self._ensure_loaded()

        new_emb = np.array(embeddings, dtype=np.float32)
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new_emb = new_emb / norms

        # Remove any existing chunks with the same IDs (re-indexing)
        existing_ids = {c.chunk_id for c in chunks} & set(self._id_to_row.keys())
        if existing_ids:
            self._remove_rows({self._id_to_row[cid] for cid in existing_ids})

        # Append
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        else:
            self._embeddings = new_emb

        base = len(self._chunks)
        self._chunks.extend(chunks)

        for i, chunk in enumerate(chunks):
            row = base + i
            self._id_to_row[chunk.chunk_id] = row
            self._item_key_to_rows.setdefault(chunk.item_key, []).append(row)

        self.save()
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        item_keys: Optional[list[str]] = None,
        chunk_types: Optional[list[ChunkType]] = None,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        self._ensure_loaded()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity = dot product (both normalized)
        scores = self._embeddings @ q

        # Build mask for filters
        mask = np.ones(len(self._chunks), dtype=bool)

        if item_keys:
            key_set = set(item_keys)
            mask &= np.array([c.item_key in key_set for c in self._chunks])

        if chunk_types:
            type_set = set(chunk_types)
            mask &= np.array([c.chunk_type in type_set for c in self._chunks])

        if min_score > 0:
            mask &= scores >= min_score

        # Apply mask, get top results
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]
        top_k = min(limit, len(valid_indices))
        top_local = _top_k_indices(valid_scores, top_k)
        top_indices = valid_indices[top_local]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def find_similar(
        self,
        chunk_id: str,
        limit: int = 5,
        exclude_same_document: bool = True,
    ) -> list[tuple[DocumentChunk, float]]:
        self._ensure_loaded()

        if chunk_id not in self._id_to_row:
            return []

        row = self._id_to_row[chunk_id]
        embedding = self._embeddings[row]

        if exclude_same_document:
            item_key = self._chunks[row].item_key
        else:
            item_key = None

        scores = self._embeddings @ embedding

        mask = np.ones(len(self._chunks), dtype=bool)
        mask[row] = False  # Exclude self

        if exclude_same_document and item_key:
            mask &= np.array([c.item_key != item_key for c in self._chunks])

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]
        top_k = min(limit, len(valid_indices))
        top_local = _top_k_indices(valid_scores, top_k)
        top_indices = valid_indices[top_local]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        self._ensure_loaded()
        row = self._id_to_row.get(chunk_id)
        if row is None:
            return None
        return self._chunks[row]

    def delete_by_item_key(self, item_key: str) -> int:
        self._ensure_loaded()

        rows = self._item_key_to_rows.get(item_key, [])
        if not rows:
            return 0

        count = len(rows)
        self._remove_rows(set(rows))
        self.save()

        logger.info(f"Deleted {count} chunks for item {item_key}")
        return count

    def get_stats(self) -> dict:
        self._ensure_loaded()

        size_bytes = 0
        for path in [self._embeddings_path, self._metadata_path]:
            if path.exists():
                size_bytes += path.stat().st_size

        return {
            "total_chunks": len(self._chunks),
            "size_mb": size_bytes / (1024 * 1024),
        }

    def save(self) -> None:
        """Persist to disk via atomic temp-file + rename."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if self._embeddings is not None and len(self._embeddings) > 0:
            # Atomic save for embeddings
            # np.save appends .npy if missing, so use a .npy suffix for the temp file
            fd, tmp = tempfile.mkstemp(
                dir=str(self.persist_directory), suffix=".npy"
            )
            os.close(fd)
            try:
                np.save(tmp, self._embeddings)
                # np.save with a .npy-ending path writes in-place (no extra suffix)
                os.replace(tmp, str(self._embeddings_path))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

            # Atomic save for metadata
            fd, tmp = tempfile.mkstemp(
                dir=str(self.persist_directory), suffix=".pkl"
            )
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    pickle.dump(self._chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, str(self._metadata_path))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        else:
            # Empty store — clean up files
            for path in [self._embeddings_path, self._metadata_path]:
                if path.exists():
                    path.unlink()

    def _remove_rows(self, rows_to_remove: set[int]) -> None:
        """Remove rows by index and compact arrays."""
        if not rows_to_remove:
            return

        keep = [i for i in range(len(self._chunks)) if i not in rows_to_remove]

        if keep and self._embeddings is not None:
            self._embeddings = self._embeddings[keep]
        else:
            self._embeddings = None

        self._chunks = [self._chunks[i] for i in keep]
        self._rebuild_indices()
