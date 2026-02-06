"""Tests for mcp_zotero.rag.vector_store — numpy-based vector store."""

import pytest

numpy = pytest.importorskip("numpy")

from mcp_zotero.models import ChunkType, DocumentChunk
from mcp_zotero.rag.vector_store import NumpyVectorStore


def _make_chunk(item_key: str, index: int, text: str = "test") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{item_key}_{index}",
        item_key=item_key,
        chunk_index=index,
        text=text,
        start_line=0,
        end_line=1,
        chunk_type=ChunkType.BODY,
    )


def _random_embedding(dim: int = 384) -> list[float]:
    return numpy.random.randn(dim).tolist()


class TestNumpyVectorStore:
    def test_add_and_search(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")

        chunks = [_make_chunk("A", 0, "hello"), _make_chunk("A", 1, "world")]
        embeddings = [_random_embedding(), _random_embedding()]

        count = store.add_chunks(chunks, embeddings)
        assert count == 2

        results = store.search(embeddings[0], limit=2)
        assert len(results) == 2
        # First result should be the exact match
        assert results[0][0].chunk_id == "A_0"
        assert results[0][1] > 0.99  # Near-perfect match

    def test_get_chunk(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")
        chunk = _make_chunk("B", 0, "content")
        store.add_chunks([chunk], [_random_embedding()])

        retrieved = store.get_chunk("B_0")
        assert retrieved is not None
        assert retrieved.text == "content"

        assert store.get_chunk("NONEXISTENT_0") is None

    def test_delete_by_item_key(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")

        chunks = [
            _make_chunk("X", 0),
            _make_chunk("X", 1),
            _make_chunk("Y", 0),
        ]
        embeddings = [_random_embedding() for _ in chunks]
        store.add_chunks(chunks, embeddings)

        deleted = store.delete_by_item_key("X")
        assert deleted == 2

        assert store.get_chunk("X_0") is None
        assert store.get_chunk("Y_0") is not None

    def test_find_similar(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")

        # Create chunks from different documents with known embeddings
        e1 = numpy.array([1.0, 0.0, 0.0]).tolist()
        e2 = numpy.array([0.9, 0.1, 0.0]).tolist()  # similar to e1
        e3 = numpy.array([0.0, 0.0, 1.0]).tolist()  # different

        chunks = [
            _make_chunk("A", 0),
            _make_chunk("B", 0),
            _make_chunk("C", 0),
        ]
        store.add_chunks(chunks, [e1, e2, e3])

        similar = store.find_similar("A_0", limit=2, exclude_same_document=True)
        assert len(similar) == 2
        assert similar[0][0].chunk_id == "B_0"  # Most similar

    def test_persistence(self, tmp_path):
        path = tmp_path / "vectors"

        # Create and populate
        store1 = NumpyVectorStore(path)
        chunk = _make_chunk("P", 0, "persistent")
        store1.add_chunks([chunk], [_random_embedding()])

        # Reload from disk
        store2 = NumpyVectorStore(path)
        retrieved = store2.get_chunk("P_0")
        assert retrieved is not None
        assert retrieved.text == "persistent"

    def test_get_stats(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")
        stats = store.get_stats()
        assert stats["total_chunks"] == 0

        store.add_chunks([_make_chunk("S", 0)], [_random_embedding()])
        stats = store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["size_mb"] >= 0

    def test_search_with_filters(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")

        chunks = [
            _make_chunk("A", 0),
            _make_chunk("B", 0),
        ]
        emb = _random_embedding()
        store.add_chunks(chunks, [emb, _random_embedding()])

        # Filter by item_keys
        results = store.search(emb, limit=10, item_keys=["A"])
        assert len(results) == 1
        assert results[0][0].item_key == "A"

    def test_reindex_replaces_existing(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")

        chunk_v1 = _make_chunk("R", 0, "version1")
        store.add_chunks([chunk_v1], [_random_embedding()])

        chunk_v2 = _make_chunk("R", 0, "version2")
        store.add_chunks([chunk_v2], [_random_embedding()])

        retrieved = store.get_chunk("R_0")
        assert retrieved.text == "version2"

        stats = store.get_stats()
        assert stats["total_chunks"] == 1

    def test_empty_search(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")
        results = store.search(_random_embedding(), limit=10)
        assert results == []

    def test_mismatched_chunks_embeddings(self, tmp_path):
        store = NumpyVectorStore(tmp_path / "vectors")
        with pytest.raises(ValueError, match="must match"):
            store.add_chunks([_make_chunk("E", 0)], [_random_embedding(), _random_embedding()])
