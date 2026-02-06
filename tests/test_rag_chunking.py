"""Tests for mcp_zotero.rag.chunking."""

import pytest

from mcp_zotero.models import ChunkType
from mcp_zotero.rag.extractor import ExtractionResult
from mcp_zotero.rag.chunking import ArticleChunker, BookChunker, get_chunker


def _make_extraction(text: str) -> ExtractionResult:
    return ExtractionResult(
        text=text,
        lines=text.split("\n"),
        content_hash="abc123",
    )


class TestArticleChunker:
    def test_basic_chunking(self):
        text = "\n\n".join([f"Paragraph {i}. " * 40 for i in range(10)])
        extraction = _make_extraction(text)
        chunker = ArticleChunker(chunk_size=200, chunk_overlap=50)
        result = chunker.chunk(extraction, "TEST_KEY")

        assert len(result.chunks) > 1
        assert all(c.item_key == "TEST_KEY" for c in result.chunks)
        assert all(c.chunk_type == ChunkType.BODY for c in result.chunks)

    def test_abstract_detection(self):
        text = "Abstract\n\nThis is the abstract of the paper.\n\nIntroduction\n\nBody text here."
        extraction = _make_extraction(text)
        chunker = ArticleChunker(chunk_size=500, min_chunk_size=5)
        result = chunker.chunk(extraction, "ABS_KEY")

        abstract_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.ABSTRACT]
        assert len(abstract_chunks) == 1
        assert "abstract of the paper" in abstract_chunks[0].text

    def test_chunk_ids_sequential(self):
        text = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(5)])
        extraction = _make_extraction(text)
        chunker = ArticleChunker(chunk_size=100)
        result = chunker.chunk(extraction, "SEQ")

        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == f"SEQ_{i}"
            assert chunk.chunk_index == i


class TestBookChunker:
    def test_chapter_detection(self):
        text = "Chapter 1 Introduction\n\n" + "Text. " * 100 + "\n\nChapter 2 Methods\n\n" + "More text. " * 100
        extraction = _make_extraction(text)
        chunker = BookChunker(chunk_size=200)
        result = chunker.chunk(extraction, "BOOK")

        chapter_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CHAPTER]
        assert len(chapter_chunks) >= 2

    def test_fallback_to_article(self):
        text = "Just plain text without chapters.\n\n" + "Body paragraph. " * 50
        extraction = _make_extraction(text)
        chunker = BookChunker(chunk_size=100)
        result = chunker.chunk(extraction, "FLAT")

        assert len(result.chunks) >= 1
        assert all(c.chunk_type in (ChunkType.BODY, ChunkType.ABSTRACT) for c in result.chunks)


class TestGetChunker:
    def test_article_default(self):
        chunker = get_chunker()
        assert isinstance(chunker, ArticleChunker)

    def test_book_type(self):
        chunker = get_chunker("book")
        assert isinstance(chunker, BookChunker)

    def test_journal_article(self):
        chunker = get_chunker("journalArticle")
        assert isinstance(chunker, ArticleChunker)

    def test_kwargs_passed(self):
        chunker = get_chunker(chunk_size=256, chunk_overlap=32)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32
