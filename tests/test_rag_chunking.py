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


class TestExtractionQualityWarnings:
    def test_normal_extraction_no_warnings(self):
        text = "This is a normal academic paper. " * 100
        result = ExtractionResult(
            text=text, lines=text.split("\n"), content_hash="h1", page_count=10
        )
        assert result.quality_warnings() == []

    def test_empty_text(self):
        result = ExtractionResult(text="", lines=[], content_hash="h2", page_count=5)
        warnings = result.quality_warnings()
        assert len(warnings) == 1
        assert "no_text" in warnings[0]

    def test_scanned_pdf_low_density(self):
        # 10 pages, only 50 chars total -> 5 chars/page
        result = ExtractionResult(
            text="Title line\nAuthor line",
            lines=["Title line", "Author line"],
            content_hash="h3",
            page_count=10,
        )
        warnings = result.quality_warnings()
        assert any("low_density" in w for w in warnings)

    def test_very_short_text(self):
        result = ExtractionResult(
            text="Fig. S1a. Mediation model.",
            lines=["Fig. S1a. Mediation model."],
            content_hash="h4",
            page_count=1,
        )
        warnings = result.quality_warnings()
        assert any("very_short" in w for w in warnings)

    def test_mostly_blank_lines(self):
        lines = ["Title"] + [""] * 80 + ["Caption"]
        result = ExtractionResult(
            text="\n".join(lines),
            lines=lines,
            content_hash="h5",
            page_count=1,
        )
        warnings = result.quality_warnings()
        assert any("mostly_blank" in w for w in warnings)

    def test_no_page_count_skips_density_check(self):
        result = ExtractionResult(
            text="Short text.", lines=["Short text."], content_hash="h6"
        )
        warnings = result.quality_warnings()
        # Should still flag very_short, but not low_density
        assert any("very_short" in w for w in warnings)
        assert not any("low_density" in w for w in warnings)


class TestDocumentIndexMetadata:
    """Tests for metadata storage and retrieval in DocumentIndex."""

    def _make_index(self, tmp_path):
        from mcp_zotero.rag.document_index import DocumentIndex

        return DocumentIndex(tmp_path / "test.db")

    def test_add_and_get_with_metadata(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="K1",
            title="Housing Instability",
            content_hash="abc",
            chunk_count=5,
            metadata={"authors": ["Jane Smith", "Bob Jones"], "year": 2020},
        )
        doc = idx.get_document("K1")
        assert doc is not None
        assert doc.authors == ["Jane Smith", "Bob Jones"]
        assert doc.year == 2020

    def test_get_without_metadata(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="K2",
            title="Old Item",
            content_hash="def",
            chunk_count=3,
        )
        doc = idx.get_document("K2")
        assert doc is not None
        assert doc.authors is None
        assert doc.year is None

    def test_list_documents_with_metadata(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="K3",
            title="Paper A",
            content_hash="ghi",
            chunk_count=2,
            metadata={"authors": ["Alice"], "year": 2015},
        )
        docs = idx.list_documents()
        assert len(docs) == 1
        assert docs[0].authors == ["Alice"]
        assert docs[0].year == 2015

    def test_filter_by_year(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="Y1", title="Early", content_hash="a", chunk_count=1,
            metadata={"authors": ["A"], "year": 2010},
        )
        idx.add_document(
            item_key="Y2", title="Middle", content_hash="b", chunk_count=1,
            metadata={"authors": ["B"], "year": 2018},
        )
        idx.add_document(
            item_key="Y3", title="Late", content_hash="c", chunk_count=1,
            metadata={"authors": ["C"], "year": 2023},
        )

        keys = idx.get_item_keys_by_metadata(year_from=2015, year_to=2020)
        assert keys == ["Y2"]

        keys = idx.get_item_keys_by_metadata(year_from=2018)
        assert set(keys) == {"Y2", "Y3"}

        keys = idx.get_item_keys_by_metadata(year_to=2010)
        assert keys == ["Y1"]

    def test_filter_by_author(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="A1", title="Paper 1", content_hash="a", chunk_count=1,
            metadata={"authors": ["Jane Smith", "Bob Jones"], "year": 2020},
        )
        idx.add_document(
            item_key="A2", title="Paper 2", content_hash="b", chunk_count=1,
            metadata={"authors": ["Alice Brown"], "year": 2019},
        )

        keys = idx.get_item_keys_by_metadata(authors=["Smith"])
        assert keys == ["A1"]

        keys = idx.get_item_keys_by_metadata(authors=["jones"])
        assert keys == ["A1"]  # case-insensitive

        keys = idx.get_item_keys_by_metadata(authors=["Nobody"])
        assert keys == []

    def test_filter_no_filters_returns_none(self, tmp_path):
        idx = self._make_index(tmp_path)
        result = idx.get_item_keys_by_metadata()
        assert result is None

    def test_filter_skips_null_metadata(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_document(
            item_key="N1", title="No Meta", content_hash="a", chunk_count=1,
        )
        idx.add_document(
            item_key="N2", title="With Meta", content_hash="b", chunk_count=1,
            metadata={"authors": ["Smith"], "year": 2020},
        )
        keys = idx.get_item_keys_by_metadata(year_from=2019)
        assert keys == ["N2"]


class TestCitationFormatting:
    """Tests for the citation string builder used in semantic_search."""

    def _build_citation(self, authors, year):
        """Reproduce the citation logic from tools.py."""
        parts = []
        if authors:
            if len(authors) == 1:
                parts.append(authors[0].split()[-1])
            elif len(authors) == 2:
                parts.append(" & ".join(a.split()[-1] for a in authors))
            else:
                parts.append(f"{authors[0].split()[-1]} et al.")
        if year:
            parts.append(str(year))
        return ", ".join(parts) if parts else None

    def test_single_author(self):
        assert self._build_citation(["Jane Smith"], 2020) == "Smith, 2020"

    def test_two_authors(self):
        assert self._build_citation(
            ["Jane Smith", "Bob Jones"], 2020
        ) == "Smith & Jones, 2020"

    def test_three_authors(self):
        assert self._build_citation(
            ["Jane Smith", "Bob Jones", "Alice Brown"], 2020
        ) == "Smith et al., 2020"

    def test_no_year(self):
        assert self._build_citation(["Jane Smith"], None) == "Smith"

    def test_no_authors(self):
        assert self._build_citation([], 2020) == "2020"

    def test_no_authors_no_year(self):
        assert self._build_citation([], None) is None

    def test_none_authors(self):
        assert self._build_citation(None, 2020) == "2020"


class TestYearExtraction:
    """Tests for year extraction from Zotero date strings."""

    def _extract_year(self, date_str):
        import re
        if not date_str:
            return None
        match = re.search(r"\b((?:19|20)\d{2})\b", date_str)
        return int(match.group(1)) if match else None

    def test_year_only(self):
        assert self._extract_year("2020") == 2020

    def test_full_date(self):
        assert self._extract_year("2015-03-01") == 2015

    def test_month_year(self):
        assert self._extract_year("March 2018") == 2018

    def test_empty(self):
        assert self._extract_year("") is None

    def test_none(self):
        assert self._extract_year(None) is None

    def test_no_year_in_string(self):
        assert self._extract_year("undated") is None

    def test_old_year(self):
        assert self._extract_year("1995") == 1995
