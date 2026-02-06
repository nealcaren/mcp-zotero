"""Chunking strategies for different document types."""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from mcp_zotero.models import ChunkType, DocumentChunk
from mcp_zotero.rag.extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of chunking a document."""

    chunks: list[DocumentChunk]
    total_tokens: int


class Chunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(self, extraction: ExtractionResult, item_key: str) -> ChunkResult:
        pass

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    @staticmethod
    def find_sentence_boundary(text: str, position: int, direction: int = 1) -> int:
        sentence_endings = re.compile(r"[.!?]\s+")
        if direction > 0:
            match = sentence_endings.search(text, position)
            if match:
                return match.end()
            return len(text)
        else:
            text_before = text[:position]
            matches = list(sentence_endings.finditer(text_before))
            if matches:
                return matches[-1].end()
            return 0


class ArticleChunker(Chunker):
    """Chunker optimized for academic articles."""

    ABSTRACT_PATTERN = re.compile(
        r"(?:^|\n)(?:abstract|summary)[\s:]*\n+(.*?)(?=\n(?:introduction|keywords|1\.|#|\Z))",
        re.IGNORECASE | re.DOTALL,
    )
    SECTION_PATTERN = re.compile(
        r"^(?:#+\s*)?(\d+\.?\s*)?([A-Z][^.\n]{2,50})$", re.MULTILINE
    )

    def chunk(self, extraction: ExtractionResult, item_key: str) -> ChunkResult:
        chunks = []
        chunk_index = 0
        text = extraction.text
        lines = extraction.lines

        abstract_match = self.ABSTRACT_PATTERN.search(text)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            if abstract_text and self.estimate_tokens(abstract_text) >= self.min_chunk_size:
                start_line, end_line = self._find_line_range(
                    lines, abstract_match.start(1), abstract_match.end(1), text
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{item_key}_{chunk_index}",
                        item_key=item_key,
                        chunk_index=chunk_index,
                        text=abstract_text,
                        start_line=start_line,
                        end_line=end_line,
                        section_title="Abstract",
                        chunk_type=ChunkType.ABSTRACT,
                    )
                )
                chunk_index += 1

        sections = self._extract_sections(text, lines)
        body_start = abstract_match.end() if abstract_match else 0
        body_text = text[body_start:].strip()

        if body_text:
            body_chunks = self._chunk_body(
                body_text, lines, item_key, chunk_index, body_start, text, sections
            )
            chunks.extend(body_chunks)

        total_tokens = sum(self.estimate_tokens(c.text) for c in chunks)
        return ChunkResult(chunks=chunks, total_tokens=total_tokens)

    def _find_line_range(
        self, lines: list[str], char_start: int, char_end: int, full_text: str
    ) -> tuple[int, int]:
        start_line = full_text[:char_start].count("\n")
        end_line = full_text[:char_end].count("\n")
        return start_line, end_line

    def _extract_sections(self, text: str, lines: list[str]) -> list[dict]:
        sections = []
        for match in self.SECTION_PATTERN.finditer(text):
            title = match.group(2).strip()
            start_pos = match.start()
            start_line = text[:start_pos].count("\n")
            sections.append(
                {"title": title, "start_pos": start_pos, "start_line": start_line}
            )
        return sections

    def _get_section_for_position(
        self, position: int, sections: list[dict]
    ) -> Optional[str]:
        current_section = None
        for section in sections:
            if section["start_pos"] <= position:
                current_section = section["title"]
            else:
                break
        return current_section

    def _chunk_body(
        self,
        body_text: str,
        lines: list[str],
        item_key: str,
        start_index: int,
        body_offset: int,
        full_text: str,
        sections: list[dict],
    ) -> list[DocumentChunk]:
        chunks = []
        chunk_index = start_index
        paragraphs = re.split(r"\n\n+", body_text)

        current_chunk: list[str] = []
        current_tokens = 0
        chunk_start_pos = body_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.estimate_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_end_pos = chunk_start_pos + len(chunk_text)

                start_line, end_line = self._find_line_range(
                    lines, chunk_start_pos, chunk_end_pos, full_text
                )
                section_title = self._get_section_for_position(
                    chunk_start_pos, sections
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{item_key}_{chunk_index}",
                        item_key=item_key,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_line=start_line,
                        end_line=end_line,
                        section_title=section_title,
                        chunk_type=ChunkType.BODY,
                    )
                )
                chunk_index += 1

                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_tokens = self.estimate_tokens(overlap_text)
                    chunk_start_pos = chunk_end_pos - len(overlap_text)
                else:
                    current_chunk = []
                    current_tokens = 0
                    chunk_start_pos = chunk_end_pos

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = "\n\n".join(current_chunk)
            chunk_end_pos = chunk_start_pos + len(chunk_text)

            start_line, end_line = self._find_line_range(
                lines, chunk_start_pos, min(chunk_end_pos, len(full_text)), full_text
            )
            section_title = self._get_section_for_position(chunk_start_pos, sections)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{item_key}_{chunk_index}",
                    item_key=item_key,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_line=start_line,
                    end_line=end_line,
                    section_title=section_title,
                    chunk_type=ChunkType.BODY,
                )
            )

        return chunks


class BookChunker(Chunker):
    """Chunker optimized for books and longer documents."""

    CHAPTER_PATTERN = re.compile(
        r"^(?:#+\s*)?(chapter\s+\d+|part\s+\d+|\d+\.\s+[A-Z])[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def chunk(self, extraction: ExtractionResult, item_key: str) -> ChunkResult:
        chunks = []
        text = extraction.text
        lines = extraction.lines
        chapters = self._find_chapters(text)

        if chapters:
            chunk_index = 0
            for i, chapter in enumerate(chapters):
                chapter_start = chapter["start"]
                chapter_end = (
                    chapters[i + 1]["start"] if i + 1 < len(chapters) else len(text)
                )
                chapter_text = text[chapter_start:chapter_end].strip()
                chapter_chunks = self._chunk_chapter(
                    chapter_text,
                    chapter["title"],
                    lines,
                    item_key,
                    chunk_index,
                    chapter_start,
                    text,
                )
                chunks.extend(chapter_chunks)
                chunk_index += len(chapter_chunks)
        else:
            article_chunker = ArticleChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_size=self.min_chunk_size,
            )
            return article_chunker.chunk(extraction, item_key)

        total_tokens = sum(self.estimate_tokens(c.text) for c in chunks)
        return ChunkResult(chunks=chunks, total_tokens=total_tokens)

    def _find_chapters(self, text: str) -> list[dict]:
        chapters = []
        for match in self.CHAPTER_PATTERN.finditer(text):
            chapters.append({"title": match.group(0).strip(), "start": match.start()})
        return chapters

    def _chunk_chapter(
        self,
        chapter_text: str,
        chapter_title: str,
        lines: list[str],
        item_key: str,
        start_index: int,
        chapter_offset: int,
        full_text: str,
    ) -> list[DocumentChunk]:
        chunks = []
        chunk_index = start_index
        paragraphs = re.split(r"\n\n+", chapter_text)

        current_chunk: list[str] = []
        current_tokens = 0
        chunk_start_pos = chapter_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.estimate_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_end_pos = chunk_start_pos + len(chunk_text)

                start_line = full_text[:chunk_start_pos].count("\n")
                end_line = full_text[:chunk_end_pos].count("\n")

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{item_key}_{chunk_index}",
                        item_key=item_key,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_line=start_line,
                        end_line=end_line,
                        section_title=chapter_title,
                        chunk_type=ChunkType.CHAPTER,
                    )
                )
                chunk_index += 1

                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_tokens = self.estimate_tokens(overlap_text)
                    chunk_start_pos = chunk_end_pos - len(overlap_text)
                else:
                    current_chunk = []
                    current_tokens = 0
                    chunk_start_pos = chunk_end_pos

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = "\n\n".join(current_chunk)
            chunk_end_pos = chunk_start_pos + len(chunk_text)

            start_line = full_text[:chunk_start_pos].count("\n")
            end_line = full_text[: min(chunk_end_pos, len(full_text))].count("\n")

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{item_key}_{chunk_index}",
                    item_key=item_key,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_line=start_line,
                    end_line=end_line,
                    section_title=chapter_title,
                    chunk_type=ChunkType.CHAPTER,
                )
            )

        return chunks


def get_chunker(item_type: Optional[str] = None, **kwargs) -> Chunker:
    """Get an appropriate chunker based on item type."""
    book_types = {"book", "booksection", "chapter"}
    if item_type and item_type.lower() in book_types:
        return BookChunker(**kwargs)
    return ArticleChunker(**kwargs)
