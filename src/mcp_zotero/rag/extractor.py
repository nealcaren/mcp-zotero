"""PDF text extraction — pymupdf primary, docling OCR fallback.

Primary: pymupdf (fitz) for standard PDFs — fast, lightweight.
Fallback: docling for scanned/OCR PDFs — requires mcp-zotero[ocr] extra.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Characters-per-page threshold — below this, PDF is likely scanned
_SCANNED_THRESHOLD = 100


@dataclass
class ExtractionResult:
    """Result of text extraction from a document."""

    text: str
    lines: list[str]
    content_hash: str
    page_count: Optional[int] = None
    chapter_count: Optional[int] = None
    has_ocr: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def get_lines(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start:end])


class ExtractionError(Exception):
    def __init__(self, message: str, file_path: Optional[Path] = None):
        self.file_path = file_path
        super().__init__(message)


def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class PdfExtractor:
    """PDF text extractor: pymupdf primary, docling OCR fallback."""

    def __init__(
        self,
        enable_ocr: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        self.enable_ocr = enable_ocr
        self.cache_dir = cache_dir

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def extract(self, file_path: Path) -> ExtractionResult:
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}", file_path)
        if not self.supports(file_path):
            raise ExtractionError(f"Unsupported file type: {file_path.suffix}", file_path)

        content_hash = compute_hash(file_path)

        # Check cache
        cached = self._load_cache(content_hash)
        if cached:
            logger.info(f"Loaded from cache: {file_path.name}")
            return cached

        # Primary extraction with pymupdf
        logger.info(f"Extracting text from: {file_path.name}")
        result = self._extract_pymupdf(file_path, content_hash)

        # Detect scanned PDFs and fall back to docling OCR
        if result.page_count and result.page_count > 0:
            chars_per_page = len(result.text) / result.page_count
            if chars_per_page < _SCANNED_THRESHOLD and self.enable_ocr:
                logger.info(
                    f"Low text density ({chars_per_page:.0f} chars/page), "
                    f"attempting OCR fallback for: {file_path.name}"
                )
                ocr_result = self._extract_docling_ocr(file_path, content_hash)
                if ocr_result:
                    result = ocr_result

        self._save_cache(content_hash, result)
        return result

    def _extract_pymupdf(self, file_path: Path, content_hash: str) -> ExtractionResult:
        try:
            import fitz  # pymupdf

            doc = fitz.open(str(file_path))
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()

            text = "\n\n".join(pages)
            lines = text.split("\n")

            return ExtractionResult(
                text=text,
                lines=lines,
                content_hash=content_hash,
                page_count=len(pages),
                has_ocr=False,
            )
        except Exception as e:
            raise ExtractionError(f"pymupdf extraction failed: {e}", file_path)

    def _extract_docling_ocr(
        self, file_path: Path, content_hash: str
    ) -> Optional[ExtractionResult]:
        """Attempt OCR extraction with docling. Returns None if not available."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            result = converter.convert(str(file_path))
            text = result.document.export_to_markdown()
            lines = text.split("\n")

            page_count = None
            if hasattr(result.document, "pages"):
                page_count = len(result.document.pages)

            return ExtractionResult(
                text=text,
                lines=lines,
                content_hash=content_hash,
                page_count=page_count,
                has_ocr=True,
            )
        except ImportError:
            logger.warning(
                "docling not installed — skipping OCR. "
                "Install with: uv tool install mcp-zotero[rag,ocr]"
            )
            return None
        except Exception as e:
            logger.warning(f"docling OCR failed: {e}")
            return None

    # ── Cache helpers ───────────────────────────────────────────────────

    def _cache_path(self, content_hash: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{content_hash}.json"

    def _load_cache(self, content_hash: str) -> Optional[ExtractionResult]:
        path = self._cache_path(content_hash)
        if path is None or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExtractionResult(
                text=data["text"],
                lines=data["lines"],
                content_hash=data["content_hash"],
                page_count=data.get("page_count"),
                chapter_count=data.get("chapter_count"),
                has_ocr=data.get("has_ocr", False),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def _save_cache(self, content_hash: str, result: ExtractionResult) -> None:
        path = self._cache_path(content_hash)
        if path is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "text": result.text,
                        "lines": result.lines,
                        "content_hash": result.content_hash,
                        "page_count": result.page_count,
                        "chapter_count": result.chapter_count,
                        "has_ocr": result.has_ocr,
                        "metadata": result.metadata,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
