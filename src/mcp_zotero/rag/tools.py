"""RAG tools for semantic search — registered as FastMCP tools.

These tools are only available when mcp-zotero[rag] is installed.
Registration happens via register_rag_tools(mcp) called from rag/__init__.py.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Lazy-initialized components
_components = None


class _ZoteroRAGClient:
    """Lightweight Zotero client for RAG operations (finding attachments)."""

    def __init__(self, settings):
        from pyzotero import zotero as pyzotero

        self.settings = settings
        if settings.zotero_local:
            self._zot = pyzotero.Zotero(
                settings.zotero_library_id,
                settings.zotero_library_type,
                settings.read_api_key,
                local=True,
            )
        else:
            self._zot = pyzotero.Zotero(
                settings.zotero_library_id,
                settings.zotero_library_type,
                settings.read_api_key,
            )

    def get_items(
        self,
        collection_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"itemType": "-attachment"}
        if limit:
            kwargs["limit"] = limit
        if collection_key:
            return self._zot.collection_items(collection_key, **kwargs)
        return self._zot.items(**kwargs)

    def get_item(self, item_key: str) -> Optional[dict[str, Any]]:
        try:
            return self._zot.item(item_key)
        except Exception:
            return None

    def get_item_attachments(self, item_key: str) -> list[dict[str, Any]]:
        try:
            return self._zot.children(item_key, itemType="attachment")
        except Exception:
            return []

    def get_attachment_path(self, attachment: dict[str, Any]) -> Optional[Path]:
        data = attachment.get("data", {})
        link_mode = data.get("linkMode")

        if link_mode == "linked_file":
            path_str = data.get("path", "")
            if path_str:
                path = Path(path_str)
                if path.exists():
                    return path
                if self.settings.attachments_path:
                    rel_path = self.settings.attachments_path / path_str
                    if rel_path.exists():
                        return rel_path
            return None

        if link_mode in ("imported_file", "imported_url"):
            filename = data.get("filename")
            key = data.get("key") or attachment.get("key")
            if filename and key and self.settings.attachments_path:
                path = self.settings.attachments_path / key / filename
                if path.exists():
                    return path
                direct_path = self.settings.attachments_path / filename
                if direct_path.exists():
                    return direct_path

        return None

    def get_indexable_attachment(
        self, item_key: str
    ) -> Optional[tuple[dict[str, Any], Path, str]]:
        attachments = self.get_item_attachments(item_key)
        for att in attachments:
            data = att.get("data", {})
            content_type = data.get("contentType", "")
            filename = data.get("filename", "")

            if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                path = self.get_attachment_path(att)
                if path:
                    return (att, path, "pdf")

        return None

    def get_item_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        data = item.get("data", {})
        creators = data.get("creators", [])
        authors = []
        for creator in creators:
            if creator.get("creatorType") == "author":
                name = creator.get("name")
                if not name:
                    first = creator.get("firstName", "")
                    last = creator.get("lastName", "")
                    name = f"{first} {last}".strip()
                if name:
                    authors.append(name)

        return {
            "key": data.get("key") or item.get("key"),
            "title": data.get("title", "Untitled"),
            "authors": authors,
            "authors_str": "; ".join(authors) if authors else None,
            "item_type": data.get("itemType"),
        }

    def iter_items_with_attachments(
        self,
        collection_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[tuple[dict[str, Any], Path, str]]:
        items = self.get_items(collection_key=collection_key, limit=limit)
        count = 0
        for item in items:
            if limit and count >= limit:
                break
            item_key = item.get("data", {}).get("key") or item.get("key")
            if not item_key:
                continue
            result = self.get_indexable_attachment(item_key)
            if result:
                yield (item, result[1], result[2])
                count += 1


def _get_components():
    """Lazy-init all RAG components."""
    global _components
    if _components is not None:
        return _components

    from mcp_zotero.config import get_settings
    from mcp_zotero.rag.extractor import PdfExtractor
    from mcp_zotero.rag.encoder import EmbeddingEncoder
    from mcp_zotero.rag.vector_store import NumpyVectorStore
    from mcp_zotero.rag.document_index import DocumentIndex

    settings = get_settings()
    settings.ensure_directories()

    extractor = PdfExtractor(
        enable_ocr=settings.rag_enable_ocr,
        cache_dir=settings.cache_dir,
    )
    encoder = EmbeddingEncoder(model_name=settings.rag_embedding_model)
    vector_store = NumpyVectorStore(settings.vectors_dir)
    doc_index = DocumentIndex(settings.sqlite_path)
    zotero_client = _ZoteroRAGClient(settings)

    _components = (settings, extractor, encoder, vector_store, doc_index, zotero_client)
    return _components


def register_rag_tools(mcp) -> None:
    """Register all RAG tools with the FastMCP instance."""

    from mcp_zotero.models import (
        ChunkType,
        FindSimilarChunksRequest,
        GetChunkContextRequest,
        IndexItemsRequest,
        IndexLibraryRequest,
        ListIndexedItemsRequest,
        RemoveFromIndexRequest,
        SemanticSearchRequest,
    )
    from mcp_zotero.rag.chunking import get_chunker

    @mcp.tool()
    def index_library(req: IndexLibraryRequest) -> Dict[str, Any]:
        """Build or update the semantic search index from Zotero library or collection."""
        settings, extractor, encoder, vector_store, doc_index, zot = _get_components()

        indexed = 0
        skipped = 0
        errors: list[str] = []

        try:
            for item, file_path, file_type in zot.iter_items_with_attachments(
                collection_key=req.collection_key, limit=req.limit
            ):
                metadata = zot.get_item_metadata(item)
                item_key = metadata["key"]

                try:
                    from mcp_zotero.rag.extractor import compute_hash

                    content_hash = compute_hash(file_path)

                    if not req.force_reindex and not doc_index.needs_reindex(
                        item_key, content_hash
                    ):
                        skipped += 1
                        continue

                    extraction = extractor.extract(file_path)

                    chunker = get_chunker(
                        metadata.get("item_type"),
                        chunk_size=settings.rag_chunk_size,
                        chunk_overlap=settings.rag_chunk_overlap,
                    )
                    chunk_result = chunker.chunk(extraction, item_key)

                    if not chunk_result.chunks:
                        continue

                    vector_store.delete_by_item_key(item_key)

                    texts = [c.text for c in chunk_result.chunks]
                    embeddings = encoder.encode(texts, show_progress=False)
                    vector_store.add_chunks(chunk_result.chunks, embeddings)

                    doc_index.add_document(
                        item_key=item_key,
                        title=metadata["title"],
                        content_hash=content_hash,
                        chunk_count=len(chunk_result.chunks),
                        file_path=str(file_path),
                        file_type=file_type,
                    )
                    doc_index.store_extracted_text(
                        item_key, content_hash, extraction.lines
                    )

                    indexed += 1

                except Exception as e:
                    errors.append(f"{item_key}: {e}")

            result = {
                "indexed": indexed,
                "skipped": skipped,
                "errors": len(errors),
            }
            if errors:
                result["error_details"] = errors[:10]
            return result

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def index_items(req: IndexItemsRequest) -> Dict[str, Any]:
        """Index specific Zotero items by their keys."""
        settings, extractor, encoder, vector_store, doc_index, zot = _get_components()

        indexed = 0
        skipped = 0
        errors: list[str] = []

        for item_key in req.item_keys:
            try:
                item = zot.get_item(item_key)
                if not item:
                    errors.append(f"{item_key}: Item not found")
                    continue

                attachment_info = zot.get_indexable_attachment(item_key)
                if not attachment_info:
                    errors.append(f"{item_key}: No PDF attachment found")
                    continue

                _, file_path, file_type = attachment_info
                metadata = zot.get_item_metadata(item)

                from mcp_zotero.rag.extractor import compute_hash

                content_hash = compute_hash(file_path)

                if not req.force_reindex and not doc_index.needs_reindex(
                    item_key, content_hash
                ):
                    skipped += 1
                    continue

                extraction = extractor.extract(file_path)
                chunker = get_chunker(
                    metadata.get("item_type"),
                    chunk_size=settings.rag_chunk_size,
                    chunk_overlap=settings.rag_chunk_overlap,
                )
                chunk_result = chunker.chunk(extraction, item_key)

                if not chunk_result.chunks:
                    errors.append(f"{item_key}: No chunks generated")
                    continue

                vector_store.delete_by_item_key(item_key)
                texts = [c.text for c in chunk_result.chunks]
                embeddings = encoder.encode(texts)
                vector_store.add_chunks(chunk_result.chunks, embeddings)

                doc_index.add_document(
                    item_key=item_key,
                    title=metadata["title"],
                    content_hash=content_hash,
                    chunk_count=len(chunk_result.chunks),
                    file_path=str(file_path),
                    file_type=file_type,
                )
                doc_index.store_extracted_text(item_key, content_hash, extraction.lines)
                indexed += 1

            except Exception as e:
                errors.append(f"{item_key}: {e}")

        result = {"indexed": indexed, "skipped": skipped, "errors": len(errors)}
        if errors:
            result["error_details"] = errors
        return result

    @mcp.tool()
    def remove_from_index(req: RemoveFromIndexRequest) -> Dict[str, Any]:
        """Remove items from the semantic search index."""
        _, _, _, vector_store, doc_index, _ = _get_components()

        removed = 0
        not_found = 0
        for item_key in req.item_keys:
            vector_store.delete_by_item_key(item_key)
            if doc_index.remove_document(item_key):
                removed += 1
            else:
                not_found += 1

        return {"removed": removed, "not_found": not_found}

    @mcp.tool()
    def semantic_search(req: SemanticSearchRequest) -> Dict[str, Any]:
        """Search for passages by meaning using semantic similarity."""
        _, _, encoder, vector_store, doc_index, _ = _get_components()

        chunk_type_enums = None
        if req.chunk_types:
            chunk_type_enums = [ChunkType(ct) for ct in req.chunk_types]

        query_embedding = encoder.encode_query(req.query)

        results = vector_store.search(
            query_embedding=query_embedding,
            limit=req.limit,
            item_keys=req.item_keys,
            chunk_types=chunk_type_enums,
            min_score=req.min_score,
        )

        if not results:
            return {"query": req.query, "results": [], "total_results": 0}

        formatted = []
        for chunk, score in results:
            doc = doc_index.get_document(chunk.item_key)
            formatted.append(
                {
                    "score": round(score, 3),
                    "source": doc.title if doc else "Unknown",
                    "item_key": chunk.item_key,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section_title,
                    "chunk_type": chunk.chunk_type.value,
                    "text": chunk.text,
                }
            )

        return {
            "query": req.query,
            "results": formatted,
            "total_results": len(formatted),
        }

    @mcp.tool()
    def get_chunk_context(req: GetChunkContextRequest) -> Dict[str, Any]:
        """Get expanded context around a search result chunk."""
        _, _, _, vector_store, doc_index, _ = _get_components()

        chunk = vector_store.get_chunk(req.chunk_id)
        if not chunk:
            return {"error": f"Chunk not found: {req.chunk_id}"}

        lines = doc_index.get_extracted_lines(chunk.item_key)
        if not lines:
            return {"error": f"Source text not available for item: {chunk.item_key}"}

        start = max(0, chunk.start_line - req.context_lines)
        end = min(len(lines), chunk.end_line + req.context_lines)

        doc = doc_index.get_document(chunk.item_key)

        return {
            "chunk_id": req.chunk_id,
            "source": doc.title if doc else "Unknown",
            "context_before": "\n".join(lines[start : chunk.start_line]),
            "chunk_text": chunk.text,
            "context_after": "\n".join(lines[chunk.end_line : end]),
            "line_range": f"{start + 1}-{end}",
        }

    @mcp.tool()
    def find_similar_chunks(req: FindSimilarChunksRequest) -> Dict[str, Any]:
        """Find passages similar to a given chunk across indexed documents."""
        _, _, _, vector_store, doc_index, _ = _get_components()

        source_chunk = vector_store.get_chunk(req.chunk_id)
        if not source_chunk:
            return {"error": f"Chunk not found: {req.chunk_id}"}

        results = vector_store.find_similar(
            chunk_id=req.chunk_id,
            limit=req.limit,
            exclude_same_document=req.exclude_same_document,
        )

        source_doc = doc_index.get_document(source_chunk.item_key)

        formatted = []
        for chunk, score in results:
            doc = doc_index.get_document(chunk.item_key)
            formatted.append(
                {
                    "score": round(score, 3),
                    "source": doc.title if doc else "Unknown",
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
            )

        return {
            "source": source_doc.title if source_doc else "Unknown",
            "source_text": source_chunk.text[:200],
            "similar": formatted,
        }

    @mcp.tool()
    def index_status() -> Dict[str, Any]:
        """Get statistics about the semantic search index."""
        settings, _, encoder, vector_store, doc_index, _ = _get_components()

        doc_stats = doc_index.get_stats()
        vector_stats = vector_store.get_stats()
        model_info = encoder.get_model_info()

        return {
            "documents_indexed": doc_stats["total_documents"],
            "total_chunks": doc_stats["total_chunks"],
            "vector_store_size_mb": round(vector_stats["size_mb"], 2),
            "embedding_model": model_info["model_name"],
            "model_loaded": model_info["is_loaded"],
            "last_updated": doc_stats["last_updated"],
        }

    @mcp.tool()
    def list_indexed_items(req: ListIndexedItemsRequest) -> Dict[str, Any]:
        """List all items that have been indexed for semantic search."""
        _, _, _, _, doc_index, _ = _get_components()

        documents = doc_index.list_documents(limit=req.limit, offset=req.offset)

        items = []
        for doc in documents:
            items.append(
                {
                    "item_key": doc.item_key,
                    "title": doc.title,
                    "chunk_count": doc.chunk_count,
                    "file_type": doc.file_type,
                    "indexed_at": doc.indexed_at.strftime("%Y-%m-%d %H:%M"),
                }
            )

        return {"total": len(items), "items": items}

    logger.info("RAG tools registered (8 tools)")
