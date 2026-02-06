"""Pydantic models for mcp-zotero (base tools + RAG)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ── Base Zotero request models ──────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text search query")
    limit: int = Field(20, ge=1, le=100)
    start: int = Field(0, ge=0)
    item_type: Optional[str] = Field(None, description="Filter by Zotero itemType")
    qmode: Optional[str] = Field(
        None, description="Quick search mode (titleCreatorYear or everything)"
    )
    tag: Optional[Union[str, List[str]]] = Field(
        None, description="Filter by tag (string, OR/NOT syntax, or list for AND)"
    )
    sort: Optional[str] = Field(None, description="Sort field (e.g., dateAdded, title)")
    direction: Optional[str] = Field(None, description="Sort direction (asc/desc)")
    since: Optional[int] = Field(None, description="Return items since this version")


class QueryRequest(BaseModel):
    q: Optional[str] = Field(None, description="Free-text query")
    qmode: Optional[str] = Field(
        None, description="Quick search mode (titleCreatorYear or everything)"
    )
    limit: Optional[int] = Field(None, ge=1, le=100)
    start: Optional[int] = Field(None, ge=0)
    item_type: Optional[str] = Field(None, description="Filter by Zotero itemType")
    tag: Optional[Union[str, List[str]]] = Field(
        None, description="Filter by tag (string, OR/NOT syntax, or list for AND)"
    )
    sort: Optional[str] = Field(None, description="Sort field (e.g., dateAdded, title)")
    direction: Optional[str] = Field(None, description="Sort direction (asc/desc)")
    since: Optional[int] = Field(None, description="Return items since this version")


class ItemRequest(BaseModel):
    item_key: str = Field(..., description="Zotero item key")
    include_children: bool = True


class ItemKeyRequest(BaseModel):
    item_key: str = Field(..., description="Zotero item key")


class ItemTypeRequest(BaseModel):
    item_type: str = Field(..., description="Zotero item type")


class CollectionKeyRequest(BaseModel):
    collection_key: str = Field(..., description="Zotero collection key")


class CollectionQueryRequest(QueryRequest):
    collection_key: str = Field(..., description="Zotero collection key")


class CollectionCreateItem(BaseModel):
    name: str
    parentCollection: Optional[str] = None


class CollectionCreateRequest(BaseModel):
    collections: List[CollectionCreateItem]


class CollectionUpdateRequest(BaseModel):
    collection_key: str
    name: Optional[str] = None
    parentCollection: Optional[str] = None


class CollectionItemRequest(BaseModel):
    collection_key: str
    item_key: str


class SavedSearchRequest(BaseModel):
    name: str
    conditions: List[Dict[str, Any]]


class DeleteSavedSearchRequest(BaseModel):
    keys: List[str]


class DeleteTagsRequest(BaseModel):
    tags: List[str]


class DownloadAttachmentsRequest(BaseModel):
    item_key: str
    overwrite: bool = False


class AddItemRequest(BaseModel):
    item_type: str = Field(
        ..., description="Zotero item type (e.g., book, journalArticle)"
    )
    fields: Dict[str, Any] = Field(default_factory=dict)
    attachment_paths: List[str] = Field(default_factory=list)


class UpdateItemRequest(BaseModel):
    item_key: str
    fields: Dict[str, Any]


class AttachFileRequest(BaseModel):
    item_key: str
    file_path: str
    title: Optional[str] = None


class AttachLinkedFileRequest(BaseModel):
    item_key: str
    file_path: str
    title: Optional[str] = None


# ── RAG models ──────────────────────────────────────────────────────────


class ChunkType(str, Enum):
    ABSTRACT = "abstract"
    BODY = "body"
    CHAPTER = "chapter"
    SECTION = "section"


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Unique identifier: {item_key}_{chunk_index}")
    item_key: str = Field(description="Zotero item key")
    chunk_index: int = Field(description="Position within the document")
    text: str = Field(description="Chunk content")
    start_line: int = Field(description="Starting line number in source")
    end_line: int = Field(description="Ending line number in source")
    section_title: Optional[str] = Field(
        default=None, description="Section title if detected"
    )
    chunk_type: ChunkType = Field(default=ChunkType.BODY, description="Type of chunk")


class IndexedDocument(BaseModel):
    item_key: str = Field(description="Zotero item key")
    title: str = Field(description="Document title")
    content_hash: str = Field(description="SHA-256 hash of source content")
    chunk_count: int = Field(description="Number of chunks")
    indexed_at: datetime = Field(description="When the document was indexed")
    file_path: Optional[str] = Field(default=None, description="Path to source file")
    file_type: Optional[str] = Field(default=None, description="File type (pdf, epub)")


class SearchResult(BaseModel):
    chunk: DocumentChunk = Field(description="The matching chunk")
    score: float = Field(description="Similarity score (0-1)")
    item_title: Optional[str] = Field(
        default=None, description="Title of source document"
    )
    item_authors: Optional[str] = Field(
        default=None, description="Authors of source document"
    )


class SearchResponse(BaseModel):
    query: str = Field(description="The search query")
    results: list[SearchResult] = Field(description="Ranked search results")
    total_results: int = Field(description="Total number of matches")


class ChunkContext(BaseModel):
    chunk: DocumentChunk = Field(description="The original chunk")
    context_before: str = Field(description="Text before the chunk")
    context_after: str = Field(description="Text after the chunk")
    full_text: str = Field(description="Combined context + chunk text")


class IndexStats(BaseModel):
    total_documents: int = Field(description="Number of indexed documents")
    total_chunks: int = Field(description="Number of chunks in index")
    embedding_model: str = Field(description="Model used for embeddings")
    index_size_mb: float = Field(description="Approximate index size in MB")
    last_updated: Optional[datetime] = Field(
        default=None, description="Last index update time"
    )


class HealthStatus(BaseModel):
    status: str = Field(description="Overall status: healthy, degraded, or unhealthy")
    zotero_connected: bool = Field(description="Whether Zotero is accessible")
    index_accessible: bool = Field(description="Whether the index is accessible")
    embedding_model_loaded: bool = Field(
        description="Whether the embedding model is loaded"
    )
    issues: list[str] = Field(default_factory=list, description="List of detected issues")


# ── RAG request models ──────────────────────────────────────────────────


class IndexLibraryRequest(BaseModel):
    collection_key: Optional[str] = Field(
        default=None, description="Optional collection key to limit indexing"
    )
    force_reindex: bool = Field(
        default=False, description="Force re-indexing even if content hash matches"
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of items to index"
    )


class IndexItemsRequest(BaseModel):
    item_keys: list[str] = Field(description="List of Zotero item keys to index")
    force_reindex: bool = Field(
        default=False, description="Force re-indexing even if content hash matches"
    )


class RemoveFromIndexRequest(BaseModel):
    item_keys: list[str] = Field(description="List of Zotero item keys to remove")


class SemanticSearchRequest(BaseModel):
    query: str = Field(description="Natural language search query")
    limit: int = Field(default=10, description="Maximum number of results")
    collection_key: Optional[str] = Field(
        default=None, description="Limit search to a specific collection"
    )
    item_keys: Optional[list[str]] = Field(
        default=None, description="Limit search to specific items"
    )
    chunk_types: Optional[list[ChunkType]] = Field(
        default=None, description="Filter by chunk types"
    )
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold")


class GetChunkContextRequest(BaseModel):
    chunk_id: str = Field(description="The chunk ID to expand")
    context_lines: int = Field(
        default=10, description="Number of lines of context before and after"
    )


class FindSimilarChunksRequest(BaseModel):
    chunk_id: str = Field(description="The chunk ID to find similar chunks for")
    limit: int = Field(default=5, description="Maximum number of similar chunks")
    exclude_same_document: bool = Field(
        default=True, description="Exclude chunks from the same document"
    )


class ListIndexedItemsRequest(BaseModel):
    collection_key: Optional[str] = Field(
        default=None, description="Filter by collection"
    )
    limit: int = Field(default=100, description="Maximum number of items to return")
    offset: int = Field(default=0, description="Offset for pagination")
