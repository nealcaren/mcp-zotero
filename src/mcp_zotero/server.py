"""MCP server for Zotero — 38 base tools + conditional RAG registration.

Default mode assumes Zotero Local API; falls back to cloud API when ZOTERO_LOCAL=false.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pyzotero import zotero
from pyzotero import errors as zotero_errors

from mcp_zotero.config import get_settings
from mcp_zotero.models import (
    AddItemRequest,
    AttachFileRequest,
    AttachLinkedFileRequest,
    CollectionCreateRequest,
    CollectionItemRequest,
    CollectionKeyRequest,
    CollectionQueryRequest,
    CollectionUpdateRequest,
    DeleteSavedSearchRequest,
    DeleteTagsRequest,
    DownloadAttachmentsRequest,
    ItemKeyRequest,
    ItemRequest,
    ItemTypeRequest,
    QueryRequest,
    SavedSearchRequest,
    SearchRequest,
    UpdateItemRequest,
)

mcp = FastMCP("zotero")


# ── Client helpers ──────────────────────────────────────────────────────


def _get_client() -> zotero.Zotero:
    """Get client for read operations. Uses local API if configured."""
    s = get_settings()
    if not s.zotero_library_id:
        raise ValueError("ZOTERO_LIBRARY_ID is required")

    api_key = s.read_api_key

    if s.zotero_local:
        return zotero.Zotero(s.zotero_library_id, s.zotero_library_type, api_key, local=True)

    if not api_key:
        raise ValueError("ZOTERO_API_KEY is required when ZOTERO_LOCAL=false")
    return zotero.Zotero(s.zotero_library_id, s.zotero_library_type, api_key)


def _get_write_client() -> zotero.Zotero:
    """Get client for write operations. Always uses Web API."""
    s = get_settings()
    if not s.zotero_library_id:
        raise ValueError("ZOTERO_LIBRARY_ID is required")

    if not s.zotero_api_key:
        raise ValueError(
            "ZOTERO_API_KEY is required for write operations. "
            "Get one from https://www.zotero.org/settings/keys"
        )
    return zotero.Zotero(s.zotero_library_id, s.zotero_library_type, s.zotero_api_key)


def _attachments_dir() -> Path:
    s = get_settings()
    if s.attachments_path:
        s.attachments_path.mkdir(parents=True, exist_ok=True)
        return s.attachments_path
    default = Path("./zotero-attachments").resolve()
    default.mkdir(parents=True, exist_ok=True)
    return default


def _simplify_item(item: Dict[str, Any]) -> Dict[str, Any]:
    data = item.get("data", {})
    return {
        "key": item.get("key"),
        "itemType": data.get("itemType"),
        "title": data.get("title"),
        "creators": data.get("creators"),
        "date": data.get("date"),
        "tags": data.get("tags"),
        "libraryCatalog": data.get("libraryCatalog"),
        "url": data.get("url"),
    }


def _query_kwargs(req: QueryRequest) -> Dict[str, Any]:
    return {
        k: v
        for k, v in {
            "q": req.q,
            "qmode": req.qmode,
            "limit": req.limit,
            "start": req.start,
            "itemType": req.item_type,
            "tag": req.tag,
            "sort": req.sort,
            "direction": req.direction,
            "since": req.since,
        }.items()
        if v is not None
    }


# ── Base tools ──────────────────────────────────────────────────────────


@mcp.tool()
def ping() -> Dict[str, Any]:
    """Basic liveness check."""
    return {"ok": True, "server": "mcp-zotero"}


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Validate configuration without making API calls."""
    s = get_settings()
    missing = s.validate_config()
    has_write_key = bool(s.zotero_api_key)

    result: Dict[str, Any] = {
        "ok": len(missing) == 0,
        "missing": missing,
        "read_mode": "local (fast)" if s.zotero_local else "web",
        "write_enabled": has_write_key,
        "write_note": "Set ZOTERO_API_KEY for writes" if not has_write_key else "ready",
    }

    # Report RAG availability
    from mcp_zotero.rag import RAG_AVAILABLE, _IMPORT_ERROR

    result["rag_available"] = RAG_AVAILABLE
    if not RAG_AVAILABLE:
        result["rag_note"] = (
            f"Install with: uv tool install mcp-zotero[rag] ({_IMPORT_ERROR})"
        )

    return result


@mcp.tool()
def search_items(req: SearchRequest) -> Dict[str, Any]:
    """Search items by free-text query."""
    client = _get_client()
    kwargs = {
        k: v
        for k, v in {
            "q": req.query,
            "qmode": req.qmode,
            "limit": req.limit,
            "start": req.start,
            "itemType": req.item_type,
            "tag": req.tag,
            "sort": req.sort,
            "direction": req.direction,
            "since": req.since,
        }.items()
        if v is not None
    }
    items = client.items(**kwargs)
    return {
        "query": req.query,
        "count": len(items),
        "items": [_simplify_item(i) for i in items],
    }


@mcp.tool()
def get_item(req: ItemRequest) -> Dict[str, Any]:
    """Retrieve a single item and (optionally) its children."""
    client = _get_client()
    item = client.item(req.item_key)
    result: Dict[str, Any] = {"item": item}
    if req.include_children:
        children = client.children(req.item_key)
        result["children"] = children
    return result


@mcp.tool()
def top_items(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve top-level items."""
    client = _get_client()
    items = client.top(**_query_kwargs(req))
    return {"count": len(items), "items": items}


@mcp.tool()
def trash_items(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve items in the trash."""
    client = _get_client()
    items = client.trash(**_query_kwargs(req))
    return {"count": len(items), "items": items}


@mcp.tool()
def deleted_items(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve deleted items (requires since)."""
    client = _get_client()
    items = client.deleted(**_query_kwargs(req))
    return {"count": len(items), "items": items}


@mcp.tool()
def list_searches(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve saved searches."""
    client = _get_client()
    items = client.searches(**_query_kwargs(req))
    return {"count": len(items), "searches": items}


@mcp.tool()
def list_groups(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve user groups."""
    client = _get_client()
    items = client.groups(**_query_kwargs(req))
    return {"count": len(items), "groups": items}


@mcp.tool()
def list_collections(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve collections."""
    client = _get_client()
    items = client.collections(**_query_kwargs(req))
    return {"count": len(items), "collections": items}


@mcp.tool()
def list_collections_top(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve top-level collections."""
    client = _get_client()
    items = client.collections_top(**_query_kwargs(req))
    return {"count": len(items), "collections": items}


@mcp.tool()
def list_collections_sub(req: CollectionKeyRequest) -> Dict[str, Any]:
    """Retrieve subcollections for a collection."""
    client = _get_client()
    items = client.collections_sub(req.collection_key)
    return {"count": len(items), "collections": items}


@mcp.tool()
def get_collection(req: CollectionKeyRequest) -> Dict[str, Any]:
    """Retrieve a collection."""
    client = _get_client()
    return {"collection": client.collection(req.collection_key)}


@mcp.tool()
def collection_items(req: CollectionQueryRequest) -> Dict[str, Any]:
    """Retrieve items in a collection."""
    client = _get_client()
    items = client.collection_items(req.collection_key, **_query_kwargs(req))
    return {"count": len(items), "items": items}


@mcp.tool()
def collection_items_top(req: CollectionQueryRequest) -> Dict[str, Any]:
    """Retrieve top-level items in a collection."""
    client = _get_client()
    items = client.collection_items_top(req.collection_key, **_query_kwargs(req))
    return {"count": len(items), "items": items}


@mcp.tool()
def collection_tags(req: CollectionKeyRequest) -> Dict[str, Any]:
    """Retrieve tags for a collection."""
    client = _get_client()
    tags = client.collection_tags(req.collection_key)
    return {"count": len(tags), "tags": tags}


@mcp.tool()
def list_tags(req: QueryRequest) -> Dict[str, Any]:
    """Retrieve tags."""
    client = _get_client()
    tags = client.tags(**_query_kwargs(req))
    return {"count": len(tags), "tags": tags}


@mcp.tool()
def item_tags(req: ItemKeyRequest) -> Dict[str, Any]:
    """Retrieve tags for a specific item."""
    client = _get_client()
    tags = client.item_tags(req.item_key)
    return {"count": len(tags), "tags": tags}


@mcp.tool()
def item_types() -> Dict[str, Any]:
    """Retrieve item types."""
    client = _get_client()
    return {"item_types": client.item_types()}


@mcp.tool()
def item_fields() -> Dict[str, Any]:
    """Retrieve item fields."""
    client = _get_client()
    return {"item_fields": client.item_fields()}


@mcp.tool()
def item_type_fields(req: ItemTypeRequest) -> Dict[str, Any]:
    """Retrieve fields for an item type."""
    client = _get_client()
    return {"item_type_fields": client.item_type_fields(req.item_type)}


@mcp.tool()
def creator_fields() -> Dict[str, Any]:
    """Retrieve creator fields."""
    client = _get_client()
    return {"creator_fields": client.creator_fields()}


@mcp.tool()
def item_creator_types(req: ItemTypeRequest) -> Dict[str, Any]:
    """Retrieve creator types for an item type."""
    client = _get_client()
    return {"item_creator_types": client.item_creator_types(req.item_type)}


@mcp.tool()
def item_attachment_link_modes() -> Dict[str, Any]:
    """Retrieve attachment link mode types."""
    client = _get_client()
    return {"item_attachment_link_modes": client.item_attachment_link_modes()}


@mcp.tool()
def download_attachments(req: DownloadAttachmentsRequest) -> Dict[str, Any]:
    """Download all file attachments for an item to a local directory."""
    client = _get_client()
    attachments = _attachments_dir() / req.item_key
    attachments.mkdir(parents=True, exist_ok=True)

    children = client.children(req.item_key)
    saved: List[str] = []

    for child in children:
        data = child.get("data", {})
        if data.get("itemType") != "attachment":
            continue
        key = child.get("key")
        filename = data.get("filename") or f"{key}"
        out_path = attachments / filename
        if out_path.exists() and not req.overwrite:
            saved.append(str(out_path))
            continue
        try:
            client.file(key, filename=str(out_path))
            saved.append(str(out_path))
        except Exception as e:
            saved.append(f"ERROR:{key}:{e}")

    return {"item_key": req.item_key, "saved": saved}


# ── Write tools ─────────────────────────────────────────────────────────


@mcp.tool()
def add_item(req: AddItemRequest) -> Dict[str, Any]:
    """Add an item to Zotero. Optionally attach files (best-effort)."""
    client = _get_write_client()
    template = client.item_template(req.item_type)
    template.update(req.fields)

    created = client.create_items([template])
    result: Dict[str, Any] = {"created": created}

    if req.attachment_paths:
        attached = []
        for path in req.attachment_paths:
            try:
                attach = _attach_file_internal(client, created, path)
                attached.append(attach)
            except Exception as e:
                attached.append(f"ERROR:{path}:{e}")
        result["attachments"] = attached

    return result


@mcp.tool()
def update_item(req: UpdateItemRequest) -> Dict[str, Any]:
    """Modify fields on an existing item."""
    client = _get_write_client()
    item = client.item(req.item_key)
    item_data = item.get("data", {})
    item_data.update(req.fields)
    updated = client.update_item(item)
    return {"updated": updated}


@mcp.tool()
def create_collections(req: CollectionCreateRequest) -> Dict[str, Any]:
    """Create collections."""
    client = _get_write_client()
    payload = [c.model_dump(exclude_none=True) for c in req.collections]
    return {"created": client.create_collections(payload)}


@mcp.tool()
def update_collection(req: CollectionUpdateRequest) -> Dict[str, Any]:
    """Update a collection's name or parent."""
    client = _get_write_client()
    existing = client.collection(req.collection_key)
    data = existing.get("data", {})
    if req.name is not None:
        data["name"] = req.name
    if req.parentCollection is not None:
        data["parentCollection"] = req.parentCollection
    existing["data"] = data
    updated = client.update_collections([existing])
    return {"updated": updated}


@mcp.tool()
def delete_collection(req: CollectionKeyRequest) -> Dict[str, Any]:
    """Delete a collection."""
    client = _get_write_client()
    existing = client.collection(req.collection_key)
    resp = client.delete_collection(existing)
    return {"status_code": resp.status_code}


@mcp.tool()
def add_item_to_collection(req: CollectionItemRequest) -> Dict[str, Any]:
    """Add an item to a collection."""
    client = _get_write_client()
    itm = client.item(req.item_key)
    resp = client.addto_collection(req.collection_key, itm)
    return {"status_code": resp.status_code}


@mcp.tool()
def remove_item_from_collection(req: CollectionItemRequest) -> Dict[str, Any]:
    """Remove an item from a collection."""
    client = _get_write_client()
    itm = client.item(req.item_key)
    resp = client.deletefrom_collection(req.collection_key, itm)
    return {"status_code": resp.status_code}


@mcp.tool()
def delete_item(req: ItemKeyRequest) -> Dict[str, Any]:
    """Delete an item."""
    client = _get_write_client()
    itm = client.item(req.item_key)
    resp = client.delete_item(itm)
    return {"status_code": resp.status_code}


@mcp.tool()
def create_saved_search(req: SavedSearchRequest) -> Dict[str, Any]:
    """Create a saved search."""
    client = _get_write_client()
    return {"created": client.saved_search(req.name, req.conditions)}


@mcp.tool()
def delete_saved_search(req: DeleteSavedSearchRequest) -> Dict[str, Any]:
    """Delete saved searches by key."""
    client = _get_write_client()
    status = client.delete_saved_search(req.keys)
    return {"status_code": status}


@mcp.tool()
def delete_tags(req: DeleteTagsRequest) -> Dict[str, Any]:
    """Delete tags."""
    client = _get_write_client()
    resp = client.delete_tags(*req.tags)
    return {"status_code": resp.status_code}


@mcp.tool()
def attach_file(req: AttachFileRequest) -> Dict[str, Any]:
    """Attach a local file to an existing item (best-effort)."""
    client = _get_write_client()
    attached = _attach_file_internal(
        client, {"success": True, "data": [req.item_key]}, req.file_path, title=req.title
    )
    return {"attached": attached}


@mcp.tool()
def attach_linked_file(req: AttachLinkedFileRequest) -> Dict[str, Any]:
    """Attach a linked (non-uploaded) local file to an existing item."""
    client = _get_write_client()
    path = Path(req.file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    template = client.item_template("attachment", "linked_file")
    content_type, _ = mimetypes.guess_type(str(path))
    template.update(
        {
            "itemType": "attachment",
            "linkMode": "linked_file",
            "parentItem": req.item_key,
            "path": str(path),
            "title": req.title or path.name,
            "contentType": content_type or "application/octet-stream",
        }
    )

    created = client.create_items([template])
    return {"created": created}


# ── Internal helpers ────────────────────────────────────────────────────


def _attach_file_internal(
    client: zotero.Zotero,
    create_resp: Dict[str, Any],
    file_path: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    item_key = None
    if isinstance(create_resp, dict):
        if "data" in create_resp and create_resp["data"]:
            first = create_resp["data"][0]
            if isinstance(first, dict):
                item_key = first.get("key")
            elif isinstance(first, str):
                item_key = first
        elif "item_key" in create_resp:
            item_key = create_resp["item_key"]
    if not item_key:
        if "successful" in create_resp and create_resp["successful"]:
            item_key = next(iter(create_resp["successful"]))

    if not item_key:
        raise ValueError("Could not determine parent item key for attachment")

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if hasattr(client, "attachment_simple"):
        try:
            return client.attachment_simple([str(path)], parentid=item_key)
        except zotero_errors.RequestEntityTooLargeError:
            template = client.item_template("attachment", "linked_file")
            content_type, _ = mimetypes.guess_type(str(path))
            template.update(
                {
                    "itemType": "attachment",
                    "linkMode": "linked_file",
                    "parentItem": item_key,
                    "path": str(path),
                    "title": title or path.name,
                    "contentType": content_type or "application/octet-stream",
                }
            )
            return {"linked_fallback": client.create_items([template])}

    raise NotImplementedError(
        "pyzotero attachment_simple not available; implement via Zotero API upload"
    )


# ── RAG tool registration ──────────────────────────────────────────────

try:
    from mcp_zotero.rag import register_tools

    register_tools(mcp)
except Exception:
    pass  # RAG deps not installed — base tools only
