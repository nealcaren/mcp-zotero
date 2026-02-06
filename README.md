# mcp-zotero

MCP server for Zotero — 38 library tools + optional semantic search via local embeddings.

## Install

### Base (Zotero API tools only)

```bash
uv tool install mcp-zotero
```

### With semantic search

```bash
uv tool install "mcp-zotero[rag]"
```

Adds 8 RAG tools for indexing PDFs and searching by meaning. Downloads a ~80MB embedding model on first use.

### With OCR for scanned PDFs

```bash
uv tool install "mcp-zotero[rag,ocr]"
```

Adds docling-based OCR fallback for PDFs that contain scanned images instead of text.

## Configuration

Add to your Claude Code MCP settings (`.mcp.json` or `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "zotero": {
      "command": "mcp-zotero",
      "env": {
        "ZOTERO_LIBRARY_ID": "YOUR_LIBRARY_ID",
        "ZOTERO_LOCAL": "true",
        "ZOTERO_LOCAL_KEY": "YOUR_LOCAL_KEY",
        "ZOTERO_API_KEY": "YOUR_API_KEY",
        "ZOTERO_ATTACHMENTS_DIR": "~/Zotero/storage"
      }
    }
  }
}
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ZOTERO_LIBRARY_ID` | Yes | — | Your Zotero library ID |
| `ZOTERO_LIBRARY_TYPE` | No | `user` | `user` or `group` |
| `ZOTERO_LOCAL` | No | `true` | Use Zotero Local API (requires Zotero app running) |
| `ZOTERO_LOCAL_KEY` | For reads | — | Local API key |
| `ZOTERO_API_KEY` | For writes | — | Web API key from [zotero.org/settings/keys](https://www.zotero.org/settings/keys) |
| `ZOTERO_ATTACHMENTS_DIR` | For RAG | — | Path to Zotero storage directory |

#### RAG-only variables (when installed with `[rag]`)

| Variable | Default | Description |
|---|---|---|
| `RAG_INDEX_DIR` | `~/.zotero-rag` | Where to store index data |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RAG_CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RAG_ENABLE_OCR` | `false` | Enable docling OCR fallback for scanned PDFs |

## Tools

### Base tools (38)

Always available. Covers the full Zotero API:

**Search & browse**: `search_items`, `top_items`, `get_item`, `trash_items`, `deleted_items`

**Collections**: `list_collections`, `list_collections_top`, `list_collections_sub`, `get_collection`, `collection_items`, `collection_items_top`, `collection_tags`, `create_collections`, `update_collection`, `delete_collection`, `add_item_to_collection`, `remove_item_from_collection`

**Tags**: `list_tags`, `item_tags`, `delete_tags`

**Items**: `add_item`, `update_item`, `delete_item`

**Attachments**: `download_attachments`, `attach_file`, `attach_linked_file`

**Schema**: `item_types`, `item_fields`, `item_type_fields`, `creator_fields`, `item_creator_types`, `item_attachment_link_modes`

**Saved searches**: `list_searches`, `list_groups`, `create_saved_search`, `delete_saved_search`

**System**: `ping`, `health_check`

### RAG tools (8)

Available when installed with `[rag]`. Semantic search across your PDF library:

| Tool | Description |
|---|---|
| `index_library` | Index a collection or entire library |
| `index_items` | Index specific items by key |
| `remove_from_index` | Remove items from the index |
| `semantic_search` | Search passages by meaning |
| `get_chunk_context` | Get surrounding text for a search result |
| `find_similar_chunks` | Find similar passages across documents |
| `index_status` | Get index statistics |
| `list_indexed_items` | List all indexed items |

## Development

```bash
git clone https://github.com/nealcaren/mcp-zotero.git
cd mcp-zotero
uv venv && uv pip install -e ".[dev,rag]"
pytest
```

## License

MIT
