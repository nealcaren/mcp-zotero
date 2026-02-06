"""RAG (semantic search) extension for mcp-zotero.

Only available when installed with: uv tool install mcp-zotero[rag]
"""

RAG_AVAILABLE = False
_IMPORT_ERROR = ""

try:
    import numpy  # noqa: F401
    import sentence_transformers  # noqa: F401
    import fitz  # noqa: F401

    RAG_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def register_tools(mcp):
    """Register RAG tools with the FastMCP instance. No-op if deps missing."""
    if not RAG_AVAILABLE:
        return

    from mcp_zotero.rag.tools import register_rag_tools

    register_rag_tools(mcp)
