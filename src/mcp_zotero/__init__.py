"""MCP server for Zotero — library tools + optional semantic search."""

__version__ = "0.2.0"


def main() -> None:
    from mcp_zotero.server import mcp

    mcp.run()
