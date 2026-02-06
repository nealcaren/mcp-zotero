import os

import pytest

from mcp_zotero import server


def test_smoke_local_api():
    if os.getenv("ZOTERO_SMOKE_TEST") != "1":
        pytest.skip("Set ZOTERO_SMOKE_TEST=1 to run smoke test")

    local = os.getenv("ZOTERO_LOCAL", "true").lower() in {"1", "true", "yes"}
    if not local:
        pytest.skip("Smoke test is intended for Zotero Local API")

    required = ["ZOTERO_LIBRARY_ID"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env vars for smoke test: {', '.join(missing)}")

    # Accept either local key or API key, since server.py uses ZOTERO_API_KEY as fallback
    if not (os.getenv("ZOTERO_LOCAL_KEY") or os.getenv("ZOTERO_API_KEY")):
        pytest.skip("Missing ZOTERO_LOCAL_KEY or ZOTERO_API_KEY for smoke test")

    client = server._get_client()
    items = client.items(limit=1)
    assert isinstance(items, list)
