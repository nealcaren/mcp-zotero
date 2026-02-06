"""Tests for mcp_zotero.server — base Zotero tools."""

import os

import pytest

from mcp_zotero import server
from mcp_zotero.models import (
    AttachLinkedFileRequest,
    CollectionCreateItem,
    CollectionCreateRequest,
    CollectionKeyRequest,
    CollectionUpdateRequest,
    DeleteSavedSearchRequest,
    DeleteTagsRequest,
    ItemTypeRequest,
    QueryRequest,
    SavedSearchRequest,
    SearchRequest,
)


class DummyZotero:
    def __init__(self, library_id, library_type, api_key, local=False, **kwargs):
        self.library_id = library_id
        self.library_type = library_type
        self.api_key = api_key
        self.local = local
        self.calls = []

    def items(self, **kwargs):
        self.calls.append(("items", kwargs))
        return [{"key": "AAA", "data": {"itemType": "book", "title": "T", "creators": []}}]

    def top(self, **kwargs):
        self.calls.append(("top", kwargs))
        return [{"key": "TOP"}]

    def trash(self, **kwargs):
        self.calls.append(("trash", kwargs))
        return [{"key": "TRASH"}]

    def deleted(self, **kwargs):
        self.calls.append(("deleted", kwargs))
        return [{"key": "DEL"}]

    def collections(self, **kwargs):
        self.calls.append(("collections", kwargs))
        return [{"key": "C1", "data": {"name": "Test"}}]

    def collections_top(self, **kwargs):
        self.calls.append(("collections_top", kwargs))
        return [{"key": "CT"}]

    def collections_sub(self, collection, **kwargs):
        self.calls.append(("collections_sub", {"collection": collection, **kwargs}))
        return [{"key": "CS"}]

    def collection(self, collection, **kwargs):
        self.calls.append(("collection", {"collection": collection, **kwargs}))
        return {"key": collection, "data": {"name": "Test", "parentCollection": ""}}

    def collection_items(self, collection, **kwargs):
        self.calls.append(("collection_items", {"collection": collection, **kwargs}))
        return [{"key": "CI"}]

    def collection_items_top(self, collection, **kwargs):
        self.calls.append(("collection_items_top", {"collection": collection, **kwargs}))
        return [{"key": "CIT"}]

    def collection_tags(self, collection, **kwargs):
        self.calls.append(("collection_tags", {"collection": collection, **kwargs}))
        return ["tag1", "tag2"]

    def tags(self, **kwargs):
        self.calls.append(("tags", kwargs))
        return ["t1"]

    def item_tags(self, item, **kwargs):
        self.calls.append(("item_tags", {"item": item, **kwargs}))
        return ["it1"]

    def groups(self, **kwargs):
        self.calls.append(("groups", kwargs))
        return [{"id": 1}]

    def searches(self, **kwargs):
        self.calls.append(("searches", kwargs))
        return [{"key": "S1"}]

    def item_types(self):
        self.calls.append(("item_types", {}))
        return ["book"]

    def item_fields(self):
        self.calls.append(("item_fields", {}))
        return [{"field": "title"}]

    def item_type_fields(self, item_type):
        self.calls.append(("item_type_fields", {"item_type": item_type}))
        return [{"field": "title"}]

    def creator_fields(self):
        self.calls.append(("creator_fields", {}))
        return [{"field": "author"}]

    def item_creator_types(self, item_type):
        self.calls.append(("item_creator_types", {"item_type": item_type}))
        return [{"creatorType": "author"}]

    def item_attachment_link_modes(self):
        self.calls.append(("item_attachment_link_modes", {}))
        return ["linked_file"]

    def item(self, key):
        return {"key": key, "data": {"itemType": "book", "title": "T"}}

    def children(self, key):
        return []

    def create_items(self, items):
        self.calls.append(("create_items", items))
        return {"successful": {"NEWKEY": items}}

    def item_template(self, item_type, linkmode=None):
        return {"itemType": item_type, "linkMode": linkmode}

    def create_collections(self, payload):
        self.calls.append(("create_collections", payload))
        return {"successful": {"CNEW": payload}}

    def update_collections(self, payload):
        self.calls.append(("update_collections", payload))
        return True

    def delete_collection(self, payload, last_modified=None):
        self.calls.append(("delete_collection", payload))

        class Resp:
            status_code = 204

        return Resp()

    def addto_collection(self, collection, payload):
        self.calls.append(("addto_collection", {"collection": collection, "item": payload}))

        class Resp:
            status_code = 204

        return Resp()

    def deletefrom_collection(self, collection, payload):
        self.calls.append(("deletefrom_collection", {"collection": collection, "item": payload}))

        class Resp:
            status_code = 204

        return Resp()

    def delete_item(self, payload, last_modified=None):
        self.calls.append(("delete_item", payload))

        class Resp:
            status_code = 204

        return Resp()

    def saved_search(self, name, conditions):
        self.calls.append(("saved_search", {"name": name, "conditions": conditions}))
        return {"successful": {"SNEW": {}}}

    def delete_saved_search(self, keys):
        self.calls.append(("delete_saved_search", {"keys": keys}))
        return 204

    def delete_tags(self, *payload):
        self.calls.append(("delete_tags", {"tags": list(payload)}))

        class Resp:
            status_code = 204

        return Resp()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith("ZOTERO_") or k.startswith("RAG_"):
            monkeypatch.delenv(k, raising=False)
    # Reset config singleton
    import mcp_zotero.config as cfg
    cfg._settings = None
    yield
    cfg._settings = None


def test_search_items(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=True)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)

    req = SearchRequest(query="test", limit=5, start=0)
    out = server.search_items(req)
    assert out["count"] == 1
    assert out["items"][0]["key"] == "AAA"


def test_search_items_qmode_tag_list(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=True)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)

    req = SearchRequest(query="test", qmode="everything", tag=["foo", "bar"], limit=10, start=0)
    _ = server.search_items(req)
    last = dummy_client.calls[-1][1]
    assert last["qmode"] == "everything"
    assert last["tag"] == ["foo", "bar"]


def test_query_tools(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=False)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)

    req = QueryRequest(q="x", qmode="everything", limit=5, start=0)

    out = server.top_items(req)
    assert out["count"] == 1
    assert dummy_client.calls[-1][0] == "top"

    out = server.list_collections(req)
    assert out["collections"][0]["key"] == "C1"
    assert dummy_client.calls[-1][0] == "collections"

    out = server.list_tags(req)
    assert out["tags"] == ["t1"]
    assert dummy_client.calls[-1][0] == "tags"


def test_collections_crud(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=False)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)
    monkeypatch.setattr(server, "_get_write_client", lambda: dummy_client)

    create_req = CollectionCreateRequest(collections=[CollectionCreateItem(name="mcp-test")])
    out = server.create_collections(create_req)
    assert "created" in out

    update_req = CollectionUpdateRequest(collection_key="C1", name="New")
    out = server.update_collection(update_req)
    assert "updated" in out

    del_req = CollectionKeyRequest(collection_key="C1")
    out = server.delete_collection(del_req)
    assert out["status_code"] == 204


def test_item_metadata_tools(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=False)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)

    out = server.item_types()
    assert out["item_types"] == ["book"]

    out = server.item_type_fields(ItemTypeRequest(item_type="book"))
    assert "item_type_fields" in out

    out = server.item_creator_types(ItemTypeRequest(item_type="book"))
    assert "item_creator_types" in out

    out = server.item_attachment_link_modes()
    assert out["item_attachment_link_modes"] == ["linked_file"]


def test_saved_search_and_tags(monkeypatch):
    dummy_client = DummyZotero("123", "user", "abc", local=False)
    monkeypatch.setattr(server, "_get_client", lambda: dummy_client)
    monkeypatch.setattr(server, "_get_write_client", lambda: dummy_client)

    out = server.create_saved_search(SavedSearchRequest(name="S", conditions=[]))
    assert "created" in out

    out = server.delete_saved_search(DeleteSavedSearchRequest(keys=["K"]))
    assert out["status_code"] == 204

    out = server.delete_tags(DeleteTagsRequest(tags=["t"]))
    assert out["status_code"] == 204


def test_attach_linked_file(monkeypatch, tmp_path):
    file_path = tmp_path / "paper.pdf"
    file_path.write_text("x")
    dummy_client = DummyZotero("123", "user", "abc", local=False)
    monkeypatch.setattr(server, "_get_write_client", lambda: dummy_client)

    req = AttachLinkedFileRequest(item_key="AAA", file_path=str(file_path), title="My PDF")
    out = server.attach_linked_file(req)
    assert "created" in out
    call = dummy_client.calls[-1]
    assert call[0] == "create_items"
    attachment = call[1][0]
    assert attachment["linkMode"] == "linked_file"
    assert attachment["parentItem"] == "AAA"


def test_health_check_missing(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    out = server.health_check()
    assert out["ok"] is False
    assert any("ZOTERO_LIBRARY_ID" in m for m in out["missing"])


def test_ping():
    out = server.ping()
    assert out["ok"] is True
