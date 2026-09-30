"""父块加载顺序测试。"""

from types import SimpleNamespace

from app.retrieval.parent_collapse import load_keyword_parent_items


class SessionContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_keyword_parent_items_preserve_original_hit_order(monkeypatch) -> None:
    hits = [
        SimpleNamespace(chunk_key="second", score=9.0),
        SimpleNamespace(chunk_key="first", score=8.0),
    ]
    rows = [
        SimpleNamespace(
            chunk_key="first",
            parent_chunk_key=None,
            content="第一条",
            article_number="第一条",
            paragraph_number=None,
            item_number=None,
            title="法规",
            source_url="https://example.com",
            document_key="doc-1",
            document_type="法律",
            jurisdiction="中国大陆",
            issuing_authority="机关",
            effective_date=None,
            expiration_date=None,
            status=None,
        ),
        SimpleNamespace(
            chunk_key="second",
            parent_chunk_key=None,
            content="第二条",
            article_number="第二条",
            paragraph_number=None,
            item_number=None,
            title="法规",
            source_url="https://example.com",
            document_key="doc-1",
            document_type="法律",
            jurisdiction="中国大陆",
            issuing_authority="机关",
            effective_date=None,
            expiration_date=None,
            status=None,
        ),
    ]
    monkeypatch.setattr(
        "app.retrieval.parent_collapse._fetch_rows",
        lambda session, keys: rows,
    )

    items = load_keyword_parent_items(lambda: SessionContext(), hits)

    assert [item.chunk_key for item in items] == ["second", "first"]
