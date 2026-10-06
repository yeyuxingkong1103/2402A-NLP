"""关键词检索过滤与排名测试。"""

from datetime import date

from app.retrieval.keyword_search import KeywordSearcher, build_keyword_searcher_rows


def test_keyword_search_applies_legal_filters_without_changing_rank_order() -> None:
    searcher = build_keyword_searcher_rows(
        [
            ("expired", "经济补偿 经济补偿", "第四十七条"),
            ("current-second", "经济补偿", "第二十七条"),
            ("wrong-type", "经济补偿", "第一条"),
            ("noise-1", "劳动合同订立", None),
            ("noise-2", "社会保险缴纳", None),
            ("noise-3", "工伤认定程序", None),
            ("noise-4", "休息休假制度", None),
        ],
        metadata=[
            {
                "document_type": "法律",
                "jurisdiction": "中国大陆",
                "effective_date": date(2008, 1, 1),
                "expiration_date": date(2020, 1, 1),
                "is_current": False,
            },
            {
                "document_type": "行政法规",
                "jurisdiction": "中国大陆",
                "effective_date": date(2008, 9, 18),
                "expiration_date": None,
                "is_current": True,
            },
            {
                "document_type": "案例材料",
                "jurisdiction": "中国大陆",
                "effective_date": None,
                "expiration_date": None,
                "is_current": True,
            },
            {},
            {},
            {},
            {},
        ],
    )

    hits = searcher.search(
        "经济补偿",
        top_k=20,
        jurisdiction="中国大陆",
        document_types=["法律", "行政法规"],
        as_of_date="2026-09-14",
        only_current=True,
    )

    assert [hit.chunk_key for hit in hits] == ["current-second"]


def test_keyword_search_rejects_missing_metadata_when_filter_is_requested() -> None:
    searcher = build_keyword_searcher_rows(
        [
            ("unknown", "经济补偿", None),
            ("known", "经济补偿", None),
            ("noise-1", "劳动合同订立", None),
            ("noise-2", "社会保险缴纳", None),
            ("noise-3", "工伤认定程序", None),
        ],
        metadata=[
            {},
            {"document_type": "法律", "jurisdiction": "中国大陆"},
            {},
            {},
            {},
        ],
    )

    hits = searcher.search(
        "经济补偿",
        jurisdiction="中国大陆",
        document_types=["法律"],
    )

    assert [hit.chunk_key for hit in hits] == ["known"]


def test_keyword_search_returns_bm25_order_when_filters_match_all() -> None:
    searcher = build_keyword_searcher_rows(
        [
            ("first", "经济补偿 经济补偿 经济补偿", None),
            ("second", "经济补偿", None),
            ("noise-1", "劳动合同订立", None),
            ("noise-2", "社会保险缴纳", None),
            ("noise-3", "工伤认定程序", None),
            ("noise-4", "休息休假制度", None),
        ],
        metadata=[{}, {}, {}, {}, {}, {}],
    )

    hits = searcher.search("经济补偿", top_k=20)

    assert [hit.chunk_key for hit in hits] == ["first", "second"]
