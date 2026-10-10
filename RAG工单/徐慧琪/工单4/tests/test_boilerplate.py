# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""RC1 样板判据：归一化、跨页重复判据、长度护栏、缓存与索引清点。"""
import json

import pytest

from rag04.retrieve.boilerplate import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_PAGES,
    BoilerplateInfo,
    collect_chunks,
    filter_boilerplate_blocks,
    find_boilerplate,
    flagged_texts,
    load_or_build,
    normalize_text,
)
from rag04.schema import Chunk, TextBlock


def _c(cid, text, doc="d1", page=1, bt="text"):
    return Chunk(chunk_id=cid, doc_id=doc, page=page, block_type=bt,
                 source_id=cid, text=text)


def test_normalize_strips_ascii_and_fullwidth_spaces():
    assert normalize_text("武汉兴图新科电子股份有限公司   　  招股意向书") == \
        "武汉兴图新科电子股份有限公司招股意向书"
    assert normalize_text("") == "" and normalize_text(None) == ""


def test_repeated_header_flagged_boundary_exact():
    """>=K 个不同页判为样板；K-1 页不判（边界钉死）。"""
    chunks = [_c(f"h{p}", "武汉兴图新科电子股份有限公司 招股意向书", page=p)
              for p in range(1, DEFAULT_MIN_PAGES + 1)]
    chunks += [_c(f"b{p}", f"第{p}页的正文段落，内容各不相同。", page=p)
               for p in range(1, DEFAULT_MIN_PAGES + 1)]
    info = find_boilerplate(chunks)
    assert info.ids == {f"h{p}" for p in range(1, DEFAULT_MIN_PAGES + 1)}
    assert info.n_flagged == DEFAULT_MIN_PAGES and info.n_scanned == DEFAULT_MIN_PAGES * 2

    below = find_boilerplate(
        [_c(f"h{p}", "页眉文案", page=p) for p in range(1, DEFAULT_MIN_PAGES)])
    assert below.ids == frozenset()


def test_same_page_duplicates_count_once():
    """同页出现多次不累计页数——否则页面内重复的正文会被误判。"""
    chunks = [_c(f"x{i}", "同页重复文本。", page=3) for i in range(DEFAULT_MIN_PAGES + 5)]
    assert find_boilerplate(chunks).ids == frozenset()


def test_long_repeated_body_protected_by_length_guard():
    """正文长段即使跨页复现也不判样板（不得误伤正文）。"""
    long_text = "公司是一家基于网络通信的军队专用视频指挥控制系统提供商，" * 8
    assert len(long_text) > DEFAULT_MAX_CHARS
    pages = range(1, DEFAULT_MIN_PAGES + 31)
    chunks = [_c(f"l{p}", long_text, page=p) for p in pages]
    assert len(chunks) == DEFAULT_MIN_PAGES + 30
    assert find_boilerplate(chunks).ids == frozenset()
    # 同一文本在护栏内则照判（护栏是唯一差别）
    short = long_text[:40]
    chunks = [_c(f"s{p}", short, page=p) for p in pages]
    assert len(find_boilerplate(chunks).ids) == DEFAULT_MIN_PAGES + 30


def test_distinct_docs_count_as_distinct_pages():
    chunks = [_c(f"a{p}", "共用模板文本", doc="d1", page=p) for p in range(1, 11)]
    chunks += [_c(f"b{p}", "共用模板文本", doc="d2", page=p) for p in range(1, 11)]
    assert len(find_boilerplate(chunks).ids) == 20      # 10 + 10 >= K


def test_blank_and_tiny_text_ignored():
    chunks = [_c(f"e{p}", "   ", page=p) for p in range(1, 40)]
    chunks += [_c(f"t{p}", "…", page=p) for p in range(1, 40)]
    info = find_boilerplate(chunks)
    assert info.ids == frozenset() and info.n_scanned == 78


def test_min_pages_below_two_rejected():
    with pytest.raises(ValueError, match="min_pages"):
        find_boilerplate([], min_pages=1)


def test_cache_roundtrip_and_hit_avoids_scan(tmp_path):
    p = tmp_path / "bp.json"
    chunks = [_c(f"h{pg}", "页眉", page=pg) for pg in range(1, 25)]
    chunks.append(_c("body", "正文内容", page=1))
    first = load_or_build(p, chunks=chunks, counts={"text_chunks": 25})
    assert p.exists() and first.ids == frozenset(f"h{pg}" for pg in range(1, 25))

    cached = json.loads(p.read_text(encoding="utf-8"))
    assert cached["min_pages"] == DEFAULT_MIN_PAGES
    assert cached["version"] == 1 and cached["n_flagged"] == 24

    # 缓存命中时即使不给 chunks 也能返回（证明没有重新扫描）
    again = load_or_build(p, chunks=None, counts={"text_chunks": 25})
    assert again.ids == first.ids and again.n_groups == 1


def test_cache_invalidated_when_index_size_changes(tmp_path):
    p = tmp_path / "bp.json"
    chunks = [_c(f"h{pg}", "页眉", page=pg) for pg in range(1, 25)]
    load_or_build(p, chunks=chunks, counts={"text_chunks": 25})
    with pytest.raises(ValueError, match="无法推导"):
        # 索引规模已变且未给 chunks：必须报错而不是用过期缓存
        load_or_build(p, chunks=None, counts={"text_chunks": 999})
    fresh = load_or_build(p, chunks=chunks, counts={"text_chunks": 25})
    assert fresh.n_flagged == 24


def test_cache_corrupt_file_rebuilt(tmp_path):
    p = tmp_path / "bp.json"
    p.write_text("{not json", encoding="utf-8")
    chunks = [_c(f"h{pg}", "页眉", page=pg) for pg in range(1, 25)]
    info = load_or_build(p, chunks=chunks)
    assert info.n_flagged == 24


class _FakeScrollClient:
    def __init__(self, by_coll):
        self.by_coll = by_coll

    def scroll(self, coll, limit=100_000, with_payload=True):
        return self.by_coll.get(coll, []), None


class _Rec:
    def __init__(self, rid, payload):
        self.id = rid
        self.payload = payload


class _FakeStore:
    def __init__(self, by_coll):
        self.client = _FakeScrollClient(by_coll)


def test_collect_chunks_reads_three_collections():
    store = _FakeStore({
        "text_chunks": [_Rec(1, {"chunk_id": "t1", "doc_id": "d", "page": 1,
                                 "block_type": "text", "text": "正文"})],
        "table_chunks": [_Rec(2, {"chunk_id": "tb1", "doc_id": "d", "page": 2,
                                  "block_type": "table", "text": "表格"})],
        "image_chunks": [_Rec(3, {"chunk_id": "im1", "doc_id": "d", "page": 3,
                                  "block_type": "image", "text": "图"})],
    })
    chunks = collect_chunks(store)
    assert {c.chunk_id for c in chunks} == {"t1", "tb1", "im1"}
    assert {c.block_type for c in chunks} == {"text", "table", "image"}


def test_info_dict_roundtrip():
    info = BoilerplateInfo(ids=frozenset({"a", "b"}), min_pages=20, max_chars=120,
                           n_scanned=3, n_flagged=2, n_groups=1, counts={"c": 3})
    assert BoilerplateInfo.from_dict(info.to_dict()) == info
    assert info.is_boilerplate("a") and not info.is_boilerplate("c")


# ---------- RC1-in：入库侧过滤（重建第二阶段） ----------

def _t(i, text, page=1, doc="d1"):
    return TextBlock(doc_id=doc, page=page, bbox=(0, 0, 100, 10), text=text)


def _mixed_blocks():
    blocks = [_t(i, "武汉兴图新科电子股份有限公司 招股意向书", page=i)
              for i in range(1, DEFAULT_MIN_PAGES + 5)]
    blocks += [_t(100 + i, f"第{i}页正文，内容各自不同。", page=i)
               for i in range(1, DEFAULT_MIN_PAGES + 5)]
    return blocks


def test_flagged_texts_reuses_chunk_criterion():
    texts = flagged_texts(_mixed_blocks())
    assert len(texts) == 1
    assert normalize_text("武汉兴图新科电子股份有限公司 招股意向书") in texts


def test_filter_boilerplate_blocks_drops_only_headers():
    kept, dropped = filter_boilerplate_blocks(_mixed_blocks())
    assert dropped == DEFAULT_MIN_PAGES + 4
    assert len(kept) == DEFAULT_MIN_PAGES + 4
    assert all("招股意向书" not in b.text for b in kept)
    assert all(b.text.startswith("第") for b in kept)


def test_filter_boilerplate_blocks_keeps_long_repeated_body():
    """长度护栏在入库侧同样生效：长段复现不得被删。"""
    long_text = "公司是一家基于网络通信的军队专用视频指挥控制系统提供商，" * 8
    blocks = [_t(i, long_text, page=i) for i in range(1, DEFAULT_MIN_PAGES + 3)]
    kept, dropped = filter_boilerplate_blocks(blocks)
    assert dropped == 0 and len(kept) == len(blocks)


def test_filter_boilerplate_blocks_baseline_off_keeps_everything():
    """入库过滤只在 full_04 启用（baseline_03 是对照组，由调用方 gate）。"""
    kept, dropped = filter_boilerplate_blocks(_mixed_blocks(), enabled=False)
    assert dropped == 0 and len(kept) == len(_mixed_blocks())
