# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 分块模块测试"""
from src.chunker import build_chunks, stats


def _t(text, page=0, level=None):
    return {"type": "text", "text": text, "page_idx": page,
            "text_level": level, "caption": []}


def test_heading_path_and_parent_shared():
    items = [
        _t("第一节 释义", 0, 1),
        _t("本招股说明书指" + "武汉兴图新科电子股份有限公司" * 20, 0),
        _t("一、一般释义", 0, 2),
        _t("公司指" + "兴图新科" * 60, 1),
    ]
    chunks = build_chunks(items, "招股说明书1.pdf")
    assert chunks[0]["heading_path"] == ["第一节 释义"]
    assert chunks[-1]["heading_path"] == ["第一节 释义", "一、一般释义"]
    same = [c for c in chunks if c["heading_path"] == ["第一节 释义"]]
    assert len({c["parent_id"] for c in same}) == 1
    assert "第一节 释义" in same[0]["parent_text"]
    # 不同节的父块必须不同
    assert chunks[0]["parent_id"] != chunks[-1]["parent_id"]


def test_table_is_standalone_block():
    items = [
        _t("第二节 财务", 2, 1),
        {"type": "table", "text": "| 项目 | 金额 |\n|---|---|\n| 收入 | 100 |",
         "page_idx": 3, "text_level": None, "caption": []},
    ]
    chunks = build_chunks(items, "x.pdf")
    tables = [c for c in chunks if c["block_type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["text"].startswith("| 项目 |")
    assert tables[0]["page_idx"] == 3
    # 表格不并入正文流
    assert all("收入" not in c["text"] or c["block_type"] == "table" for c in chunks)


def test_child_blocks_bounded_and_split():
    big = _t("。".join(["这是句子内容" * 10] * 30), 5)
    chunks = build_chunks([_t("标题", 5, 1), big], "x.pdf")
    body = [c for c in chunks if c["block_type"] == "text" and c["text"] != "标题"]
    assert len(body) > 1                                   # 长文本被切成多块
    assert all(len(c["text"]) <= 520 for c in body)        # 上限 450 + 句子溢出余量


def test_ids_stable_and_unique():
    items = [_t("标题", 0, 1), _t("正文" * 100, 0)]
    a = build_chunks(items, "x.pdf")
    b = build_chunks(items, "x.pdf")
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert len({c["chunk_id"] for c in a}) == len(a)
    assert len({c["parent_id"] for c in a}) == 1


def test_metadata_complete():
    items = [_t("第一章", 0, 1), _t("内容" * 80, 1)]
    for c in build_chunks(items, "招股说明书1.pdf"):
        for key in ("page_idx", "heading_path", "source", "block_type",
                    "parent_id", "parent_text", "seq"):
            assert key in c, key
        assert c["source"] == "招股说明书1.pdf"


def test_stats():
    items = [_t("第一章", 0, 1), _t("内容" * 80, 1)]
    s = stats(build_chunks(items, "x.pdf"))
    assert s["n_chunks"] >= 1 and "text" in s["by_type"]


# ---------------------------------------------------------------------------
# 工单3：多文档唯一性（实测踩坑 —— 两文档 chunk_id 各自从 c000001 开始，
# 只按 chunk_id 做 Qdrant 点 ID 会让招股2 覆盖招股1：7968 点入库后只剩 5024）
# ---------------------------------------------------------------------------
def test_chunk_and_parent_ids_scoped_by_source():
    items = [_t("第一章", 0, 1), _t("内容" * 80, 1)]
    c1 = build_chunks(items, "招股说明书1.pdf")
    c2 = build_chunks(items, "招股说明书2.pdf")
    # chunk_id 相同（各自编号），但 parent_id 必须带 source 才不会跨文档合并
    assert [c["chunk_id"] for c in c1] == [c["chunk_id"] for c in c2]
    p1 = {c["parent_id"] for c in c1}
    p2 = {c["parent_id"] for c in c2}
    assert p1.isdisjoint(p2)
    assert all(pid.startswith("招股说明书1.pdf#") for pid in p1)
    assert all(pid.startswith("招股说明书2.pdf#") for pid in p2)


def test_table_parent_id_also_scoped_by_source():
    """表格 parent_id = table_id，本身已含 source，不与其他文档冲突。"""
    item = {"type": "table", "text": "表", "page_idx": 3, "caption": [],
            "table_html": "<table><tr><td>甲</td><td>乙</td></tr>"
                          "<tr><td>1</td><td>2</td></tr>"
                          "<tr><td>3</td><td>4</td></tr></table>"}
    c1 = build_chunks([item], "招股说明书1.pdf")
    c2 = build_chunks([item], "招股说明书2.pdf")
    ids1 = {c["parent_id"] for c in c1}
    ids2 = {c["parent_id"] for c in c2}
    assert ids1.isdisjoint(ids2)
