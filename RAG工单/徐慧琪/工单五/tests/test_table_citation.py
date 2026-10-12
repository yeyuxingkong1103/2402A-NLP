# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""Q1 必须溯源到《招股说明书1.pdf》的收入表格（需求文档硬性要求 4）。

设计取舍（实测得来，见 `_nearby_table_notes` 的 docstring）：该表**只做引用定位，
不进生成上下文**。原因是源文档冲突——表里「军用领域/小计/2019年1-6月」是
4,627.15，正文与验收标准答案是 4,627.14；把表喂进上下文会让模型答 4,627.15
并漏掉 14,414.16（5/5 次复现），同时使数字校正护栏失效。
"""
from __future__ import annotations

import pytest

from src import rag

TABLE = "招股说明书1.pdf#p342#t0"
CAPTION = "4、主营业务收入按行业列示"


def _hit(block_type: str, table_id: str = "", page_idx: int = 0,
         caption: str = "", row_index: int = -1, source: str = "招股说明书1.pdf",
         text: str = "t") -> dict:
    return {"block_type": block_type, "table_id": table_id, "text": text,
            "page_idx": page_idx, "table_caption": caption,
            "row_index": row_index, "source": source}


def _paragraphs(n: int) -> list[dict]:
    return [_hit("text", text=f"段落{i}") for i in range(n)]


def test_notes_carry_table_identity_for_tables_outside_the_final_context():
    """最终 6 块全是段落，同表 2 行落在窗口内 → 记一条定位（表名/页码/命中行）。"""
    ranked = _paragraphs(6) + [
        _hit("table_row", TABLE, page_idx=342, caption=CAPTION, row_index=1),
        _hit("table_row", TABLE, page_idx=342, caption=CAPTION, row_index=3),
    ]
    notes = rag._nearby_table_notes(ranked, ranked[:6])

    assert len(notes) == 1
    n = notes[0]
    assert n["table_id"] == TABLE
    assert n["table_caption"] == CAPTION
    assert n["row_indices"] == [1, 3]
    assert n["page_display"] == 343


def test_table_already_in_context_is_not_noted():
    """表已经在生成上下文里 → 无需再做定位（它会以【片段N】正常引用）。"""
    final = _paragraphs(5) + [_hit("table_row", TABLE, page_idx=342, caption=CAPTION)]
    ranked = final + [_hit("table_row", TABLE, page_idx=342, caption=CAPTION, row_index=2)]
    assert rag._nearby_table_notes(ranked, final) == []


def test_single_row_of_each_table_is_ignored():
    """不同表各命中 1 行 → 弱信号，不记（避免给普通问答题挂无关表格）。"""
    ranked = _paragraphs(6) + [
        _hit("table_row", "招股说明书1.pdf#p10#t0", page_idx=9),
        _hit("table_row", "招股说明书1.pdf#p12#t0", page_idx=11),
    ]
    assert rag._nearby_table_notes(ranked, ranked[:6]) == []


def test_attach_matches_by_heading_in_same_source():
    """按「引用块标题末级 == 表名」挂接——两张证据说的是同一处，判据确切。"""
    cmap = [
        {"index": 1, "source": "招股说明书1.pdf", "page_display": 129,
         "heading_path": ["…", "①行业集中度高的影响"], "is_table": False},
        {"index": 2, "source": "招股说明书1.pdf", "page_display": 342,
         "heading_path": ["…", CAPTION], "is_table": False},
    ]
    notes = [{"table_id": TABLE, "table_caption": CAPTION, "source": "招股说明书1.pdf",
              "page_display": 343, "row_indices": [1]}]
    rag._attach_nearby_tables(cmap, notes)

    assert cmap[1]["nearby_table"]["table_id"] == TABLE
    assert "nearby_table" not in cmap[0]


def test_attach_ignores_other_source_and_table_entries():
    """别的 PDF 的同名章节不挂；已经是表格块的引用不挂。"""
    cmap = [
        {"index": 1, "source": "招股说明书2.pdf", "page_display": 342,
         "heading_path": ["…", CAPTION], "is_table": False},
        {"index": 2, "source": "招股说明书1.pdf", "page_display": 343,
         "heading_path": ["…", CAPTION], "is_table": True},
    ]
    notes = [{"table_id": TABLE, "table_caption": CAPTION, "source": "招股说明书1.pdf",
              "page_display": 343, "row_indices": [1]}]
    rag._attach_nearby_tables(cmap, notes)

    assert all("nearby_table" not in c for c in cmap)


def test_attach_falls_back_to_page_when_table_has_no_caption():
    """无表名的表格（招股2 常见）退化为同页 ±1 匹配。"""
    cmap = [{"index": 1, "source": "招股说明书2.pdf", "page_display": 100,
             "heading_path": ["…", "某章节"], "is_table": False}]
    notes = [{"table_id": "招股说明书2.pdf#p99#t0", "table_caption": "",
              "source": "招股说明书2.pdf", "page_display": 100, "row_indices": [2]}]
    rag._attach_nearby_tables(cmap, notes)

    assert cmap[0]["nearby_table"]["table_id"] == "招股说明书2.pdf#p99#t0"


@pytest.mark.slow
def test_q1_citation_locates_the_revenue_table_without_feeding_it():
    """端到端（走完整 ask 链路）：Q1 的引用里能定位到收入表格（表名 + 命中行），
    该表**不得**进入生成上下文，且答案必须逐字给出 4 个年度数字。

    最后一条是回归闸门：把该表放进上下文时，实测 5/5 次答成
    「4,627.15…」并漏掉 14,414.16（源文档的表格与正文数字冲突）。"""
    rag.warmup()
    r = rag.ask("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                use_cache=False)

    for num in ("6,464.51", "14,414.16", "18,780.67", "4,627.14"):
        assert num in (r.answer or ""), f"答案漏了 {num}：{r.answer!r}"
    assert all(not b.get("table_id") for b in r.contexts), \
        "该表与正文数字冲突，绝不允许进入生成上下文"

    located = [c for c in r.citations if c.get("nearby_table")]
    assert located, f"Q1 的引用里没有表格定位：{r.citations}"
    assert located[0]["nearby_table"]["table_id"] == TABLE
    assert located[0]["nearby_table"]["table_caption"] == CAPTION
    assert located[0]["nearby_table"]["row_indices"], "定位必须带命中行"
