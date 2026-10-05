# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 上下文组装测试"""
from src.context import build_context, citation_map, expand_parents


def _hit(pid, text, page, heading=("第一节",), parent_text=None):
    return {"chunk_id": f"c-{pid}-{text[:4]}", "text": text, "parent_id": pid,
            "parent_text": parent_text or f"父块正文……{text}……后续内容",
            "parent_page_idx": page, "page_idx": page,
            "heading_path": list(heading), "source": "招股说明书1.pdf",
            "block_type": "text", "score": 0.9}


def test_expand_parents_dedupes_by_parent():
    hits = [_hit("s1", "注册资本为7,360万元", 10),
            _hit("s1", "法定代表人为程家明", 10),      # 同父块
            _hit("s2", "军用领域收入6,464万元", 129)]
    blocks = expand_parents(hits, max_chars=5000)
    assert len(blocks) == 2                       # 同父块去重
    assert blocks[0]["parent_id"] == "s1"
    assert len(blocks[0]["child_hits"]) == 2
    assert blocks[1]["parent_id"] == "s2"


def test_expand_parents_respects_budget_and_order():
    hits = [_hit(f"s{i}", f"内容{i}" * 50, i) for i in range(10)]
    blocks = expand_parents(hits, max_chars=2500, per_parent_chars=1200)
    total = sum(len(b["parent_text"]) for b in blocks)
    assert total <= 2500 + 400                    # 允许最后一个片段的裁剪余量
    assert [b["parent_id"] for b in blocks] == sorted(
        [b["parent_id"] for b in blocks], key=lambda x: int(x[1:]))  # 保序


def test_build_context_has_page_and_heading():
    hits = [_hit("s1", "注册资本为7,360万元", 10, heading=("第三节", "基本情况"))]
    blocks = expand_parents(hits)
    ctx = build_context(blocks)
    assert "【片段1】" in ctx
    assert "第11页" in ctx                        # page_idx 0 基 → 展示 +1
    assert "第三节 / 基本情况" in ctx


def test_citation_map():
    hits = [_hit("s1", "x", 10), _hit("s2", "y", 129)]
    cm = citation_map(expand_parents(hits))
    assert cm[0]["index"] == 1 and cm[0]["page_display"] == 11
    assert cm[1]["page_display"] == 130
