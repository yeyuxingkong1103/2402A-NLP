# -*- coding: utf-8 -*-
"""工单十五：单元测试"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.query_router import route_query, detect_figure_refs, is_image_query, rewrite_query_for_retrieval
from src.image_knowledge import ImageKnowledgeBase
from src.chunker import chunk_ocr_results


def test_detect_figure_refs():
    r = detect_figure_refs("在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？")
    assert r["fig_no"] == 3
    assert r["page"] == 11
    assert 13 in r["parts"] and 12 in r["parts"]


def test_route_query_visual():
    r = route_query("根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？")
    assert r["has_visual_ref"] is True
    assert r["needs_vlm"] is True
    assert r["fig_no"] == 3


def test_route_query_text():
    r = route_query("本发明主要涉及哪种物料的分配装置？")
    assert r["query_type"] == "text"
    assert r["needs_vlm"] is False


def test_image_knowledge():
    kb = ImageKnowledgeBase()
    desc = kb.get_description(fig_no=3)
    assert "部件13" in desc and "部件14" in desc and "顶部" in desc
    info = kb.get(page=11)
    assert info["fig_no"] == 3


def test_rewrite_query_injects_figure():
    kb = ImageKnowledgeBase()
    desc = kb.get_description(fig_no=3)
    enhanced = rewrite_query_for_retrieval("图3中部件14在哪", desc)
    assert "相关图纸描述" in enhanced


def test_chunker():
    ocr = [{"page": 1, "text": "这是第一段文字。\n\n这是第二段文字，包含图3的引用。"}]
    res = chunk_ocr_results(ocr, "test", "test.pdf")
    assert res["chunk_count"] >= 1
    assert "fig_3" in res["chunks"][0]["img_refs"]


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} 测试通过")
