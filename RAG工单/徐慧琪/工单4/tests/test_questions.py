# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest

from rag04.eval.questions import QUESTIONS, get_question, coverage


def test_all_16_questions_present():
    ids = [q.qid for q in QUESTIONS]
    assert len(ids) == 16, f"应为 16 题，实际 {len(ids)}"
    assert sorted(ids) == sorted([5, 6, 1, 2, 3, 4, 260, 95, 33, 34,
                                  957, 793, 795, 543, 531, 207])
    assert len(ids) == len(set(ids)), "题目 ID 不得重复"


def test_id5_and_id6_are_image_questions():
    """工单把 id 5、id 6 排在问题列表最前，因为它们代表本工单的核心增量。"""
    img_ids = [q.qid for q in QUESTIONS if q.block_type == "image"]
    assert sorted(img_ids) == [5, 6], f"图像题应恰为 id 5 与 id 6，实际={img_ids}"


def test_every_question_has_answer_key_and_pages():
    for q in QUESTIONS:
        assert q.answer_key, f"id {q.qid} 缺标准答案要点"
        assert q.gold_pages, f"id {q.qid} 缺出处页码"
        assert q.doc_id in ("招股说明书1", "招股说明书2")


def test_id5_is_image_question_with_10_key_points():
    q = get_question(5)
    assert q.block_type == "image"
    assert q.doc_id == "招股说明书2"
    assert 39 in q.gold_pages, "组织结构图在 1-based 第 39 页（印刷页 38）"
    assert q.strict is True
    for kw in ["渠道销售部", "电话及网络销售部", "大客户销售部", "国际贸易部",
               "北京销售处", "深圳销售处", "广州销售处", "成都销售处",
               "珠海销售处", "武汉销售处"]:
        assert kw in q.answer_key, f"id5 标准答案缺 {kw}"


def test_id6_is_image_question():
    q = get_question(6)
    assert q.block_type == "image"
    assert q.doc_id == "招股说明书2"
    assert 72 in q.gold_pages, "IC 市场图在 1-based 第 72 页（印刷页 71）"
    assert any("汽车" in k for k in q.answer_key)
    assert any("IC卡" in k or "IC 卡" in k for k in q.answer_key)


def test_questions_have_english_translation():
    for q in QUESTIONS:
        assert q.question_en and q.question_en != q.question, f"id {q.qid} 缺英文版"


def test_coverage_full_match():
    q = get_question(6)
    ans = "增长率最快的是汽车行业（14.0%），负增长的是IC卡（-2.0%）"
    assert coverage(ans, q) == 1.0


def test_coverage_partial():
    q = get_question(5)
    ans = "销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部"
    c = coverage(ans, q)
    assert 0.0 < c < 1.0, f"只答一半应得部分分，实际 {c}"


def test_coverage_id5_strict_requires_all_ten():
    q = get_question(5)
    full = "销售部下设渠道销售部、电话及网络销售部、大客户销售部、国际贸易部。\
大客户销售部下设北京销售处、深圳销售处、广州销售处、成都销售处、珠海销售处、武汉销售处。"
    assert coverage(full, q) == 1.0


def test_coverage_zero_for_unrelated():
    q = get_question(5)
    assert coverage("完全无关的回答内容", q) == 0.0


def test_get_question_raises_on_unknown():
    with pytest.raises(KeyError):
        get_question(99999)
