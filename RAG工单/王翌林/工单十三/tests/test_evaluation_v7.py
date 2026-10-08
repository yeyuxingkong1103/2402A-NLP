# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
tests/test_evaluation_v7.py —— 工单七检索评估框架单元测试（不依赖 Milvus/模型）
"""
import pytest

from src.evaluation_v7 import (normalize_text, keyword_group_hits,
                               doc_recall_at_k, reciprocal_rank,
                               cross_company_noise, evaluate_case, summarize)

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"


def test_work_order_marker():
    """工单七：评估框架模块必须标注工单编号"""
    from src import evaluation_v7
    assert evaluation_v7.WORK_ORDER == WORK_ORDER


def test_normalize_text_thousands_separator():
    """工单七：千分位逗号/空白归一化（2,768.09 与 2768.09 等价）"""
    assert normalize_text("2,768.09 亿元") == "2768.09亿元"
    assert normalize_text("6,122.65") == "6122.65"
    assert normalize_text(None) == ""


def test_keyword_group_hits_synonyms():
    """工单七：同义词组任一命中即通过"""
    r = keyword_group_hits("当年营业收入543.83亿元",
                           [["营业收入", "营收"], ["净利润"], ["543.83"]])
    assert r["hit"] == 2 and r["total"] == 3
    assert r["rate"] == pytest.approx(0.667, abs=0.001)
    assert r["missed"] == ["净利润"]


def test_keyword_group_number_normalization():
    """工单七：数字关键词归一化后命中"""
    r = keyword_group_hits("实现营业收入2,768.09亿元，同比增长6.06%",
                           [["2768.09"], ["6.06"]])
    assert r["hit"] == 2 and r["missed"] == []


def test_doc_recall_at_k_and_mrr():
    """工单七：金标文档 top-k 覆盖率与 MRR"""
    retrieved = [{"doc_id": "A"}, {"doc_id": "B"}, {"doc_id": "C"}]
    assert doc_recall_at_k(retrieved, ["A", "B"], k=5) == 1.0
    assert reciprocal_rank(retrieved, ["B"]) == pytest.approx(0.5)
    # top-k 截断：金标 C 在第 3，k=2 时不召回
    assert doc_recall_at_k(retrieved, ["A", "C"], k=2) == 0.5
    # 跨文档题：5 个金标只命中 2 个
    gold = ["A", "B", "X", "Y", "Z"]
    assert doc_recall_at_k(retrieved, gold, k=5) == pytest.approx(0.4)
    assert reciprocal_rank(retrieved, ["Z"]) == 0.0


def test_cross_company_noise():
    """工单七：top-k 串档检测"""
    retrieved = [{"doc_id": "平安银行2019年报"}, {"doc_id": "招商银行2019年报"}]
    noise = cross_company_noise(retrieved, ["平安银行2019年报"], k=5)
    assert noise == ["招商银行2019年报"]


def test_evaluate_case_full():
    """工单七：单题评估完整链路（好结果无问题标签）"""
    retrieved = [{"doc_id": "邮储银行2019年报",
                  "content": "2019年实现营业收入2,768.09亿元，同比增长6.06%。"}]
    m = evaluate_case(retrieved,
                      "邮储银行2019年营业收入2,768.09亿元，同比增长6.06%。",
                      ["邮储银行2019年报"],
                      [["2768.09"], ["6.06"], ["营业收入"]],
                      latency_ms=1800.0)
    assert m["doc_recall@5"] == 1.0 and m["mrr"] == 1.0
    assert m["context_recall"] == 1.0 and m["answer_acc"] == 1.0
    assert m["noise_docs"] == [] and m["issues"] == []


def test_evaluate_case_classifies_issues():
    """工单七：漏检/串档/漏点/超时自动归类"""
    retrieved = [{"doc_id": "招商银行2019年报",
                  "content": "金融科技相关内容"}]
    m = evaluate_case(retrieved, "金融科技",
                      ["邮储银行2019年报"], [["2768.09"]], latency_ms=4000.0)
    assert m["doc_recall@5"] == 0.0 and m["mrr"] == 0.0
    assert m["noise_docs"] == ["招商银行2019年报"]
    assert any("金标文档漏检" in i for i in m["issues"])
    assert any("跨公司串档" in i for i in m["issues"])
    assert any("答案要点遗漏" in i for i in m["issues"])
    assert any("超时" in i for i in m["issues"])


def test_summarize_aggregation():
    """工单七：多题汇总指标与问题分布"""
    good = {"metrics": {"doc_recall@5": 1.0, "mrr": 1.0,
                        "context_recall": 1.0, "answer_acc": 1.0,
                        "latency_ms": 1000.0, "issues": []}}
    bad = {"metrics": {"doc_recall@5": 0.0, "mrr": 0.0,
                       "context_recall": 0.5, "answer_acc": 0.0,
                       "latency_ms": 4000.0,
                       "issues": ["答案要点遗漏(答案未命中全部金标关键词)",
                                  "响应超时(>3s)"]}}
    s = summarize([good, bad])
    assert s["case_count"] == 2
    assert s["doc_recall@5_avg"] == 0.5
    assert s["answer_accuracy"] == 0.5
    assert s["latency_avg_ms"] == 2500.0
    assert s["latency_ok_rate"] == 0.5
    assert s["issue_distribution"].get("响应超时(>3s)") == 1
