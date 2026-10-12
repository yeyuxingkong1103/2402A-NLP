# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""查询改写测试（意图/分解/扩展/核心查询剥离）。"""
from src import query_rewrite as qr


def test_intent_detection():
    assert qr.detect_intent("注册资本是多少？") == "numeric"
    assert qr.detect_intent("下游主要包括哪些行业？") == "list"
    assert qr.detect_intent("法定代表人是谁？") == "fact"
    assert qr.detect_intent("军用领域收入与民用领域收入有何差异？".replace("有何差异", "的区别")) == "compare"


def test_disambiguate_is_passthrough():
    """工单5 **有意行为变更**：disambiguate 只做空白规整，不再替换实体。

    历史：工单3 只有一家公司的语料，这里把「该公司/其/公司/发行人」硬编码
    替换成武汉兴图新科。工单5 起语料含两家公司且支持多轮，硬编码替换会把
    Q4「那武汉力源…呢」错误地拉回兴图新科 —— 消歧改由 `src/coref.resolve`
    基于对话焦点（dialogue.DialogueState）完成，本层退化为纯直通。
    """
    # 代词句不再被替换成任何实体（消歧在 coref 层）
    assert qr.disambiguate("该公司的注册资本是多少？") == "该公司的注册资本是多少？"
    # 只做空白规整
    out2 = qr.disambiguate("  武汉兴图新科电子股份有限公司法定代表人是谁？\n ")
    assert out2 == "武汉兴图新科电子股份有限公司法定代表人是谁？"
    assert out2.count("武汉兴图新科电子股份有限公司") == 1


def test_expansion_for_domain_terms():
    a = qr.analyze("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
    assert a["intent"] == "numeric"
    assert any("军用领域" in e for e in a["expansions"])


def test_split_sub_questions_only_when_multiple():
    # 多个问号 → 分解
    subs = qr.split_sub_questions("公司参与制定了哪个技术标准？由谁牵头制定？", "fact")
    assert len(subs) == 2
    # 单一问题的"分别是多少"不分解
    subs2 = qr.split_sub_questions("报告期内，公司来自军用领域的收入分别是多少？", "numeric")
    assert subs2 == []


def test_language_detection():
    assert qr.detect_language("What is the registered capital?") == "en"
    assert qr.detect_language("注册资本是多少？") == "zh"


def test_multi_queries_bounded():
    a = qr.analyze("武汉兴图新科电子股份有限公司的注册资本和法定代表人是谁？")
    qs = qr.multi_queries(a, max_n=3)
    assert 1 <= len(qs) <= 3
    assert qs[0] == a["resolved"]


def test_keywords_extraction():
    a = qr.analyze("根据招股意向书，电子信息行业的上游涉及哪些企业？")
    assert "上游" in a["keywords"]
