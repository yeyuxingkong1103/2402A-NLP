# -*- coding: utf-8 -*-
"""核心单元测试（二）：意图识别/检索路由 + 安全护栏 + 引用展示。

从 test_core.py 拆出，保证每个测试文件也不超过 150 行。
全部纯逻辑，不连外部服务。
"""
import sys  # 标准库：改模块搜索路径
from pathlib import Path  # 拼项目根路径

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 本文件在 tests/ 下，向上一级即项目根
sys.path.insert(0, str(PROJECT_ROOT))  # 插到最前：pytest 从任意目录运行都能 import src


# ========== 意图识别与检索路由 ==========

def test_detect_intent_knowledge():  # 通用知识问句的意图分流
    """通用知识问句不得走追问链路。"""
    from src.intent import KNOWLEDGE, detect_intent  # 被测：意图常量与识别函数

    assert detect_intent("高血压吃什么药") == KNOWLEDGE  # 无"我"无时间词 → 知识类
    assert detect_intent("高血压不能吃什么") == KNOWLEDGE  # 清单型问句同样是知识类


def test_detect_intent_personal():  # 个人病情信号识别
    """带"我/时间"信号才算个人病情。"""
    from src.intent import PERSONAL, detect_intent  # 被测

    assert detect_intent("我最近血压偏高") == PERSONAL  # "我"+"最近" → 个人病情类


def test_needs_clarify_on_dangling_input():  # 断句追问判断
    """断句/缺宾语必须先追问（如"遗传的高血压，有吃"）。"""
    from src.query_tools import needs_clarify  # 被测

    assert needs_clarify("遗传的高血压，有吃")  # 缺宾语的断句 → 需要追问
    assert not needs_clarify("高血压吃什么药")  # 完整问句不追问


def test_multi_route_expands_drug_question():  # 用药类问题的多路拆检
    """用药类问题要拆成多路检索，避免只召回一两条。"""
    from src.intent import is_multi_route, multi_queries  # 被测：多路判定与子查询拆分

    assert is_multi_route("高血压吃什么药")  # 用药问句触发多路
    assert len(multi_queries("高血压吃什么药")) >= 5  # 至少拆出 5 路子查询保证找全
    assert multi_queries("今天股市怎么样") == []  # 无关问题不拆（返回空列表）


def test_anchor_query_only_for_on_topic_phrases():  # 情绪/症状问句补疾病锚点
    """情绪/症状类口语问句补疾病锚点；无关问题绝不补（保住阈值拦截能力）。"""
    from src.query_tools import anchor_query  # 被测

    assert anchor_query("我最近很焦虑") == "我最近很焦虑 高血压"  # 焦虑是高血压相关情绪 → 补锚点
    assert anchor_query("我失眠好几天了") == "我失眠好几天了 高血压"  # 失眠同理补锚点
    assert anchor_query("高血压患者情绪波动大") == "高血压患者情绪波动大"  # 已含疾病主体 → 不重复补
    assert anchor_query("今天股市怎么样") == "今天股市怎么样"  # 无关问题绝不补，保住阈值拦截能力


# ========== 安全护栏 ==========

def test_emergency_hits():  # 急症关键词拦截
    """急症词必须能被拦截。"""
    from src.safety import emergency_hits  # 被测

    assert emergency_hits("我突然胸痛、说不出话") == ["胸痛", "说不出话"]  # 两个急症词都要命中且按序返回
    assert emergency_hits("高血压吃什么药") == []  # 普通问句零命中


def test_unverified_drugs_flagged():  # 药名白名单校验
    """资料原文没写的药名应被白名单校验标记。"""
    from src.safety import unverified_drugs  # 被测

    assert unverified_drugs("可以吃氨氯地平", "资料里只提到CCB类") == ["氨氯地平"]  # 原文没写的药名被标记（防 LLM 瞎编药）
    assert unverified_drugs("可以吃氨氯地平", "资料里提到氨氯地平") == []  # 原文写过的放行


# ========== 引用展示 ==========

def test_citation_highlight_marks_matched_sentence():  # 命中句高亮
    """命中句要高亮，且必须做 HTML 转义。"""
    from src.citation import highlight  # 被测

    html = highlight("高血压患者应低盐饮食。规律运动有益。", "低盐")  # 对命中词"低盐"所在句做高亮
    assert "<mark>" in html and "低盐" in html  # 高亮标签存在且原词保留
