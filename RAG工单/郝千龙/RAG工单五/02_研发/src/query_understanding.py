# -*- coding: utf-8 -*-
# 【Query理解模块 · query_understanding.py】指代消解、省略补全、查询改写、术语归一
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""Query 理解层（工单五核心）：

把用户多轮对话中的省略/指代查询改写为独立完整的检索查询，使下游检索无需
理解对话历史也能命中正确文档。

处理流水线：
1. 术语归一：把口语/同义词归一到招股书用词（如“军用”→“国防”）；
2. 实体抽取：从当前 query 与历史中识别公司全称/简称；
3. 指代消解：把“他/这个公司/该公司/其”等替换为历史中最近的话题实体；
4. 省略补全：对“那 X 呢”类省略句，继承上一轮的问题意图（谓词+宾语）；
5. 查询改写：输出实体+意图组合后的完整检索查询。
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 招股书术语归一表（口语 → 招股书原文用词）
TERM_NORMALIZE = {
    "军用": "国防",
    "军品": "国防",
    "军工领域": "国防领域",
    "军用领域": "国防领域",
    "军工业": "国防",
    "组织结构": "组织结构",  # 保持不变，但保留以支持“组织结构图”
    "组织架构": "组织结构",
}

# 公司全称正则（含股份/有限/有限责任公司）
_COMPANY_FULL_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z]{2,30}(?:股份有限公司|有限责任公司|有限公司)")

# 指代词集合
PRONOUNS = ("他", "她", "它", "这个公司", "该公司", "这家公司",
            "那个公司", "其", "该企业", "这个企业")

# 省略触发词：“那 X 呢”“那么 X 呢”“X 呢”等
ELLIPSIS_PATTERNS = [
    re.compile(r"^那(?:么)?\s*(.+?)\s*(?:呢|怎么样|如何)?\s*[？?]?$"),
    re.compile(r"^(.+?)\s*(?:呢|怎么样|如何)\s*[？?]?$"),
]


@dataclass
class ParsedQuery:
    """单轮 Query 解析结果。"""

    original: str                     # 用户原始问题
    rewritten: str                    # 改写后的完整检索查询
    entity: str                       # 当前话题实体（公司名）
    intent: str                       # 问题意图（去掉实体后的核心谓词+宾语）
    has_coreference: bool = False     # 是否发生了指代消解
    has_ellipsis: bool = False        # 是否发生了省略补全


def normalize_terms(text: str) -> str:
    """术语归一：把口语同义词替换为招股书用词。

    :param text: 原始文本
    :return: 归一化后的文本
    """
    result = text
    # 按长度降序替换，避免短词先替换破坏长词
    for src in sorted(TERM_NORMALIZE, key=len, reverse=True):
        if src in result:
            result = result.replace(src, TERM_NORMALIZE[src])
    return result


def extract_company(text: str) -> Optional[str]:
    """从文本中抽取第一个公司全称。

    :param text: 待抽取文本
    :return: 公司全称，未命中返回 None
    """
    m = _COMPANY_FULL_RE.search(text)
    return m.group(0) if m else None


def extract_intent(query: str, entity: str) -> str:
    """从问题中抽取意图（去掉实体后剩余的谓词+宾语部分）。

    :param query: 原始问题
    :param entity: 已识别的公司实体
    :return: 意图字符串
    """
    intent = query
    if entity:
        intent = intent.replace(entity, "")
    # 去掉常见前缀词与标点
    intent = re.sub(r"^(报告期内|根据|请问|请问一下|一下|那么|那)\s*", "", intent)
    intent = intent.strip(" ，,。.？?！!；;")
    return intent


def is_ellipsis_query(query: str) -> bool:
    """判断当前 query 是否为省略句（“那 X 呢”类）。

    :param query: 当前问题
    :return: 是省略句返回 True
    """
    if "呢" not in query and "怎么样" not in query and "如何" not in query:
        return False
    # 含明确实体且含“呢”→ 典型省略切换话题
    if extract_company(query):
        return True
    # 纯代词 + 呢（如“那他呢”）
    if any(p in query for p in PRONOUNS) and ("呢" in query or "怎么样" in query):
        return True
    return False


def resolve_coreference(query: str, history_entities: List[str]) -> Tuple[str, bool]:
    """指代消解：把 query 中的指代词替换为最近的话题实体。

    :param query: 归一化后的当前问题
    :param history_entities: 历史话题实体列表（按时间倒序，最新在前）
    :return: (消解后的query, 是否发生消解)
    """
    has = False
    resolved = query
    # 取最近一个有效实体
    antecedent = next((e for e in history_entities if e), None)
    if not antecedent:
        return resolved, has
    for pronoun in sorted(PRONOUNS, key=len, reverse=True):
        if pronoun in resolved:
            resolved = resolved.replace(pronoun, antecedent)
            has = True
    return resolved, has


def fill_ellipsis(query: str, last_intent: str, last_entity: str) -> Tuple[str, str, bool]:
    """省略补全：为“那 X 呢”类查询补全意图。

    策略：
    - 先剥离实体与省略触发词，看剩余部分是否有实质意图（谓词+宾语）；
    - 有实质意图：用“当前实体 + 当前意图”组合（不继承上一轮）；
    - 无实质意图（纯“那 X 呢”）：继承上一轮意图。

    :param query: 当前问题（可能含新实体）
    :param last_intent: 上一轮的问题意图
    :param last_entity: 上一轮的话题实体
    :return: (补全后query, 当前实体, 是否发生省略补全)
    """
    new_entity = extract_company(query)
    entity = new_entity or last_entity
    # 剥离实体、省略触发词与标点，看剩余意图
    remainder = query
    if new_entity:
        remainder = remainder.replace(new_entity, "")
    remainder = re.sub(r"^(那|那么|那末)\s*", "", remainder)
    remainder = re.sub(r"(呢|怎么样|如何)\s*[？?]?$", "", remainder)
    remainder = remainder.strip(" ，,。.？?！!；:：")

    if remainder:
        # 当前 query 自带意图，直接组合
        filled = f"{entity}{remainder}"
        return filled, entity, True
    # 纯省略：继承上一轮意图
    if entity and last_intent:
        filled = f"{entity}{last_intent}"
        return filled, entity, True
    return query, entity, False


class QueryUnderstanding:
    """Query 理解器：维护历史实体与意图，完成指代消解与省略补全。"""

    def __init__(self) -> None:
        """初始化（无状态，状态由 dialogue_manager 传入）。"""
        pass

    def rewrite(self, query: str, history_entities: List[str],
                last_intent: str, last_entity: str) -> ParsedQuery:
        """执行完整 Query 理解流水线。

        :param query: 用户原始问题
        :param history_entities: 历史话题实体（最新在前）
        :param last_intent: 上一轮意图
        :param last_entity: 上一轮实体
        :return: 解析后的结构化结果
        """
        # ① 术语归一
        normalized = normalize_terms(query)

        # ② 抽取当前实体
        current_entity = extract_company(normalized) or ""

        # ③ 指代消解（仅在当前无明确实体且含指代词时触发）
        has_coref = False
        if not current_entity and any(p in normalized for p in PRONOUNS):
            normalized, has_coref = resolve_coreference(normalized, history_entities)
            current_entity = extract_company(normalized) or ""

        # ④ 省略补全（“那 X 呢”类）
        has_ellipsis = False
        if is_ellipsis_query(normalized) and last_intent:
            normalized, ent, has_ellipsis = fill_ellipsis(
                normalized, last_intent, last_entity)
            current_entity = ent or current_entity

        # ⑤ 若当前仍无实体且历史有实体，默认继承上一轮实体（兜底）
        if not current_entity and last_entity:
            current_entity = last_entity

        # ⑥ 抽取意图（用于下一轮的省略补全）
        intent = extract_intent(normalized, current_entity)

        return ParsedQuery(
            original=query,
            rewritten=normalized,
            entity=current_entity,
            intent=intent,
            has_coreference=has_coref,
            has_ellipsis=has_ellipsis,
        )
