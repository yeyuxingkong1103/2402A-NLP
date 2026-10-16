# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""问题分流：把用户问题拆成「局部关键词」与「全局关键词」（查询响应流程第 ① 步）。

工单原文口径：
  * **局部关键词** —— 具体内容（实体名、人名、机构名、数值、比例、日期）
    → 走向量库做局部检索；
  * **全局关键词** —— 概念性词（行业、主题、产业链、组织结构…）
    → 沿知识图谱的关系扩展做全局检索。

判定顺序（阈值全部来自 ``cfg.routing`` / ``部署/配置/lightrag.json``）：
  1. **实体词典命中**：问题里出现的图谱节点名（长名优先，避免「力源」吞掉「武汉力源…」），
     由 ``entity_dict_from_graph`` 决定词典是否取自图谱；
  2. **数值命中**：正则抽出金额/比例/年份 → 局部；
  3. **抽象词命中**：``abstract_suffixes``（37 条）与 ``abstract_char_hints``（17 条）→ 全局；
  4. **局部白名单**：``whitelist_local`` 命中强制并入局部；
  5. 两路都命中或都判不出 → ``route_fallback``（默认 ``both``）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Cfg

# 数值/比例/年份（含千分位与百分号）
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# 概念型实体类型：命中这些类型的实体同时计入全局关键词
CONCEPT_TYPES = ("行业", "产品", "财务指标")


@dataclass
class Route:
    """一次分流的完整结果（全部可回放，便于逐题复核）。"""

    question: str
    route: str                                  # local | global | both
    local_keywords: list[str] = field(default_factory=list)
    global_keywords: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    matched_abstract: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "local_keywords": self.local_keywords,
            "global_keywords": self.global_keywords,
            "matched_entities": self.matched_entities,
            "matched_abstract": self.matched_abstract,
            "numbers": self.numbers,
            "reason": self.reason,
        }


def norm_ws(text: str) -> str:
    """去空白 + 全角括号归一（命中统计与分流通用口径）。

    实测依据：PDF 抽取会在数字与术语中插空格（``军队 视频指挥 领域``），
    不归一化会系统性漏匹配（``军用领域`` 20→24 块、``销售处`` 26→29 块）。
    """
    s = re.sub(r"\s+", "", str(text or ""))
    return s.translate(str.maketrans("（）【】", "()[]"))


def build_entity_dict(nodes: Sequence[dict[str, Any]], *, min_len: int = 2, max_len: int = 40) -> dict[str, str]:
    """图谱节点 → ``{实体名: 类型}``（分流用词典）。"""
    out: dict[str, str] = {}
    for node in nodes:
        name = norm_ws(str(node.get("name") or ""))
        if min_len <= len(name) <= max_len:
            out[name] = str(node.get("type") or "")
    return out


def _match_entities(text_norm: str, entity_dict: dict[str, str]) -> list[str]:
    """长名优先匹配；被更长命中包含的短名丢弃（避免「力源」与全称重复计分）。"""
    hits = [name for name in entity_dict if name and name in text_norm]
    hits.sort(key=len, reverse=True)
    pruned: list[str] = []
    for name in hits:
        if any(name != other and name in other for other in hits):
            continue
        pruned.append(name)
    return pruned


def route_question(question: str, entity_dict: dict[str, str], cfg: Cfg) -> Route:
    """按配置口径分流一个问题。"""
    r = cfg.routing
    text = norm_ws(question)

    matched = _match_entities(text, entity_dict) if bool(r.get("entity_dict_from_graph", True)) else []
    numbers = NUMBER_RE.findall(text)
    abstract = [w for w in (r.get("abstract_suffixes") or []) if norm_ws(str(w)) in text]
    abstract += [w for w in (r.get("abstract_char_hints") or []) if norm_ws(str(w)) in text]
    abstract = list(dict.fromkeys(abstract))
    whitelist = [w for w in (r.get("whitelist_local") or []) if norm_ws(str(w)) in text]

    concept_entities = [n for n in matched if entity_dict.get(n) in CONCEPT_TYPES]
    local_kw = list(dict.fromkeys(matched + numbers + whitelist))
    global_kw = list(dict.fromkeys(abstract + concept_entities))
    # 白名单是明确的局部信号：命中则从全局里剔除同名项
    global_kw = [w for w in global_kw if w not in whitelist]

    has_local, has_global = bool(local_kw), bool(global_kw)
    fallback = str(r.get("route_fallback") or "both")
    if has_local and not has_global:
        route = "local"
        reason = "命中实体/数值，未命中抽象词"
    elif has_global and not has_local:
        route = "global"
        reason = "命中抽象词/概念型实体，未命中具体实体"
    elif has_local and has_global:
        route = "both"
        reason = "同时命中具体实体与抽象词"
    else:
        route = fallback
        reason = "未命中任何关键词，按 route_fallback 兜底"

    if int(r.get("global_min_keywords") or 0) > len(global_kw) and route == "global":
        route = fallback
        reason = f"全局关键词不足 global_min_keywords={r.get('global_min_keywords')}，兜底为 {fallback}"

    return Route(question=question, route=route, local_keywords=local_kw,
                 global_keywords=global_kw, matched_entities=matched,
                 matched_abstract=abstract, numbers=numbers, reason=reason)
