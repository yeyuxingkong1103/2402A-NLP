# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
Query 改写模块：基于对话历史，把用户的追问/指代/切换问题改写成完整独立问题，
再交给下游检索与回答模块。这是 Query 理解优化的核心。

策略：
    1. 规则前置：
       - 识别"那XX呢？"模式 → 直接替换主语，复用上一轮的谓语；
       - 已有完整公司实体 → 不改写；
    2. LLM 改写：
       - 把历史 N 轮 + 当前问题交给 DeepSeek，输出独立完整问题；
       - 使用低温（0.0）保证稳定；
       - 启用 LRU 缓存避免重复改写；
    3. 失败回退：改写失败/返回为空 → 用启发式"最后实体 + 当前问题"拼接。

支持的 4 类追问模式：
    - 指代消解："他参与的哪个工程..." → "武汉兴图新科电子股份有限公司参与的..."
    - 主语继承："这个公司的法定代表人是谁？" → 继承上一论实体
    - 主语切换："那武汉力源信息技术股份有限公司呢？" → 复用上一轮的谓语
    - 顺承追问："报告期内，来自军用领域的收入分别是多少？" → 加上主语
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import httpx

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from logger import get_logger
from session_manager import (
    Session, detect_query_type, extract_entities,
    get_history_for_rewrite, update_session_entity,
)

logger = get_logger(__name__)


# ==================== 改写结果缓存 ====================
class RewriteCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, history_key: str) -> str:
        return hashlib.md5(f"{query}||{history_key}".encode("utf-8")).hexdigest()

    def get(self, query: str, history_key: str) -> Optional[str]:
        k = self._key(query, history_key)
        if k in self.cache:
            self.hits += 1
            self.cache.move_to_end(k)
            return self.cache[k]
        self.misses += 1
        return None

    def set(self, query: str, history_key: str, value: str):
        k = self._key(query, history_key)
        if k in self.cache:
            self.cache.move_to_end(k)
        self.cache[k] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get_info(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 2) if total > 0 else 0,
        }


_rewrite_cache = RewriteCache()


def get_rewrite_cache_info() -> dict:
    return _rewrite_cache.get_info()


# ==================== Prompt 模板 ====================
REWRITE_SYSTEM_PROMPT = """你是一个专业的 Query 理解优化助手。任务：根据对话历史，把用户最新的问题改写成一个完整、独立、无歧义的问题，便于检索系统理解。

改写规则：
1. 如果当前问题已经包含明确的主体（公司名、人名等），不需要改写，直接返回原问题；
2. 如果当前问题含有"他/她/它/这个/那个/该/其/这家/那家"等指代词，请将其替换为历史中最近提到的明确主体；
3. 如果当前问题是"那XX呢？"的形式，意思是"把上一轮的谓语应用到XX上"，请组合上一轮的谓语与新的主语XX；
4. 如果当前问题省略了主语但是显然承接上文，请补全主语；
5. 不要添加额外信息，不要改变问题的语义；
6. 只输出改写后的问题本身，不要任何解释、引号或前缀。"""


def _build_rewrite_prompt(query: str, history_pairs: List[Tuple[str, str]]) -> str:
    """构建改写 prompt。"""
    if not history_pairs:
        return f"对话历史为空。\n\n当前问题：{query}\n\n请输出改写后的问题："

    lines = ["对话历史："]
    for i, (q, a) in enumerate(history_pairs, 1):
        # 答案截断，避免 prompt 过长
        a_short = a[:120] + "..." if len(a) > 120 else a
        lines.append(f"第{i}轮 用户：{q}")
        lines.append(f"第{i}轮 助手：{a_short}")
    lines.append(f"\n当前问题：{query}")
    lines.append("\n请输出改写后的问题（独立完整、含明确主体）：")
    return "\n".join(lines)


# ==================== LLM 调用 ====================
def _call_llm_for_rewrite(prompt: str) -> str:
    """调用 DeepSeek API 进行改写。低温、小 token，保证速度。"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ==================== 启发式回退 ====================
def _fallback_rewrite(query: str, session: Session) -> str:
    """LLM 失败时的兜底改写：简单拼接最近实体。"""
    if session.last_entity and session.last_entity not in query:
        # 处理"那XX呢"切换
        import re
        m = re.match(r"^(那|那么|而)\s*(.+?)\s*呢[?？]?$", query)
        if m and session.history:
            new_entity = m.group(2).strip()
            # 找最近一轮用户问题作为模板
            for msg in reversed(session.history):
                if msg["role"] == "user":
                    prev_q = msg["content"]
                    # 把 prev_q 中的旧实体替换为新实体
                    old = session.last_entity
                    if old and old in prev_q:
                        return prev_q.replace(old, new_entity)
                    return f"{new_entity}{prev_q}" if not prev_q.startswith(new_entity) else prev_q
        # 简单加主语
        return f"{session.last_entity}{query}"
    return query


# ==================== 主入口 ====================
def rewrite_query(query: str, session: Session, use_llm: bool = True) -> dict:
    """根据会话历史改写用户问题。

    Args:
        query: 用户原始问题
        session: 当前会话
        use_llm: 是否使用 LLM 改写（默认 True，关闭则只用启发式）

    Returns:
        dict: {
            "original": 原问题,
            "rewritten": 改写后问题,
            "query_type": 问题类型,
            "rewritten_by": "none|llm|cache|fallback",
            "elapsed_sec": 改写耗时
        }
    """
    t0 = time.time()
    q_type = detect_query_type(query)

    # 已经是独立问题，无需改写
    if q_type == "standalone":
        # 即使不改写也要更新实体
        update_session_entity(session, query)
        return {
            "original": query,
            "rewritten": query,
            "query_type": q_type,
            "rewritten_by": "none",
            "elapsed_sec": round(time.time() - t0, 3),
        }

    # 构造历史指纹用于缓存
    history_pairs = get_history_for_rewrite(session, max_turns=3)
    history_key = "|".join(f"{q[:30]}:{a[:30]}" for q, a in history_pairs)

    # 查缓存
    cached = _rewrite_cache.get(query, history_key)
    if cached:
        logger.info(f"[Rewrite] 缓存命中：'{query[:30]}' -> '{cached[:40]}'")
        # 更新实体
        new_ents = extract_entities(cached)
        if new_ents:
            session.last_entity = new_ents[0]
        return {
            "original": query,
            "rewritten": cached,
            "query_type": q_type,
            "rewritten_by": "cache",
            "elapsed_sec": round(time.time() - t0, 3),
        }

    # 规则快速路径：主语切换 "那XX呢？"
    if q_type == "switch":
        import re
        m = re.match(r"^(那|那么|而)\s*(.+?)\s*呢[?？]?$", query)
        if m:
            new_entity = m.group(2).strip()
            # 补全"公司"后缀：如果 new_entity 没有后缀但 last_entity 是股份有限公司，尝试补全
            if not re.search(r"(股份有限公司|有限责任公司|有限公司|集团|公司)$", new_entity):
                if session.last_entity:
                    suffix_m = re.search(r"(股份有限公司|有限责任公司|有限公司|集团|公司)$", session.last_entity)
                    if suffix_m:
                        new_entity_full = new_entity + suffix_m.group(1)
                    else:
                        new_entity_full = new_entity
                else:
                    new_entity_full = new_entity
            else:
                new_entity_full = new_entity

            # 找上一轮用户问题作为模板
            prev_q = None
            for msg in reversed(session.history):
                if msg["role"] == "user":
                    prev_q = msg["content"]
                    break
            if prev_q and session.last_entity and session.last_entity in prev_q:
                rewritten = prev_q.replace(session.last_entity, new_entity_full)
                session.last_entity = new_entity_full
                _rewrite_cache.set(query, history_key, rewritten)
                logger.info(f"[Rewrite] 主语切换：'{query}' -> '{rewritten}'")
                return {
                    "original": query,
                    "rewritten": rewritten,
                    "query_type": q_type,
                    "rewritten_by": "rule_switch",
                    "elapsed_sec": round(time.time() - t0, 3),
                }

    # LLM 改写
    if use_llm:
        try:
            prompt = _build_rewrite_prompt(query, history_pairs)
            rewritten = _call_llm_for_rewrite(prompt)
            # 清洗：去掉可能的引号、前后缀
            rewritten = rewritten.strip().strip('"').strip("'").strip('“”‘’')
            # 去掉模型可能输出的"改写后的问题："等前缀
            for prefix in ["改写后的问题：", "改写后的问题:", "改写结果：", "改写结果:", "问题：", "问题:"]:
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()
            if rewritten and len(rewritten) >= 4:
                _rewrite_cache.set(query, history_key, rewritten)
                # 更新实体
                new_ents = extract_entities(rewritten)
                if new_ents:
                    session.last_entity = new_ents[0]
                logger.info(f"[Rewrite] LLM 改写：'{query[:30]}' -> '{rewritten[:50]}'")
                return {
                    "original": query,
                    "rewritten": rewritten,
                    "query_type": q_type,
                    "rewritten_by": "llm",
                    "elapsed_sec": round(time.time() - t0, 3),
                }
        except Exception as e:
            logger.warning(f"[Rewrite] LLM 改写失败（{e}），使用启发式回退")

    # 启发式回退
    rewritten = _fallback_rewrite(query, session)
    new_ents = extract_entities(rewritten)
    if new_ents:
        session.last_entity = new_ents[0]
    logger.info(f"[Rewrite] 启发式改写：'{query[:30]}' -> '{rewritten[:50]}'")
    return {
        "original": query,
        "rewritten": rewritten,
        "query_type": q_type,
        "rewritten_by": "fallback",
        "elapsed_sec": round(time.time() - t0, 3),
    }
