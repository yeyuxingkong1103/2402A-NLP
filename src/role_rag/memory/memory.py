"""记忆系统。

分两层，各自物理隔离，互不污染：

**短期记忆（Redis）**
* List ``sess:<sid>:msgs``   —— 最近若干轮原文，直接进提示词；
* String ``sess:<sid>:summary`` —— 会话摘要，超过阈值后由本地模型压缩生成；
* Hash ``sess:<sid>:meta``   —— 会话元信息（轮数、角色、时间）。

**长期记忆（Milvus ``role_memory``）**
* 规则抽取的用户事实（姓名、目标、偏好等）+ 会话摘要，向量化后写入，
  检索时按 ``user_id`` + 角色做过滤，召回与该用户最相关的历史记忆。
* Redis 侧同步保留最近 50 条事实，避免每次都走向量检索。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Config, get_config
from ..logging_conf import get_logger
from ..models.embedder import get_embedder
from ..store.milvus_store import get_milvus
from ..store.redis_store import get_redis

logger = get_logger(__name__)

# (正则, 记忆类型, 重要度, 输出模板)
FACT_PATTERNS: list[tuple[str, str, int, str]] = [
    (r"(?:请)?记住[：:,，]?\s*([^。！？\n]{2,60})", "explicit", 5, "用户要求记住：{0}"),
    (r"我(?:的名字)?叫\s*([^\s，。！？,.!?]{1,20})", "name", 5, "用户的名字是 {0}"),
    (r"我的名字是\s*([^\s，。！？,.!?]{1,20})", "name", 5, "用户的名字是 {0}"),
    (r"我今年\s*(\d{1,3})\s*岁", "age", 4, "用户年龄 {0} 岁"),
    (r"我住在\s*([^\s，。！？,.!?]{2,20})", "location", 3, "用户常住地：{0}"),
    (r"我的(?:目标|计划|打算)是\s*([^。！？\n]{2,60})", "goal", 4, "用户的目标：{0}"),
    (r"我(?:喜欢|偏好|倾向于|更看重|不喜欢)\s*([^。！？\n]{2,40})", "preference", 4, "用户偏好：{0}"),
]

# 角色专属抽取规则
ROLE_FACT_PATTERNS: dict[str, list[tuple[str, str, int, str]]] = {
    "financial_planner": [
        (r"(?:风险承受能力|风险偏好)(?:是|为|：|:)?\s*([^。！？\n]{2,30})", "risk_profile", 5,
         "用户风险偏好：{0}"),
        (r"我(?:每月|每年)(?:可以|能|大概)?(?:结余|存|投(?:入)?)\s*([^。！？\n]{1,20})", "cashflow", 4,
         "用户可投入资金：{0}"),
        (r"我(?:想要|希望)\s*(\d{1,3})\s*年(?:后|之内)?", "horizon", 4, "用户投资期限：{0} 年"),
    ],
    "scientist": [
        (r"我(?:的)?研究方向是\s*([^。！？\n]{2,40})", "research_area", 5, "用户研究方向：{0}"),
        (r"我(?:在)?研究\s*([^。！？\n]{2,40})", "research_area", 4, "用户正在研究：{0}"),
        (r"我(?:是|读)\s*(博士|硕士|本科|博后|研究员)", "identity", 3, "用户身份：{0}"),
    ],
    "lawyer": [
        (r"我(?:遇到|碰到|涉及)(?:了)?\s*([^。！？\n]{2,40})", "case", 5, "用户遇到的问题：{0}"),
        (r"(?:合同|协议)(?:是|签订于|签于)\s*([^。！？\n]{2,30})", "contract", 4, "合同信息：{0}"),
        (r"我(?:是)?(?:劳动仲裁|起诉|起诉中|被起诉)", "procedure", 4, "用户处在诉讼/仲裁阶段"),
    ],
}


@dataclass(slots=True)
class RecallFact:
    """召回的一条长期记忆。"""

    text: str
    kind: str = "fact"
    importance: int = 3
    score: float = 0.0
    source: str = "milvus"
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "importance": self.importance,
            "score": round(self.score, 6),
            "source": self.source,
        }


@dataclass(slots=True)
class MemoryBundle:
    """一次提问所需的全部记忆上下文。"""

    recent: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    facts: list[RecallFact] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "summary": self.summary,
            "recent_messages": len(self.recent),
            "facts": [item.to_dict() for item in self.facts],
            "profile": self.profile,
        }


class MemoryManager:
    """短期记忆 + 长期记忆的统一入口。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.redis = get_redis(self.config)
        self.milvus = get_milvus(self.config)
        self.embedder = get_embedder(self.config)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 读取
    def load(self, user_id: str, role_id: str, session_id: str, query: str = "") -> MemoryBundle:
        turns = int(self.config.get("memory.short_term_turns", 5))
        recent = self.redis.recent_context(session_id, turns)
        summary = self.redis.get_summary(session_id)
        profile = self.redis.profile(user_id)
        facts = self._recall_facts(user_id, role_id, query)
        return MemoryBundle(recent=recent, summary=summary, facts=facts, profile=profile,
                            turns=self.redis.message_count(session_id) // 2)

    def _recall_facts(self, user_id: str, role_id: str, query: str) -> list[RecallFact]:
        """长期记忆召回：Milvus 向量召回 + Redis 最近事实兜底。"""

        limit = int(self.config.get("memory.long_term_top_k", 4))
        min_score = float(self.config.get("memory.long_term_min_score", 0.35))
        collected: list[RecallFact] = []
        seen: set[str] = set()

        if query:
            try:
                vector, sparse = self.embedder.encode_query(query)
                expr = f'user_id == "{user_id}" and (role_id == "{role_id}" or role_id == "*")'
                hits = self.milvus.search_memory(vector.tolist(), sparse, expr, limit)
                for hit in hits:
                    score = float(hit.get("score", 0.0))
                    # Milvus COSINE 相似度；RRF 融合分数则不做阈值裁剪
                    if score < min_score and score <= 1.0:
                        continue
                    text = str(hit.get("text", "")).strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    collected.append(
                        RecallFact(
                            text=text,
                            kind=str(hit.get("kind", "fact")),
                            importance=int(hit.get("importance", 3) or 3),
                            score=score,
                            source="milvus",
                            ts=float(hit.get("ts", 0) or 0),
                        )
                    )
            except Exception as exc:  # pragma: no cover - 记忆失败不应影响问答
                logger.warning("长期记忆向量召回失败：%s", exc)

        for item in self.redis.recent_facts(user_id, role_id, limit=limit):
            text = str(item.get("text", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            collected.append(
                RecallFact(
                    text=text,
                    kind=str(item.get("kind", "fact")),
                    importance=int(item.get("importance", 3) or 3),
                    score=1.0,
                    source="redis",
                    ts=float(item.get("ts", 0) or 0),
                )
            )
        collected.sort(key=lambda item: (-item.importance, -item.score))
        return collected[: max(limit, 1) * 2]

    # ------------------------------------------------------------------ 抽取
    def extract_facts(self, text: str, role_id: str) -> list[dict[str, Any]]:
        """规则式事实抽取（确定性、零成本、可解释）。"""

        if not bool(self.config.get("memory.extract_rules", True)):
            return []
        patterns = list(FACT_PATTERNS) + list(ROLE_FACT_PATTERNS.get(role_id, []))
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pattern, kind, importance, template in patterns:
            for match in re.finditer(pattern, text or ""):
                value = match.group(1).strip() if match.groups() else ""
                sentence = template.format(value) if "{" in template and value else template
                if sentence in seen:
                    continue
                seen.add(sentence)
                facts.append({"kind": kind, "text": sentence, "importance": importance})
        return facts[:5]

    # ------------------------------------------------------------------ 写入
    def remember(
        self,
        user_id: str,
        role_id: str,
        session_id: str,
        question: str,
        answer: str = "",
    ) -> dict[str, Any]:
        """一轮问答结束后写回记忆。"""

        if not bool(self.config.get("memory.write_back", True)):
            return {"facts": 0, "written": 0}

        facts = self.extract_facts(question, role_id)
        if facts:
            self.redis.remember_facts(user_id, role_id, facts)
            written = self._write_facts_to_milvus(user_id, role_id, session_id, facts)
        else:
            written = 0

        summary = self.maybe_refresh_summary(user_id, role_id, session_id)
        self.redis.update_profile(user_id, last_role=role_id)
        self.redis.incr_profile(user_id, "questions")
        return {"facts": len(facts), "written": written, "summary_updated": bool(summary),
                "fact_texts": [item["text"] for item in facts]}

    def _write_facts_to_milvus(
        self, user_id: str, role_id: str, session_id: str, facts: Sequence[dict[str, Any]]
    ) -> int:
        try:
            texts = [str(item["text"]) for item in facts]
            dense, sparse = self.embedder.encode(texts)
            rows = [
                {
                    "dense": dense[index].tolist(),
                    "sparse": sparse[index],
                    "text": texts[index],
                    "user_id": user_id,
                    "role_id": role_id,
                    "kind": str(facts[index].get("kind", "fact")),
                    "importance": int(facts[index].get("importance", 3)),
                    "ts": int(time.time()),
                    "session_id": session_id,
                }
                for index in range(len(facts))
            ]
            return self.milvus.insert_memories(rows)
        except Exception as exc:  # pragma: no cover
            logger.warning("长期记忆写入失败：%s", exc)
            return 0

    # ------------------------------------------------------------------ 摘要
    def maybe_refresh_summary(self, user_id: str, role_id: str, session_id: str) -> str:
        """当会话轮数达到阈值时刷新摘要（Redis String + Milvus 各存一份）。"""

        threshold = int(self.config.get("memory.summary_trigger_turns", 5))
        messages = self.redis.messages(session_id)
        turns = len([item for item in messages if item.get("role") == "user"])
        if turns < threshold or turns % threshold != 0:
            return ""

        previous = self.redis.get_summary(session_id)
        lines = []
        for item in messages[-threshold * 2:]:
            speaker = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            lines.append(f"{speaker}：{content[:200]}")
        transcript = "\n".join(lines)

        summary = ""
        try:
            from ..models.llm import get_llm

            prompt = (
                "你是会话压缩器。请把下面的对话压缩成不超过 120 字的中文摘要，"
                "只保留用户身份信息、目标、偏好、结论与待办，不要添加任何评论。\n"
                f"{'已有摘要：' + previous if previous else ''}\n对话：\n{transcript}\n摘要："
            )
            result = get_llm(self.config).complete(
                [{"role": "user", "content": prompt}], max_new_tokens=160, temperature=0.0
            )
            summary = result.text.strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("摘要生成失败，退化为拼接：%s", exc)

        if not summary:
            summary = (previous + " " + transcript)[-300:]
        self.redis.set_summary(session_id, summary)

        try:
            dense, sparse = self.embedder.encode([f"[会话摘要] {summary}"])
            self.milvus.insert_memories(
                [
                    {
                        "dense": dense[0].tolist(),
                        "sparse": sparse[0],
                        "text": f"[会话摘要] {summary}",
                        "user_id": user_id,
                        "role_id": role_id,
                        "kind": "summary",
                        "importance": 4,
                        "ts": int(time.time()),
                        "session_id": session_id,
                    }
                ]
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("摘要写入长期记忆失败：%s", exc)
        logger.info("会话摘要已刷新：user=%s session=%s", user_id, session_id)
        return summary

    # ------------------------------------------------------------------ 管理
    def clear(self, user_id: str, role_id: str | None = None) -> dict[str, Any]:
        """清空某个用户的长期记忆（可选限定角色）。"""

        expr = f'user_id == "{user_id}"'
        if role_id:
            expr += f' and role_id == "{role_id}"'
        removed = self.milvus.delete_memories(expr)
        if role_id:
            self.redis.client.delete(self.redis.key("user", user_id, "facts", role_id))
        else:
            for key in self.redis.client.scan_iter(match=self.redis.key("user", user_id, "facts", "*")):
                self.redis.client.delete(key)
        return {"deleted": removed, "user_id": user_id, "role_id": role_id or "*"}

    def inspect(self, user_id: str, role_id: str) -> dict[str, Any]:
        """查看某用户在某角色下的记忆（前端记忆面板）。"""

        facts = self.redis.recent_facts(user_id, role_id, limit=20)
        expr = f'user_id == "{user_id}" and role_id == "{role_id}"'
        try:
            vector, sparse = self.embedder.encode_query("用户信息 目标 偏好")
            hits = self.milvus.search_memory(vector.tolist(), sparse, expr, 10)
        except Exception:
            hits = []
        return {
            "user_id": user_id,
            "role_id": role_id,
            "profile": self.redis.profile(user_id),
            "redis_facts": facts,
            "milvus_memories": [
                {"text": hit.get("text"), "kind": hit.get("kind"), "score": round(float(hit.get("score", 0)), 4)}
                for hit in hits
            ],
        }


_memory: MemoryManager | None = None
_memory_lock = threading.Lock()


def get_memory(config: Config | None = None) -> MemoryManager:
    global _memory
    with _memory_lock:
        if _memory is None:
            _memory = MemoryManager(config)
        return _memory
