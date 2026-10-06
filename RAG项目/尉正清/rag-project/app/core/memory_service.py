# app/core/memory_service.py
"""双层记忆：短期（Redis）+ 长期（Milvus）。

短期：最近 N 轮对话原文，存在 Redis List，读写都是 O(1)。
长期：短期窗口滑出的旧对话，先由大模型压缩成摘要，再连同向量写入 Milvus，
      按 user_id + role_id 隔离，每轮按当前问题语义召回最相关的若干条。

这样既保证「最近说过的话」一字不差，又能让「很久以前说过的关键信息」在
需要时被重新想起来，而不是简单丢弃。
"""
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from app.config import settings
from app.db import redis_conn
from app.db.milvus_conn import expr_eq, get_milvus
from app.config.components import get_embeddings, get_llm
import logging

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(self):
        self.milvus = get_milvus()
        self.collection = settings.MILVUS_MEMORY_COLLECTION
        self.short_turns = settings.SHORT_TERM_TURNS
        self.long_top_k = settings.LONG_TERM_TOP_K
        self.summary_trigger = settings.MEMORY_SUMMARY_TRIGGER
        self._ready = False

    def _ensure(self):
        if not self._ready:
            self.milvus.ensure_memory_collection(self.collection)
            self._ready = True

    # ---------------- 读 ----------------
    def get_context(self, user_id: str, role_id: str, session_id: str,
                    question: str) -> Tuple[List[str], List[str]]:
        """返回 (短期记忆行, 长期记忆摘要列表)。"""
        history = redis_conn.get_recent_turns(
            user_id, role_id, session_id, self.short_turns)
        memories = self.recall(user_id, role_id, question)
        return history, memories

    def recall(self, user_id: str, role_id: str, question: str) -> List[str]:
        """按语义召回该用户在该角色下的长期记忆。"""
        if not question:
            return []
        try:
            self._ensure()
            if self.milvus.count(self.collection) == 0:
                return []
            vec = get_embeddings().embed_query(question)
            expr = expr_eq(user_id=user_id, role_id=role_id)
            hits = self.milvus.dense_search(
                self.collection, vec, expr=expr,
                limit=self.long_top_k, output_fields=["summary", "created_at"])
            # 不设相似度阈值：实测在这个嵌入空间里，相关与无关问句的
            # 稠密相似度几乎重叠（0.44 vs 0.43），任何固定阈值都会
            # 静默丢掉真正相关的记忆。召回的是用户自己的历史，
            # 多带一条的代价远小于漏掉一条，因此按 top-k 取用即可。
            return [h.get("summary", "") for h in hits if h.get("summary")]
        except Exception as e:
            logger.warning("召回长期记忆失败: %s", e)
            return []

    # ---------------- 写 ----------------
    def save_turn(self, user_id: str, role_id: str, session_id: str,
                  question: str, answer: str) -> None:
        """写入一轮对话；短期窗口滑出的旧记录归档进长期记忆。"""
        try:
            overflow = redis_conn.push_turn(user_id, role_id, session_id,
                                            question, answer,
                                            keep=self.short_turns)
            if not overflow:
                return
            # 每轮滑出的量是固定的（约 2 行），先攒进缓冲区，
            # 够一批再摘要，避免为两三句话就调用一次大模型
            pending = redis_conn.add_pending(user_id, role_id, session_id, overflow)
            if pending >= self.summary_trigger:
                self._archive(redis_conn.take_pending(user_id, role_id, session_id),
                              user_id, role_id)
        except Exception as e:
            logger.warning("保存对话记忆失败: %s", e)

    def _archive(self, lines: List[str], user_id: str, role_id: str) -> None:
        """把滑出短期窗口的对话摘要后写入 Milvus。"""
        text = "\n".join(lines)
        if len(text) < 80:
            return
        summary = get_llm().summarize(text, max_chars=300)
        if not summary:
            return
        self.store(summary, user_id, role_id)
        logger.info("归档长期记忆: %s 行 / %s 字 -> 摘要 %s 字",
                    len(lines), len(text), len(summary))

    def store(self, summary: str, user_id: str, role_id: str) -> None:
        """把一条记忆摘要写入 Milvus。"""
        try:
            self._ensure()
            emb = get_embeddings()
            dense, sparse = emb.embed_both([summary])
            row = {
                "id": uuid.uuid4().hex,
                "user_id": user_id,
                "role_id": role_id,
                "summary": summary,
                "vector": dense[0],
                "sparse_vector": sparse[0],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.milvus.insert(self.collection, [row])
        except Exception as e:
            logger.warning("写入长期记忆失败: %s", e)

    def clear(self, user_id: str, role_id: str, session_id: str) -> None:
        redis_conn.clear_history(user_id, role_id, session_id)

    def stats(self, user_id: str, role_id: str) -> Dict[str, int]:
        try:
            self._ensure()
            rows = self.milvus.query(
                self.collection, expr_eq(user_id=user_id, role_id=role_id),
                output_fields=["id"], limit=1000)
            return {"long_term_count": len(rows)}
        except Exception:                                   # pragma: no cover
            return {"long_term_count": 0}


# ============================================================
# 短期记忆 -> LangChain 消息对象
# ============================================================
def _line_to_message(line: str):
    """把「用户：xxx / 助手：xxx」还原成 LangChain 消息对象。

    rag_service 组装提示词时用它把 Redis 里的历史转成 messages，
    再交给 chain_service 的 ChatPromptTemplate。
    """
    from langchain_core.messages import AIMessage, HumanMessage
    line = (line or "").strip()
    if line.startswith("用户："):
        return HumanMessage(content=line[3:].strip())
    if line.startswith("助手："):
        return AIMessage(content=line[3:].strip())
    return HumanMessage(content=line)


_memory: MemoryService = None


def get_memory() -> MemoryService:
    global _memory
    if _memory is None:
        _memory = MemoryService()
    return _memory
