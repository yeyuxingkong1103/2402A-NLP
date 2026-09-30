# -*- coding: utf-8 -*-
"""长期记忆（跨会话沉淀）：接 **Milvus 独立 collection** ``legal_rag_memory``。

设计意图：短期记忆（Redis 滑窗）达到阈值后，调用 LLM 把一段对话总结成 Summary，
把 Summary 向量化后写入**独立的**向量 collection，实现跨会话的用户画像沉淀。

为什么用独立 collection
-----------------------
1. 长期记忆与知识库的**生命周期完全不同**：知识库可整批重建（rebuild），
   用户记忆不能被人重跑索引时误删；
2. 混在一个 collection 里，检索与删除就得靠 ``where`` 过滤兜底，容易误伤；分开存互不影响；
3. 仍然遵守红线：**只新建** ``legal_rag_memory``，绝不触碰既有的
   ``user_long_term_memory`` / ``user_long_term_memory_bge`` / ``ragas_test_collection``。

默认关闭（``LONGTERM_ENABLED=false``）：关闭时 ``enabled=False``，``remember()`` 直接返回
None，不产生任何写入，也不额外建连接。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..embedding.base import Embedder
from ..schemas import Chunk, Message, SearchHit
from ..store.base import VectorStore, build_store
from ..utils import now_ts, stable_id

logger = logging.getLogger(__name__)

#: 长期记忆专用 collection（与知识库 collection 分开）
DEFAULT_MEMORY_COLLECTION = "legal_rag_memory"

#: 记录主键前缀：``mem-<message_id>`` ⇒ **同一 message_id 重复写入落在同一主键上**
#: （Milvus ``upsert`` 语义 = 覆盖）⇒ 天然幂等（`AC-ST-8`）。
MEMORY_ID_PREFIX = "mem-"
#: 记录的可回溯来源前缀（带用户维度；`AC-ST-10` 的三元组从这里与字段里取）。
MEMORY_SOURCE_PREFIX = "memory://u/"
#: 用户维度的**可过滤字段**：落在 `doc_id` 上（Milvus 标量字段，`where` 可等值过滤）。
#: 为什么不用新列：`legal_rag_memory` 复用既有 schema，不新增字段也能做**服务端强制过滤**。
MEMORY_DOC_PREFIX = "memory-u-"


def memory_record_id(message_id: str) -> str:
    """长期记忆记录的主键（由 ``message_id`` 派生 ⇒ 幂等）。"""
    return f"{MEMORY_ID_PREFIX}{message_id}"


def memory_source(user_id: str, session_id: str, message_id: str) -> str:
    """记录来源：``memory://u/<user>/s/<session>/m/<message_id>``（可回溯）。"""
    return f"{MEMORY_SOURCE_PREFIX}{user_id}/s/{session_id}/m/{message_id}"


def memory_doc_id(user_id: str) -> str:
    """用户维度的可过滤取值（服务端从登录态派生，`AC-CI-9`）。"""
    return f"{MEMORY_DOC_PREFIX}{user_id}"


class LongTermMemory:
    def __init__(self, store: VectorStore, embedder: Embedder,
                 enabled: bool = False, threshold: int = 20,
                 *, collection: str = DEFAULT_MEMORY_COLLECTION,
                 config: Any = None) -> None:
        self.store = store
        self.embedder = embedder
        self.enabled = enabled
        self.threshold = max(int(threshold), 1)
        self.collection = collection
        self.config = config

    # ---------- 判定与摘要 ----------
    def should_summarize(self, history: list[Message]) -> bool:
        return self.enabled and len(history) >= self.threshold

    def summarize(self, history: list[Message]) -> str:
        """默认用规则化摘要；接入 LLM 后可替换为模型总结。"""
        lines = []
        for message in history:
            speaker = "用户" if message.role == "user" else "助手"
            lines.append(f"{speaker}：{message.content.strip()[:120]}")
        return "\n".join(lines)[:2000]

    # ---------- B-9：逐条消息写入（只写滚出窗口的旧轮次） ----------
    def remember_messages(self, user_id: str, role_id: str, session_id: str,
                          entries: list[dict]) -> dict:
        """把**滚出上下文窗口**的消息逐条写入长期记忆；返回**可判定**的结果字典。

        * ``entries`` 每项：``{message_id, role, content, created_at, turn_index}``；
        * **幂等**：记录主键 = ``mem-<message_id>``，重复投递走 ``upsert`` 覆盖 ⇒
          同一 message_id 的记录数恒为 1，总量不因重复而增长（`AC-ST-8`）；
        * **可回溯**：``source`` 带 ``u/<user>/s/<session>/m/<message_id>``，
          另有 ``doc_id = memory-u-<user>`` 供服务端强制过滤（`AC-ST-10`/`AC-CI-9`）；
        * **失败不吞**：异常一律转成 ``ok=False`` + WARNING 日志 + 计数器
          （``longterm_write_total{status=failed}``），由调用方决定"暂不裁剪 Redis"
          （`AC-ST-9`）。
        """
        from .. import metrics as M
        try:
            from ..observability import current_request_id
            request_id = current_request_id() or "-"
        except Exception:  # noqa: BLE001 - 观测模块缺席不影响写入
            request_id = "-"

        collection = self.collection
        payload = [dict(item) for item in (entries or []) if str(item.get("message_id") or "")]
        if not self.enabled:
            M.counter("longterm_skip_total").inc(reason="disabled", collection=collection)
            return {"enabled": False, "ok": True, "written": 0, "failed": 0,
                    "skipped": "disabled", "message_ids": []}
        if not payload:
            M.counter("longterm_skip_total").inc(reason="no_rollout", collection=collection)
            return {"enabled": True, "ok": True, "written": 0, "failed": 0,
                    "skipped": "no_rollout", "message_ids": []}

        try:
            chunks = [self._memory_chunk(user_id, role_id, session_id, item)
                      for item in payload]
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.vector = list(vector)
            self.store.upsert(chunks)
        except Exception as exc:  # noqa: BLE001 - 写失败必须可见，不静默
            M.counter("longterm_write_total").inc(status="failed", collection=collection)
            logger.warning("长期记忆写入失败（collection=%s，%d 条，session=%s，user=%s，"
                           "request_id=%s）：%s: %s —— **本轮不裁剪 Redis**，下次问答会自动补写",
                           collection, len(payload), session_id, user_id, request_id,
                           type(exc).__name__, exc)
            return {"enabled": True, "ok": False, "written": 0, "failed": len(payload),
                    "error": f"{type(exc).__name__}: {exc}",
                    "message_ids": [str(i["message_id"]) for i in payload]}

        M.counter("longterm_write_total").inc(status="ok", collection=collection)
        logger.info("长期记忆写入成功（**裁剪前、同请求内**）：collection=%s 条数=%d session=%s "
                    "user=%s request_id=%s message_ids=%s",
                    collection, len(chunks), session_id, user_id, request_id,
                    ",".join(c.id for c in chunks))
        return {"enabled": True, "ok": True, "written": len(chunks), "failed": 0,
                "message_ids": [str(i["message_id"]) for i in payload]}

    def _memory_chunk(self, user_id: str, role_id: str, session_id: str,
                      item: dict) -> Chunk:
        message_id = str(item.get("message_id") or "")
        content = str(item.get("content") or "")
        created_at = float(item.get("created_at") or 0.0) or now_ts()
        role = str(item.get("role") or "")
        # ⚠️ 助手记录的主键取自**配对的用户消息**（``pair_id``），不是回答正文：
        #    回答是生成出来的、逐字会变，用正文派生会让"同一个问题重问"每次多写一条
        #    （实测：同会话重问 4 个问题 ⇒ 记忆里多出 4 条助手记录，长期会稀释召回）。
        #    加 ``pair-`` 前缀是为了**不与用户记录撞主键**（两者都是 ``mem-<...>``）。
        #    ``pair_id`` 缺席时（老数据/没有配对信息）退回原来的内容寻址，行为不变。
        pair_id = str(item.get("pair_id") or "")
        key = f"pair-{pair_id}" if (pair_id and role != "user") else message_id
        meta = {"role": role, "turn_index": int(item.get("turn_index") or 0),
                "message_id": message_id, "pair_id": pair_id}
        chunk = Chunk(
            id=memory_record_id(key),
            text=content,
            source=memory_source(user_id, session_id, message_id),
            doc_id=memory_doc_id(user_id),
            is_parent=False,
            role_id=role_id,
            summary=json.dumps(meta, ensure_ascii=False),
            created_at=created_at,
            updated_at=now_ts(),
        )
        return chunk

    def records_for(self, message_ids: list[str]) -> list[Chunk]:
        """按 ``message_id`` 回读记录（幂等/可回溯证据用；不命中就给空列表）。

        **为什么必须先全量扫、而不能只按记录主键查**：助手记录的 id 是
        ``mem-pair-<问题键>``（见 `_memory_chunk`），**从 message_id 推不出来**；
        只按 ``mem-<message_id>`` 查会让"按 message_id 回读助手记录"**永远查不到**
        （`AC-ST-10` 的可回溯证据就断了）。所以：

        1. **全量扫一遍**，按记录元数据里的 ``message_id`` 建索引 —— 这是**唯一可靠**的口径，
           同时兼容位置寻址/内容寻址两种 ID 口径（单用户记忆量级是几十到几百条，扫得起）；
        2. 还缺的再用单条 ``get(mem-<message_id>)`` 兜一把（某些后端可能不回传元数据）。
        """
        wanted = [str(mid) for mid in (message_ids or []) if str(mid)]
        if not wanted:
            return []
        wanted_set = set(wanted)
        found: dict[str, Chunk] = {}
        try:
            for chunk in self.store.all_chunks():
                meta = {}
                try:
                    meta = json.loads(getattr(chunk, "summary", "") or "{}")
                except (TypeError, ValueError):
                    meta = {}
                mid = str(meta.get("message_id") or "")
                if mid in wanted_set and mid not in found:
                    found[mid] = chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆回读失败（collection=%s）：%s: %s",
                           self.collection, type(exc).__name__, exc)
        for mid in [m for m in wanted if m not in found]:
            try:
                rows = self.store.get(memory_record_id(mid))
            except Exception as exc:  # noqa: BLE001 - 各后端 get 约定不同
                logger.debug("长期记忆回读 get() 不可用（%s: %s）", type(exc).__name__, exc)
                continue
            if isinstance(rows, (list, tuple)):
                for chunk in rows:
                    if getattr(chunk, "id", None):
                        found[mid] = chunk
                        break
            elif rows is not None and getattr(rows, "id", None):
                found[mid] = rows
        return [found[mid] for mid in wanted if mid in found]

    def count_for_user(self, user_id: str) -> int:
        """该用户在长期记忆里的记录数（服务端强制按用户维度过滤；`AC-ST-11`）。"""
        if not self.enabled:
            return 0
        try:
            rows = self.store.all_chunks({"doc_id": memory_doc_id(user_id)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆按用户计数失败（user=%s）：%s: %s",
                           user_id, type(exc).__name__, exc)
            return 0
        return len(rows or [])

    # ---------- 写入 / 召回 ----------
    def remember(self, user_id: str, role_id: str, session_id: str,
                 history: list[Message]) -> Chunk | None:
        if not self.should_summarize(history):
            return None

        summary = self.summarize(history)
        if not summary.strip():
            return None

        chunk = Chunk(
            id=stable_id("longterm", user_id, role_id, session_id, str(len(history))),
            text=summary,
            source=f"memory://{user_id}/{role_id}/{session_id}",
            doc_id=f"memory-{session_id}",
            is_parent=True,
            role_id=role_id,
            summary=summary[:120],
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        chunk.vector = self.embedder.embed_query(summary)
        self.store.upsert([chunk])
        logger.info("长期记忆已沉淀：user=%s role=%s session=%s collection=%s chunk=%s",
                    user_id, role_id, session_id, self.collection, chunk.id)
        return chunk

    def recall(self, query: str, *, user_id: str = "", role_id: str = "",
               top_k: int = 3) -> list[SearchHit]:
        """按语义召回长期记忆（**服务端强制按当前用户过滤**；`AC-ST-11`/`AC-CI-9`）。

        两层都在，缺一不可：

        1. **存储层 filter**（第一道）：``where={"doc_id": memory_doc_id(user_id)}`` ——
           取值由**服务端从登录态派生**，检索请求本身**不接受**任何客户端 filter /
           ``user_id`` 入参（本方法的 ``user_id`` 只由调用方从登录态传进来）；
        2. **应用层复核**（第二道）：命中记录的 ``source`` 必须带
           ``memory://u/<user_id>/`` 前缀，否则丢弃 —— 防止将来有人改坏第一道。
        """
        if not self.enabled:
            return []
        where: dict[str, Any] = {}
        if user_id:
            where["doc_id"] = memory_doc_id(user_id)
        if role_id:
            where["role_id"] = role_id
        vector = self.embedder.embed_query(query)
        hits = self.store.search(vector, top_k=top_k, where=where or None)
        if user_id:
            marker = f"{MEMORY_SOURCE_PREFIX}{user_id}/"
            hits = [hit for hit in hits if hit.chunk.source.startswith(marker)]
        return hits

    def count(self) -> int:
        if not self.enabled:
            return 0
        try:
            return int(self.store.count())
        except Exception as exc:  # noqa: BLE001 - 统计失败不影响主链路
            logger.warning("长期记忆计数失败（collection=%s）：%s", self.collection, exc)
            return 0

    # ---------- 运维 ----------
    def health(self) -> dict:
        """健康信息（既有键 ``enabled`` / ``threshold`` 保留，不破坏旧契约）。"""
        info: dict = {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "collection": self.collection,
            "store": getattr(self.store, "name", "unknown"),
            "backend": type(self.store).__name__,
        }
        if self.enabled:
            try:
                info["chunks"] = int(self.store.count())
            except Exception as exc:  # noqa: BLE001
                info["error"] = f"{type(exc).__name__}: {exc}"
            reason = getattr(self.store, "fallback_reason", "")
            if reason:
                info["degraded"] = True
                info["degraded_reason"] = reason
        return info


def build_longterm_memory(config: Any, embedder: Embedder,
                          fallback_store: VectorStore,
                          *, collection: str = DEFAULT_MEMORY_COLLECTION,
                          ) -> LongTermMemory:
    """按配置构建长期记忆。

    * ``LONGTERM_ENABLED`` 为假 -> 返回 ``enabled=False`` 的空壳，**不建任何连接**；
    * 为真 -> 用 t2 的 ``build_store()`` 给 ``legal_rag_memory`` 建一个**独立** store
      （Milvus 连不上时按既有降级链回落内存库，并在 ``fallback_reason`` 里写明原因）。
    """
    enabled = bool(getattr(config, "longterm_enabled", False))
    threshold = int(getattr(config, "longterm_threshold", 20) or 20)
    if not enabled:
        logger.info("长期记忆未启用（LONGTERM_ENABLED=false），不创建 %s", collection)
        return LongTermMemory(fallback_store, embedder, enabled=False,
                              threshold=threshold, collection=collection, config=config)

    provider = str(getattr(config, "vector_store", "memory"))
    persist_dir = getattr(config, "index_dir", None)
    # ⚠️ 内存兜底后端**不认 collection 名**，只认 ``persist_dir/memory_store.pkl`` ——
    # 与知识库共用同一个 persist_dir 会把长期记忆写进**知识库的同一个 pickle**，
    # 后果是"用户对话文本被当成知识召回"，在跨用户维度上等于把 A 的对话喂给 B。
    # ⇒ 给长期记忆单独一个持久化目录（Milvus 后端本来就靠 collection 名隔开）。
    if provider.strip().lower() in ("memory", "mem", "inmemory", "fallback") and persist_dir:
        persist_dir = str(Path(str(persist_dir)) / "longterm")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
    try:
        store = build_store(provider, persist_dir=persist_dir,
                            collection=collection, config=config, embedder=embedder)
        reason = getattr(store, "fallback_reason", "")
        if store is fallback_store:
            logger.warning("长期记忆与知识库**共用同一个向量库实例**（collection=%s）："
                           "两者会互相污染召回，请检查 VECTOR_STORE/persist_dir 配置",
                           collection)
        if reason:
            logger.warning("长期记忆向量库已降级为 %s（collection=%s）：%s",
                           getattr(store, "name", "?"), collection, reason)
        else:
            logger.info("长期记忆使用向量库 %s（collection=%s，store=%s）",
                        provider, collection, getattr(store, "name", "?"))
    except Exception as exc:  # noqa: BLE001 - 长期记忆失败不能拖垮服务启动
        logger.warning("长期记忆向量库构建失败（%s），回落主向量库：%s: %s",
                       collection, type(exc).__name__, exc)
        store = fallback_store

    return LongTermMemory(store, embedder, enabled=True, threshold=threshold,
                          collection=collection, config=config)


__all__ = ["LongTermMemory", "build_longterm_memory", "DEFAULT_MEMORY_COLLECTION"]
