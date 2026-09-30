# -*- coding: utf-8 -*-
"""长期记忆：把跨会话的对话要点存入向量库，按语义召回。

与短期记忆的分工
----------------
    短期（memory.py, Redis LIST）—— **本次会话**最近 N 轮，精确、有界、会过期
    长期（本模块, 向量库）        —— **跨会话**的相关往事，按语义召回、按用户隔离

举例：用户上周聊过「我对 ACEI 类药过敏」，这周新开会话问「推荐什么降压药」——
短期记忆是空的（新会话），长期记忆能召回上周那句话。

Collection 设计
---------------
独立集合 `kb_memory`，与知识库集合分开，避免污染检索。
每条记忆 = **一轮问答**（问+答拼成一段文本），带 `user_id` 做隔离。

为什么按 user_id 过滤而不是分集合
---------------------------------
每用户一个 collection 会让集合数量随用户数增长，且 Milvus 对集合数量有限制。
用标量字段过滤是标准做法。

⚠️ 依赖 Milvus
--------------
长期记忆走 `VECTOR_STORE` 指定的向量库，但**实现只覆盖 Milvus**
（默认后端）。`VECTOR_STORE=qdrant` 时本模块自动禁用并记一条 warning ——
而不是假装能用然后静默返回空。
"""
import hashlib
import time

from ..core import config
from ..core.logging import get_logger
from . import embed

log = get_logger("long_memory")

COLLECTION = "kb_memory"

# 记忆条目的文本上限（问答拼接过长时截断，避免单条占满上下文）
MAX_MEMORY_CHARS = 1200
# 召回时注入 prompt 的最大条数
RECALL_K = 3

_enabled_warned = False


def _enabled() -> bool:
    """长期记忆是否可用（需要 Milvus）。"""
    global _enabled_warned
    if config.VECTOR_STORE != "milvus":
        if not _enabled_warned:
            log.warning(
                "长期记忆需要 Milvus（当前 VECTOR_STORE=%s），已禁用", config.VECTOR_STORE
            )
            _enabled_warned = True
        return False
    return True


def _mid(user_id: int, conversation_id: int, text: str) -> int:
    """稳定主键：同一条记忆重复写入为覆盖。

    ⚠️ 与 milvus_store 一致，必须按位与成**有符号** INT64 —— 直接取 sha1
    前 16 位是 64 位无符号，约半数会超出 Milvus 上限并报
    `DataNotMatchError: Value out of range`。
    """
    raw = f"{user_id}|{conversation_id}|{text}"
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16) & (2 ** 63 - 1)


def _ensure_collection():
    from pymilvus import DataType

    from . import milvus_store
    c = milvus_store.get_client()
    if c.has_collection(COLLECTION):
        return
    schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=config.EMBED_DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("user_id", DataType.INT64)          # 用户隔离
    schema.add_field("conversation_id", DataType.INT64)
    schema.add_field("created_at", DataType.INT64)

    ip = c.prepare_index_params()
    ip.add_index(field_name="dense", index_type="HNSW",
                 metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP")
    c.create_collection(collection_name=COLLECTION, schema=schema, index_params=ip)
    log.info("创建长期记忆集合: %s", COLLECTION)


def remember(user_id: int, conversation_id: int,
             question: str, answer: str) -> bool:
    """把一轮问答写入长期记忆。

    ⚠️ **失败不抛异常**：长期记忆是增强能力，不可用不应影响主对话流程。
    但会记 warning，不静默吞掉。
    """
    if not _enabled():
        return False

    text = f"用户问：{question.strip()}\n助手答：{answer.strip()}"[:MAX_MEMORY_CHARS]
    if len(text) < 10:
        return False

    try:
        from . import milvus_store
        _ensure_collection()

        dense, sparse = embed.encode([text], device="cpu")
        row = {
            "id": _mid(user_id, conversation_id, text),
            "dense": dense[0],
            "sparse": milvus_store._to_milvus_sparse(sparse[0]),
            "text": text,
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            "created_at": int(time.time()),
        }
        milvus_store.upsert_rows(COLLECTION, [row])
        return True
    except Exception as e:
        log.warning("写入长期记忆失败（不影响主流程）: %s", str(e)[:200])
        return False


def recall(user_id: int, query: str,
           k: int = RECALL_K, exclude_conversation: int | None = None) -> list[dict]:
    """按语义召回该用户的相关历史记忆。

    `exclude_conversation`：排除当前会话 —— 当前会话的内容走短期记忆，
    再被长期记忆召回一次是重复注入，会白占上下文。

    返回 [{"text": ..., "conversation_id": ..., "score": ...}]，失败返回 []。
    """
    if not _enabled():
        return []
    if not query or len(query.strip()) < 4:
        return []

    try:
        from pymilvus import AnnSearchRequest, RRFRanker

        from . import milvus_store
        c = milvus_store.get_client()
        if not c.has_collection(COLLECTION):
            return []
        c.load_collection(COLLECTION)

        dense, sparse = embed.encode([query], device="cpu")
        expr = f"user_id == {int(user_id)}"
        if exclude_conversation is not None:
            expr += f" and conversation_id != {int(exclude_conversation)}"

        # ⚠️ 过滤必须挂在**每个 AnnSearchRequest** 上。
        #    `MilvusClient.hybrid_search(filter=...)` 会被 `**kwargs` 静默吞掉，
        #    传了等于没传 —— 实测表现为 `recall(user_id=999999)` 返回了别的用户的
        #    记忆（用户隔离失效）。此坑在 milvus_store 与本文件各踩过一次。
        reqs = [
            AnnSearchRequest(data=[dense[0]], anns_field="dense",
                             param={"metric_type": "COSINE"}, limit=k * 3,
                             expr=expr),
            AnnSearchRequest(data=[milvus_store._to_milvus_sparse(sparse[0])],
                             anns_field="sparse", param={"metric_type": "IP"},
                             limit=k * 3, expr=expr),
        ]
        res = c.hybrid_search(
            collection_name=COLLECTION, reqs=reqs, ranker=RRFRanker(),
            limit=k, output_fields=["text", "conversation_id", "created_at"],
        )
        out = []
        for group in res:
            for h in group:
                ent = h.get("entity", {}) or {}
                if ent.get("text"):
                    out.append({
                        "text": ent["text"],
                        "conversation_id": ent.get("conversation_id"),
                        "created_at": ent.get("created_at"),
                        "score": round(float(h.get("distance", 0.0)), 6),
                    })
        if out:
            log.info("长期记忆召回 %d 条（user=%s）", len(out), user_id)
        return out
    except Exception as e:
        log.warning("召回长期记忆失败（不影响主流程）: %s", str(e)[:200])
        return []


def forget(user_id: int) -> int:
    """删除某用户的全部长期记忆。返回删除条数（失败返回 -1）。"""
    if not _enabled():
        return -1
    try:
        from . import milvus_store
        c = milvus_store.get_client()
        if not c.has_collection(COLLECTION):
            return 0
        res = c.delete(collection_name=COLLECTION, filter=f"user_id == {int(user_id)}")
        n = int(res.get("delete_count", 0)) if isinstance(res, dict) else 0

        # ⚠️ **必须 flush**（2026-09-20 实测）：Milvus 的删除是**最终一致**的，
        #    不 flush 的话，刚删掉的记录**仍会被搜索返回** ——
        #    用户点了「清除记忆」却还查得到，是可见的隐私问题。
        #    实测：加 flush 前 `forget()` 后仍能搜到该用户的记录。
        #    （与之前 `get_collection_stats` 删后计数不降是同一类「Milvus 延迟可见」现象。）
        try:
            c.flush(COLLECTION)
        except Exception as e:
            log.warning("flush 失败（删除可能延迟生效）: %s", str(e)[:120])

        log.info("已删除用户 %s 的 %d 条长期记忆", user_id, n)
        return n
    except Exception as e:
        log.warning("删除长期记忆失败: %s", str(e)[:200])
        return -1


def count(user_id: int | None = None) -> int:
    if not _enabled():
        return -1
    try:
        from . import milvus_store
        c = milvus_store.get_client()
        if not c.has_collection(COLLECTION):
            return 0
        expr = f"user_id == {int(user_id)}" if user_id is not None else ""
        res = c.query(collection_name=COLLECTION, filter=expr,
                      output_fields=["count(*)"])
        if not res:
            return 0
        first = res[0]
        return int(first.get("count(*)", 0)) if isinstance(first, dict) else int(str(first).strip("'\""))
    except Exception:
        return -1


def format_for_prompt(memories: list[dict]) -> str:
    """把召回的记忆渲染成 prompt 片段。无记忆时返回空串（调用方据此跳过注入）。

    ⚠️ 引导语的分寸很重要（2026-09-20 实测调整）
    -------------------------------------------
    初版写的是「仅在与当前问题相关时参考」—— 结果**模型几乎完全忽略这些记忆**：
    用户上一轮交代了「我有糖尿病」，下一轮问「用药注意什么」，
    回答里对糖尿病只字未提（而日志显示记忆确实被召回了）。

    过于保守的措辞会把「可用信息」降级成「可能无关的噪声」。
    现在改为**明确要求结合已了解的用户信息作答**，同时仍禁止把它当引用来源 ——
    这两件事必须分开说：*要用来理解用户*，*不能用来做引用标注*。
    """
    if not memories:
        return ""
    lines = [
        "【你与该用户过往对话中的相关记录】",
        "以下是这位用户以前和你说过的话。**如果其中有关于他的个人情况"
        "（既往病史、正在服用的药物、过敏史、偏好、已确认过的事实），"
        "请在回答时结合这些信息**，例如据此调整建议的适用范围或提醒注意事项。",
        "",
    ]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m['text']}")
    lines += [
        "",
        "（注意：以上是**对话历史**，不是本轮检索到的知识库资料。"
        "不要为它们标注 [n] 引用编号。）",
    ]
    return "\n".join(lines)
