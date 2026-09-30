"""src/memory/long_term.py —— 长期记忆（Milvus 向量检索）。

在链路中的位置：
    src/online/chain.py → 【本文件】 → Milvus 长期记忆集合
                                      → src/offline/embedder.py（向量化）
                                      → src/offline/milvus_store.py（检索）

与短期记忆（src/memory/short_term.py）的根本区别：
    短期  按**时间**取最近 N 轮 —— 解决"刚才说过什么"
    长期  按**语义相似度**跨会话召回 —— 解决"他之前提到过什么相关的"

举例说明这个区别的价值：
    用户三周前说过"我对花生过敏"。今天新开一个会话问"这道菜我能吃吗"，
    短期记忆里什么都没有，但长期记忆能按语义把那条过敏史捞出来。

实现上它复用了知识库那一整套存取设施：
    同一个向量化器、同一个 MilvusStore、同样的 dense+sparse 双路召回。
    区别只是集合名不同（milvus_memory_collection）和过滤条件不同 ——
    记忆本身就是"属于某个用户的一段文本"，和知识片段在数据结构上是一回事。

全程 fail-open（每个函数都 try/except 返回空/False）：
    记忆是增强能力，不能因为 Milvus 没起就让用户没法聊天。
"""
from __future__ import annotations

import time
from typing import Any

from configs.settings import get_settings
from src.offline.embedder import embed
from src.offline.milvus_store import store


def _expr(user_id: int, role_id: str, tenant_id: str) -> str:
    """构造长期记忆检索的过滤表达式。

    参数：
        user_id / role_id / tenant_id: 三层隔离维度
    返回：
        Milvus 过滤表达式字符串。

    三个条件缺一不可，各自防住一类串数据：
        user_id   防止 A 的记忆被 B 检索到（隐私）
        role_id   防止"医生"角色下的健康隐私串到"虚拟朋友"角色里（人格一致性）
        tenant_id 防止跨租户越界（数据安全）

    与 src/online/retriever.py 的 _expr 不同，这里**没有**做转义：
        user_id 是 int，本就无需转义；role_id / tenant_id 在此场景下由内部代码传入。
        但这种"内部可信"的假设是脆弱的 —— 如果将来 role_id 能被用户自定义并传到这里，
        就必须补上转义（可参考 milvus_store._escape 的写法）。
    """
    return f'user_id == "{user_id}" and role_id == "{role_id}" and tenant_id == "{tenant_id}"'


def ensure_memory_collection() -> str:
    """确保长期记忆集合存在且已加载。

    返回：
        实际使用的集合名。

    单独建一个集合（而不是和知识库共用一个）：
        两者的生命周期和用途完全不同 ——
        知识库是人工整理、长期稳定的；记忆是随对话持续增长、需要按用户过滤的。
        混在一起既不利于各自的索引调优，也会让"删除某用户所有记忆"变得麻烦。
    """
    settings = get_settings()
    return store.ensure_collection(settings.milvus_memory_collection)


def remember_turn(user_id: int, role_id: str, session_id: int, text: str, tenant_id: str = "default") -> bool:
    """把一轮用户发言写入长期记忆。

    参数：
        user_id / role_id / session_id / tenant_id: 归属信息
        text: 用户发言原文
    返回：
        写入成功 True；内容过短或写入异常返回 False。

    两个设计决定：
        1. 太短不入库（< 8 字符）："嗯""好的""谢谢"这类短语没有记忆价值，
           入库只会稀释后续的语义检索质量
        2. 只记用户说的话、不记模型的回答（由 chain.py 调用时决定）：
           需要被记住的是用户的偏好和处境，模型每轮的组织措辞没有记忆价值

    复用知识片段的记录结构（content/summary/page/doc_id/doc_source…）：
        这样就能直接用 store.insert_chunks 和 hybrid_search 这套现成的设施，
        不必为记忆单独写一套存取逻辑。
        几个字段的填法：
            doc_id    借用 session_id —— 记忆不属于某个文档，用会话 id 占位
            doc_source 记成 "memory:{user_id}:{session_id}" —— 可读的来源标识，
                       排查问题时一眼能看出这条记忆来自哪个会话
            page      -1 —— 记忆没有页码概念

    这里用 insert_chunks（而非 upsert）：
        允许同一条内容被重复记住 —— 用户在不同时间说同一句话，
        是两个独立的记忆事件，都应该留下记录。
    """
    text = (text or "").strip()
    if len(text) < 8:
        return False
    try:
        settings = get_settings()
        vectors = embed([text])
        now = int(time.time())
        record = {"dense_vector": vectors.dense[0], "sparse_vector": vectors.sparse[0], "content": text[:8192], "summary": text[:512], "parent_id": 0, "doc_id": int(session_id), "role_id": role_id, "tenant_id": tenant_id, "doc_source": f"memory:{user_id}:{session_id}", "page": -1, "create_time": now, "update_time": now, "user_id": str(user_id)}
        store.insert_chunks([record], settings.milvus_memory_collection)
        return True
    except Exception:
        # fail-open：Milvus 不可用、向量化失败都不该让这一轮对话失败
        return False


def retrieve_memory(user_id: int, role_id: str, query: str, tenant_id: str = "default", limit: int = 4) -> list[dict[str, Any]]:
    """按当前问题检索该用户在该角色下的长期记忆。

    参数：
        user_id / role_id / tenant_id: 隔离维度
        query: 当前用户消息（用它去找语义相关的历史记忆）
        limit: 返回条数，默认 4
    返回：
        [{"text": 记忆内容, "score": 相关性分数}, ...]；异常时返回空列表。

    为什么用"当前问题"而不是"整个会话"去检索：
        检索需要的是一个具体的查询向量。用当前问题最直接 ——
        用户正在关注什么，就把相关的历史记忆捞出来。
        这也天然实现了"按需回忆"：不相关的记忆不会挤进提示词。

    先 ensure_memory_collection 再检索：
        集合必须处于已加载状态才能检索。首次对话时集合可能刚被创建，
        不先 ensure 就会报 "collection not loaded"。

    limit 默认只取 4 条：
        长期记忆是"补充参考"而非"事实依据"，
        取太多会挤占提示词预算，还可能把不相关的旧记忆误当成本轮上下文。

    score 的取值用 hit.get("distance", hit.get("rrf_score", 0))：
        兼容不同返回结构 —— 纯向量检索返回 distance，
        而 hybrid_search 融合后的结果带的是 rrf_score。
    """
    try:
        settings = get_settings()
        ensure_memory_collection()
        vectors = embed([query])
        hits = store.hybrid_search(vectors.dense[0], vectors.sparse[0], limit=limit, expression=_expr(user_id, role_id, tenant_id), collection=settings.milvus_memory_collection)
        return [{"text": (hit.get("entity") or hit).get("content", ""), "score": hit.get("distance", hit.get("rrf_score", 0))} for hit in hits]
    except Exception:
        # fail-open：记忆检索失败就当作"这次没有相关记忆"，不影响正常对话
        return []
