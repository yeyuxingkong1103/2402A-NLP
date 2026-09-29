"""src/online/retriever.py —— 在线检索入口（改写 → 缓存 → 双路召回 → 阈值过滤）。

在链路中的位置：
    src/online/chain.py → 【本文件】 → src/offline/milvus_store.py（双路召回 + RRF 融合）
                                     → src/online/query_transform.py（改写）
                                     → src/memory/short_term.py（检索结果缓存）

本文件做了三件让在线检索"又快又稳"的事：
    1. 缓存 —— 同一个角色 + 租户 + 知识库范围 + 改写后查询，结果直接复用
    2. 指代消解 —— 把"它有什么要求"这类依赖上下文的追问补全成完整问题
    3. 阈值过滤 —— 低于相似度门限的候选在这里就被拦掉，不进后续精排

多租户隔离的落地点在 _expr()：所有检索都强制带 role_id + tenant_id 过滤，
并可选用 bound_kb 把检索范围限制在角色绑定的若干份文档内。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from configs.settings import get_settings
from src.memory.short_term import retrieval_cache_get, retrieval_cache_set
from src.offline.embedder import embed
from src.offline.milvus_store import store
from src.online.query_transform import expand_query, resolve_coreference


def _expr(role_id: str, tenant_id: str, bound_kb: list[str] | None = None) -> str:
    """构造 Milvus 过滤表达式。

    参数：
        role_id: 角色 id（隔离维度一）
        tenant_id: 租户 id（隔离维度二）
        bound_kb: 该角色绑定的文档来源列表；空/None 表示不限制
    返回：
        Milvus 过滤表达式字符串。

    基础条件 role_id + tenant_id 永远存在、不可绕过：
        这是数据隔离的底线 —— 缺了它，A 角色能检索到 B 角色的私有知识库。
        bound_kb 是可选的额外收窄：让"智能客服"角色只引用产品手册，
        而不被国标文档干扰。

    生成 sources 时先过滤掉空串：
        调用方可能传入 ['', None] 之类的脏值，
        直接拼进去会得到 `doc_source in [""]` 这种把所有内容都排除掉的条件。

    `if sources:` 这层判断不能省：
        bound_kb 非空但全是空串时，拼接结果会是空的 in 列表 ——
        与其生成一个非法表达式让 Milvus 报错，不如退回基础条件（等于不做范围限制）。
    """
    base = f'role_id == "{_escape(role_id)}" and tenant_id == "{_escape(tenant_id)}"'
    if bound_kb:
        sources = ", ".join(f'"{_escape(source)}"' for source in bound_kb if source)
        if sources:
            return f"({base}) and doc_source in [{sources}]"
    return base


def _escape(value: str) -> str:
    """转义过滤表达式里的字符串字面量（先反斜杠、后引号，顺序不可反）。

    参数：
        value: 待转义的值
    返回：
        转义后的字符串。

    不转义的后果不只是报错：role_id 里含引号时会提前闭合字面量，
    后面的内容被当成表达式解析 —— 过滤条件被改写，可能检索到不该看到的数据。
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def retrieve(query: str, role_id: str, tenant_id: str, history: list[dict] | None = None, top_k: int | None = None, bound_kb: list[str] | None = None) -> dict[str, Any]:
    """在线检索主流程。

    参数：
        query: 用户原始问题
        role_id / tenant_id: 隔离维度
        history: 短期历史消息，用于指代消解
        top_k: 召回条数，不传则取配置默认值
        bound_kb: 限定检索的文档来源列表
    返回：
        {"query": 原问题, "rewritten_query": 改写后查询, "expansions": 扩展词,
         "hits": 命中片段, "trace": 链路明细}

    五步流程：
        1. 指代消解 resolve_coreference —— 用历史把"它/这个"还原成具体名词
        2. 查询扩展 expand_query —— 补全专业术语（规则化，不调模型）
        3. 查缓存 —— 命中就直接返回，省掉向量化和 Milvus 检索
        4. 向量化 + 双路召回（hybrid_search 在存储层已完成 RRF 融合）
        5. 阈值过滤后写缓存并返回

    缓存键的构成（role_id + tenant_id + bound_kb + 改写后查询）：
        这四个要素共同决定了检索结果。少任何一个都会串数据：
        例如不含 tenant_id，A 租户的检索结果会被 B 租户复用 —— 严重的数据泄露。
        不含 bound_kb 则角色换了绑定文档后仍拿到旧结果。
        键用改写后的查询而非原问题：不同问法改写后可能等价，这样才能真正命中缓存。

    向量化为空时的早退（`if not vectors.dense`）：
        hash 后端遇到空文本、或 api 后端返回异常时会出现这种情况。
        这里返回一个结构完整的空结果（含 trace 说明 embedding 为空），
        而不是抛异常 —— 让上层能正常走完流程并如实告诉用户"没检索到"。

    那个看起来复杂的 filtered 条件：
        `distance/score/rrf_score` 三者取第一个存在的值，与相似度阈值比较；
        后面的 `or hit.get("rrf_score", 0) > 0` 是一个豁免条款 ——
        只要 RRF 融合分大于 0（说明至少被一路召回过），就保留。
        原因：稀疏路的 IP 分数与稠密路的 COSINE 分数不在同一量纲，
        单看 distance 会把只被稀疏路命中的正确结果误杀。
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    # 先消解指代、再扩展术语：顺序不能反，否则会对"它"这种代词做无意义的术语扩展
    rewritten = expand_query(resolve_coreference(query, history))
    cache_key = hashlib.md5(f"{role_id}:{tenant_id}:{bound_kb}:{rewritten.rewritten}".encode("utf-8")).hexdigest()
    cached = retrieval_cache_get(cache_key)
    if cached:
        return json.loads(cached)  # 缓存里存的是 JSON 字符串，反序列化后结构与下面返回的一致
    vectors = embed([rewritten.rewritten])
    if not vectors.dense:
        return {"query": query, "rewritten_query": rewritten.rewritten, "hits": [], "trace": [{"step": "embedding", "status": "empty"}]}
    hits = store.hybrid_search(vectors.dense[0], vectors.sparse[0], limit=top_k, expression=_expr(role_id, tenant_id, bound_kb))
    filtered = [hit for hit in hits if float(hit.get("distance", hit.get("score", hit.get("rrf_score", 0)))) >= settings.similarity_threshold or hit.get("rrf_score", 0) > 0]
    result = {"query": query, "rewritten_query": rewritten.rewritten, "expansions": rewritten.expansions, "hits": filtered, "trace": [{"step": "query_transform", "status": "done", "detail": rewritten.rewritten}, {"step": "hybrid_retrieve", "status": "done", "detail": f"{len(filtered)} hits"}]}
    # ensure_ascii=False 让中文以原文存入缓存，体积更小、人工查看缓存时也可读
    retrieval_cache_set(cache_key, json.dumps(result, ensure_ascii=False))
    return result
