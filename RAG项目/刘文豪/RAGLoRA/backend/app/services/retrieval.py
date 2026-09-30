# -*- coding: utf-8 -*-
"""混合检索：dense ∥ sparse 双路召回 → RRF 融合。

关于分数（M0 实测结论）：
    RRF 融合分是 1/(k+rank) 的量化值，会出现大量并列（如不相关文档也拿 0.5 分）。
    **不要用 RRF 分做相关性阈值过滤**，阈值判断请用精排分。

向量库后端可切换（config.VECTOR_STORE）
---------------------------------------
    milvus  —— Docker standalone（**默认**，需先 `bash tools/demo/up.sh milvus`）
    qdrant  —— 嵌入式，已验证基线（无需 Docker，可作回退）
    both    —— 两库都查，用于现场对比召回差异

两条路径产出**同样结构**的 hit（text/source/page/law_name/article_no/
collection/score），上层 `rerank` 与 `persona` 无需感知底层是哪个库。
"""
import re
import time

from qdrant_client import models as qm

from ..core import config
from ..core.logging import get_logger
from .embed import encode_one
from .qdrant_store import get_client

log = get_logger("retrieval")


def _build_filter(filters: dict | None) -> qm.Filter | None:
    if not filters:
        return None
    must = [
        qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
        for k, v in filters.items() if v
    ]
    return qm.Filter(must=must) if must else None


def _qdrant_hybrid(dense: list[float], sparse: dict, collection: str,
                   recall_k: int, filters: dict | None) -> list[dict]:
    """Qdrant 侧混合检索（原有实现，未改动）。"""
    client = get_client()
    if not client.collection_exists(collection):
        log.warning("collection 不存在: %s", collection)
        return []

    flt = _build_filter(filters)
    result = client.query_points(
        collection_name=collection,
        prefetch=[
            # 多路召回：稠密语义 + 稀疏词权重
            #
            # ⚠️ `filter` 必须挂在**每个 Prefetch 上**，不能只放顶层
            #    （2026-09-16 修的真实 bug）。原因：顶层 `query_filter` 对
            #    `FusionQuery` 融合不起作用 —— 融合只是把两路的排名合并，
            #    并不重新过滤。实测对照：
            #
            #      顶层 filter + fusion        -> 传入不存在的过滤值仍返回 5 条（被忽略）
            #      每个 Prefetch 挂 filter      -> 正确返回 0 条
            #
            #    在这之前 `检索调试台` 的过滤条件一直是被静默忽略的。
            qm.Prefetch(query=dense, using="dense", limit=recall_k, filter=flt),
            qm.Prefetch(query=qm.SparseVector(**sparse), using="sparse",
                        limit=recall_k, filter=flt),
        ],
        # RRF 融合两路排名
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=recall_k,
        with_payload=True,
    )

    hits = []
    for p in result.points:
        payload = p.payload or {}
        hits.append({
            "id": p.id,
            "score": round(float(p.score), 6),
            "text": payload.get("text", ""),
            "source": payload.get("source"),
            "page": payload.get("page"),
            "law_name": payload.get("law_name"),
            "article_no": payload.get("article_no"),
            "collection": collection,
        })
    return hits


def _interleave_hits(groups: dict[str, list[dict]], limit: int) -> list[dict]:
    """把多个库的召回结果**交错**归并，并标注每条的来源库。

    `both` 模式专用（现场对比两库召回差异）。

    ⚠️ 原实现是「Qdrant 全部在前、满了就截断」，导致 **Milvus 的结果永远不出现**
    —— Qdrant 单独通常就能填满 recall_k。那样 `both` 模式根本达不到
    「对比两库召回」的目的（2026-09-16 修）。

    现在改为**按排名轮流取**：各库第 1 名、各库第 2 名、……
    这样两库的结果都必然出现在结果里，差异一眼可见。

    另：每条 hit 打上 `_from` 字段标注来源库，调用方可直接分辨。
    """
    merged: list[dict] = []
    seen: set[tuple] = set()
    max_len = max((len(g) for g in groups.values()), default=0)

    for rank in range(max_len):
        for store_name, group in groups.items():
            if rank >= len(group):
                continue
            h = dict(group[rank])          # 复制，避免污染原结果
            h["_from"] = store_name
            key = (h.get("source"), (h.get("text") or "")[:80])
            if key in seen:                # 两库都召回到的同一条：保留先到的，但补记来源
                continue
            seen.add(key)
            merged.append(h)
            if len(merged) >= limit:
                return merged
    return merged


def hybrid_search(query: str, collection: str, recall_k: int | None = None,
                  filters: dict | None = None) -> tuple[list[dict], str]:
    """在单个 collection 内做混合检索。

    返回 (hits, rewritten_query)。hits 每项含 text/source/score/payload。

    后端由 `config.VECTOR_STORE` 决定：qdrant（默认）/ milvus / both。
    """
    recall_k = recall_k or config.RECALL_TOP_K
    dense, sparse = encode_one(query, device="cpu")

    store = config.VECTOR_STORE

    if store == "milvus":
        from . import milvus_store
        hits = milvus_store.hybrid_search(dense, sparse, collection, recall_k, filters)
    elif store == "both":
        from . import milvus_store
        groups = {
            "qdrant": _qdrant_hybrid(dense, sparse, collection, recall_k, filters),
            "milvus": milvus_store.hybrid_search(dense, sparse, collection,
                                                 recall_k, filters),
        }
        hits = _interleave_hits(groups, limit=recall_k)
        log.info("both 模式 | qdrant %d 条 + milvus %d 条 -> 交错归并 %d 条",
                 len(groups["qdrant"]), len(groups["milvus"]), len(hits))
    elif store == "qdrant":
        hits = _qdrant_hybrid(dense, sparse, collection, recall_k, filters)
    else:
        log.warning("未知 VECTOR_STORE=%r，回退 qdrant", store)
        hits = _qdrant_hybrid(dense, sparse, collection, recall_k, filters)

    return hits, query


def search(query: str, collection: str | None = None,
           recall_k: int | None = None, filters: dict | None = None,
           multi: bool | None = None) -> tuple[list[dict], list[str]]:
    """检索入口。collection 为 None 时检索所有集合并按分数归并。

    `multi=True` 时走三路召回 + RRF 融合（见 `multi_recall`）；
    默认取 `config.MULTI_RECALL_ENABLED`。

    返回 (hits, collections_searched)。
    """
    recall_k = recall_k or config.RECALL_TOP_K
    use_multi = config.MULTI_RECALL_ENABLED if multi is None else multi

    if use_multi and collection:
        hits, _ = multi_recall(query, collection, recall_k, filters)
        return hits, [collection]

    if collection:
        hits, _ = hybrid_search(query, collection, recall_k, filters)
        return hits, [collection]

    # 全库检索：每个 collection 各召回，再按融合分归并
    names = collection_names()
    merged: list[dict] = []
    for name in names:
        hits, _ = hybrid_search(query, name, recall_k, filters)
        merged.extend(hits)
    merged.sort(key=lambda h: -h["score"])
    return merged[:recall_k], names


# ================================================================ 多路召回
def rrf_fuse(groups: dict[str, list[dict]], k: int = 60,
             weights: dict[str, float] | None = None,
             limit: int | None = None) -> list[dict]:
    """RRF（Reciprocal Rank Fusion）融合多路召回。

    每路的贡献是 `weight / (k + rank)`，按 `(source, text 前 80 字)` 归并同一条。

    ⚠️ 与项目既有结论一致：**RRF 分是量化值，不能用于相关性阈值过滤**
    （见本文件顶部注释）。它只负责排序，相关性判断一律用**精排分**。

    这里额外把**每路的名次**记进结果（`_ranks`），便于排查
    「某条为什么排这么高」；也便于回答「这一路到底有没有贡献」。
    """
    weights = weights or {}
    k = max(1, k)
    acc: dict[tuple, dict] = {}

    for path, hits in groups.items():
        w = weights.get(path, 1.0)
        for rank, h in enumerate(hits, start=1):
            key = (h.get("source"), (h.get("text") or h.get("law_name") or "")[:80])
            e = acc.get(key)
            if e is None:
                e = dict(h)
                e["rrf"] = 0.0
                e["_ranks"] = {}
                # 保留首见路作为主来源标注
                e["_paths"] = []
                acc[key] = e
            e["rrf"] += w / (k + rank)
            e["_ranks"][path] = rank
            if path not in e["_paths"]:
                e["_paths"].append(path)

    merged = sorted(acc.values(), key=lambda x: -x["rrf"])
    for e in merged:
        e["score"] = round(e.pop("rrf"), 6)
    return merged[:limit] if limit else merged


def multi_recall(query: str, collection: str,
                 recall_k: int | None = None,
                 filters: dict | None = None) -> tuple[list[dict], dict]:
    """三路召回 → RRF 融合。

        路1 向量库  语义相似（dense ∥ sparse → RRF）
        路2 MySQL   文档级结构匹配（**判断指名文档是否在库**）
        路3 Neo4j   法条关系（精确定位 / 引用 / 相邻）

    三路的分工不是「同一个东西查三遍」，而是**各补对方短处**：

        - 向量路不擅长精确定位（问「民法典第五百七十七条」未必排第一）→ 图谱路精确命中
        - 向量路不擅长回答「某文档在不在库」→ MySQL 路直接判定
        - 图谱路没有正文 → 由向量库按 source 取回原文

    ⚠️ 路2 返回的是**文档**不是 chunk，无法直接进 RRF。因此它的贡献是
    **文档级约束**（见下方 doc_meta），而非一路 chunk 排名：
        * 命中文档 → 该文档内的语义检索作为独立一路（缩小范围，比全库更准）
        * 未命中文档 → 记录为「语料缺口」，供上层告知模型「不要编造」

    返回 (hits, trace)。
    """
    recall_k = recall_k or config.RECALL_TOP_K
    from . import doc_store, graph_store

    t_all = time.time()
    trace: dict = {"paths": {}}
    groups: dict[str, list[dict]] = {}

    # ---- 路1：向量库 ----
    t = time.time()
    v_hits, _ = hybrid_search(query, collection, recall_k, filters)
    groups["vector"] = v_hits
    trace["paths"]["vector"] = {"n": len(v_hits), "ms": round((time.time() - t) * 1000)}

    # ---- 路2：MySQL 文档级 ----
    t = time.time()
    doc_hits, doc_meta = ([], {})
    try:
        doc_hits, doc_meta = doc_store.doc_lookup(query, collection)
    except Exception as e:
        log.warning("文档路失败（不影响其它路）: %s", e)
    trace["paths"]["doc"] = {"n": len(doc_hits), "ms": round((time.time() - t) * 1000),
                             "in_library": doc_meta.get("in_library"),
                             "missing": doc_meta.get("missing", [])}

    # 命中文档 -> 限定在该文档范围内再检索一次。
    # ⚠️ 这一路**不能作为独立一路参与 RRF**（2026-09-17 踩过）：
    #    它本质是「向量检索 + source 过滤」，与路1 高度重叠，
    #    同一条会在两路里各得一次票、分数翻倍，把**图谱路的精确定中挤出前三**
    #    （实测：问「民法典第五百七十七条」时图谱命中从第1位掉出前3）。
    #    正确用法是：把它当作**对指名文档内命中的加分**，以及**补充路1漏掉的条文**。
    scoped: list[dict] = []
    if doc_hits:
        t = time.time()
        for d in doc_hits[:2]:
            h, _ = hybrid_search(query, collection, recall_k,
                                 filters={"source": d["source"]})
            scoped.extend(h)
        trace["paths"]["doc_scoped"] = {"n": len(scoped),
                                        "ms": round((time.time() - t) * 1000)}

    # ---- 路3：Neo4j 图谱 ----
    t = time.time()
    g_hits: list[dict] = []
    try:
        g_hits = graph_store.graph_recall(query, collection, k=recall_k)
    except Exception as e:
        log.warning("图谱路失败（不影响其它路）: %s", e)
    if g_hits:
        # 图谱只存关系，正文要回向量库按 (law_name, article_no) 补齐
        g_hits = _hydrate_graph_hits(g_hits, collection)
        groups["graph"] = g_hits
    trace["paths"]["graph"] = {"n": len(g_hits), "ms": round((time.time() - t) * 1000)}

    # ---- 融合：只对**真正独立**的两路做 RRF ----
    fused = rrf_fuse(groups, k=config.RRF_K,
                     weights=config.MULTI_RECALL_WEIGHTS,
                     limit=None)

    # 把限定检索的结果作为**补充**并入（已存在的只记票，不重复计分）
    if scoped:
        seen = {(h.get("source"),
                 (h.get("text") or h.get("law_name") or "")[:80]) for h in fused}
        added = 0
        for h in scoped:
            key = (h.get("source"),
                   (h.get("text") or h.get("law_name") or "")[:80])
            if key not in seen:
                seen.add(key)
                fused.append({**h, "score": 0.0, "_paths": ["doc_scoped"],
                              "_ranks": {}})
                added += 1
        if added:
            log.debug("限定检索补充 %d 条路1未覆盖的命中", added)

    # 指名文档内的命中加分（而不是重复计票）
    #
    # ⚠️ 口径要对齐（2026-09-17 踩过）：三路的 `source` 字段**含义不同**——
    #    向量路/限定路来自 payload，是**文件名**（`中华人民共和国民法典.txt`）；
    #    图谱路来自图节点，是**法律名**（`中华人民共和国民法典`，无扩展名）。
    #    直接比对会匹配不上，加分静默失效、图谱命中排不上来。
    #    因此这里按「去掉扩展名后的名字」双向包含匹配。
    if doc_hits:
        named = {_bare_name(d["source"]) for d in doc_hits}
        for h in fused:
            hn = _bare_name(h.get("source") or h.get("law_name"))
            if hn and any(hn == n or hn in n or n in hn for n in named):
                h["score"] = round((h.get("score") or 0.0) * config.DOC_BOOST, 6)
                if "doc" not in h.get("_paths", []):
                    h["_paths"] = list(h.get("_paths", [])) + ["doc"]

    fused.sort(key=lambda x: -(x.get("score") or 0.0))
    fused = fused[:recall_k]

    trace["total_ms"] = round((time.time() - t_all) * 1000)
    trace["doc_meta"] = doc_meta
    log.info("多路召回 | " + " ".join(
        f"{p}:{v['n']}" for p, v in trace["paths"].items()) +
        f" -> 融合 {len(fused)} 条 | {trace['total_ms']}ms")
    return fused, trace


def _bare_name(s: str | None) -> str:
    """去掉扩展名与「中华人民共和国」前缀，用于跨路对齐文档名。

    三路的 source 口径不同（文件名 vs 法律名），必须归一后才能比较。
    """
    if not s:
        return ""
    s = re.sub(r"\.(txt|pdf|md)$", "", s)
    return s.replace("中华人民共和国", "").strip()


def _hydrate_graph_hits(hits: list[dict], collection: str) -> list[dict]:
    """给图谱命中补上正文（图谱里只存了关系，不存文本）。"""
    from . import milvus_store
    out = []
    for h in hits:
        if h.get("text"):
            out.append(h)
            continue
        text = None
        law, art = h.get("law_name"), h.get("article_no")
        if law and art:
            try:
                if config.VECTOR_STORE == "milvus":
                    res = milvus_store.get_client().query(
                        collection_name=collection,
                        filter=f'law_name == "{law}" and article_no == "{art}"',
                        output_fields=["text"], limit=1)
                else:
                    from .qdrant_store import get_client as qc
                    pts, _ = qc().scroll(
                        collection_name=collection, limit=1, with_payload=True,
                        scroll_filter=qm.Filter(must=[
                            qm.FieldCondition(key="law_name",
                                              match=qm.MatchValue(value=law)),
                            qm.FieldCondition(key="article_no",
                                              match=qm.MatchValue(value=art)),
                        ]))
                    res = [p.payload for p in pts] if pts else []
                if res:
                    text = (res[0] or {}).get("text")
            except Exception as e:
                log.debug("图谱命中补正文失败 %s %s: %s", law, art, e)
        out.append({**h, "text": text or ""})
    return [h for h in out if h.get("text")]


def collection_names() -> list[str]:
    """当前向量库后端中的集合名列表。

    注意要按 `config.VECTOR_STORE` 走对应后端 —— 否则在 milvus 模式下
    会去问 Qdrant 要集合列表（碰巧同名时看不出问题，是隐性耦合）。
    """
    if config.VECTOR_STORE == "milvus":
        from . import milvus_store
        return sorted(milvus_store.get_client().list_collections())
    return [c.name for c in get_client().get_collections().collections]


def format_source(hit: dict) -> str:
    """把命中项格式化成人类可读的来源标注。"""
    if hit.get("law_name") and hit.get("article_no"):
        return f"《{hit['law_name']}》{hit['article_no']}"
    if hit.get("source") and hit.get("page"):
        return f"{hit['source']} 第{hit['page']}页"
    return hit.get("source") or "未知来源"
