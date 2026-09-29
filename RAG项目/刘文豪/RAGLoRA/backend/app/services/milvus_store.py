# -*- coding: utf-8 -*-
"""Milvus 向量库客户端（Docker standalone）。

与 `qdrant_store` 的关系
------------------------
两者是**并存**的两个向量库后端，由 `config.VECTOR_STORE` 选择：

    qdrant   —— 嵌入式，已验证基线（无需 Docker，但独占文件锁、只能单 worker）
    milvus   —— Docker standalone，支持并发/水平扩展
    both     —— 双写，用于对比

**以 Qdrant 为权威源**：Milvus 的数据可随时由 Qdrant 全量重建
（`scripts/migrate_to_milvus.py`）。

pymilvus 3.x 的 API 与 2.x 有破坏性差异
----------------------------------------
本文件按 3.0.1 实测的 API 编写，**照抄网上 2.x 的写法会踩坑**：

    ✗ `from pymilvus import IndexParams`      —— 3.x 顶层已不导出
    ✗ `client.add_index(...)`                 —— MilvusClient 上不存在该方法
    ✓ `ip = client.prepare_index_params()`    —— 正确入口
      `ip.add_index(field_name=..., index_type=..., metric_type=...)`
      `client.create_index(collection_name, ip)`

    另：`search` 前必须 `load_collection`，否则报 collection not loaded。

主键为什么不用自增 `auto_id=True`
---------------------------------
清单写的是「ID（主键：唯一性，自增ID）」，但本项目**刻意选择内容哈希做主键**
（`auto_id=False`），理由：

    Qdrant 侧的主键就是内容哈希 `sha1(collection|source|article_no|text)`，
    它保证了**重复入库幂等**（同一条内容覆盖而非追加）。

    若改用自增 ID，重新入库同一份文档会产生**重复行** —— 这是功能退化。
    且迁移时无法把 Qdrant 与 Milvus 的同一条记录对应起来，无法做一致性核对。

因此主键满足「唯一性」，但**不是自增**。这是有依据的取舍，不是疏漏。

字段与清单的对应
----------------
清单要求 Milvus collection 含：ID / 向量 / 原文 / 混合检索BM25 / 创建时间 /
修改时间 / 文档来源 / 摘要。本 schema 全部覆盖，并额外保留 Qdrant 侧已有的
`page` / `law_name` / `article_no` 三个元数据字段，使检索结果与 Qdrant 侧等价。
"""
import threading
import time

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from ..core import config
from ..core.logging import get_logger

log = get_logger("milvus_store")

_lock = threading.Lock()
_client: MilvusClient | None = None

# 输出字段：检索时回传给上层的 payload。
_OUTPUT_FIELDS = [
    "text", "source", "page", "law_name", "article_no", "summary",
    "created_at", "updated_at",
]

# Milvus INT64 的有符号上限。见 to_milvus_id()。
_INT64_MAX = 2 ** 63 - 1

# 各 VARCHAR 字段的最大长度（Milvus 必填，且 <= 65535）
_MAX_LEN = {
    "text": 8192,
    "source": 1024,
    "law_name": 512,
    "article_no": 128,
    "summary": 4096,
}


# ---------------------------------------------------------------- 连接
def get_client() -> MilvusClient:
    """Milvus 客户端单例。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                log.info("连接 Milvus: %s", config.MILVUS_URI)
                _client = MilvusClient(uri=config.MILVUS_URI)
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def health() -> dict:
    """探活：返回 {ok, ms, collections} 或 {ok: False, error}。"""
    t = time.time()
    try:
        names = get_client().list_collections()
        return {"ok": True, "ms": round((time.time() - t) * 1000),
                "collections": sorted(names)}
    except Exception as e:
        return {"ok": False, "ms": round((time.time() - t) * 1000),
                "error": str(e)[:200]}


# ---------------------------------------------------------------- 建表
def _build_schema():
    """构造 collection schema（字段对应清单要求，见模块 docstring）。"""
    schema = get_client().create_schema(auto_id=False, enable_dynamic_field=False)

    # 主键：内容哈希（见模块 docstring「主键为什么不用自增」）
    schema.add_field("id", DataType.INT64, is_primary=True)
    # 向量：稠密 + 稀疏（混合检索）
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=config.EMBED_DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    # 原文与来源
    schema.add_field("text", DataType.VARCHAR, max_length=_MAX_LEN["text"])
    schema.add_field("source", DataType.VARCHAR, max_length=_MAX_LEN["source"])
    # 元数据（与 Qdrant 侧对齐）
    schema.add_field("page", DataType.INT64)
    schema.add_field("law_name", DataType.VARCHAR, max_length=_MAX_LEN["law_name"])
    schema.add_field("article_no", DataType.VARCHAR, max_length=_MAX_LEN["article_no"])
    # 摘要（清单要求；当前留空，供后续数据增强写入）
    schema.add_field("summary", DataType.VARCHAR, max_length=_MAX_LEN["summary"])
    # 时间戳
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("updated_at", DataType.INT64)
    return schema


def _build_index_params():
    """索引：稠密 HNSW/COSINE，稀疏倒排/IP。"""
    ip = get_client().prepare_index_params()
    ip.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP")
    return ip


def ensure_collection(name: str) -> None:
    """collection 不存在则创建（含 schema 与索引）。"""
    client = get_client()
    if client.has_collection(name):
        return
    with _lock:
        if client.has_collection(name):
            return
        client.create_collection(
            collection_name=name,
            schema=_build_schema(),
            index_params=_build_index_params(),
        )
        log.info("创建 Milvus collection: %s", name)


# ---------------------------------------------------------------- 数据转换
def _to_milvus_sparse(sparse) -> dict:
    """Qdrant 稀疏向量 -> Milvus 的 {index: value} 字典。

    ⚠️ Qdrant 有**两种**返回形态，必须都兼容：
        * `query_points`（在线检索）-> `dict{"indices": [...], "values": [...]}`
        * `scroll`（迁移导出）      -> `SparseVector` 对象，取 `.indices` / `.values`

    实测踩过：只按 dict 写，迁移时抛 `TypeError: 'SparseVector' object is not
    subscriptable`。而 `sparse_vectors` 适配层用的是前者，所以在线检索看不出来。
    """
    if isinstance(sparse, dict):
        idx, val = sparse.get("indices", []), sparse.get("values", [])
    else:                      # qdrant_client.http.models.SparseVector
        idx, val = sparse.indices, sparse.values
    return {int(i): float(v) for i, v in zip(idx, val)}


def to_milvus_id(raw_id: int) -> int:
    """Qdrant 的 64 位无符号 ID -> Milvus 的有符号 INT64。

    ⚠️ 这是一个**必须处理**的类型鸿沟（实测踩过）：
        Qdrant 的 ID 是 `int(sha1(...)[:16], 16)`，取值范围为完整 64 位无符号
        （0 ~ 2^64-1）；而 Milvus 的 INT64 是**有符号**的（0 ~ 2^63-1）。
        约**半数** ID 会超出 Milvus 上限，直接写入会抛
        `DataNotMatchException: Value out of range`。

    这里按位与掉最高位，把值映射到有符号区间。

    **这仍是一个确定性函数**，所以：
        * 同一条文档每次迁移得到同一个 ID —— 迁移可重复执行且幂等；
        * 但 Milvus 侧 ID 与 Qdrant 侧 ID **不相等**（差一个符号位），
          跨库比对需要各自用本函数换算，不能直接比原始 ID。

    碰撞风险：14955 条落在 63 位空间，生日碰撞概率约 1e-11，可忽略；
    迁移脚本会实测校验无碰撞。
    """
    return int(raw_id) & _INT64_MAX


def row_from_qdrant(payload: dict, _id: int, dense, sparse) -> dict:
    """把一条 Qdrant 记录转成 Milvus 行（供迁移脚本使用）。"""
    now = int(time.time())
    src = payload.get("source") or ""
    return {
        "id": to_milvus_id(_id),
        "dense": list(dense),
        "sparse": _to_milvus_sparse(sparse),
        "text": (payload.get("text") or "")[:_MAX_LEN["text"]],
        "source": src[:_MAX_LEN["source"]],
        "page": int(payload.get("page") or 0),
        "law_name": (payload.get("law_name") or "")[:_MAX_LEN["law_name"]],
        "article_no": (payload.get("article_no") or "")[:_MAX_LEN["article_no"]],
        "summary": (payload.get("summary") or "")[:_MAX_LEN["summary"]],
        # Qdrant 侧无时间字段，迁移时统一打上迁移时刻
        "created_at": int(payload.get("created_at") or now),
        "updated_at": now,
    }


# ---------------------------------------------------------------- 写入
def upsert_rows(collection: str, rows: list[dict], batch: int = 200) -> int:
    """批量写入（upsert：主键相同则覆盖，保证重复入库幂等）。"""
    client = get_client()
    ensure_collection(collection)
    total = 0
    for i in range(0, len(rows), batch):
        client.upsert(collection_name=collection, data=rows[i:i + batch])
        total += len(rows[i:i + batch])
    return total


def flush(collection: str) -> None:
    get_client().flush(collection_name=collection)


def delete_by_source(collection: str, source: str) -> int:
    """按来源删除（与 Qdrant 侧 delete_document 对应）。"""
    client = get_client()
    if not client.has_collection(collection):
        return 0
    res = client.delete(collection_name=collection,
                        filter=f'source == "{source}"')
    return int(res.get("delete_count", 0)) if isinstance(res, dict) else 0


# ---------------------------------------------------------------- 检索
def build_filter_expr(filters: dict | None) -> str:
    """把 `{"law_name": "民法典", "page": 3}` 转成 Milvus 过滤表达式。

    值为 None/空串的键跳过；数字用等值，字符串用**包含匹配**（`like "%值%"`）。

    ⚠️ 两个后端的字符串匹配语义**不同**（2026-09-16 实测）：

        过滤条件                  Qdrant（精确 MatchValue）   Milvus（like 包含）
        law_name="民法典"          0 条                        5 条
        law_name="中华人民共和国民法典"  5 条                        5 条

    原因是库里 `law_name` 存的是**全称**（`中华人民共和国民法典`），
    而调用方（`schemas.py:132` 的示例 `{"law_name": "民法典"}`）习惯传简称。
    包含匹配对简称友好，精确匹配则要求调用方写全称。

    **这是刻意的行为差异，不是 bug** —— 选包含匹配是因为「传简称查不到」比
    「匹配略宽」更影响可用性。若需要两库严格一致，必须统一到其中一种语义。

    字符串值统一走 `_escape`（转义反斜杠与双引号），因为这里是手工拼表达式。
    """
    if not filters:
        return ""

    parts: list[str] = []
    for key, val in filters.items():
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            parts.append(f"{key} == {str(val).lower()}")
        elif isinstance(val, (int, float)):
            parts.append(f"{key} == {val}")
        else:
            safe = _escape(str(val))
            parts.append(f'{key} like "%{safe}%"')
    return " and ".join(parts)


def _escape(s: str) -> str:
    """转义 Milvus 表达式里的字符串字面量。"""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def hybrid_search(dense: list[float], sparse: dict, collection: str,
                  recall_k: int | None = None,
                  filters: dict | None = None) -> list[dict]:
    """稠密 ∥ 稀疏 双路召回，RRF 融合。

    与 Qdrant 侧的 `retrieval.hybrid_search` 等价：
    两路各召回 recall_k 条，再用 RRF 融合排名，取 recall_k 条。

    返回的 hit 结构与 Qdrant 侧一致（text/source/page/law_name/article_no/
    collection/score），使上层 `rerank` 与 `persona` 无需感知底层用的是哪个库。
    """
    recall_k = recall_k or config.RECALL_TOP_K
    client = get_client()
    if not client.has_collection(collection):
        log.warning("Milvus collection 不存在: %s", collection)
        return []

    client.load_collection(collection)

    # ⚠️ 过滤条件必须挂在**每个 AnnSearchRequest** 上（2026-09-20 修的真实 bug）
    # ------------------------------------------------------------------
    # `MilvusClient.hybrid_search(...)` **没有 `filter` 形参**，
    # 它会被 `**kwargs` 静默吞掉 —— 传了等于没传，且**不报任何错**。
    #
    # 实测后果（长期记忆的用户隔离）：
    #     recall(user_id=999999) 返回了 user_id=41 的记录 —— 用户能看到别人的记忆。
    # 这与之前在 Qdrant 上修过的是**同一类**问题（prefetch+fusion 下顶层
    # query_filter 被忽略），只是换了个后端、换个位置重演。
    #
    # 正确做法：把表达式下推到每个 AnnSearchRequest 的 `expr`。
    # 这与 Qdrant 侧「filter 挂在每个 Prefetch 上」是同一个道理：
    # **融合只合并排名，不重新过滤**，过滤必须在各路检索时就生效。
    flt = build_filter_expr(filters)
    reqs = [
        AnnSearchRequest(data=[dense], anns_field="dense",
                         param={"metric_type": "COSINE"}, limit=recall_k,
                         expr=flt or None),
        AnnSearchRequest(data=[_to_milvus_sparse(sparse)], anns_field="sparse",
                         param={"metric_type": "IP"}, limit=recall_k,
                         expr=flt or None),
    ]
    # ⚠️ 多路召回必须用 `hybrid_search`，不能用 `search`。
    #    pymilvus 3.x 里 `search(data=[AnnSearchRequest, ...])` 会抛
    #    `ParamError: search_data ... is illegal` —— 那是 2.x 的写法。
    #    3.x 的 `search` 只接受向量本身，多路+重排走 `hybrid_search(reqs=...)`。
    res = client.hybrid_search(
        collection_name=collection,
        reqs=reqs,
        ranker=RRFRanker(),
        limit=recall_k,
        output_fields=_OUTPUT_FIELDS,
        # 这里**刻意不再传 filter** —— 它会被 kwargs 吞掉（见上方注释）。
        # 过滤已下推到 reqs 里。
    )

    hits: list[dict] = []
    # search 返回 list[list[hit]]，混合检索时为单层结果集
    for group in res:
        for h in group:
            ent = h.get("entity", {}) or {}
            hits.append({
                "id": h.get("id"),
                "score": round(float(h.get("distance", 0.0)), 6),
                "text": ent.get("text", ""),
                "source": ent.get("source"),
                "page": ent.get("page") or None,
                "law_name": ent.get("law_name") or None,
                "article_no": ent.get("article_no") or None,
                "collection": collection,
            })
    return hits


def count_rows(collection: str) -> int:
    """权威行数：用 `count(*)` 聚合，而不是 `get_collection_stats`。

    ⚠️ `get_collection_stats().row_count` 是**陈旧值**（2026-09-16 实测）：
    删除数据后它只增不减，直到后台压缩才修正。实测删除 3 条后：
        get_collection_stats  -> 639  ❌
        count(*) 聚合          -> 636  ✓

    因此**任何一致性核对都必须用本函数**，否则删过数据后会误报不一致。
    """
    client = get_client()
    if not client.has_collection(collection):
        return -1
    res = client.query(collection_name=collection, filter="",
                       output_fields=["count(*)"])
    if not res:
        return 0
    first = res[0]
    # pymilvus 返回形如 {'count(*)': 636}，但不同版本可能是 str 或 dict
    if isinstance(first, dict):
        return int(first.get("count(*)", 0))
    return int(str(first).strip("'\""))


def collection_stats() -> list[dict]:
    """各 collection 的条数统计（用权威计数，见 count_rows）。"""
    client = get_client()
    out = []
    for name in client.list_collections():
        try:
            out.append({"name": name, "count": count_rows(name)})
        except Exception as e:
            out.append({"name": name, "count": -1, "error": str(e)[:120]})
    return out
