# -*- coding: utf-8 -*-
"""
Milvus 向量库访问模块

职责边界：
    Milvus 只负责回答「哪些块与问题相关」（相似度计算）。
    「这块来自哪一页、属于哪份文档」一律回 MySQL 取回，
    本模块冗余保存的 page_no / content 仅用于日志回显与快速排查。

V2 扩展说明：
    V2 混合检索需要新增稀疏向量字段 `sparse_embedding`（SPARSE_FLOAT_VECTOR）。
    本模块已预留 create_collection(enable_sparse=...) 开关，
    V1 阶段不建该字段，保持集合结构最简。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pymilvus import DataType, MilvusClient

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

# 各标量字段的容量上限
_MAX_LEN_CHUNK_ID = 64
_MAX_LEN_DOC_ID = 64
_MAX_LEN_CONTENT = 8192

_client: Optional[MilvusClient] = None


def _truncate_utf8(text: str, max_bytes: int, *, field: str = "") -> str:
    """
    按 UTF-8 **字节数**截断字符串。

    为什么不能用 text[:max_bytes]：
        Milvus 的 VARCHAR(max_length=N) 以 **UTF-8 字节**计数，而 Python 的
        字符串切片按**字符**计数。一个中文字符占 3 字节，因此对中文内容做
        text[:8192] 实际会产生 24576 字节 —— 超出字段上限，导致
        MilvusException(code=1100) 并使**整条插入失败**（不是静默截断）。

    实测：本函数修复前，向 content 写入 20000 个中文字符会直接抛
    「length of varchar field content exceeds max length, length: 24576,
    max length: 8192」。

    errors="ignore" 用于避免在多字节字符中间切断时抛 UnicodeDecodeError。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    logger.warning(
        "字段超长已按 UTF-8 字节截断 | 字段=%s | 原长=%d 字节 | 截断至=%d 字节",
        field or "(未知)", len(encoded), max_bytes,
    )
    return truncated


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------

def get_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        _client = MilvusClient(uri=uri)
        logger.info("Milvus 客户端已连接：%s", uri)
    return _client


def health_check() -> bool:
    """连通性检查，供 /api/health 使用"""
    try:
        get_client().list_collections()
        return True
    except Exception as exc:
        logger.error("Milvus 健康检查失败：%s", exc)
        return False


# ---------------------------------------------------------------------------
# 集合管理
# ---------------------------------------------------------------------------

def collection_exists(name: Optional[str] = None) -> bool:
    """判断集合是否存在"""
    name = name or settings.milvus_collection
    try:
        return get_client().has_collection(collection_name=name)
    except Exception as exc:
        logger.error("查询 Milvus 集合是否存在失败：%s", exc)
        return False


def create_collection(
    name: Optional[str] = None,
    *,
    drop_existing: bool = False,
    enable_sparse: bool = False,
) -> None:
    """
    创建集合与索引。

    参数：
        drop_existing : 为 True 时先删除同名集合（重建知识库时使用）
        enable_sparse : V2 混合检索开关，为 True 时额外创建稀疏向量字段
    """
    name = name or settings.milvus_collection
    client = get_client()

    if collection_exists(name):
        if not drop_existing:
            logger.info("Milvus 集合已存在，跳过创建：%s", name)
            return
        client.drop_collection(collection_name=name)
        logger.info("已删除旧的 Milvus 集合：%s", name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="chunk_id", datatype=DataType.VARCHAR,
        max_length=_MAX_LEN_CHUNK_ID, is_primary=True,
    )
    schema.add_field(
        field_name="doc_id", datatype=DataType.VARCHAR, max_length=_MAX_LEN_DOC_ID,
    )
    schema.add_field(field_name="page_no", datatype=DataType.INT64)
    schema.add_field(
        field_name="content", datatype=DataType.VARCHAR, max_length=_MAX_LEN_CONTENT,
    )
    schema.add_field(
        field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=settings.milvus_dim,
    )
    if enable_sparse:
        # V2 混合检索的稀疏通道（BGE-M3 lexical weights）
        schema.add_field(
            field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type=settings.milvus_metric_type,
        params={
            "M": settings.milvus_hnsw_m,
            "efConstruction": settings.milvus_hnsw_ef_construction,
        },
    )
    if enable_sparse:
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    logger.info("Milvus 集合创建完成：%s（稀疏字段=%s）", name, enable_sparse)


def drop_collection(name: Optional[str] = None) -> None:
    """删除集合（重建知识库时使用）"""
    name = name or settings.milvus_collection
    if collection_exists(name):
        get_client().drop_collection(collection_name=name)
        logger.info("已删除 Milvus 集合：%s", name)


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def insert_vectors(
    chunk_ids: Sequence[str],
    doc_ids: Sequence[str],
    page_nos: Sequence[int],
    contents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    *,
    sparse_embeddings: Optional[Sequence[Dict[int, float]]] = None,
    name: Optional[str] = None,
    batch_size: int = 256,
) -> int:
    """
    批量写入向量与随行标量字段。

    sparse_embeddings 仅 V2 使用，为「词元ID -> 权重」的稀疏字典列表。
    """
    name = name or settings.milvus_collection
    total = len(chunk_ids)
    if total == 0:
        return 0

    # 稀疏字段是 V2 混合检索的前置条件。集合一旦建立，schema 就不会再变
    # （create_collection 对已存在集合直接跳过），因此若用旧集合写入稀疏向量，
    # Milvus 只会抛一句难以理解的字段错误。这里提前拦住并给出可执行的指引。
    if sparse_embeddings is not None and not has_sparse_field(name):
        raise RuntimeError(
            f"集合 {name} 中没有 sparse_embedding 字段，无法写入稀疏向量。\n"
            f"该字段需要重建集合才能加上，请执行：\n"
            f"    .venv/Scripts/python.exe -m scripts.embedding_store --rebuild"
        )

    client = get_client()
    written = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        rows: List[Dict[str, Any]] = []
        for i in range(start, end):
            row: Dict[str, Any] = {
                "chunk_id": _truncate_utf8(
                    str(chunk_ids[i]), _MAX_LEN_CHUNK_ID, field="chunk_id"
                ),
                "doc_id": _truncate_utf8(
                    str(doc_ids[i]), _MAX_LEN_DOC_ID, field="doc_id"
                ),
                "page_no": int(page_nos[i]),
                # 必须按字节截断：content 含中文，Milvus 以 UTF-8 字节计数
                "content": _truncate_utf8(
                    str(contents[i]), _MAX_LEN_CONTENT, field="content"
                ),
                "embedding": list(embeddings[i]),
            }
            if sparse_embeddings is not None:
                row["sparse_embedding"] = sparse_embeddings[i]
            rows.append(row)

        client.insert(collection_name=name, data=rows)
        written += len(rows)

    # 写入后立即 flush：否则新数据在 Milvus 自动 flush 前不可检索，
    # 而该失败是静默的（不报错、只少召回），会直接影响重建后立刻进行的评测。
    if written:
        flush_collection(name)

    logger.info("Milvus 写入向量完成：%d 条 -> %s", written, name)
    return written


def delete_by_doc_id(doc_id: str, name: Optional[str] = None) -> None:
    """按文档 ID 删除该文档的全部向量（重建单份文档时使用）"""
    name = name or settings.milvus_collection
    if not collection_exists(name):
        return
    get_client().delete(collection_name=name, filter=f'doc_id == "{doc_id}"')
    logger.info("已按文档删除向量：doc_id=%s", doc_id)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def search_dense(
    query_vector: Sequence[float],
    *,
    top_k: Optional[int] = None,
    expr: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    稠密向量检索（V1 的唯一检索方式）。

    返回：[{chunk_id, score, page_no, doc_id, content}, ...]，按相似度降序。
    """
    name = name or settings.milvus_collection
    top_k = top_k or settings.retrieve_top_k
    output_fields = output_fields or ["chunk_id", "doc_id", "page_no", "content"]

    # 集合不存在（还没跑过离线建库）就直接返回空，让上层走「未找到」分支，
    # 而不是抛异常把整次问答打断。
    if not collection_exists(name):
        logger.warning("Milvus 集合不存在，检索返回空：%s", name)
        return []

    # 真正发起向量检索：以问题向量为查询，取最相似的 top_k 条。
    results = get_client().search(
        collection_name=name,
        data=[list(query_vector)],
        # ★ 必须显式指定 anns_field ★
        # V1 时期集合只有一个向量字段，Milvus 能自动推断；V2 加入稀疏字段后
        # 集合内出现两个向量字段，不指定就会报：
        #   code=65535, "multiple anns_fields exist, please specify a anns_field"
        # 即：加稀疏字段这个动作本身会让 V1 的稠密检索失效，必须显式声明。
        # 指明要检索的是稠密向量字段（集合里还有稀疏字段，必须点清是哪一个）
        anns_field="embedding",
        # 要返回多少条结果
        limit=top_k,
        # 除了向量本身，还要顺带取回哪些标量字段（供拼装结果用）
        output_fields=output_fields,
        search_params={
            # 度量方式：本项目配的是 COSINE 余弦相似度。
            # 入库时向量做过 L2 归一化，因此余弦与内积是等价的。
            "metric_type": settings.milvus_metric_type,
            # ef 是 HNSW 索引的搜索范围参数：越大越准、也越慢。
            # 这里跟着 top_k 一起放大，保证要的条数多时候选范围也够宽。
            "params": {"ef": max(64, top_k * 4)},
        },
        # 附加过滤条件（默认不过滤），需要时可用来限定在某份文档内检索
        filter=expr or "",
    )

    hits: List[Dict[str, Any]] = []
    # results 是"每个查询向量各自的结果列表"，这里只查了一个向量，所以取 [0]。
    for hit in (results[0] if results else []):
        # entity 里装着 output_fields 指定的那些标量字段
        entity = hit.get("entity", {}) or {}

        # ★ 主键取值需做兼容 ★
        # pymilvus 的 Hit 中，主键字段名与集合 schema 里定义的主键字段名保持一致
        # （本项目为 chunk_id），而**不是**固定的 "id"。
        # 不同 pymilvus 版本此处行为有差异，因此依次尝试多个来源，
        # 避免因取不到主键导致下游「元数据回填」全部失败、检索结果被静默丢弃。
        chunk_id = (
            hit.get("chunk_id")
            or hit.get("id")
            or entity.get("chunk_id")
        )

        # 统一成下游要用的结构。
        # 注意这里的 page_no / content 只是 Milvus 里的冗余副本，仅用于日志回显；
        # 对外返回的溯源信息一律以 MySQL 为准（见 V1/V2 链路里的元数据回填步骤）。
        hits.append({
            "chunk_id": chunk_id,
            # distance 即相似度得分（余弦越大越相似），结果按它降序排列
            "score": float(hit.get("distance", 0.0)),
            "doc_id": entity.get("doc_id"),
            "page_no": entity.get("page_no"),
            "content": entity.get("content", ""),
        })
    return hits


def has_sparse_field(name: Optional[str] = None) -> bool:
    """判断集合是否含稀疏向量字段（V2 混合检索的前置条件）"""
    name = name or settings.milvus_collection
    try:
        # 读出集合的字段定义，看有没有一个叫 sparse_embedding 的字段。
        desc = get_client().describe_collection(name)
        return any(f.get("name") == "sparse_embedding" for f in desc.get("fields", []))
    except Exception as exc:
        # 读不出来一律当作"没有"：宁可退化成只用稠密检索，也不要让调用方报错。
        logger.warning("读取集合字段失败 | %s | %s", name, exc)
        return False


def flush_collection(name: Optional[str] = None) -> None:
    """
    强制落盘，使已写入的数据**立即可被检索到**。

    为什么需要显式 flush：
        Milvus 在默认一致性级别下，新插入的数据要等自动 flush（周期性触发）
        之后才进入可检索状态。实测：同一进程内先 insert 再 search，
        不 flush 时命中数为 **0**；flush 后立刻命中。
        这个失败是**静默的** —— 不报错，只是少召回甚至完全查不到，
        属于最难察觉的一类故障。因此写入完成后必须显式 flush。

    参数：
        name : 集合名，默认取 settings.milvus_collection
    """
    name = name or settings.milvus_collection
    try:
        # 让刚写进去的数据立刻进入"可被检索到"的状态。
        get_client().flush(collection_name=name)
    except Exception as exc:
        # flush 失败不应让整批写入失败：数据已写入，只是可见性延迟
        logger.warning("Milvus flush 失败，数据可能延迟可见 | %s | %s", name, exc)


def search_sparse(
    query_sparse: Dict[int, float],
    *,
    top_k: Optional[int] = None,
    output_fields: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    稀疏向量检索（V2 混合检索的稀疏通道）。

    度量方式为 IP（内积）—— BGE-M3 的 lexical weights 非负且不归一化，
    内积即为加权词匹配得分，这是 BGE-M3 稀疏表示的标准用法。

    返回结构与 search_dense() 对称，含 chunk_id / score / doc_id / page_no / content。

    注意：这里的 page_no / content 来自 Milvus 的冗余标量字段，
    仅用于 Trace 日志回显；对外溯源信息一律以 MySQL 为准（ADR-005）。
    """
    # 查询的稀疏权重是空的（问题里没有任何能映射成词元的成分），
    # 那就压根不存在字面线索，直接返回空 —— 这一路不参与融合即可。
    if not query_sparse:
        return []

    name = name or settings.milvus_collection
    client = get_client()
    limit = top_k or settings.retrieve_top_k

    # 与 search_dense 的实质差别只有这两个参数：
    #   anns_field 指向稀疏向量字段（而不是稠密那个）
    #   metric_type 用 IP 内积（而不是 COSINE）
    results = client.search(
        collection_name=name,
        data=[query_sparse],
        anns_field="sparse_embedding",
        search_params={"metric_type": "IP", "params": {}},
        limit=limit,
        output_fields=output_fields or ["chunk_id", "doc_id", "page_no", "content"],
    )

    hits: List[Dict[str, Any]] = []
    for hit in (results[0] if results else []):
        entity = hit.get("entity", {}) or {}

        # ★ 与 search_dense 保持同样的主键取值兼容 ★
        # pymilvus 不同版本下主键可能出现在 hit 顶层（字段名同 schema）或 entity 内，
        # 依次尝试多个来源，避免取不到主键导致下游元数据回填全部失败。
        chunk_id = (
            hit.get("chunk_id")
            or hit.get("id")
            or entity.get("chunk_id")
        )

        # 返回结构与 search_dense 完全对称 —— 正因如此，
        # 上层的 RRF 融合才能把两路结果同等对待、只看名次不看分数。
        hits.append({
            "chunk_id": chunk_id,
            # 这里是内积得分（越大越相关）。它的量纲与稠密路的余弦相似度不同，
            # 所以两路的分数绝不能拿去比大小，只能交给 RRF 按名次融合。
            "score": float(hit.get("distance", 0.0)),
            "doc_id": entity.get("doc_id"),
            "page_no": entity.get("page_no"),
            "content": entity.get("content", ""),
        })
    return hits


def count_entities(name: Optional[str] = None) -> int:
    """统计集合内实体总数，供知识库状态接口使用"""
    name = name or settings.milvus_collection
    if not collection_exists(name):
        return 0
    try:
        stats = get_client().get_collection_stats(collection_name=name)
        return int(stats.get("row_count", 0))
    except Exception as exc:
        logger.error("统计 Milvus 实体数失败：%s", exc)
        return 0
