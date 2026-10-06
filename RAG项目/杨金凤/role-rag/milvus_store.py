"""Milvus 向量存储访问层：客户端懒加载单例 + collection 管理 + 检索/全量拉取。

职责
----
封装 pymilvus，提供知识库 collection 与长期记忆 collection 的建表（schema + HNSW 索引）、
写入、按 source 覆盖删除、向量检索、全量拉取等操作。USE_MILVUS 开关与 MILVUS_URI /
MILVUS_COLLECTION / EMBED_DIM 等常量集中于此，供 ingest.py / retrieval.py 复用。

依赖
----
- pymilvus（MilvusClient / DataType / exceptions.MilvusException）：延迟 import，避免模块加载即建连

输入输出契约
------------
- search(collection_name, query_vector, top_k) -> [{"id","content","page","similarity",...}]
- query_all(collection_name) -> list[dict]（BM25 数据源，含 id/content/page/source/domain 等）
- insert(collection_name, records) -> result
- delete_by_source(collection_name, source) -> None
- create_collection / create_memory_collection：幂等建表

异常策略：异常直接抛，不吞（由上层决定降级或中断）；仅「collection 不存在」这一过渡态
（角色已建、库还没灌数据）在检索侧降级返回 []。
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

USE_MILVUS = os.getenv("USE_MILVUS", "true").lower() in ("1", "true", "yes")
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hypertension_guide")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# 字段名常量（知识库 schema 共 10 个字段）
ID_FIELD = "id"                         # 主键：chunk 唯一标识（如 p12_c0）
EMBED_FIELD = "embedding"               # 向量字段：BGE-m3 生成的 1024 维 embedding
CONTENT_FIELD = "content"               # 正文：子块内容（~400 字），检索与 BM25 的语料
PAGE_FIELD = "page"                     # 页码：来源 PDF 页码，供溯源引用
SOURCE_FIELD = "source"                 # 来源文件名：覆盖式删除的过滤键
DOMAIN_FIELD = "domain"                 # 业务域：多角色场景区分语料归属
CREATED_AT_FIELD = "created_at"         # 入库时间戳（秒）
UPDATED_AT_FIELD = "updated_at"         # 更新时间戳（秒）
SUMMARY_FIELD = "summary"               # 预生成摘要：供列表页直接展示，免二次 LLM 摘要
PARENT_CONTENT_FIELD = "parent_content"  # 父块全文：检索在子块做、最终喂给 LLM 的是父块

# 检索/全量拉取时返回的标量字段（向量字段体积大，检索结果不回传）
_SCALAR_FIELDS = [
    CONTENT_FIELD, PAGE_FIELD, SOURCE_FIELD, DOMAIN_FIELD,
    CREATED_AT_FIELD, UPDATED_AT_FIELD, SUMMARY_FIELD, PARENT_CONTENT_FIELD,
]

# 相对旧 schema 新增的字段；create_collection 据此检测旧 collection 是否需 drop 重建
_REQUIRED_SCALAR_FIELDS = {
    CREATED_AT_FIELD, UPDATED_AT_FIELD, SUMMARY_FIELD, PARENT_CONTENT_FIELD,
}

# 长期记忆 collection（跨会话保留用户客观事实）
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "user_memory")
USER_ID_FIELD = "user_id"


@lru_cache(maxsize=1)
def get_milvus_client():
    """懒加载 MilvusClient 单例（进程内缓存）。

    Returns:
        MilvusClient 实例。

    Raises:
        Milvus 连接失败时向上抛（不降级，向量检索是主链路依赖）。

    延迟 import 避免模块加载即建连；@lru_cache 保证整个进程只建一次连接。
    """
    from pymilvus import MilvusClient

    logger.info("连接 Milvus：%s ...", MILVUS_URI)
    return MilvusClient(uri=MILVUS_URI)


def create_collection(name: str, dim: int = EMBED_DIM):
    """创建（幂等）+ 建 HNSW/COSINE 索引 + 载入内存。

    Args:
        name: collection 名。
        dim: 向量维度（默认 EMBED_DIM=1024）。

    Returns:
        None。

    Raises:
        RuntimeError: collection 已存在但缺少新字段（created_at/updated_at/summary/
        parent_content）时抛出，提示手动 drop 后重跑。

    schema 变更检测：Milvus 不支持给已有 collection 加字段，只能 drop 重建；
    故已存在时校验字段集，缺新字段直接抛错指引重建，避免写入时才发现字段对不上。
    """
    from pymilvus import DataType

    client = get_milvus_client()
    if client.has_collection(name):
        existing = {f["name"] for f in client.describe_collection(name)["fields"]}
        missing = _REQUIRED_SCALAR_FIELDS - existing
        if missing:
            raise RuntimeError(
                f"collection {name} 缺少新字段 {sorted(missing)}，"
                "请先 milvus_store.drop_collection 后重新 python ingest.py"
            )
        client.load_collection(name)
        return

    schema = client.create_schema()
    # 主键用 VARCHAR（chunk id 形如 p12_c0），非自增 int，便于按 source/page 生成稳定 id
    schema.add_field(ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(EMBED_FIELD, DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(CONTENT_FIELD, DataType.VARCHAR, max_length=65535)  # 正文可能长，取 VARCHAR 上限
    schema.add_field(PAGE_FIELD, DataType.INT64)
    schema.add_field(SOURCE_FIELD, DataType.VARCHAR, max_length=255)
    schema.add_field(DOMAIN_FIELD, DataType.VARCHAR, max_length=255)
    schema.add_field(CREATED_AT_FIELD, DataType.INT64)
    schema.add_field(UPDATED_AT_FIELD, DataType.INT64)
    schema.add_field(SUMMARY_FIELD, DataType.VARCHAR, max_length=512)  # 摘要短，512 够用
    schema.add_field(PARENT_CONTENT_FIELD, DataType.VARCHAR, max_length=65535)  # 父块 ~1500 字

    index_params = client.prepare_index_params()
    # HNSW 参数：M=16 是每个节点最大连接数（16 是内存/召回率的折中，Milvus 推荐 4~64）；
    # efConstruction=200 是建图时的搜索宽度（更大则构图更精细、召回更高，但建库更慢，
    # 200 是常见默认）。查询时的 ef 由 Milvus 运行时按 nprobe/limit 自适应，无需显式设。
    index_params.add_index(
        field_name=EMBED_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(name, schema=schema, index_params=index_params)
    client.load_collection(name)
    logger.info("已创建 collection：%s（dim=%d）", name, dim)


def insert(collection_name: str, records: list[dict]):
    """写入 records（list[dict]，每条含 id/embedding/content/page/source/domain 等）。

    Args:
        collection_name: collection 名。
        records: 待写入的记录列表。

    Returns:
        Milvus insert 的返回值（含主键列表等）。

    Raises:
        Milvus 写入异常向上抛。

    写后 flush，让数据立即可检索（否则 bounded consistency 下 search/query 可能看不到刚写入的数据）。
    """
    client = get_milvus_client()
    result = client.insert(collection_name, data=records)
    client.flush(collection_name)
    return result


def search(collection_name: str, query_vector: list[float], top_k: int, filter_expr: str | None = None):
    """向量召回 top_k 条，返回 [{"id","content","page","similarity",...}, ...]。

    Args:
        collection_name: collection 名。
        query_vector: 查询向量（与库内同维度、已归一化）。
        top_k: 返回条数。
        filter_expr: 可选过滤表达式（如 source/domain 过滤），None 则不过滤。

    Returns:
        命中列表，每项含 id/content/page/similarity/created_at/updated_at/summary/parent_content。
        similarity 语义：Milvus COSINE 下 distance 字段即余弦相似度（越大越相似），直接采用；
        （与 Chroma 相反——Chroma 的 distance 是距离、越小越近，故 retrieval.py 用 1-dist 转成相似度。）

    Raises:
        collection 不存在以外的 MilvusException 向上抛；「collection not found」降级返回 []。
    """
    from pymilvus.exceptions import MilvusException

    client = get_milvus_client()
    try:
        res = client.search(
            collection_name,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=_SCALAR_FIELDS,
        )
    except MilvusException as e:
        # 降级路径：collection 不存在（角色已建、库还没灌数据的过渡期）返回空上下文
        if "collection not found" in str(e).lower():
            logger.warning("collection %s 不存在，降级为空上下文", collection_name)
            return []
        raise
    hits = []
    for hit in res[0]:
        entity = hit.get("entity", {})
        hits.append({
            "id": hit[ID_FIELD],
            "content": entity.get(CONTENT_FIELD, ""),
            "page": int(entity.get(PAGE_FIELD, 0)),
            "similarity": round(float(hit["distance"]), 4),  # COSINE 的 distance 即相似度，越大越好
            "created_at": int(entity.get(CREATED_AT_FIELD, 0)),
            "updated_at": int(entity.get(UPDATED_AT_FIELD, 0)),
            "summary": entity.get(SUMMARY_FIELD, ""),
            "parent_content": entity.get(PARENT_CONTENT_FIELD, ""),
        })
    return hits


def delete_by_source(collection_name: str, source: str):
    """按 source 过滤删除该来源旧数据，并 flush 让删除生效（否则计数瞬时不准）。

    Args:
        collection_name: collection 名。
        source: 来源文件名。

    Returns:
        None。

    Raises:
        Milvus 删除异常向上抛。
    """
    client = get_milvus_client()
    client.delete(collection_name, filter=f'{SOURCE_FIELD} == "{source}"')
    client.flush(collection_name)


def drop_collection(name: str):
    """删除 collection（用于 schema 变更后重建）。

    Args:
        name: collection 名。

    Returns:
        None。
    """
    get_milvus_client().drop_collection(name)


def list_collections():
    """列出所有 collection 名。

    Returns:
        list[str]。
    """
    return get_milvus_client().list_collections()


def query_all(collection_name: str) -> list[dict]:
    """全量拉取（BM25 数据源），返回 list[dict]（含 id/content/page/source/domain 等）。

    Args:
        collection_name: collection 名。

    Returns:
        全量 chunk 列表；collection 为空则返回 []。

    Raises:
        Milvus 查询异常向上抛（BM25 构建由 ingest.load_keyword_index 捕获降级）。

    用 query_iterator 分批拉取，避免一次性拉全量撑爆内存。
    """
    client = get_milvus_client()
    it = client.query_iterator(
        collection_name, batch_size=1000, output_fields=[ID_FIELD] + _SCALAR_FIELDS
    )
    rows = []
    while True:
        batch = it.next()
        if not batch:
            break
        rows.extend(batch)
    return rows


def list_documents(collection_name: str) -> list[dict]:
    """按 source 聚合统计，返回 [{"source","chunk_count","created_at"}, ...]。

    Args:
        collection_name: collection 名。

    Returns:
        每个来源一条聚合记录；collection 不存在时降级返回 []。

    Raises:
        collection 不存在以外的 MilvusException 向上抛。

    created_at 取该 source 下 chunk 的最小入库时间戳（同一文档各 chunk 时间一致，取 min 稳妥）。
    """
    from pymilvus.exceptions import MilvusException

    client = get_milvus_client()
    try:
        it = client.query_iterator(
            collection_name,
            batch_size=1000,
            output_fields=[SOURCE_FIELD, CREATED_AT_FIELD],
        )
        docs: dict[str, dict] = {}
        while True:
            batch = it.next()
            if not batch:
                break
            for row in batch:
                source = row.get(SOURCE_FIELD, "")
                ts = int(row.get(CREATED_AT_FIELD, 0))
                if source not in docs:
                    docs[source] = {"source": source, "chunk_count": 0, "created_at": ts}
                docs[source]["chunk_count"] += 1
                docs[source]["created_at"] = min(docs[source]["created_at"], ts)
    except MilvusException as e:
        msg = str(e).lower()
        # 降级路径：collection 不存在（未灌数据）返回空列表
        if "collection not found" in msg or "can't find collection" in msg:
            logger.warning("collection %s 不存在，跳过", collection_name)
            return []
        raise
    return list(docs.values())


def create_memory_collection(name: str = MEMORY_COLLECTION, dim: int = EMBED_DIM):
    """创建（幂等）user_memory collection：id/user_id/content/embedding + HNSW/COSINE 索引。

    Args:
        name: collection 名（默认 MEMORY_COLLECTION=user_memory）。
        dim: 向量维度。

    Returns:
        None。

    Raises:
        Milvus 建表异常向上抛。

    memory schema 与知识库不同（无 page/source/domain/summary 等），故独立建，不复用 create_collection。
    用 user_id 字段 + 检索时过滤，实现「同一用户的多条记忆」隔离。
    """
    from pymilvus import DataType

    client = get_milvus_client()
    if client.has_collection(name):
        client.load_collection(name)
        return

    schema = client.create_schema()
    schema.add_field(ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(USER_ID_FIELD, DataType.VARCHAR, max_length=255)  # 记忆归属键（user_id），检索时过滤
    schema.add_field(CONTENT_FIELD, DataType.VARCHAR, max_length=65535)
    schema.add_field(EMBED_FIELD, DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    # HNSW 参数同 create_collection（M=16 / efConstruction=200）
    index_params.add_index(
        field_name=EMBED_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(name, schema=schema, index_params=index_params)
    client.load_collection(name)
    logger.info("已创建 memory collection：%s（dim=%d）", name, dim)


def search_memory(
    user_id: str,
    query_vector: list[float],
    top_k: int,
    collection_name: str = MEMORY_COLLECTION,
) -> list[str]:
    """按 user_id 过滤召回 top_k 条记忆，返回 content 列表；collection 不存在降级 []。

    Args:
        user_id: 用户标识（过滤键）。
        query_vector: 查询向量。
        top_k: 返回条数。
        collection_name: memory collection 名。

    Returns:
        content 字符串列表；collection 不存在时降级返回 []。

    Raises:
        collection 不存在以外的 MilvusException 向上抛。
    """
    from pymilvus.exceptions import MilvusException

    client = get_milvus_client()
    try:
        res = client.search(
            collection_name,
            data=[query_vector],
            limit=top_k,
            filter=f'{USER_ID_FIELD} == "{user_id}"',  # 只召回该用户的记忆
            output_fields=[CONTENT_FIELD],
        )
    except MilvusException as e:
        # 降级路径：memory collection 不存在时返回空记忆（首次使用未建库）
        if "collection not found" in str(e).lower():
            logger.warning("memory collection %s 不存在，降级为空记忆", collection_name)
            return []
        raise
    return [hit.get("entity", {}).get(CONTENT_FIELD, "") for hit in res[0]]
"""
`milvus_store.py`是 Milvus 向量库的访问封装层，分成三部分。第一，`get_milvus_client()`懒加载 Milvus 客户端，只在第一次使用时才建立连接；`create_collection()`用来创建知识库集合，定义字段并建立 HNSW 向量索引；`create_memory_collection()`创建单独的记忆集合，用`user_id`区分不同用户的记忆。第二，数据写入和删除：`insert()`把 chunk 记录写入 Milvus；`delete_by_source()`按文档来源删除旧数据；`drop_collection()`删除整个集合，`list_collections()`查看所有集合，`list_documents()`统计每个文档的块数量。第三，数据查询：`search()`做向量召回，找不到集合就返回空；`query_all()`分批读出全部 chunk，给 BM25 建索引用；`search_memory()`根据用户 ID，召回该用户的长期记忆。除集合不存在时返回空以外，其他异常直接向上抛出，交给上层代码处理。
"""