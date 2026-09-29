"""Milvus 向量库封装。

包含两个 Collection：
1. persona_knowledge    知识库分块（稠密向量 + BM25 稀疏向量，partition key = persona_id）
2. user_long_term_memory 长期记忆摘要（稠密向量，partition key = persona_id，标量过滤 user_id）

检索方式：稠密向量 ANN（HNSW/COSINE） + 稀疏向量 BM25 混合检索 + RRFRanker 融合。

为什么需要「混合检索」而不是只用向量？
- 稠密向量擅长语义相似（“我很焦虑”能召回“紧张不安”的段落），
  但对专有名词、术语缩写、精确短语不敏感；
- BM25 稀疏检索恰好相反：字面命中强、语义泛化弱。
两者用 RRF（Reciprocal Rank Fusion）按名次融合，兼顾语义与关键词，
是本项目检索质量的关键。

**为什么 partition key 选 persona_id？**
三个心理医生角色的知识库彼此独立，检索时必定带 persona_id 条件。
Milvus 的分区键会把同一 persona 的数据物理聚集在一起，
查询时只扫描对应分区，避免全库扫描，检索延迟和内存占用都显著下降。
注意：分区键必须是查询里最常见的过滤维度，选错反而更慢；
且分区键字段不支持频繁更新，所以它更适合这种“写入时即确定、几乎不变”的字段。
"""
import threading
import time
from typing import Any, Dict, List, Optional

from pymilvus import (AnnSearchRequest, DataType, Function, FunctionType, MilvusClient,
                      RRFRanker)
from pymilvus.exceptions import MilvusException

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("db.milvus")

# 进程级单例。MilvusClient 内部维护 gRPC 连接，创建/销毁成本高，
# 必须全局复用，否则每次检索都新建连接会把 Milvus 的连接数打满。
_client: Optional[MilvusClient] = None
# 互斥锁：FastAPI 是多线程执行同步函数的（线程池），
# 多个线程可能同时首次调用 get_client()。若没有锁，会创建出多个客户端
# 并互相覆盖全局变量，造成连接泄漏（被覆盖的那个没人再引用但连接还开着）。
_lock = threading.Lock()


def get_client() -> MilvusClient:
    """返回全局唯一的 MilvusClient（双重检查锁单例）。"""
    global _client
    # 第一次检查（无锁）：绝大多数调用时 _client 已存在，直接返回，
    # 完全避开加锁开销——这是性能关键路径。
    if _client is None:
        with _lock:
            # 第二次检查（持锁）：拿到锁的瞬间，可能已被别的线程创建好了，
            # 所以必须再判一次，否则还是会重复创建。这就是“双重检查锁”的由来。
            if _client is None:
                _client = MilvusClient(uri=settings.milvus_uri)
                logger.info("Milvus 客户端已创建：%s", settings.milvus_uri)
    return _client


# ============================ Collection 定义 ============================
def ensure_knowledge_collection(drop: bool = False) -> None:
    """创建知识库 Collection（不存在时）。

    :param drop: True 时先删除再重建（仅用于开发/重建知识库，会清空数据）
    """
    client = get_client()
    name = settings.milvus_collection
    if client.has_collection(name):
        if drop:
            # 删除是不可逆操作，用 warning 级别留痕，方便事后追溯数据为何消失。
            client.drop_collection(name)
            logger.warning("已删除已存在的 Collection：%s", name)
        else:
            # 已存在直接返回：保证本函数可反复调用（幂等），
            # 服务每次启动都能安全执行。
            return

    # auto_id=True：主键由 Milvus 自增生成，应用无需维护 ID 生成器，
    # 也避免多实例并发写入时主键冲突。
    # enable_dynamic_field=False：关闭动态字段，未在 schema 中声明的字段会被拒绝，
    # 好处是数据严格受约束（不会悄悄写进拼错的字段名）。
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    # 主键：INT64 自增。用整型而非字符串，索引更小、比较更快。
    schema.add_field("id", DataType.INT64, is_primary=True)
    # 分区键：按角色物理分区，检索时只扫目标分区（见模块 docstring 的说明）。
    schema.add_field("persona_id", DataType.INT64, is_partition_key=True)
    # doc_id/chunk_id 用于关联 MySQL 里的文档与分块元数据，
    # 形成「MySQL 存正文与元数据、Milvus 存向量」的分工。
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("chunk_id", DataType.INT64)
    # text 是分块原文：既是返回给 LLM 的上下文，也是 BM25 的输入字段。
    # max_length 用 65535（VARCHAR 上限）是因为分块长度不固定，留足余量。
    # enable_analyzer=True 让该字段参与分词，BM25 Function 才能工作。
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    # summary 是分块的简短摘要，用于列表展示与低成本预览，长度上限 2048 足够。
    schema.add_field("summary", DataType.VARCHAR, max_length=2048)
    # source 记录来源（文件名/URL），便于前端展示引用出处，增强可信度。
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    # 时间戳统一用 INT64（Unix 秒）而不是 DATETIME：
    # Milvus 对整型的过滤/排序更高效，且不受时区影响。
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("updated_at", DataType.INT64)
    # 稠密向量，维度由配置决定（必须与 Embedding 模型输出维度严格一致，
    # 否则写入会报维度不匹配——这是换模型时最容易踩的坑）。
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
    # 稀疏向量：不用应用自己算，而是由下面的 BM25 Function 从 text 自动生成。
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # Function 是 Milvus 的“内置加工函数”：写入时自动把 text 分词并生成
    # sparse_vector，查询时把 query 文本同样处理后做 BM25 匹配。
    # 好处是应用层无需引入额外的 BM25 实现，索引与数据天然一致、不会不同步。
    schema.add_function(
        Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    index_params = client.prepare_index_params()
    # 稠密索引用 HNSW：图索引，查询快、召回高，适合本场景的数据量级。
    # - M=16：每个节点的邻居数。越大召回越好但内存占用越高（经验值 16-64）。
    # - efConstruction=200：建索引时的候选队列长度，越大索引质量越好但建得越慢。
    #   这两个参数只影响“建索引阶段”，查询时另有 ef 参数。
    # - COSINE：余弦相似度。文本向量经归一化后，余弦只关注方向（语义），
    #   不受文本长度影响，是文本检索的常规选择。
    index_params.add_index(
        field_name="vector", index_type="HNSW", metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 稀疏索引用倒排索引（SPARSE_INVERTED_INDEX）：本质是倒排表，
    # 每个词项指向包含它的文档列表，这正是 BM25 需要的结构。
    index_params.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
    )
    # 注意：创建 Collection 时必须同时传入 schema 和 index_params，
    # 这样 Milvus 会在建表时一并建好索引；否则后续要手动 create_index，
    # 且在建立索引前无法 load/检索。
    client.create_collection(name, schema=schema, index_params=index_params)
    logger.info("知识库 Collection 创建成功：%s", name)


def ensure_memory_collection(drop: bool = False) -> None:
    """创建长期记忆 Collection（不存在时）。

    与知识库的区别：这里存的是「用户自己的对话摘要」，只做纯稠密检索
    （记忆没有关键词匹配的需求，且摘要长度短，BM25 收益不大）。
    """
    client = get_client()
    name = settings.milvus_memory_collection
    if client.has_collection(name):
        if drop:
            client.drop_collection(name)
            logger.warning("已删除已存在的 Collection：%s", name)
        else:
            return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    # 同样以 persona_id 为分区键：记忆天然按角色隔离
    #（对 CBT 角色倾诉的内容，不该在人本角色下被检索到）。
    schema.add_field("persona_id", DataType.INT64, is_partition_key=True)
    # user_id 是普通标量字段而不是分区键，因为分区键只能有一个；
    # 用户维度靠下面的 INVERTED 索引 + filter 表达式来加速。
    schema.add_field("user_id", DataType.INT64)
    schema.add_field("conversation_id", DataType.INT64)
    # 摘要比知识分块长（要容纳一段对话的压缩结果），上限放到 8192。
    schema.add_field("summary", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", index_type="HNSW", metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 给 user_id 建标量倒排索引：检索时条件是 persona_id + user_id，
    # 没有索引就要逐行比对 user_id；有索引可直接定位到该用户的少量记录，
    # 在数据量大时对延迟影响非常明显。
    index_params.add_index(field_name="user_id", index_type="INVERTED")
    client.create_collection(name, schema=schema, index_params=index_params)
    logger.info("长期记忆 Collection 创建成功：%s", name)


def ensure_collections() -> None:
    """初始化全部 Collection 并加载到内存。"""
    ensure_knowledge_collection()
    ensure_memory_collection()
    # 创建完必须 load：Milvus 的检索只作用于「已加载」的集合。
    # 未加载时查询会报 "collection not loaded"，所以这两步必须成对出现。
    load_collection(settings.milvus_collection)
    load_collection(settings.milvus_memory_collection)


def load_collection(name: str) -> None:
    """把 Collection 加载进查询节点内存（检索前置条件）。"""
    try:
        get_client().load_collection(name)
    except MilvusException as exc:  # pragma: no cover
        # 加载失败不抛出：可能只是已经加载过了（重复 load 会报错），
        # 记日志即可，不应让启动流程中断。
        logger.error("加载 Collection 失败 %s：%s", name, exc)


def drop_all() -> None:
    """删除本项目的全部 Collection（危险操作，仅供测试/重置环境使用）。"""
    client = get_client()
    for name in (settings.milvus_collection, settings.milvus_memory_collection):
        if client.has_collection(name):
            client.drop_collection(name)
            logger.warning("已删除 Collection：%s", name)


# ============================ 知识库写入 / 检索 ============================
def insert_chunks(records: List[Dict[str, Any]]) -> List[int]:
    """写入知识分块，返回 Milvus 主键列表（用于回写 MySQL）。

    :return: 与 records 顺序一致的主键列表；空输入返回空列表
    """
    if not records:
        # 提前返回：避免对空列表调用 insert（部分版本会报参数错误），
        # 也省一次无意义的网络往返。
        return []
    client = get_client()
    result = client.insert(settings.milvus_collection, records)
    # flush 把数据从内存缓冲落盘并让其可被检索。
    # 代价是较慢（是一次同步操作），所以批量写入时应在最后统一 flush，
    # 而不是每条都 flush。这里因为已按批调用，所以可以接受。
    client.flush(settings.milvus_collection)
    # result.get("ids") 在异常情况下可能为 None，用 or [] 兜底防止迭代报错。
    ids = result.get("ids", []) or []
    logger.info("Milvus 写入知识分块 %d 条", len(records))
    # 转成 int：Milvus 返回的 ID 可能是 numpy.int64 或字符串，
    # 直接写回 MySQL 的 INT 字段会因类型不匹配报错。
    return [int(i) for i in ids]


def hybrid_search_knowledge(persona_id: int, query_dense: List[float], query_text: str,
                            top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """稠密 + BM25 稀疏混合检索，按 persona_id 过滤。

    :param query_dense: 已向量化的问题向量（由上层 Embedder 生成）
    :param query_text: 问题原文，供 BM25 分词匹配
    """
    top_k = top_k or settings.retrieve_top_k
    # 过滤表达式用 f-string 内嵌 int(persona_id)：这里必须先转 int，
    # 既避免字符串注入风险，也保证表达式类型与字段类型一致。
    # 由于 persona_id 是分区键，该条件还能触发分区裁剪（只扫一个分区）。
    expr = f"persona_id == {int(persona_id)}"
    reqs = [
        # 两路召回各自取 top_k，再由 RRF 融合。注意两路的 param 必须与
        # 建索引时的 metric_type 保持一致，否则会报错。
        AnnSearchRequest(data=[query_dense], anns_field="vector",
                         param={"metric_type": "COSINE"}, limit=top_k, expr=expr),
        # 稀疏路传入的是「原始文本」而不是向量：Milvus 会用 bm25_fn 自动编码。
        AnnSearchRequest(data=[query_text], anns_field="sparse_vector",
                         param={"metric_type": "BM25"}, limit=top_k, expr=expr),
    ]
    # 记录耗时用于性能观测：检索变慢时能从日志立刻看出是哪一环。
    start = time.time()
    try:
        res = get_client().hybrid_search(
            settings.milvus_collection, reqs,
            # RRFRanker(60)：用 RRF 算法融合两路结果，60 是公式中的平滑常数 k。
            # RRF 只看名次不看分数，因此无需把余弦相似度与 BM25 分数归一化对齐，
            # 这正是它在混合检索里被广泛使用的原因。
            ranker=RRFRanker(60), limit=top_k,
            # output_fields 显式列出要返回的字段：只取需要的数据，
            # 避免把大字段（如完整向量）也拉回来浪费带宽。
            output_fields=["doc_id", "chunk_id", "text", "summary", "source", "persona_id"],
        )
    except MilvusException as exc:
        # 降级返回空列表：上层会得到“没有参考资料”，仍能生成不依赖知识库的回复。
        # 若在这里抛异常，用户会看到 500——对陪伴类产品体验伤害很大。
        logger.error("Milvus 混合检索失败 persona=%s：%s", persona_id, exc)
        return []

    hits: List[Dict[str, Any]] = []
    # res 是「每个查询一组结果」的列表，这里只有 1 个查询，所以取 res[0]。
    # res 为 None 时用 [] 兜底，保证下面的 for 安全。
    for hit in res[0] if res else []:
        # entity 可能为 None（例如某些版本在无 output_fields 时返回空），用 {} 兜底。
        entity = hit.get("entity", {}) or {}
        hits.append({
            "milvus_id": hit.get("id"),
            "doc_id": entity.get("doc_id"),
            "chunk_id": entity.get("chunk_id"),
            # persona_id 优先取实体值，取不到则回退到入参（保证字段永不为 None）。
            "persona_id": entity.get("persona_id", persona_id),
            # 文本字段统一兜底成空字符串，避免上层做字符串拼接时出现 None。
            "text": entity.get("text", ""),
            "summary": entity.get("summary", ""),
            "source": entity.get("source", ""),
            # distance 在 COSINE 下是相似度（越大越相似），转 float 是为了
            # 把 numpy 标量变成原生类型，便于 JSON 序列化。
            "score": float(hit.get("distance", 0.0)),
        })
    logger.info("Milvus 混合检索 persona=%s 命中 %d 条，耗时 %.0fms",
                persona_id, len(hits), (time.time() - start) * 1000)
    return hits


def dense_search_knowledge(persona_id: int, query_dense: List[float],
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """纯稠密向量检索（混合检索降级方案），返回余弦相似度。

    使用场景：BM25 稀疏索引不可用、或 query_text 缺失时的兜底路径。
    虽然召回质量略低于混合检索，但保证功能始终可用。
    """
    top_k = top_k or settings.retrieve_top_k
    try:
        res = get_client().search(
            settings.milvus_collection, data=[query_dense], anns_field="vector",
            # 这里用 filter 参数（单路 search），而混合检索里是放在
            # AnnSearchRequest 的 expr 上——两处 API 命名不同，容易混淆。
            filter=f"persona_id == {int(persona_id)}", limit=top_k,
            output_fields=["doc_id", "chunk_id", "text", "summary", "source"],
            # 单路检索的 search_params 直接平铺，不像混合检索那样包在 req 里。
            search_params={"metric_type": "COSINE"},
        )
    except MilvusException as exc:
        logger.error("Milvus 稠密检索失败 persona=%s：%s", persona_id, exc)
        return []

    hits = []
    for hit in res[0] if res else []:
        entity = hit.get("entity", {}) or {}
        hits.append({
            "milvus_id": hit.get("id"),
            "doc_id": entity.get("doc_id"),
            "chunk_id": entity.get("chunk_id"),
            # 未请求 persona_id 字段，直接用入参填充（语义上等价）。
            "persona_id": persona_id,
            "text": entity.get("text", ""),
            "summary": entity.get("summary", ""),
            "source": entity.get("source", ""),
            "score": float(hit.get("distance", 0.0)),
        })
    return hits


def delete_knowledge_by_doc(doc_id: int, persona_id: int) -> int:
    """删除某文档对应的全部向量。

    为什么删除条件要同时带 doc_id 和 persona_id？
    因为 doc_id 只在单个角色内唯一（不同角色各自从 1 编号），
    只按 doc_id 删会误删其他角色的同名文档——这是典型的数据串库风险。
    """
    try:
        res = get_client().delete(
            settings.milvus_collection, filter=f"doc_id == {int(doc_id)} and persona_id == {int(persona_id)}"
        )
        # 删除后 flush，让删除立即生效（否则可能仍被检索到旧数据）。
        get_client().flush(settings.milvus_collection)
        count = int(res.get("delete_count", 0))
        logger.info("删除文档向量 doc_id=%s persona=%s 共 %d 条", doc_id, persona_id, count)
        return count
    except MilvusException as exc:
        # 返回 0 表示“没删成功”：调用方据此判断是否需要提示管理员，
        # 但不抛异常（删文档失败不该让整个接口报错）。
        logger.error("删除文档向量失败 doc_id=%s：%s", doc_id, exc)
        return 0


def delete_knowledge_by_persona(persona_id: int) -> int:
    """删除某角色下的全部知识向量（清空该角色的知识库）。"""
    try:
        res = get_client().delete(settings.milvus_collection, filter=f"persona_id == {int(persona_id)}")
        get_client().flush(settings.milvus_collection)
        return int(res.get("delete_count", 0))
    except MilvusException as exc:
        logger.error("删除角色向量失败 persona=%s：%s", persona_id, exc)
        return 0


def count_knowledge(persona_id: Optional[int] = None) -> int:
    """统计知识向量条数；persona_id 为空时统计整表。"""
    try:
        if persona_id is None:
            # 整表计数走 collection stats（读元数据，极快），
            # 而不是 query count(*)（需要真实扫描）。
            stats = get_client().get_collection_stats(settings.milvus_collection)
            return int(stats.get("row_count", 0))
        # 按角色计数只能走 query count(*)：Milvus 的聚合统计支持有限，
        # 但因为有分区裁剪，实际只扫该角色一个分区，成本可控。
        res = get_client().query(
            settings.milvus_collection, filter=f"persona_id == {int(persona_id)}",
            output_fields=["count(*)"],
        )
        # 结果形如 [{"count(*)": 12}]，取第一行的值；无结果时返回 0。
        return int(res[0].get("count(*)", 0)) if res else 0
    except MilvusException as exc:
        logger.error("统计向量数量失败：%s", exc)
        return 0


# ============================ 长期记忆写入 / 检索 ============================
def insert_memory(persona_id: int, user_id: int, conversation_id: int,
                  summary: str, vector: List[float]) -> Optional[int]:
    """写入一条长期记忆（对话摘要）。

    :return: 新记录的主键；失败时返回 None（调用方据此决定是否重试）
    """
    try:
        # 用服务端时间戳而非客户端传入：避免多实例时钟不一致导致排序错乱。
        now = int(time.time())
        result = get_client().insert(settings.milvus_memory_collection, [{
            # 显式 int() 转换：确保写入类型与 schema 声明的 INT64 完全一致，
            # 防止上层传 numpy 整型或字符串导致写入失败。
            "persona_id": int(persona_id),
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            "summary": summary,
            "created_at": now,
            "vector": vector,
        }])
        get_client().flush(settings.milvus_memory_collection)
        ids = result.get("ids", []) or []
        logger.info("写入长期记忆 user=%s persona=%s", user_id, persona_id)
        # 只插了一条，取 ids[0]；空列表时返回 None。
        return int(ids[0]) if ids else None
    except MilvusException as exc:
        # 记忆写入失败属于“可丢”数据：用户不会因此感知到错误，
        # 只是这次对话没被记住，因此记 error 日志即可，不抛出。
        logger.error("写入长期记忆失败：%s", exc)
        return None


def search_memory(persona_id: int, user_id: int, query_dense: List[float],
                  top_k: int = 3) -> List[Dict[str, Any]]:
    """按用户 + 角色检索长期记忆摘要。

    双重过滤的意义：既不能让 A 用户看到 B 用户的记忆（user_id），
    也不能让角色之间互相串味（persona_id）。
    默认 top_k=3 是刻意的：记忆只作为背景补充，取太多会稀释当前对话的权重，
    还可能把模型带偏到很久以前的话题上。
    """
    try:
        res = get_client().search(
            settings.milvus_memory_collection, data=[query_dense], anns_field="vector",
            filter=f"persona_id == {int(persona_id)} and user_id == {int(user_id)}",
            limit=top_k, output_fields=["conversation_id", "summary", "created_at"],
            search_params={"metric_type": "COSINE"},
        )
    except MilvusException as exc:
        logger.error("检索长期记忆失败：%s", exc)
        return []

    items = []
    for hit in res[0] if res else []:
        entity = hit.get("entity", {}) or {}
        items.append({
            "conversation_id": entity.get("conversation_id"),
            "summary": entity.get("summary", ""),
            "created_at": entity.get("created_at"),
            "score": float(hit.get("distance", 0.0)),
        })
    return items


def health_check() -> bool:
    """健康检查：能列出集合即视为可用（最轻量的探测调用）。"""
    try:
        get_client().list_collections()
        return True
    except Exception as exc:
        # 这里捕获的是宽泛的 Exception 而非仅 MilvusException：
        # 建连失败可能抛 gRPC 层的异常（如 StatusCode.UNAVAILABLE），
        # 健康检查必须把这些也归为“不可用”，而不是让异常冒泡成 500。
        logger.error("Milvus 健康检查失败：%s", exc)
        return False