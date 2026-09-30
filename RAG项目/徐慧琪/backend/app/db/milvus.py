"""Milvus 集合 schema、索引与读写。

存在的理由（技术方案 4.4 + 5.2）：law_chunks 要在一个 collection 里同时存 dense 与
sparse 两种向量，才能在**一次请求**里发两个 AnnSearchRequest 再由内置 RRFRanker 融合——
这正是"复合向量 + RRF 粗排"的落点。拆成两个集合就退化成手工融合，多一次往返。

主键用 chunk_id 而非技术方案 4.4 写的 auto_id：auto_id 会让重跑入库生成一份新 id、
旧数据原样保留（集合从 3388 变 6776 且不报错），4.5 的增量 upsert 也无从实现。
详见设计文档第七节。

本模块**不导入任何模型代码**，因此它的测试不必加载 2.3GB 的 bge-m3。
"""
from __future__ import annotations

import time

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

MILVUS_URI = "http://127.0.0.1:19530"
# 生产集合名，Task 5 的 3388 条向量就灌在这里。质量闸门是 python -m pytest backend/tests，
# 测试若默认操作这个名字，每跑一次全量回归都会把生产数据静默清空——所以下面每个
# 要指定集合的函数都留了显式 name 参数，测试必须传自己的名字，不要靠 monkeypatch
# 改本模块的全局量：测试顶部 `from ... import COLLECTION` 早已把名字绑定进它自己的
# 命名空间，改这里影响不到那份副本，只会造出"以为隔离了其实没有"的假象。
COLLECTION = "law_chunks"
DENSE_DIM = 1024

# 每批写入条数。3388 条一次性 upsert 会让请求体过大，分批还能让进度可见
WRITE_BATCH = 500

# 每批最多尝试几次与退避基数（设计文档第六节"Milvus 写入失败 | 重试 3 次后抛错"）。
# 落地为"每批最多调用 3 次（首次 + 2 次重试），3 次都失败才抛"。
# 为什么要重试：一趟灌库 16.7 分钟，网络瞬时抖动不该让它整个白跑。
# 为什么只 3 次：真正故障（schema 不符、服务掉了）重试多少次都不会好，
# 无限重试只会把"炸了"拖成"看起来卡住"
UPSERT_ATTEMPTS = 3
RETRY_BACKOFF = 0.5

# (字段名, 类型, 参数)。顺序无关，但集中声明才能一眼看出与设计文档 4.2 的对应关系
FIELDS: list[tuple[str, DataType, dict]] = [
    ("chunk_id", DataType.VARCHAR, {"is_primary": True, "max_length": 16}),
    ("dense", DataType.FLOAT_VECTOR, {"dim": DENSE_DIM}),
    ("sparse", DataType.SPARSE_FLOAT_VECTOR, {}),
    ("law_id", DataType.VARCHAR, {"max_length": 64, "is_partition_key": True}),
    ("law_version", DataType.VARCHAR, {"max_length": 32}),
    ("article_no", DataType.INT64, {}),
    ("article_no_cn", DataType.VARCHAR, {"max_length": 32}),
    ("paragraph_no", DataType.INT64, {"nullable": True}),
    ("item_no", DataType.VARCHAR, {"max_length": 16, "nullable": True}),
    # 实测最长 50 字，留 5 倍余量：定短了 Milvus 会截断且不报错
    ("path", DataType.VARCHAR, {"max_length": 255}),
    ("status", DataType.VARCHAR, {"max_length": 16}),
    ("effective_date", DataType.VARCHAR, {"max_length": 10}),
    ("parent_id", DataType.VARCHAR, {"max_length": 16, "nullable": True}),
    ("chunk_type", DataType.VARCHAR, {"max_length": 16}),
    ("source_hash", DataType.VARCHAR, {"max_length": 32}),
    # 实测最长 400 字，同样留 5 倍余量
    ("text", DataType.VARCHAR, {"max_length": 2000}),
]

# 供 upsert 与检索返回用的字段全集，与 FIELDS 顺序保持一致
ALL_FIELDS = [f[0] for f in FIELDS]

# 5.2 的过滤条件落在 status / chunk_type 上；effective_date 供时效过滤，
# article_no 供将来的条款号精确通路——四个都建标量索引
SCALAR_INDEX_FIELDS = ["status", "effective_date", "chunk_type", "article_no"]

# 参数取自技术方案 4.3 / 4.4
DENSE_INDEX = {"index_type": "HNSW", "metric_type": "COSINE",
               "params": {"M": 16, "efConstruction": 200}}
SPARSE_INDEX = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}

# 检索返回给上层的字段：够组装引用（条款号 + 原文 + 路径 + 父子指针）即可
OUTPUT_FIELDS = ["chunk_id", "article_no", "article_no_cn", "paragraph_no", "item_no",
                 "path", "chunk_type", "parent_id", "text"]

# 参数取自技术方案 5.2 的"起始值"。本期不调参——正式评估集尚未建立，
# 没有调参依据（见设计文档第九节）
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_K = 60
DEFAULT_EF = 128

# 可被召回的块类型：父块有向量但语义太宽，会挤掉精确的子块
CHILD_TYPES = ["paragraph", "item"]


def get_client(uri: str = MILVUS_URI) -> MilvusClient:
    """建连。Milvus 跑在 WSL2 里，Windows 侧经 localhost 转发访问，已实测可达。"""
    return MilvusClient(uri=uri)


def build_schema():
    """按 FIELDS 声明构造 CollectionSchema。

    auto_id 显式关掉，enable_dynamic_field 也关掉——后者会让拼错的字段名
    静默写进动态字段而不报错，属于"看起来成功"的失败。
    """
    client = get_client()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, dtype, params in FIELDS:
        schema.add_field(name, dtype, **params)
    return schema


def build_index_params():
    """构造索引参数：dense 走 HNSW/COSINE，sparse 走倒排/IP，标量走 INVERTED。"""
    client = get_client()
    idx = client.prepare_index_params()
    idx.add_index("dense", **DENSE_INDEX)
    idx.add_index("sparse", **SPARSE_INDEX)
    for name in SCALAR_INDEX_FIELDS:
        idx.add_index(name, index_type="INVERTED")
    return idx


def create_collection(client: MilvusClient, drop_existing: bool = False,
                      name: str = COLLECTION) -> None:
    """建集合与索引。drop_existing 只在测试与需要重建时用，默认不删数据。

    name 显式可传，是为了让测试落在自己的集合上——否则测试里的
    drop_existing 与 finally 里的 drop_collection 会直接砸生产集合。
    """
    if drop_existing and client.has_collection(name):
        client.drop_collection(name)
    if client.has_collection(name):
        return
    client.create_collection(name, schema=build_schema(),
                             index_params=build_index_params())


def ensure_collection(client: MilvusClient, name: str = COLLECTION) -> None:
    """集合不存在才建。重跑入库走这个入口，不会误删已有数据。"""
    create_collection(client, drop_existing=False, name=name)


def describe(client: MilvusClient, name: str = COLLECTION) -> list[dict]:
    """返回集合字段描述。

    注意 list_indexes() 返回的是索引名字符串列表而不是 dict 列表，
    想看字段结构必须走 describe_collection()['fields']——两者别混。
    """
    return client.describe_collection(name)["fields"]


def build_filter_expr(status: str = "现行有效", child_only: bool = True) -> str:
    """构造检索过滤表达式。

    效力过滤来自 FR-3.3：已废止法条绝不能被召回。chunk_type 过滤是
    "3388 块全编码"的配套措施——父块向量与子块同集合，不滤就会互相挤占。
    """
    expr = f'status == "{status}"'
    if child_only:
        joined = ", ".join(f'"{t}"' for t in CHILD_TYPES)
        expr += f" and chunk_type in [{joined}]"
    return expr


# 默认过滤条件。提成常量是为了让调用方与测试引用同一份定义，
# 避免表达式在两处各写一遍、改一处漏一处
FILTER_EXPR = build_filter_expr()


def hybrid_search(client: MilvusClient, dense_vec: list, sparse_vec: dict,
                  top_k: int = 10, status: str = "现行有效",
                  child_only: bool = True) -> list[dict]:
    """dense + sparse 双路召回，由 Milvus 内置 RRFRanker 融合。

    融合放在 Milvus 侧而非取回两路结果后手工合并：一次请求少一次往返，
    且 RRF 的排序口径与"RRF 粗排"这道要求天然对齐（技术方案 5.2）。
    """
    expr = build_filter_expr(status, child_only)
    reqs = [
        AnnSearchRequest(data=[dense_vec], anns_field="dense",
                         param={"metric_type": "COSINE", "params": {"ef": DEFAULT_EF}},
                         limit=DENSE_TOP_K, expr=expr),
        AnnSearchRequest(data=[sparse_vec], anns_field="sparse",
                         param={"metric_type": "IP"},
                         limit=SPARSE_TOP_K, expr=expr),
    ]
    res = client.hybrid_search(COLLECTION, reqs, ranker=RRFRanker(k=RRF_K),
                               limit=top_k, output_fields=OUTPUT_FIELDS)
    # 返回结构是 list[list[hit]]，每个 hit 的字段在 hit["entity"] 里
    return [dict(hit["entity"]) for hit in res[0]]


def entity_count(client: MilvusClient, name: str = COLLECTION) -> int:
    """返回集合里**当前真实的实体数**。

    不能用 get_collection_stats()['row_count']：Milvus 3.0.1 实测它是**滞后的
    最终一致值**——同主键 upsert 两次后读到 2（真实 1）、delete 后仍是 2、
    重跑灌库 3388 条后立刻读到 6776，但约半小时后（分段合并完成）会回落到 3388。
    会收敛意味着基于它的检查时对时错，比稳定地错更难排查。
    而 upsert 覆盖正是本设计选 chunk_id 作主键想要的幂等行为，
    用错口径会把"重跑幂等成功"误报成"行数翻倍"。

    count(*) 走查询通路，反映的是覆盖与删除之后仍在的实体，才是要数的东西。
    filter 写 chunk_id != '' 是因为 Milvus 的 count(*) 需要一个过滤表达式，
    而 chunk_id 是主键、非空，这个条件对结果无影响。
    """
    got = client.query(name, filter="chunk_id != ''", output_fields=["count(*)"])
    return int(got[0]["count(*)"])


def insert_chunks(client: MilvusClient, rows: list[dict], name: str = COLLECTION) -> int:
    """按 chunk_id upsert，分批下发，返回 Milvus **实际接受**的行数。

    用 upsert 而非 insert：主键是 chunk_id，重跑时应当覆盖而不是报主键冲突。

    单批写失败最多尝试 UPSERT_ATTEMPTS 次（瞬时抖动不该废掉整趟，但也不能无限重试），
    仍失败则把原异常抛出——不做部分成功，让上层看到的是"这一跑没成功"。
    重试之所以安全，是因为主键 upsert 幂等：上一次若其实已经写进服务端（超时误判），
    再写一遍也只是同主键覆盖，不会像 insert 那样添出重复行。

    返回的是 upsert 回报的 upsert_count 累加，而不是 len(rows)：后者会让调用方的
    校验退化成 input == input，Milvus 少收也照样报满额成功（Task 5 的 main() 正是
    拿这个返回值卡 Global Constraint）。数目对不上就抛异常中断——条数不符意味着
    有行没进库，静默继续等于把丢失留到检索阶段才暴露。
    """
    if not rows:
        return 0
    written = 0
    for start in range(0, len(rows), WRITE_BATCH):
        batch = rows[start:start + WRITE_BATCH]
        for attempt in range(UPSERT_ATTEMPTS):
            try:
                written += client.upsert(name, batch)["upsert_count"]
                break
            except Exception:
                # 最后一次仍失败就把原异常整个抛出，保留 pymilvus 的原始信息——
                # 换成自己拼的 RuntimeError 会把 schema 不符这类真因盖成"写入失败"
                if attempt == UPSERT_ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_BACKOFF * 2 ** attempt)
    # 先校验再 flush：数目不符要中断时，不必再做落盘这个多余动作
    if written != len(rows):
        raise RuntimeError(
            f"Milvus 接受的行数与提交的行数不符：集合 {name} 提交 {len(rows)} 条，"
            f"实际接受 {written} 条，差 {len(rows) - written} 条")
    client.flush(name)
    return written
