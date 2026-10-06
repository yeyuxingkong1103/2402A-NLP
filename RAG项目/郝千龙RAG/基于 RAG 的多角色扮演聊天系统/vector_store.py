# -*- coding: utf-8 -*-
# 上一行为文件编码声明，保证中文注释和字符串被正确解析（按 PEP 263 必须独占一行）
"""【向量数据库 · vector_store.py】Milvus 长期知识库：一角色一分区物理隔离，COSINE 余弦检索；Milvus 未启动时自动跳过降级。"""  # 模块级文档字符串：中文名 + 文件名 + 一句话作用
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，使 set[str] | None 等新语法在旧版 Python 也可用

from datetime import datetime  # 导入 datetime，用于给向量记录打 created_at/updated_at 时间戳
from typing import Any  # 导入 Any，用于 search_vectors 返回值的字典类型标注

from config import MILVUS_COLLECTION, MILVUS_ENABLED, MILVUS_HOST, MILVUS_PORT  # 从全局配置读取集合名、启用开关、连接地址端口
from embeddings import embedding_dim  # 导入 embedding_dim 获取向量维度（BGE-m3 维度），建集合时需传入
from logger import log  # 导入统一日志器，记录连接失败/重建集合等关键事件

_connected = False  # 连接标记：True 表示已与 Milvus 建立连接，避免每次检索重复握手
_partition_cache: set[str] | None = None  # 已确认存在的分区名缓存（避免每次写入前重复 RPC 拉取分区列表）


def _connect():  # 私有方法：建立并复用与 Milvus 的连接（单例式握手）
    """连接 Milvus；未启用或失败返回 None（调用方跳过向量检索）。"""
    global _connected  # 声明使用模块级全局变量，以便成功后改写为 True
    if not MILVUS_ENABLED:  # 配置未开启 Milvus：直接返回 None，调用方据此降级到纯 BM25
        return None
    if _connected:  # 已连接直接复用，避免每次检索重复握手
        return True
    try:  # 尝试导入并连接，pymilvus 可能未安装，故放 try 内惰性导入
        from pymilvus import connections  # 延迟导入：仅在真正需要连 Milvus 时才加载 pymilvus

        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)  # 用默认别名连接指定 host:port
        _connected = True  # 标记已连接，后续调用直接短路返回 True
        return True  # 返回 True 表示连接可用，调用方可继续后续操作
    except Exception as exc:  # 连接失败（服务未启动 / 网络异常 / 库缺失）时降级
        log.warning("milvus connect failed: %s", exc)  # 记录警告日志便于排查，不抛异常以免阻断主流程
        return None  # 返回 None 让调用方跳过向量检索走纯关键词检索


def _list_partitions(col) -> set[str]:  # 私有方法：返回集合所有分区名（带进程内缓存）
    """返回当前集合已有分区名的集合（进程内缓存，drop/重建集合时失效）。"""
    global _partition_cache  # 引用全局缓存变量
    if _partition_cache is None:  # 缓存为空说明是首次调用或集合刚重建，需要远程拉取
        _partition_cache = {p.name for p in col.partitions}  # 遍历 col.partitions 提取分区名构造集合
    return _partition_cache  # 命中缓存直接返回，避免每次写入前重复 RPC 调用


def _ensure_partition(col, role_code: str) -> str | None:  # 确保某角色分区存在，返回分区名或 None
    """确保角色分区存在并返回分区名；失败返回 None（回退默认分区写入）。

    一个角色一个分区（partition）：写入定向、检索裁剪，物理隔离比
    纯标量 expr 过滤少扫数据，是 Milvus 多租户的推荐做法。
    """
    try:  # 创建分区可能因并发或权限失败，包一层异常保护
        names = _list_partitions(col)  # 先查（缓存）分区列表，判断是否需要新建
        if role_code not in names:  # 该角色分区尚未创建
            col.create_partition(role_code)  # 分区名规则与集合名一致，role_code 均为合法标识符
            names.add(role_code)  # 同步更新缓存集合，下次命中无需再查
        return role_code  # 返回分区名，供 insert/search 指定 partition_name
    except Exception as exc:  # 创建分区失败（如并发冲突 / 已存在）时降级
        log.warning("milvus create partition %s failed: %s", role_code, exc)  # 记录警告，不阻断写入流程
        return None  # 返回 None，调用方据此回退到 _default 默认分区


# 新 schema 必须包含的字段名（用于检测旧集合并触发自动重建）
REQUIRED_FIELDS = {"id", "vector", "text", "role_code", "chunk_uid", "source", "summary", "created_at", "updated_at"}  # 字段名集合，ensure_collection 用 issubset 比对，缺失任一字段即 drop 重建


def _build_schema(dim: int):  # 构建 Milvus collection 的 schema（字段定义），入参为向量维度
    """构建多角色 collection schema：在原字段基础上增加 role_code / chunk_uid。"""
    from pymilvus import CollectionSchema, DataType, FieldSchema  # 延迟导入 schema 构建相关类，避免无 Milvus 时模块加载失败

    fields = [  # 字段列表，顺序即后续 insert 列式数据的写入顺序
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 自增主键
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),  # BGE-m3 向量
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),  # 向量对应原文（问答拼接）
        FieldSchema(name="role_code", dtype=DataType.VARCHAR, max_length=32),  # 角色标记，用于按角色过滤
        FieldSchema(name="chunk_uid", dtype=DataType.VARCHAR, max_length=64),  # 与 SQL knowledge_chunks 对应
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),  # 文档来源
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),  # 摘要
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),  # 创建时间
        FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32),  # 修改时间
    ]
    return CollectionSchema(fields, description="roleplay knowledge (multi-role)")  # 用字段列表构造 CollectionSchema 并返回，描述=多角色知识库


def ensure_collection():  # 入口方法：获取或首次创建 collection，含旧 schema 自动迁移
    """获取（或首次创建/自动迁移）collection：字段含 id/向量/原文/角色/chunk_uid/来源/摘要/时间。"""
    global _partition_cache  # 声明引用全局缓存，schema 变更后需置空以重新拉取
    if not _connect():  # 先尝试连接 Milvus，未启用或失败则返回 None 让调用方降级
        return None
    from pymilvus import Collection, utility  # 延迟导入 Collection（集合操作）和 utility（管理工具）

    dim = embedding_dim()  # 取当前 embedding 模型维度（如 1024 维），建集合时作为 vector 字段 dim
    if utility.has_collection(MILVUS_COLLECTION):  # 集合已存在的情况
        col = Collection(MILVUS_COLLECTION)  # 用名字拿到已存在集合的句柄
        field_names = {f.name for f in col.schema.fields}  # 提取已有 schema 的全部字段名集合
        if REQUIRED_FIELDS.issubset(field_names):  # 旧 schema 字段是否完整覆盖 REQUIRED_FIELDS
            col.load()  # 已存在且 schema 完整：加载进内存供检索
            _partition_cache = None  # 重新拉取分区列表，防止跨进程建分区后缓存过期
            return col  # 返回可用的 collection 句柄
        # 旧 schema（无 role_code/chunk_uid）：空集合或历史结构，drop 后重建
        log.warning("milvus collection schema outdated, dropping and recreating: %s", MILVUS_COLLECTION)  # 记录迁移日志
        utility.drop_collection(MILVUS_COLLECTION)  # 删除旧集合，随后用新 schema 重建（自动迁移）
    # 首次创建：定义 schema
    col = Collection(MILVUS_COLLECTION, _build_schema(dim))  # 用新 schema 创建集合，dim 来自 embedding_dim
    # IVF_FLAT 倒排索引 + 余弦度量；nlist 分桶 / nprobe 查桶数，精度速度权衡
    col.create_index("vector", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})  # 在 vector 字段上建 IVF_FLAT 索引，nlist=128 分桶，度量=余弦
    col.load()  # 把集合加载进内存，使后续 search 可用
    _partition_cache = None  # 新建集合后分区缓存置空，等首次写入时再拉取
    log.info("milvus collection created: %s", MILVUS_COLLECTION)  # 记录集合创建成功日志
    return col  # 返回新建的 collection 句柄


def upsert_texts(  # 批量写入（插入）向量与原文，供知识入库使用
    texts: list[str],  # 原文文本列表（问答拼接后的句子）
    vectors: list[list[float]],  # 与 texts 一一对应的 embedding 向量列表
    source: str,  # 文档来源（文件名 / URL），写入 source 字段
    summary: str = "",  # 该批知识的摘要，写入 summary 字段（可空）
    role_code: str = "default",  # 角色标记（teacher/doctor/lawyer/psychologist/scientist）
    chunk_uids: list[str] | None = None,  # 业务键列表，与 SQL knowledge_chunks.chunk_uid 对应；None 时自动生成
) -> int:  # 返回实际写入的条数
    """批量写入向量与原文（知识入库），返回写入条数。

    role_code：角色标记（teacher/doctor/lawyer/psychologist/scientist）。
    chunk_uids：与 SQL knowledge_chunks.chunk_uid 一一对应的业务键；缺省按序号生成。
    """
    col = ensure_collection()  # 确保 collection 存在并已 load，未启用 Milvus 时返回 None
    if col is None or not vectors:  # 集合不可用或没有向量数据，直接返回 0 条写入
        return 0
    now = datetime.now().isoformat(timespec="seconds")  # 当前时间字符串（秒精度），用于 created_at/updated_at
    n = len(texts)  # 本批要写入的文本条数
    uids = chunk_uids or [f"{role_code}-{abs(hash(t)) & 0xFFFFFFFF:08x}" for t in texts]  # 没传 chunk_uids 则按文本哈希生成 8 位十六进制 UID
    # 定向写入角色分区；建分区失败时回退默认分区（数据仍可用 role_code 标量过滤）
    partition_name = _ensure_partition(col, role_code) or "_default"  # 确保角色分区存在，失败则用 _default
    # 按字段顺序组织列式数据（与 schema 一致，id 自增不传）
    col.insert(  # 调用 Milvus insert 写入数据
        [  # 列式数据：每个子列表对应一个字段的所有记录
            vectors,  # vector 字段：向量列表
            texts,  # text 字段：原文列表
            [role_code] * n,  # role_code 字段：每条都填同一角色码
            uids,  # chunk_uid 字段：业务键列表
            [source] * n,  # source 字段：每条都填同一来源
            [summary[:500]] * n,  # summary 字段：截取前 500 字，与 schema max_length=512 留余量
            [now] * n,  # created_at 字段：每条都填当前时间
            [now] * n,  # updated_at 字段：每条都填当前时间
        ],
        partition_name=partition_name,  # 指定写入哪个分区，实现角色物理隔离
    )
    col.flush()  # 刷盘，保证立即可检索
    return n  # 返回写入条数，调用方据此记录日志或更新 SQL


def search_vectors(  # 向量近邻检索（ANN）入口
    vector: list[float],  # 查询向量（需与库内向量同维度，且已归一化以便余弦度量）
    top_k: int = 8,  # 返回的近邻数量上限，默认 8
    role_code: str | None = None,  # 角色码：非空时只在该角色分区内检索；None 则全集合扫描
) -> list[dict[str, Any]]:  # 返回命中结果的字典列表
    """向量近邻检索（余弦），返回 text/role_code/chunk_uid/source/summary/score 列表。

    role_code 非空时只在该角色的分区内检索（按角色物理隔离）；
    分区不存在说明该角色还没有任何知识，直接返回空列表。
    """
    col = ensure_collection()  # 确保 collection 可用并已 load
    if col is None:  # Milvus 不可用，返回空列表让上层走 BM25 回退
        return []
    partition_names = None  # 默认 None 表示不限定分区（全集合搜索）
    if role_code:  # 指定了角色码，需要做分区裁剪
        if role_code not in _list_partitions(col):  # 该角色分区尚不存在
            return []  # 该角色分区尚不存在：无知识可检索
        partition_names = [role_code]  # 只搜该角色分区，避免全集合扫描
    results = col.search(  # 执行 ANN 检索
        data=[vector],  # 查询向量（外层列表表示可批量查询，这里单条）
        anns_field="vector",  # 指定在 vector 字段上做近邻检索
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},  # 搜 10 个桶
        limit=top_k,  # 每条查询返回的近邻数上限
        partition_names=partition_names,  # 分区裁剪：None 全扫，[role_code] 只扫该分区
        output_fields=["text", "role_code", "chunk_uid", "source", "summary"],  # 顺带取回标量字段
    )
    hits = []  # 收集命中的结果字典
    for hit in results[0]:  # results[0] 是第一条查询的命中列表（因为只查了一条向量）
        hits.append(  # 把每条命中转成普通字典返回给上层
            {
                "text": hit.entity.get("text"),  # 命中向量对应的原文
                "role_code": hit.entity.get("role_code"),  # 命中记录所属的角色码
                "chunk_uid": hit.entity.get("chunk_uid"),  # 业务键，用于回表 SQL knowledge_chunks
                "source": hit.entity.get("source"),  # 命中记录的文档来源
                "summary": hit.entity.get("summary"),  # 命中记录的摘要
                "score": float(hit.score),  # 余弦相似度得分，转 float 便于序列化
            }
        )
    return hits  # 返回结果列表，供 RAG 上游拼上下文


def role_data_stats() -> dict[str, int] | None:  # 统计各角色（分区）的向量条数，供前端过滤角色列表
    """按分区统计各角色向量条数：{role_code: num_entities}。

    供角色列表过滤使用（没有数据的角色不出现在前端选择框）。
    Milvus 不可用时返回 None，由调用方决定回退策略。
    """
    col = ensure_collection()  # 确保 collection 可用
    if col is None:  # Milvus 不可用，返回 None 让调用方走回退（如全角色列表）
        return None
    try:  # 统计可能因 Milvus 异常失败，包一层异常保护
        col.flush()  # 确保 num_entities 反映最新写入
        return {p.name: int(p.num_entities) for p in col.partitions}  # 遍历各分区，把 num_entities 转 int 构造 {分区名: 条数} 字典
    except Exception as exc:  # 统计失败时降级
        log.warning("milvus partition stats failed: %s", exc)  # 记录警告日志
        return None  # 返回 None，由调用方决定回退策略


# =====================================================================
# 知识点说明（RAG：向量存储 / Milvus）
# ---------------------------------------------------------------------
# 1. 向量数据库：Milvus 存句子向量（Embedding），支持 ANN 近似最近邻
#    检索。IVF_FLAT 索引：倒排文件分桶 nlist=128，查询只搜 nprobe=10 个
#    桶——"牺牲少量精度换大幅提速"的典型权衡。
# 2. 相似度度量：COSINE 余弦相似度；BGE 系列向量需先归一化
#    （embeddings.py 的 normalize_embeddings=True），归一化后内积即余弦。
# 3. Collection 设计：id（自增主键）/ vector（1024 维 BGE-m3）/
#    text（问答拼接原文）/ role_code（角色隔离，检索时用 expr 过滤）/
#    chunk_uid（与 SQL knowledge_chunks 回表关联）/ source（文档来源）/
#    summary（摘要）/ created_at / updated_at。
# 4. 多角色隔离——分区（Partition）方案：一个角色一个分区，写入时
#    insert(partition_name=role_code) 定向落分区，检索时
#    search(partition_names=[role_code]) 只扫该分区的数据，比
#    "全集合扫 + role_code 标量 expr 过滤"少读无关向量，是 Milvus
#    多租户/多角色的推荐物理隔离手段；标量过滤 expr 仍可叠加在分区内
#    做二级过滤。分区不存在 = 该角色无知识，检索直接短路返回空。
#    role_data_stats() 读各分区 num_entities，供前端"无数据角色不下发"。
# 5. 在 RAG 中的位置：属于"长期知识库"存储层，与 Redis 短期记忆
#    （memory_store.py）相对：知识是全局共享的资料，记忆是用户个人的。
# 6. 降级设计：未启动 Milvus 时 _connect 返回 None，系统回退纯 BM25。
# =====================================================================
