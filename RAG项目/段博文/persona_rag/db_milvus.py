# -*- coding: utf-8 -*-
"""
Milvus 数据层：多集合知识库入库 + 混合检索 + 长期记忆。

在系统中的位置：
    上游是入库脚本（doc_parser 解析文件 → text_splitter 切分 → 调本模块 ingest_*）和
    检索链路（retriever 调向量召回、main 调长期记忆读写）；本模块是唯一直接持有
    Milvus 连接的地方。入库完成后由 retriever.refresh_bm25() 重建关键词索引，
    本模块只负责向量这一路。

架构：
    rag_legal         ← 法律顾问知识库
    rag_psychology    ← 心理专家知识库
    rag_companion     ← 虚拟朋友知识库
    rag_long_term_memory ← 长期记忆（共用）

每个角色检索自己对应的集合，互不干扰。

关键设计取舍：
    1. 模型与连接都做进程内单例（_embedding / _vectorstores / _milvus_client）：
       加载 bge-m3 权重要几秒、建连接也有开销，而 FastAPI 是常驻进程，复用即可；
    2. 主键由自己生成 UUID（auto_id=False），再配合 (source, chunk_index) 做幂等去重，
       同一份文件重复入库不会产生重复向量；
    3. 同时使用 LangChain 封装（Milvus）和原生客户端（MilvusClient）：前者用于
       带向量的写入，后者用于不带向量的全量 query（重建 BM25 时不需要向量，
       走原生客户端更快也更省事）。
"""

import uuid  # 生成主键
from datetime import datetime  # 时间戳
from typing import List, Optional, Dict  # 类型标注

from langchain_huggingface import HuggingFaceEmbeddings  # 本地向量模型（直接加载 bge-m3 权重）
from langchain_milvus import Milvus  # LangChain 的 Milvus 封装
from langchain_core.documents import Document  # 统一文档结构
from pymilvus import MilvusClient  # 原生客户端（全量查询用）

from config import (  # 配置
    MILVUS_URI, MILVUS_COLLECTION, MILVUS_COLLECTIONS,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, MILVUS_MEMORY_COLLECTION,
    EMBED_MODEL_PATH, EMBED_DEVICE, EMBED_DIM,
    get_collection_for_role,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# ==================== 模型单例（进程内只加载一次） ====================
_embedding: Optional[HuggingFaceEmbeddings] = None  # 向量模型单例
_vectorstores: Dict[str, Milvus] = {}  # 集合名 -> 向量存储（多集合）；字典本身也充当缓存
_milvus_client: Optional[MilvusClient] = None  # 原生客户端单例


def get_embedding() -> HuggingFaceEmbeddings:
    """
    获取向量模型单例（首次调用时加载本地 bge-m3，之后复用）

    返回：
        HuggingFaceEmbeddings 实例，已绑定本地权重路径与运行设备。
    说明：
        normalize_embeddings=True 是关键：向量先做 L2 归一化，配合集合的 COSINE
        度量，相似度就等价于余弦值，分数天然落在 [-1,1]，方便直接和阈值比较；
        若改成 False，分数分布会变化，SCORE_THRESHOLD 就需要重新标定。
        用 global 声明是因为要「写」模块级变量；模型文件缺失时构造阶段就会抛异常，
        这里不做兜底，属于启动期配置错误，应当早失败。
    """
    global _embedding
    if _embedding is None:  # 未初始化
        _embedding = HuggingFaceEmbeddings(  # 加载本地权重，不联网
            model_name=EMBED_MODEL_PATH,  # 模型目录
            model_kwargs={"device": EMBED_DEVICE},  # 运行设备
            encode_kwargs={"normalize_embeddings": True},  # 归一化，配合 COSINE 度量
        )
        logger.info(f"向量模型已加载：{EMBED_MODEL_PATH}")  # 日志
    return _embedding


def get_vectorstore(collection_name: str = None) -> Milvus:
    """
    获取向量存储单例（按集合名缓存）
    首次调用时建连接 + 配置索引，之后复用

    参数：
        collection_name：Milvus 集合名；传 None 时使用 config.MILVUS_COLLECTION
                          （默认 rag_legal），兼容旧代码和入库脚本。
    返回：
        langchain_milvus.Milvus 实例（已绑定向量模型与索引配置）。
    说明：
        用字典按集合名做缓存，而不是只留一个全局变量：本项目是多集合架构，
        同一进程内要同时访问法务/心理/朋友/长期记忆四个集合；
        集合不存在时由 langchain-milvus 自动创建，维度取自 EMBED_DIM，
        这也是换向量模型必须重建集合的原因（维度在建模时已固化）。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    if collection_name not in _vectorstores:  # 该集合未初始化
        index_params = {  # 索引参数（langchain-milvus 0.4 要求字典格式）
            "index_type": MILVUS_INDEX_TYPE,  # AUTOINDEX（交给 Milvus 按数据量自动挑索引）
            "metric_type": MILVUS_METRIC_TYPE,  # COSINE（与向量归一化配套）
            "params": {},  # AUTOINDEX 下留空；若改成 HNSW 才需要填 M / efConstruction
        }
        _vectorstores[collection_name] = Milvus(  # 初始化
            embedding_function=get_embedding(),  # 向量模型
            collection_name=collection_name,  # 集合名
            connection_args={"uri": MILVUS_URI},  # 连接参数
            index_params=index_params,  # 索引配置
            auto_id=False,  # 主键我们自己给（这样才能用固定 UUID 做幂等去重）
        )
        logger.info(f"Milvus 向量存储已连接：{MILVUS_URI}/{collection_name}")  # 日志
    return _vectorstores[collection_name]


def get_milvus_client() -> MilvusClient:
    """
    获取原生客户端单例（用于全量查询拉文本重建 BM25）

    返回：
        pymilvus.MilvusClient 实例（连接 MILVUS_URI，不做向量运算）。
    说明：
        与 LangChain 封装并存的原因：重建 BM25 只需要 text 字段，用原生 query
        可以直接分页拉取、不必触发 embedding，省时省算力。
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)  # 连接
        logger.info(f"Milvus 原生客户端已连接：{MILVUS_URI}")  # 日志
    return _milvus_client


# ==================== 入库操作 ====================

def ingest_documents(docs: List[Document], batch_size: int = 10,
                     collection_name: str = None) -> int:
    """
    向量化并写入指定 Milvus 集合，返回当前语料总数

    metadata 必须包含固定键：
        source      : 来源文件名
        section     : 章节路径（Markdown 有，txt 为空）
        chunk_index : 块序号
        summary     : 块摘要（可选，用于展示）

    参数：
        docs：待入库的 Document 列表（来自 text_splitter 切分结果）。
        batch_size：每批写入条数，默认 10；调大能减少网络往返，但单批太大时
                    向量化会一次占用较多内存。
        collection_name：目标集合名；None 时用 MILVUS_COLLECTION。
    返回：
        int，入库完成后集合内的语料总数；无新文档入库（空输入或全部重复）时返回 0。
    异常与降级：
        拉取已有键失败只在日志里告警并跳过去重（继续入库）；最终统计语料数时
        若集合不存在则退化为本次入库条数。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    if not docs:  # 空列表
        logger.warning("入库文档列表为空，跳过")  # 警告日志
        return 0

    # ---- 幂等去重：用 (source, chunk_index) 唯一键，跳过已存在的记录 ----
    client_m = get_milvus_client()  # 原生客户端
    existing_keys = set()  # 已存在键集合（放内存里做 O(1) 命中判断）
    if client_m.has_collection(collection_name):  # 集合已存在才需要查,has_collection = 集合存不存在
        try:
            offset = 0  # 分页偏移
            page_size = 500  # 每页条数（避免一次拉太多撑爆内存）
            while True:  # 分页拉取已存在的 (source, chunk_index)
                rows = client_m.query(  # 查询
                    collection_name=collection_name,
                    filter="source != ''",  # 只取有来源的记录
                    output_fields=["source", "chunk_index"],  # 只取幂等键字段
                    limit=page_size,
                    offset=offset,
                )
                if not rows:  # 没数据了
                    break
                for r in rows:  # 逐个收集键
                    existing_keys.add((r.get("source", ""), r.get("chunk_index", -1)))
                if len(rows) < page_size:  # 最后一页
                    break
                offset += page_size  # 翻页
        except Exception as e:  # 查询失败不阻断入库，仅警告
            logger.warning(f"查询已有键失败（{e}），本次跳过去重")  # 警告；最坏结果只是产生重复数据，比整批入库失败好

    # 过滤掉已存在的文档
    filtered = []  # 真正要入库的
    skipped = 0  # 已存在被跳过的数量
    for d in docs:  # 逐个文档
        key = (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1))  # 幂等键（缺失时给默认值，保证可比较）
        if key in existing_keys:  # 已存在
            skipped += 1  # 跳过计数
            continue
        existing_keys.add(key)  # 标记本次批次已计划入库（防止批内重复）
        filtered.append(d)  # 加入待入库

    if not filtered:  # 全部重复
        logger.info(f"集合 {collection_name} 无新文档需要入库（跳过 {skipped} 条重复）")  # 日志
        return 0

    # ---- 写入（带进度条） ----
    vs = get_vectorstore(collection_name)  # 获取该集合的向量存储
    try:
        from tqdm import tqdm  # 进度条
    except ImportError:
        tqdm = None  # 未安装则退化为普通循环
    total_batches = (len(filtered) + batch_size - 1) // batch_size  # 总批次数（向上取整的除法写法）
    bar = tqdm(total=total_batches, desc=f"[{collection_name}]", unit="batch") if tqdm else None
    for i in range(0, len(filtered), batch_size):  # 按 batch_size 步长切片，天然处理最后一批不足的情况
        batch = filtered[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]  # 每批现场生成 UUID 字符串主键（同一文档重复入库也是新主键，交由上面的去重拦截）
        vs.add_documents(documents=batch, ids=batch_ids)
        if bar:  # 进度条
            bar.set_postfix_str(f"{len(batch)}条/批")  # 尾部信息
            bar.update(1)  # 前进一格
        else:
            logger.info(f"  写入批次 {i // batch_size + 1}/{(len(filtered) - 1) // batch_size + 1}（{len(batch)} 条）")
    if bar:
        bar.close()  # 关闭进度条

    client = get_milvus_client()  # 原生客户端
    total = client.get_collection_stats(collection_name).get("row_count", 0) if client.has_collection(collection_name) else len(filtered)
    logger.info(f"入库 {len(filtered)} 条（跳过重复 {skipped} 条）到集合 {collection_name}，当前语料 {total} 条")  # filtered 本次入库数
    return total


def ingest_jsonl(file_path: str, collection_name: str = None) -> int:
    """
    读取一个已处理好的 JSONL 文件并入库到指定 Milvus 集合。

    JSONL 每行格式（与预处理输出对齐）：
        {"text": "...", "metadata": {"source": "...", "section": "...",
                                     "chunk_index": 0, "summary": "...", ...}}
    也兼容直接形如 {"page_content": "...", "metadata": {...}} 的记录。

    参数：
        file_path：JSONL 文件绝对/相对路径。
        collection_name：目标集合名；None 时用 MILVUS_COLLECTION。
    返回：入库条数。文件读取失败或为空时返回 0。
    异常与降级：
        文件不存在只记 error 并返回 0，不抛异常；单行 JSON 解析失败只跳过该行；
        完全不含内容字段的行也被跳过。
    """
    import json  # JSON 解析
    from langchain_core.documents import Document  # 统一文档结构

    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    docs = []  # 收集解析出的文档
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # 逐行读（显式 utf-8，避免 Windows 默认 GBK 读中文乱码）
            for line in f:
                line = line.strip()  # 去掉首尾空白（含换行符）
                if not line:
                    continue  # 跳过空行
                try:
                    rec = json.loads(line)  # 解析一行
                except json.JSONDecodeError:
                    logger.warning(f"  跳过损坏行：{line[:80]}...")  # 坏行日志（只截前 80 字，防止刷屏）
                    continue

                # 兼容两种字段命名：text/page_content
                if "text" in rec:  # 项目自己的预处理链路
                    content, meta = rec["text"], rec.get("metadata") or {}  # or {} 兜底 null,避免后面 setdefault报错
                    #  metadata 附加信息
                elif "page_content" in rec:  # LangChain Document 序列化
                    content, meta = rec["page_content"], rec.get("metadata") or {}
                else:
                    continue  # 缺少内容字段，跳过

                if not content or not str(content).strip():
                    continue  # 空内容跳过（空串入库只会污染检索结果）
                # 补齐 metadata 固定键
                meta.setdefault("source", "jsonl")  # 兜底来源文件名，保证幂等键可构造
                meta.setdefault("section", "")  # # 章节路径
                meta.setdefault("chunk_index", len(docs))  # 兜底块号用当前下标，避免全部落到 -1 而互相覆盖,块序号
                docs.append(Document(page_content=str(content), metadata=meta))
    except FileNotFoundError:
        logger.error(f"文件不存在：{file_path}")  # 找不到文件
        return 0

    if not docs:
        logger.warning(f"JSONL 无有效记录：{file_path}")  # 空文件
        return 0

    logger.info(f"读取 {file_path}：{len(docs)} 条有效记录，入库集合 {collection_name}")  # 日志
    return ingest_documents(docs, collection_name=collection_name)  # 复用统一的入库/去重逻辑


# ==================== 全量拉取（BM25 重建用） ====================

def fetch_all_text(collection_name: str = None) -> List[Document]:
    """
    分页拉取指定集合内全部文本，返回 Document 列表（用于重建 BM25）

    参数：
        collection_name：集合名；None 时用 MILVUS_COLLECTION。
    返回：
        List[Document]，每个元素只有 page_content（原文）和 metadata["pk"]；
        集合不存在时返回空列表 []（不抛异常，调用方按「该角色暂无知识」处理）。
    说明：
        用原生 query 而不走向量检索：这里要的是「全量文本」而不是「相似结果」，
        BM25 索引必须覆盖全部语料；分页 offset 逐页推进，直到某页不足 page_size
        或返回空，即认为已取完。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()  # 原生客户端
    if not client.has_collection(collection_name):  # 集合不存在
        return []  # 返回空
    all_docs: List[Document] = []  # 收集文档
    offset = 0  # 分页偏移
    page_size = 500  # 每页条数（太小网络往返多，太大单次响应占内存）
    while True:  # 分页循环
        rows = client.query(  # 全量查询
            collection_name=collection_name,
            filter="",  # 空过滤 = 匹配全部
            output_fields=["pk", "text"],  # 只取主键和原文（不取向量，省带宽）
            limit=page_size,
            offset=offset,
        )
        if not rows:  # 没数据了
            break
        for row in rows:  # 逐行转 Document
            all_docs.append(Document(page_content=row["text"], metadata={"pk": row["pk"]}))
        if len(rows) < page_size:  # 最后一页
            break
        offset += page_size  # 翻页
    logger.info(f"从 Milvus 集合 {collection_name} 拉取 {len(all_docs)} 条文档")  # 日志
    return all_docs


# ==================== 长期记忆存储 ====================

def save_long_term_memory(user_id: int, role_id: int, content: str, summary: str = ""):
    """
    将重要对话存入 Milvus 作为长期记忆（与知识库分离，用独立集合）

    长期记忆 vs 短期记忆：
        短期（Redis）：最近 20 条对话，滑动窗口，2 小时过期
        长期（Milvus）：重要事实永久存储，用向量检索找回

    参数：
        user_id：用户 ID，写入 user_id 字段，检索时用它过滤，防止串到别人的记忆。
        role_id：角色 ID，记录记忆是哪个角色产生的。
        content：要长期保存的原文（会被向量化，长度受字段上限 65535 限制）。
        summary：可选摘要，便于展示时先看摘要再看全文。
    返回：
        None。
    说明：
        集合不存在时惰性建集合：字段与知识库集合不同（多了 user_id / role_id /
        summary），所以这里手写 schema，而不是复用 get_vectorstore 的自动建表。
    """
    embedding = get_embedding()  # 向量模型
    vec = embedding.embed_query(content)  # 向量化（query 与 document 走同一模型，保证同空间可比）
    client = get_milvus_client()  # 原生客户端

    # 使用配置的集合名（修复之前硬编码的 bug）
    memory_collection = MILVUS_MEMORY_COLLECTION

    if not client.has_collection(memory_collection):
        from pymilvus import CollectionSchema, FieldSchema, DataType  # schema 构造器
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True),  # 主键：UUID 字符串，不自动生成
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # 记忆原文（VARCHAR 上限约 64KB）, 16 位无符号整数
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),  # 向量：维度必须与 EMBED_DIM 一致,浮点向量
            FieldSchema(name="user_id", dtype=DataType.INT64),  # 归属用户，检索时做标量过滤
            FieldSchema(name="role_id", dtype=DataType.INT64),  # 归属角色
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),  # 摘要
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),  # 创建时间（ISO 字符串，比时间戳更易读）
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)  # 关闭动态字段：字段固定，写入错字段会报错便于发现
        client.create_collection(memory_collection, schema=schema)
        index_params = client.prepare_index_params()  # 原生建索引接口
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")  # 与知识库保持同一度量，分数口径一致,AUTOINDEX自动选择索引类型
        client.create_index(memory_collection, index_params=index_params)
        logger.info(f"长期记忆集合已创建：{memory_collection}")

    client.insert(memory_collection, {  # 直接 insert，不走 LangChain：这里显式控制每个字段
        "pk": str(uuid.uuid4()), "text": content, "vector": vec,
        "user_id": user_id, "role_id": role_id, "summary": summary,
        "created_at": datetime.utcnow().isoformat(),  # UTC 时间字符串，避免多机时区不一致
    })
    logger.info(f"长期记忆已保存：user={user_id} role={role_id}")


def search_long_term_memory(user_id: int, query: str, top_k: int = 3) -> List[Document]:
    """
    检索某用户的长期记忆（按角色过滤）

    参数：
        user_id：用户 ID，作为标量过滤条件——只在本人的记忆里检索。
        query：用户的当前提问，会被向量化后做相似检索。
        top_k：返回条数，默认 3（长期记忆是补充信息，取太多反而干扰回答）。
    返回：
        List[Document]，page_content 为记忆原文，metadata 含 "summary"；
        集合不存在时返回 []，检索无结果时也返回 []（均不抛异常）。
    说明：
        filter 用表达式字符串 "user_id == {user_id}"，user_id 来自内部整型参数
        而非用户输入文本，因此不存在注入风险。
    """
    embedding = get_embedding()
    vec = embedding.embed_query(query)
    client = get_milvus_client()

    memory_collection = MILVUS_MEMORY_COLLECTION  # 使用配置的集合名（不硬编码）

    if not client.has_collection(memory_collection):
        return []  # 还没人写过长期记忆，属于正常情况
    results = client.search(
        collection_name=memory_collection,
        data=[vec],
        anns_field="vector",  # anns_field 指向量字段名
        filter=f"user_id == {user_id}",  # 标量过滤：只看自己的记忆
        limit=top_k,  # 返回条数
        output_fields=["text", "summary", "created_at"],  # 返回字段
    )
    if not results or not results[0]:  # results 是「每条 query 一个列表」的嵌套结构，这里只有一条 query
        return []
    return [Document(page_content=r["entity"]["text"], metadata={"summary": r["entity"].get("summary", "")})
            for r in results[0]]
