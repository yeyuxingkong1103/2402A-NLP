import asyncio  # 导入 asyncio，用于异步执行与线程卸载
import logging  # 导入日志模块

from pymilvus import DataType, MilvusClient  # 导入 pymilvus 的数据类型和客户端

from ..config import get_settings  # 从上层 config 模块导入配置获取函数
from ..core.providers.base import EmbeddingProvider  # 导入 embedding 提供者基类
from ..core.rrf import rrf_fuse  # 导入 RRF 融合函数

logger = logging.getLogger(__name__)  # 获取当前模块的 logger

DIM = 1024  # 稠密向量维度常量


class MilvusStore:  # 定义 Milvus 存储封装类
    def __init__(self, embedding: EmbeddingProvider, host: str | None = None, port: int | None = None):  # 构造函数
        settings = get_settings()  # 读取全局配置
        self.embedding = embedding  # 保存 embedding 提供者
        self.client = MilvusClient(uri=f"http://{host or settings.milvus_host}:{port or settings.milvus_port}")  # 创建 Milvus 客户端，host/port 可覆盖配置

    async def init_collections(self, retries: int | None = None, interval: float | None = None) -> None:  # 异步初始化所有集合
        # Milvus 启动较慢，api 启动时可能尚未就绪；这里重试等待而非直接退出。
        settings = get_settings()  # 读取配置
        retries = settings.milvus_connect_retries if retries is None else retries  # 重试次数，未传则用配置
        interval = settings.milvus_connect_interval if interval is None else interval  # 重试间隔，未传则用配置
        last_exc: Exception | None = None  # 记录最后一次异常
        for attempt in range(1, retries + 1):  # 按重试次数循环
            try:  # 尝试初始化一次
                await self._init_collections_once()  # 调用单次初始化
                return  # 成功则返回
            except Exception as exc:  # 捕获异常
                last_exc = exc  # 记录异常
                if attempt < retries:  # 如果还没到最后一次
                    logger.warning(  # 打印警告日志
                        "Milvus 未就绪，%.0fs 后重试（%s/%s）：%s", interval, attempt, retries, exc
                    )
                    await asyncio.sleep(interval)  # 等待间隔后重试
        raise last_exc  # 全部失败后抛出最后一次异常

    async def _init_collections_once(self) -> None:  # 单次初始化三个集合
        schema_settings = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)  # 创建角色设定集合 schema，主键自增、允许动态字段
        schema_settings.add_field("id", DataType.INT64, is_primary=True)  # 主键 id
        schema_settings.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)  # 稠密向量字段
        schema_settings.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量字段
        schema_settings.add_field("character_id", DataType.INT64)  # 角色 ID
        schema_settings.add_field("setting_type", DataType.VARCHAR, max_length=64)  # 设定类型
        schema_settings.add_field("text", DataType.VARCHAR, max_length=4096)  # 设定正文
        await asyncio.to_thread(self._ensure, "character_settings", schema_settings, "dense_vector")  # 用线程执行确保集合存在

        schema_mem = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)  # 创建长期记忆集合 schema
        schema_mem.add_field("id", DataType.INT64, is_primary=True)  # 主键 id
        schema_mem.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)  # 稠密向量字段
        schema_mem.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量字段
        schema_mem.add_field("user_id", DataType.INT64)  # 用户 ID
        schema_mem.add_field("character_id", DataType.INT64)  # 角色 ID
        schema_mem.add_field("memory_type", DataType.VARCHAR, max_length=32)  # 记忆类型
        schema_mem.add_field("content", DataType.VARCHAR, max_length=4096)  # 记忆正文
        schema_mem.add_field("importance", DataType.INT8)  # 重要度
        await asyncio.to_thread(self._ensure, "long_term_memory", schema_mem, "dense_vector")  # 确保记忆集合存在

        schema_docs = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)  # 创建文档集合 schema
        schema_docs.add_field("id", DataType.INT64, is_primary=True)  # 主键 id
        schema_docs.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)  # 稠密向量字段
        schema_docs.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量字段
        schema_docs.add_field("doc_id", DataType.INT64)  # 文档 ID
        schema_docs.add_field("chunk_index", DataType.INT64)  # 分块序号
        schema_docs.add_field("source", DataType.VARCHAR, max_length=255)  # 文档来源名
        schema_docs.add_field("text", DataType.VARCHAR, max_length=4096)  # 文档片段正文
        schema_docs.add_field("created_at", DataType.INT64)  # 创建时间
        await asyncio.to_thread(self._ensure, "documents", schema_docs, "dense_vector")  # 确保文档集合存在

    def _ensure(self, name, schema, vector_field):  # 确保集合存在、有索引、已加载
        # 集合已存在但可能因历史失败而无索引（Milvus 2.4.0 建 HNSW 索引若未显式传
        # M/efConstruction 会报 efConstruction out of range），这里自愈补齐。
        if not self.client.has_collection(name):  # 如果集合不存在
            self.client.create_collection(name, schema=schema)  # 创建集合
        if not self.client.list_indexes(name):  # 如果集合没有索引
            idx = self.client.prepare_index_params()  # 准备索引参数
            idx.add_index(field_name=vector_field, index_type="HNSW", metric_type="COSINE", M=16, efConstruction=200)  # 稠密向量 HNSW 索引
            idx.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")  # 稀疏向量倒排索引
            self.client.create_index(name, idx)  # 创建索引
        # 每次启动确保已加载到内存（load 幂等），否则重启后 search 会报 collection not loaded
        self.client.load_collection(name)  # 加载集合到内存

    async def upsert_settings(self, character_id: int, chunks: list[dict]) -> None:  # 异步写入角色设定分块
        texts = [c["text"] for c in chunks]  # 提取所有文本
        dense = await self.embedding.encode_dense(texts)  # 批量编码稠密向量
        sparse = await self.embedding.encode_sparse(texts)  # 批量编码稀疏向量
        rows = [  # 组装待插入的行
            {
                "dense_vector": d,  # 稠密向量
                "sparse_vector": s,  # 稀疏向量
                "character_id": character_id,  # 角色 ID
                "setting_type": c["setting_type"],  # 设定类型
                "text": c["text"],  # 设定正文
            }
            for d, s, c in zip(dense, sparse, chunks)  # 三元组同步遍历
        ]
        await asyncio.to_thread(self.client.insert, "character_settings", rows)  # 线程中插入集合

    async def upsert_memories(self, memories: list[dict]) -> None:  # 异步写入长期记忆
        texts = [m["content"] for m in memories]  # 提取所有记忆正文
        dense = await self.embedding.encode_dense(texts)  # 批量编码稠密向量
        sparse = await self.embedding.encode_sparse(texts)  # 批量编码稀疏向量
        rows = [  # 组装待插入的行
            {
                "dense_vector": d,  # 稠密向量
                "sparse_vector": s,  # 稀疏向量
                "user_id": m["user_id"],  # 用户 ID
                "character_id": m["character_id"],  # 角色 ID
                "memory_type": m["memory_type"],  # 记忆类型
                "content": m["content"],  # 记忆正文
                "importance": m["importance"],  # 重要度
            }
            for d, s, m in zip(dense, sparse, memories)  # 三元组同步遍历
        ]
        await asyncio.to_thread(self.client.insert, "long_term_memory", rows)  # 线程中插入集合

    async def upsert_documents(self, chunks: list[dict]) -> None:  # 异步写入文档分块
        texts = [c["text"] for c in chunks]  # 提取所有文本
        dense = await self.embedding.encode_dense(texts)  # 批量编码稠密向量
        sparse = await self.embedding.encode_sparse(texts)  # 批量编码稀疏向量
        rows = [  # 组装待插入的行
            {
                "dense_vector": d,  # 稠密向量
                "sparse_vector": s,  # 稀疏向量
                "doc_id": c["doc_id"],  # 文档 ID
                "chunk_index": c["chunk_index"],  # 分块序号
                "source": c["source"],  # 文档来源名
                "text": c["text"],  # 文档片段正文
                "created_at": c["created_at"],  # 创建时间
            }
            for d, s, c in zip(dense, sparse, chunks)  # 三元组同步遍历
        ]
        await asyncio.to_thread(self.client.insert, "documents", rows)  # 线程中插入集合

    async def hybrid_search(self, collection: str, query: str, filter_expr: str, top_k: int) -> list[dict]:  # 异步混合检索
        dense_q, sparse_q = await self.embedding.encode_query(query)  # 把 query 编码成稠密 + 稀疏查询向量

        def _run():  # 定义同步执行体，便于卸载到线程
            dense_res = self.client.search(  # 稠密向量检索
                collection, data=[dense_q], anns_field="dense_vector", limit=top_k,  # 集合、查询向量、字段、条数
                filter=filter_expr, output_fields=["*"],  # 过滤条件、输出全部字段
            )[0]  # 取第一个查询的结果
            sparse_res = self.client.search(  # 稀疏向量检索
                collection, data=[sparse_q], anns_field="sparse_vector", limit=top_k,  # 集合、查询向量、字段、条数
                filter=filter_expr, output_fields=["*"],  # 过滤条件、输出全部字段
            )[0]  # 取第一个查询的结果
            dense_ids = [h["id"] for h in dense_res]  # 提取稠密结果 ID 列表
            sparse_ids = [h["id"] for h in sparse_res]  # 提取稀疏结果 ID 列表
            fused_ids = rrf_fuse(dense_ids, sparse_ids)  # 用 RRF 融合两路 ID
            by_id = {h["id"]: h["entity"] for h in dense_res + sparse_res}  # 建立 id -> entity 映射
            return [{"id": i, **by_id[i]} for i in fused_ids if i in by_id]  # 按融合顺序返回带 id 的实体

        return await asyncio.to_thread(_run)  # 在线程中执行并返回结果

    async def delete_by_filter(self, collection: str, filter_expr: str) -> None:  # 异步按过滤条件删除
        await asyncio.to_thread(self.client.delete, collection, filter=filter_expr)  # 线程中执行删除

    async def close(self) -> None:  # 异步关闭客户端
        await asyncio.to_thread(self.client.close)  # 线程中关闭