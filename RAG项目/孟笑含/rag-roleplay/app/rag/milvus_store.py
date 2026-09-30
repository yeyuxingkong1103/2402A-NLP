# -*- coding: utf-8 -*-
"""Milvus 向量库：每角色一个 Collection，稠密向量 + BM25 稀疏向量混合检索（RRF）。"""
import logging
# 解析：日志模块

from pymilvus import (
    # 解析：导入 pymilvus 所需组件
    AnnSearchRequest,
    # 解析：向量检索请求（混合检索的两路之一）
    Collection,
    # 解析：集合对象
    CollectionSchema,
    # 解析：集合 schema
    DataType,
    # 解析：字段类型枚举
    FieldSchema,
    # 解析：字段定义
    RRFRanker,
    # 解析：RRF 融合排序器（混合检索合并两路结果）
    connections,
    # 解析：连接管理
    utility,
    # 解析：工具函数（has_collection 等）
)

logger = logging.getLogger("rag-roleplay.milvus")
# 解析：本模块 logger

DENSE_DIM = 1024  # BGE-m3 稠密向量维度
# 解析：稠密向量维度常量（BGE-m3 输出 1024 维）
TEXT_MAX_LEN = 65535
# 解析：文本字段最大长度（VARCHAR 上限）


# Milvus 向量库封装：每角色一个 Collection（v2 含摘要字段），稠密+稀疏混合检索
class MilvusStore:
    def __init__(self, host: str = "127.0.0.1", port: int = 19530):
        # 解析：构造——保存连接参数（惰性连接）
        self.host = host
        # 解析：主机地址
        self.port = port
        # 解析：端口（Milvus 默认 19530）
        self._connected = False
        # 解析：连接标记（惰性建连）

    def _connect(self) -> None:
        # 解析：确保已连接
        if not self._connected:
            # 解析：未连接
            # 使用默认别名连接，Collection/utility 操作无需再传 using
            connections.connect("default", host=self.host, port=str(self.port))
            # 解析：用默认别名建连——后续所有操作自动走该连接
            self._connected = True
            # 解析：标记已连接

    @staticmethod
    def collection_name(role_id: int) -> str:
        # 解析：知识库集合名
        # v3：schema 含 summary 与 parent_id 字段（旧版本库不支持加字段，保留不动）
        return f"role_kb_v3_{role_id}"
        # 解析：每角色独立集合（v3 为 schema 演进版本号）

    @staticmethod
    def parent_collection_name(role_id: int) -> str:
        """父子块模式下的父块集合（纯文本无向量）。"""
        return f"role_parents_{role_id}"
        # 解析：父块集合名（父子块模式专用）

    def ensure_collection(self, role_id: int) -> None:
        # 解析：确保知识库集合存在
        self.ensure_named_collection(self.collection_name(role_id), f"角色 {role_id} 知识库")
        # 解析：委托通用方法（带描述）

    def ensure_named_collection(self, name: str, description: str = "") -> None:
        """按任意名字建 Collection（知识库/长期记忆共用）。"""
        self._connect()
        # 解析：确保连接
        if utility.has_collection(name):
            # 解析：集合已存在
            return
            # 解析：幂等返回
        fields = [
            # 解析：定义集合字段
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            # 解析：主键（自增）
            FieldSchema("text", DataType.VARCHAR, max_length=TEXT_MAX_LEN),
            # 解析：块原文
            FieldSchema("summary", DataType.VARCHAR, max_length=1024),
            # 解析：抽取式摘要
            FieldSchema("parent_id", DataType.VARCHAR, max_length=64),
            # 解析：父块链接（父子块模式）
            FieldSchema("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM),
            # 解析：BGE-m3 稠密向量
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            # 解析：BM25 稀疏向量
            FieldSchema("source", DataType.VARCHAR, max_length=512),
            # 解析：文档来源
            FieldSchema("created_at", DataType.INT64),
            # 解析：创建时间
            FieldSchema("updated_at", DataType.INT64),
            # 解析：更新时间
        ]
        collection = Collection(
            # 解析：创建集合
            name, schema=CollectionSchema(fields, description=description or name)
            # 解析：名称与 schema
        )
        collection.create_index(
            # 解析：建稠密向量索引
            "dense",
            # 解析：索引字段
            {"index_type": "HNSW", "metric_type": "COSINE",
             "params": {"M": 16, "efConstruction": 200}},
            # 解析：HNSW 图索引 + 余弦度量 + 构图参数
        )
        collection.create_index(
            # 解析：建稀疏向量索引
            "sparse",
            # 解析：索引字段
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
            # 解析：倒排索引 + 内积度量（BM25 标准）
        )
        collection.load()
        # 解析：加载进内存（可检索状态）
        logger.info("已创建知识库 Collection: %s", name)
        # 解析：记录日志

    def insert_chunks(self, role_id: int, records: list[dict]) -> None:
        # 解析：向角色知识库插入块
        self.insert_into(self.collection_name(role_id), records)
        # 解析：委托通用插入

    def insert_into(self, name: str, records: list[dict]) -> None:
        """records: [{text, summary, parent_id, dense, sparse, source, created_at, updated_at}]"""
        self.ensure_named_collection(name)
        # 解析：确保集合存在
        collection = Collection(name)
        # 解析：打开集合
        collection.insert(
            # 解析：批量插入（按列组织）
            [
                [r["text"] for r in records],
                # 解析：text 列
                [r["summary"] for r in records],
                # 解析：summary 列
                [r.get("parent_id", "") for r in records],
                # 解析：parent_id 列（缺省空串）
                [r["dense"] for r in records],
                # 解析：dense 列
                [r["sparse"] for r in records],
                # 解析：sparse 列
                [r["source"] for r in records],
                # 解析：source 列
                [r["created_at"] for r in records],
                # 解析：created_at 列
                [r["updated_at"] for r in records],
                # 解析：updated_at 列
            ]
        )
        collection.flush()
        # 解析：刷盘（数据落段，保证可检索）

    def ensure_parent_collection(self, role_id: int) -> None:
        """父子块模式：父块集合（id / parent_id / text / source / created_at）。"""
        name = self.parent_collection_name(role_id)
        # 解析：父集合名
        self._connect()
        # 解析：确保连接
        if utility.has_collection(name):
            # 解析：集合已存在
            collection = Collection(name)
            # 解析：打开
            if not collection.indexes:  # Milvus v3 要求有索引才能 load
                # 解析：老集合没索引（历史遗留）
                collection.create_index("dummy", {"index_type": "FLAT", "metric_type": "L2"})
                # 解析：补建占位索引
                collection.load()
                # 解析：加载
            return
            # 解析：返回
        fields = [
            # 解析：父集合字段（纯文本，无业务向量）
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            # 解析：主键
            FieldSchema("parent_id", DataType.VARCHAR, max_length=64),
            # 解析：父块 ID
            FieldSchema("text", DataType.VARCHAR, max_length=TEXT_MAX_LEN),
            # 解析：父块全文
            FieldSchema("dummy", DataType.FLOAT_VECTOR, dim=2),  # Milvus 集合必须有向量字段（占位，不检索）
            # 解析：占位向量——Milvus 强制集合含向量字段
            FieldSchema("source", DataType.VARCHAR, max_length=512),
            # 解析：文档来源
            FieldSchema("created_at", DataType.INT64),
            # 解析：创建时间
            FieldSchema("updated_at", DataType.INT64),
            # 解析：更新时间
        ]
        collection = Collection(name, schema=CollectionSchema(fields, description=f"角色 {role_id} 父块"))
        # 解析：创建父集合
        collection.create_index("dummy", {"index_type": "FLAT", "metric_type": "L2"})
        # 解析：占位向量建 FLAT 索引（v3 要求有索引才能 load）
        collection.load()
        # 解析：加载
        logger.info("已创建父块 Collection: %s", name)
        # 解析：记录日志

    def insert_parents(self, role_id: int, records: list[dict]) -> None:
        """records: [{parent_id, text, source, created_at, updated_at}]"""
        name = self.parent_collection_name(role_id)
        # 解析：父集合名
        self.ensure_parent_collection(role_id)
        # 解析：确保存在
        collection = Collection(name)
        # 解析：打开
        collection.insert(
            # 解析：按列插入
            [
                [r["parent_id"] for r in records],
                # 解析：parent_id 列
                [r["text"] for r in records],
                # 解析：text 列
                [[0.0, 0.0] for _ in records],
                # 解析：dummy 列（全零占位）
                [r["source"] for r in records],
                # 解析：source 列
                [r["created_at"] for r in records],
                # 解析：created_at 列
                [r["updated_at"] for r in records],
                # 解析：updated_at 列
            ]
        )
        collection.flush()
        # 解析：刷盘

    def delete_parents_by_source(self, role_id: int, source: str) -> int:
        """删除某文档的全部父块（文档级替换用）。"""
        name = self.parent_collection_name(role_id)
        # 解析：父集合名
        self._connect()
        # 解析：确保连接
        if not utility.has_collection(name):
            # 解析：集合不存在
            return 0
            # 解析：无可删
        collection = Collection(name)
        # 解析：打开
        result = collection.delete(expr=f'source == "{source}"')
        # 解析：按来源表达式删除
        collection.flush()
        # 解析：刷盘（删除立即可见）
        return getattr(result, "delete_count", 0)
        # 解析：返回删除条数

    def get_parents(self, role_id: int, parent_ids: list[str]) -> dict[str, str]:
        """按 parent_id 批量取父块文本：{parent_id: text}。"""
        name = self.parent_collection_name(role_id)
        # 解析：父集合名
        self._connect()
        # 解析：确保连接
        if not utility.has_collection(name) or not parent_ids:
            # 解析：集合不存在或无请求
            return {}
            # 解析：返回空字典
        collection = Collection(name)
        # 解析：打开
        collection.load()
        # 解析：加载
        ids = ", ".join(f'"{p}"' for p in parent_ids)
        # 解析：拼查询表达式（parent_id in ["a", "b"]）
        result = collection.query(
            # 解析：条件查询
            expr=f"parent_id in [{ids}]", output_fields=["parent_id", "text"],
            # 解析：表达式与返回字段
            limit=16384, consistency_level="Strong",
            # 解析：上限与强一致性
        )
        return {row["parent_id"]: row["text"] for row in result}
        # 解析：转成 {parent_id: text} 字典返回

    def hybrid_search(
        self, role_id: int, query_dense: list[float], query_sparse: dict, limit: int
    ) -> list[str]:
        # 解析：知识库混合检索（角色级接口）
        return self.hybrid_search_in(self.collection_name(role_id), query_dense, query_sparse, limit)
        # 解析：委托通用实现

    def hybrid_search_in(
        self, name: str, query_dense: list[float], query_sparse: dict, limit: int
    ) -> list[str]:
        """稠密 + BM25 稀疏混合检索，RRF 融合，返回文本列表。"""
        self._connect()
        # 解析：确保连接
        if not utility.has_collection(name):
            # 解析：集合不存在
            return []
            # 解析：返回空
        collection = Collection(name)
        # 解析：打开
        collection.load()
        # 解析：加载
        dense_req = AnnSearchRequest(
            # 解析：稠密检索请求
            [query_dense], "dense",
            # 解析：查询向量与字段
            {"metric_type": "COSINE", "params": {"ef": 128}}, limit=limit,
            # 解析：余弦度量 + 搜索参数 + 召回数
        )
        sparse_req = AnnSearchRequest(
            # 解析：稀疏检索请求
            [query_sparse], "sparse", {"metric_type": "IP"}, limit=limit
            # 解析：查询向量、字段、内积度量、召回数
        )
        hits = collection.hybrid_search(
            # 解析：双路混合检索
            [dense_req, sparse_req], rerank=RRFRanker(60),
            # 解析：两路请求 + RRF 融合（k=60）
            limit=limit, output_fields=["text"], consistency_level="Strong",
            # 解析：返回数、输出字段、强一致性
        )
        return [hit.entity.get("text") for hit in hits[0]]
        # 解析：提取文本列表返回

    def hybrid_search_with_parents(
        self, name: str, query_dense: list[float], query_sparse: dict, limit: int
    ) -> list[tuple[str, str]]:
        """混合检索并返回 (子块文本, parent_id)——父子块模式用。"""
        self._connect()
        # 解析：确保连接
        if not utility.has_collection(name):
            # 解析：集合不存在
            return []
            # 解析：返回空
        collection = Collection(name)
        # 解析：打开
        collection.load()
        # 解析：加载
        dense_req = AnnSearchRequest(
            # 解析：稠密检索请求
            [query_dense], "dense",
            # 解析：查询向量与字段
            {"metric_type": "COSINE", "params": {"ef": 128}}, limit=limit,
            # 解析：余弦度量+参数+召回数
        )
        sparse_req = AnnSearchRequest(
            # 解析：稀疏检索请求
            [query_sparse], "sparse", {"metric_type": "IP"}, limit=limit
            # 解析：查询向量、字段、内积度量、召回数
        )
        hits = collection.hybrid_search(
            # 解析：双路混合检索
            [dense_req, sparse_req], rerank=RRFRanker(60),
            # 解析：两路请求 + RRF 融合
            limit=limit, output_fields=["text", "parent_id"], consistency_level="Strong",
            # 解析：输出文本与父链接
        )
        return [(hit.entity.get("text"), hit.entity.get("parent_id", "")) for hit in hits[0]]
        # 解析：返回 (子块文本, parent_id) 对列表

    def all_texts(self, role_id: int) -> list[str]:
        # 解析：取知识库全部块文本
        return self.all_texts_in(self.collection_name(role_id))
        # 解析：委托通用实现

    def all_texts_in(self, name: str) -> list[str]:
        # 解析：取指定集合全部文本
        self._connect()
        # 解析：确保连接
        if not utility.has_collection(name):
            # 解析：集合不存在
            return []
            # 解析：返回空
        collection = Collection(name)
        # 解析：打开
        collection.load()
        # 解析：加载
        result = collection.query(
            # 解析：全量查询
            expr="id >= 0", output_fields=["text"],
            # 解析：取全部记录（id>=0 恒真）的文本字段
            limit=16384, consistency_level="Strong",
            # 解析：上限与强一致性
        )
        return [row["text"] for row in result]
        # 解析：提取文本列表

    def list_sources(self, role_id: int) -> list[dict]:
        # 解析：文档列表（按来源聚合）
        self._connect()
        # 解析：确保连接
        name = self.collection_name(role_id)
        # 解析：集合名
        if not utility.has_collection(name):
            # 解析：集合不存在
            return []
            # 解析：返回空
        collection = Collection(name)
        # 解析：打开
        collection.load()
        # 解析：加载
        result = collection.query(
            # 解析：全量查询
            expr="id >= 0", output_fields=["source", "summary", "updated_at"],
            # 解析：取来源、摘要、更新时间
            limit=16384, consistency_level="Strong",
            # 解析：上限与强一致性
        )
        grouped: dict[str, dict] = {}
        # 解析：按来源聚合
        for row in result:
            # 解析：逐块
            item = grouped.setdefault(
                # 解析：该来源的聚合项
                row["source"],
                {"source": row["source"], "chunks": 0, "summary": "", "updated_at": 0},
                # 解析：初始结构
            )
            item["chunks"] += 1
            # 解析：块数 +1
            if not item["summary"]:
                # 解析：摘要还没取
                item["summary"] = row.get("summary", "")
                # 解析：取第一个块的摘要
            item["updated_at"] = max(item["updated_at"], row["updated_at"])
            # 解析：更新时间取最大值
        return list(grouped.values())
        # 解析：返回聚合结果列表

    def delete_source(self, role_id: int, source: str) -> int:
        # 解析：删除某文档全部块
        self._connect()
        # 解析：确保连接
        name = self.collection_name(role_id)
        # 解析：集合名
        if not utility.has_collection(name):
            # 解析：集合不存在
            return 0
            # 解析：无可删
        collection = Collection(name)
        # 解析：打开
        result = collection.delete(expr=f'source == "{source}"')
        # 解析：按来源表达式删除
        collection.flush()  # 让删除立即可见
        # 解析：刷盘——强一致性查询才能立刻看到删除
        return getattr(result, "delete_count", 0)
        # 解析：返回删除条数
