"""向量存储模块"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema,
    DataType, utility, MilvusException
)

from src.config import config
from src.utils.logger import logger
from src.utils.tools import generate_id


class VectorStore:
    """向量数据库"""
    
    def __init__(self, collection_name: Optional[str] = None):
        self.host = config.get('milvus.host', 'localhost')
        self.port = config.get('milvus.port', 19530)
        self.collection_name = collection_name or config.get('milvus.collection_name', 'roleplay_knowledge')
        self.dimension = config.get('milvus.dimension', 1024)
        self.collection = None
        self.connected = False
        self.loaded = False
        self._field_names = None
    
    def connect(self):
        """连接Milvus"""
        if self.connected:
            return
        
        try:
            logger.info(f"连接Milvus: {self.host}:{self.port}")
            connections.connect(host=self.host, port=self.port)
            self.connected = True
            logger.info("Milvus连接成功")
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise
    
    def disconnect(self):
        """断开连接"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
    
    def create_collection(self, drop_existing: bool = False):
        """创建集合"""
        self.connect()
        
        if utility.has_collection(self.collection_name):
            if drop_existing:
                utility.drop_collection(self.collection_name)
                logger.info(f"删除已存在的集合: {self.collection_name}")
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                self._field_names = None
                return
        
        # 定义字段（对应文档.txt：ID、向量、原文、创建/修改时间、文档来源、摘要）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="create_time", dtype=DataType.INT64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="modify_time", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(fields=fields, description="RAG角色扮演知识库")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",  # 内积相似度
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)

        self._field_names = None
        logger.info(f"创建集合成功: {self.collection_name}")

    def _get_field_names(self) -> List[str]:
        """读取集合实际存在的字段名（兼容新旧 schema，按定义顺序）。"""
        if self._field_names is None:
            if self.collection is None:
                self.load_collection()
            self._field_names = [f.name for f in self.collection.schema.fields]
        return self._field_names
    
    def load_collection(self):
        """加载集合"""
        self.connect()
        self.collection = Collection(self.collection_name)
        self.collection.load()
        self.loaded = True
        logger.info(f"加载集合: {self.collection_name}")
    
    def insert(
        self,
        texts: List[str],
        vectors: np.ndarray,
        chunk_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        summaries: Optional[List[str]] = None
    ) -> List[int]:
        """插入数据"""
        if self.collection is None:
            self.load_collection()

        import json
        from datetime import datetime

        chunk_ids = chunk_ids or [generate_id(t) for t in texts]
        sources = sources or ["unknown"] * len(texts)
        page_numbers = page_numbers or [1] * len(texts)
        metadata = metadata or [{}] * len(texts)
        summaries = summaries or [""] * len(texts)
        now = int(datetime.now().timestamp())

        # 字段名 -> 列值（不含 auto_id 主键）
        columns = {
            "chunk_id": chunk_ids,
            "text": texts,
            "vector": vectors.tolist(),
            "source": sources,
            "page_number": page_numbers,
            "metadata": [json.dumps(m, ensure_ascii=False) for m in metadata],
            "create_time": [now] * len(texts),
            "summary": summaries,
            "modify_time": [now] * len(texts),
        }

        # 按集合实际 schema 顺序组装（跳过主键 id，跳过旧集合中不存在的 summary/modify_time）
        entities = [
            columns[f] for f in self._get_field_names()
            if f != "id" and f in columns
        ]

        insert_result = self.collection.insert(entities)
        self.collection.flush()

        logger.info(f"插入 {len(texts)} 条数据")
        return insert_result.primary_keys
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        if self.collection is None or not self.loaded:
            self.load_collection()
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        # 只请求集合中实际存在的字段（旧集合无 summary/modify_time 时自动降级）
        desired = ["chunk_id", "text", "source", "page_number", "metadata", "summary"]
        output_fields = [f for f in desired if f in self._get_field_names()]

        results = self.collection.search(
            data=query_vector.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )

        return self._parse_search_results(results)

    def _parse_search_results(self, results) -> List[Dict[str, Any]]:
        """解析搜索结果"""
        parsed = []

        for hits in results:
            for hit in hits:
                import json
                try:
                    meta = json.loads(hit.entity.get('metadata', '{}'))
                except:
                    meta = {}

                parsed.append({
                    'id': hit.id,
                    'chunk_id': hit.entity.get('chunk_id'),
                    'text': hit.entity.get('text'),
                    'source': hit.entity.get('source'),
                    'page_number': hit.entity.get('page_number'),
                    'metadata': meta,
                    'summary': hit.entity.get('summary', ''),
                    'distance': hit.distance
                })

        return parsed
    
    def delete(self, expr: str):
        """删除数据"""
        if self.collection is None or not self.loaded:
            self.load_collection()

        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"删除数据: {expr}")

    def delete_by_source(self, source: str):
        """按来源删除该文档的所有分块（用于重新索引同一文件时去重）"""
        if self.collection is None or not self.loaded:
            self.load_collection()

        # 转义来源名中的双引号，避免破坏表达式
        escaped = source.replace('"', '\\"')
        expr = f'source == "{escaped}"'
        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"删除来源分块: {source}")

    def get_count(self) -> int:
        """获取数据数量"""
        if self.collection is None or not self.loaded:
            self.load_collection()
        return self.collection.num_entities

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """取出集合内全部分块的 chunk_id/text/source/page_number（供 BM25 建索引）。

        注意：仅适用于中小规模知识库；大规模时建议改为持久化稀疏索引。
        """
        if self.collection is None or not self.loaded:
            self.load_collection()

        count = self.collection.num_entities
        if count == 0:
            return []

        rows = self.collection.query(
            expr="",
            output_fields=["chunk_id", "text", "source", "page_number"],
            limit=count,
        )
        return [
            {
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "page_number": r.get("page_number", 1),
            }
            for r in rows
        ]

    def get_distinct_sources(self) -> List[str]:
        """获取知识库中的所有文档来源（用于知识库列表/动态更新）。"""
        if self.collection is None or not self.loaded:
            self.load_collection()

        count = self.collection.num_entities
        if count == 0:
            return []

        rows = self.collection.query(
            expr="",
            output_fields=["source"],
            limit=count,
        )
        return sorted({r.get("source", "") for r in rows if r.get("source")})


class HybridStore(VectorStore):
    """混合检索向量库（支持BM25）"""
    
    def __init__(self, collection_name: Optional[str] = None):
        super().__init__(collection_name)
        self.bm25_index = None  # 待实现BM25
    
    def search_hybrid(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        # 向量检索
        vector_results = self.search(query_vector, top_k * 2)
        
        # BM25检索（待实现）
        # bm25_results = self.bm25_search(query_text, top_k * 2)
        
        # 合并结果
        # TODO: 实现合并排序
        return vector_results
