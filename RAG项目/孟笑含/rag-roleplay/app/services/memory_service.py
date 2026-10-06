# -*- coding: utf-8 -*-
"""长期记忆服务：用户与角色的对话沉淀到 Milvus（每 user×role 一个记忆库）。"""
import time
# 解析：生成入库时间戳

from app.rag.bm25 import BM25Encoder
# 解析：BM25 稀疏向量编码器（记忆也做混合检索）
from app.rag.chunking import chunk_text
# 解析：分块器（超长对话轮次切块）
from app.rag.cleaning import make_summary
# 解析：抽取式摘要（记忆条目摘要字段）


class MemoryService:
    """对话记忆：每轮对话作为一条记忆条目向量化入库；检索时按相似度阈值过滤。"""

    def __init__(self, embedder, reranker, milvus, similarity_threshold: float = 0.5):
        # 解析：构造——依赖注入（与知识库服务共享 BGE 模型与 Milvus 连接）
        self.embedder = embedder
        # 解析：向量化模型（BGE-m3）
        self.reranker = reranker
        # 解析：重排序模型（BGE-reranker）
        self.milvus = milvus
        # 解析：Milvus 向量库
        self.similarity_threshold = similarity_threshold
        # 解析：记忆相似度阈值——低于此值的候选记忆不注入提示词
        self._bm25_cache: dict[str, BM25Encoder] = {}
        # 解析：BM25 词表缓存（按记忆库名键控，入库后失效重建）

    @staticmethod
    def _memory_collection(user_id: int, role_id: int) -> str:
        """记忆库按 用户×角色 隔离（互不串扰）。"""
        return f"user_memory_{user_id}_{role_id}"
        # 解析：每个用户×角色组合一个独立 Milvus 集合（与短期记忆隔离原则一致）

    def remember(
        # 解析：沉淀一轮对话为记忆
        self, user_id: int, role_id: int, role_name: str, user_input: str, reply: str
        # 解析：用户ID、角色ID、角色名、用户输入、角色回复
    ) -> None:
        """把一轮对话原文沉淀为记忆条目（每轮一条，零额外 LLM 调用）。"""
        entry = f"用户：{user_input}\n{role_name}：{reply}"
        # 解析：组装一轮对话原文（"用户：xx / 角色名：xx"格式）
        chunks = chunk_text(entry, chunk_size=700, overlap=0)
        # 解析：超长轮次分块（通常一轮一两条块）
        name = self._memory_collection(user_id, role_id)
        # 解析：取该用户×角色的记忆库名
        self.milvus.ensure_named_collection(name)
        # 解析：确保记忆库集合存在（幂等）

        dense_vectors = self.embedder.embed_documents(chunks)
        # 解析：BGE-m3 稠密向量化
        corpus = self.milvus.all_texts_in(name) + chunks
        # 解析：BM25 语料 = 已有记忆 + 新条目（保证新旧统计一致）
        encoder = BM25Encoder()
        # 解析：新建编码器
        encoder.fit(corpus)
        # 解析：在完整语料上统计词表与 IDF
        sparse_rows = encoder.encode_texts(chunks)
        # 解析：生成新条目的 BM25 稀疏向量

        now = int(time.time())
        # 解析：当前 Unix 时间戳（秒）
        records = [
            # 解析：组装入库记录列表
            {
                "text": chunk,
                # 解析：记忆原文
                "summary": make_summary(chunk),
                # 解析：抽取式摘要（前 80 字）
                "dense": dense_vectors[i],
                # 解析：稠密向量
                "sparse": sparse_rows[i],
                # 解析：稀疏向量
                "source": "对话记忆",
                # 解析：来源标记（与知识库文档区分）
                "created_at": now,
                # 解析：创建时间
                "updated_at": now,
                # 解析：更新时间
            }
            for i, chunk in enumerate(chunks)
            # 解析：逐块生成记录
        ]
        self.milvus.insert_into(name, records)
        # 解析：写入 Milvus 记忆库
        self._bm25_cache.pop(name, None)
        # 解析：词表已变，缓存失效（下次检索重建）

    def retrieve(
        # 解析：检索相关长期记忆
        self, user_id: int, role_id: int, query: str, top_k: int = 3, recall_k: int = 10
        # 解析：双方ID、当前问题、返回条数、召回条数
    ) -> list[str]:
        """检索相关长期记忆：混合检索召回 → 余弦相似度阈值过滤不相关记忆。"""
        name = self._memory_collection(user_id, role_id)
        # 解析：取记忆库名
        encoder = self._get_bm25(name)
        # 解析：取该库的 BM25 编码器（带缓存）
        query_sparse = encoder.encode_query(query)
        # 解析：查询的稀疏向量
        query_dense = self.embedder.embed_query(query)
        # 解析：查询的稠密向量
        candidates = self.milvus.hybrid_search_in(name, query_dense, query_sparse, recall_k)
        # 解析：混合检索（稠密+稀疏 RRF 融合）召回候选
        if not candidates:
            # 解析：无候选（记忆库为空等）直接返回空
            return []

        related = []
        # 解析：收集通过阈值过滤的相关记忆
        for text in candidates:
            # 解析：逐条候选
            if self._cosine(query_dense, self.embedder.embed_query(text)) >= self.similarity_threshold:
                # 解析：查询向量与候选向量的余弦相似度达到阈值才算相关
                related.append(text)
                # 解析：相关记忆入列
        if self.reranker is not None and related:
            # 解析：有重排器且有候选时
            related = self.reranker.rerank(query, related)
            # 解析：BGE 重排序精排
        return related[:top_k]
        # 解析：取 Top3 相关记忆返回

    def _get_bm25(self, name: str) -> BM25Encoder:
        # 解析：取记忆库的 BM25 编码器（惰性构建+缓存）
        encoder = self._bm25_cache.get(name)
        # 解析：查缓存
        if encoder is None:
            # 解析：缓存未命中
            encoder = BM25Encoder()
            # 解析：新建编码器
            encoder.fit(self.milvus.all_texts_in(name))
            # 解析：在该库全部记忆上统计词表
            self._bm25_cache[name] = encoder
            # 解析：写入缓存
        return encoder
        # 解析：返回编码器

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        # 解析：两个向量的余弦相似度
        dot = sum(x * y for x, y in zip(a, b))
        # 解析：点积（逐元素相乘求和）
        na = sum(x * x for x in a) ** 0.5
        # 解析：向量 a 的模长
        nb = sum(y * y for y in b) ** 0.5
        # 解析：向量 b 的模长
        if na == 0 or nb == 0:
            # 解析：零向量防御（除零保护）
            return 0.0
            # 解析：零向量相似度定义为 0
        return dot / (na * nb)
        # 解析：余弦相似度 = 点积 / (模长之积)
