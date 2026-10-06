import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models.entities import Document, DocumentChunk
from embedding.bge_m3_embedder import get_embedding_service
from rerank.bge_reranker import get_reranker_service
from vectorstores.milvus_store import MilvusStore


settings = get_settings()


class RetrievalService:
    """负责 BM25 + 向量检索 + RRF 融合 + Rerank 重排。"""

    def __init__(self) -> None:
        # 加载向量模型服务，用于问题向量化。
        self.embedding_service = get_embedding_service()
        # 加载 Milvus 服务，用于向量检索。
        self.milvus = MilvusStore()
        # 加载 Reranker 服务，用于最终排序。
        self.reranker = get_reranker_service()

    def retrieve(self, db: Session, user_id: int, knowledge_base_id: int, question: str) -> list[dict]:
        # 先用 BGE-M3 把用户问题转成向量。
        query_vector = self.embedding_service.encode([question])[0]
        # 从 Milvus 做向量召回；返回结果已经包含分块文本、标题和页码。
        vector_items = self.milvus.search_chunks(query_vector, user_id, knowledge_base_id, settings.vector_top_k)
        # 从 MySQL 文本块做内置 BM25 关键词召回。
        bm25_items = self._bm25_search(db, user_id, knowledge_base_id, question)
        # 使用 RRF 算法融合两路召回结果。
        fused_items = self._rrf_fuse([vector_items, bm25_items])
        # 使用 BGE Reranker 对融合后的候选重新排序。
        reranked_items = self.reranker.rerank(question, fused_items)
        # 补充文件名，方便前端显示引用来源。
        self._attach_document_names(db, reranked_items)
        # 返回最终给 DeepSeek 的依据。
        return reranked_items

    def _bm25_search(self, db: Session, user_id: int, knowledge_base_id: int, question: str) -> list[dict]:
        # 从 MySQL 读取当前用户当前知识库的全部文本块。
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.user_id == user_id, DocumentChunk.knowledge_base_id == knowledge_base_id)
            .all()
        )
        # 如果知识库还没有分块，直接返回空结果。
        if not chunks:
            return []
        # 使用 jieba 做中文分词，BM25 需要词列表而不是原始字符串。
        corpus = [list(jieba.cut(chunk.content)) for chunk in chunks]
        # 初始化 BM25 检索器。
        bm25 = BM25Okapi(corpus)
        # 对用户问题分词。
        query_tokens = list(jieba.cut(question))
        # 计算每个文本块的 BM25 分数。
        scores = bm25.get_scores(query_tokens)
        # 把分数和文本块配对。
        ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)[: settings.bm25_top_k]
        # 转成统一候选格式。
        return [
            {
                "chunk_uid": chunk.chunk_uid,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "title_path": chunk.title_path,
                "page_number": chunk.page_number,
                "score": float(score),
                "source": "bm25",
            }
            for chunk, score in ranked
            if score > 0
        ]

    def _rrf_fuse(self, result_lists: list[list[dict]]) -> list[dict]:
        # 用 chunk_uid 去重并累加 RRF 分数。
        score_map: dict[str, float] = {}
        # 保存每个 chunk_uid 对应的完整候选对象。
        item_map: dict[str, dict] = {}
        # 遍历每一路检索结果。
        for results in result_lists:
            # rank 从 1 开始更符合 RRF 公式。
            for rank, item in enumerate(results, start=1):
                # 取当前候选的唯一编号。
                chunk_uid = item["chunk_uid"]
                # RRF 公式：1 / (k + rank)，排名越靠前贡献越大。
                score_map[chunk_uid] = score_map.get(chunk_uid, 0.0) + 1.0 / (settings.rrf_k + rank)
                # 保存候选内容，如果两路都有同一块，保留第一次出现的内容。
                item_map.setdefault(chunk_uid, dict(item))
        # 按 RRF 融合分数排序。
        ranked_uids = sorted(score_map, key=lambda uid: score_map[uid], reverse=True)[: settings.rrf_result_top_k]
        # 生成融合后的候选列表。
        fused: list[dict] = []
        for uid in ranked_uids:
            item = item_map[uid]
            item["score"] = score_map[uid]
            item["source"] = "rrf"
            fused.append(item)
        # 返回融合结果。
        return fused

    def _attach_document_names(self, db: Session, items: list[dict]) -> None:
        # 收集候选中涉及的文档编号。
        document_ids = {item["document_id"] for item in items}
        # 如果没有候选，就不需要查询数据库。
        if not document_ids:
            return
        # 查询文档表，获得文件名。
        documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
        # 转成 document_id 到 filename 的映射。
        name_map = {document.id: document.filename for document in documents}
        # 给每个候选补充 filename 字段。
        for item in items:
            item["filename"] = name_map.get(item["document_id"], "未知文档")
