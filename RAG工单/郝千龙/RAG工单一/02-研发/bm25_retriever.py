# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【BM25关键词召回 · bm25_retriever.py】基于 jieba+BM25 的词法检索路径，与向量路径互补
# 编写日期：2026-10-04
import pickle
from typing import List, Dict, Optional

import jieba
from rank_bm25 import BM25Okapi

import config


def _tokenize(text: str) -> List[str]:
    """jieba 中文分词，过滤空白；数字、年份、专有名词按词保留"""
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    """BM25 索引：构建自知识库全部 chunk，pickle 持久化避免重复构建"""

    _instance = None
    bm25: Optional[BM25Okapi] = None
    chunk_uids: List[str] = []
    chunk_infos: Dict[str, Dict] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bm25 = None
        return cls._instance

    def build(self, chunks: List[Dict]) -> None:
        """
        用全部 chunk 构建 BM25 索引并持久化
        chunks: VectorStore.get_all_chunks() 输出
        """
        corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        self.chunk_uids = [c["chunk_uid"] for c in chunks]
        self.chunk_infos = {
            c["chunk_uid"]: {
                "text": c["text"], "page": c["page"], "doc_id": c["doc_id"],
            } for c in chunks
        }
        with config.BM25_CACHE.open("wb") as f:
            pickle.dump(
                {"uids": self.chunk_uids, "infos": self.chunk_infos,
                 "bm25": self.bm25}, f,
            )
        print(f"[OK] BM25 索引构建完成: {len(chunks)} 条")

    def load(self) -> bool:
        """从磁盘加载索引，成功返回 True"""
        if self.bm25 is not None:
            return True
        if not config.BM25_CACHE.exists():
            return False
        try:
            data = pickle.loads(config.BM25_CACHE.read_bytes())
            self.bm25 = data["bm25"]
            self.chunk_uids = data["uids"]
            self.chunk_infos = data["infos"]
            return True
        except Exception as e:
            print(f"[WARN] BM25 加载失败: {e}")
            return False

    def rebuild_from_store(self) -> None:
        """从向量库全量同步重建（入库/删除文档后调用）"""
        from vector_store import VectorStore
        chunks = VectorStore.get_all_chunks()
        if chunks:
            self.build(chunks)

    def search(self, question: str, top_n: int = None) -> List[Dict]:
        """BM25 检索 Top-N，返回 [{text,page,doc_id,chunk_id,score}]"""
        if not self.load():
            return []
        top_n = top_n or config.BM25_TOP_N
        scores = self.bm25.get_scores(_tokenize(question))
        # 取分数最高的前 N 个下标
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        out = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            uid = self.chunk_uids[idx]
            info = self.chunk_infos[uid]
            out.append({
                "text": info["text"],
                "page": info["page"],
                "doc_id": info["doc_id"],
                "chunk_id": uid,
                "score": float(scores[idx]),
            })
        return out


if __name__ == "__main__":
    r = BM25Retriever()
    if r.load():
        for h in r.search("公司注册资本是多少？"):
            print(h["page"], round(h["score"], 2), h["text"][:50])
    else:
        print("BM25 索引不存在，请先完成入库")

# ====================================================================
# 技术备注：
# 1. RAG：BM25（Best Matching 25）是经典词法检索，基于词频、逆文档频率与
#    文档长度归一化打分，对公司名、年份、数字等关键词命中精准，
#    与稠密向量语义检索形成互补。
# 2. RRF 融合在 rag_engine 中完成，双路命中互为兜底。
# 3. Transformer：本模块不依赖神经网络，纯统计方法，无 GPU 开销。
# ====================================================================
