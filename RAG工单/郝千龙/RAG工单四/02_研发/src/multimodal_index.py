# -*- coding: utf-8 -*-
# 【多模态双索引模块 · multimodal_index.py】文本/表格/图像统一建块，稠密向量+BM25双路索引
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""图文统一索引层：

- 文本/表格Block经标题感知切分形成文字块，图像FigureRecord形成图像证据块；
- 稠密向量：本地BGE双语嵌入（无缓存时降级字符ngram TF-IDF）；
- 稀疏索引：jieba分词BM25倒排，兜住数字/专名；
- numpy内存矩阵毫秒级检索，全部产物持久化到index_store。
"""
import json
import logging
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jieba
import numpy as np

from config import CONFIG
from image_extractor import FigureRecord
from pdf_parser import Block

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|(?<=\.)\s+")
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


# ============================ 分块 ============================

@dataclass
class Chunk:
    """统一检索块：文字块/表格块/图像证据块。"""

    chunk_id: int
    text: str
    parent_text: str
    page_no: int
    doc: str
    heading_path: str = ""
    chunk_type: str = "text"      # text / table / image
    companies: List[str] = field(default_factory=list)
    figure: Optional[FigureRecord] = None


def _split_sentences(text: str) -> List[str]:
    """按中文/英文句读切句，过短句并入相邻句。

    :param text: 段落原文
    :return: 句子列表
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < 8:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _pack(sentences: List[str], size: int, overlap: int) -> List[str]:
    """贪心打包句子为目标长度块，块间按字符滑窗重叠。

    :param sentences: 句子列表
    :param size: 目标块字符数
    :param overlap: 重叠字符数
    :return: 块文本列表
    """
    packs: List[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > size:
            packs.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = tail + sent
        else:
            buf += sent
    if buf.strip():
        packs.append(buf.strip())
    return packs


def _main_company(blocks: List[Block], doc_tag: str) -> str:
    """按词频选出单个文档的主体公司名（发行人）。

    :param blocks: 该文档全部Block
    :param doc_tag: 文档标签
    :return: 公司全称，取不到时取配置表
    """
    counter: Counter = Counter()
    for b in blocks:
        if b.doc == doc_tag:
            counter.update(_COMPANY_RE.findall(b.text))
    for name, _cnt in counter.most_common():
        if CONFIG.doc_companies.get(doc_tag, "") in name or name in \
                CONFIG.doc_companies.get(doc_tag, ""):
            return name
    fallback = CONFIG.doc_companies.get(doc_tag, "")
    return counter.most_common(1)[0][0] if counter and not fallback else fallback


def build_chunks(blocks: List[Block],
                 figures: List[FigureRecord]) -> List[Chunk]:
    """把文本/表格Block与图像FigureRecord统一构建为检索Chunk。

    :param blocks: 全部文档Block
    :param figures: 全部图像证据记录
    :return: 带编号、页码、文档标签、类型与公司实体的Chunk列表
    """
    doc_tags = sorted({b.doc for b in blocks} | {f.doc for f in figures})
    main_company = {tag: _main_company(blocks, tag) for tag in doc_tags}
    chunks: List[Chunk] = []
    cid = 0

    # 正文父段落（Small-to-Big）：同文档同标题路径同页聚合
    parents: dict = {}
    for b in blocks:
        if b.type == "text":
            parents.setdefault((b.doc, b.heading_path, b.page_no), []).append(b.text)

    for b in blocks:
        company = main_company.get(b.doc, "")
        if b.type == "table":
            cid += 1
            prefix = f"（{b.heading_path or '表格'}）" if b.heading_path else ""
            text = f"【{b.doc}】{prefix}\n{b.text}"
            chunks.append(Chunk(cid, text, text, b.page_no, b.doc,
                                b.heading_path, "table", [company]))
            continue
        sentences = _split_sentences(b.text)
        for pack in _pack(sentences, CONFIG.chunk_size, CONFIG.chunk_overlap):
            cid += 1
            parent = "".join(parents.get((b.doc, b.heading_path, b.page_no),
                                         [b.text]))
            head = f"【{b.doc}｜{b.heading_path}】\n" if b.heading_path else f"【{b.doc}】\n"
            ent = f"主体：{company}\n" if company else ""
            chunks.append(Chunk(
                cid, f"{head}{ent}{pack}",
                parent[:1200], b.page_no, b.doc, b.heading_path, "text",
                [company] if company else []))

    # 图像证据块（伪多模态：图题+图内文字+邻近正文）
    for fig in figures:
        cid += 1
        company = main_company.get(fig.doc, "")
        chunks.append(Chunk(
            cid, fig.to_text(), fig.to_text(), fig.page_no, fig.doc,
            f"图像证据/{fig.caption or '未命名图'}", "image",
            [company] if company else [], figure=fig))
    return chunks


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba切中文，英文/数字整词保留。

    :param text: 原始文本
    :return: 词项列表
    """
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# ============================ 嵌入 ============================

class BaseEmbedder:
    """嵌入器统一接口（返回L2归一化向量）。"""

    dim: int = 0

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块（子类实现）。"""
        raise NotImplementedError

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询（子类实现）。"""
        raise NotImplementedError

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """按行L2归一化，使点积等价余弦相似度。

        :param vectors: 原始向量矩阵
        :return: 归一化矩阵
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms


class BGEEmbedder(BaseEmbedder):
    """本地BGE双语嵌入实现（sentence-transformers加载HF缓存权重）。"""

    def __init__(self) -> None:
        """加载本地BGE模型（离线，不触发联网）。"""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(CONFIG.embedding_model)
        self.dim = self.model.get_embedding_dimension()

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        vec = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=32)
        return vec.astype(np.float32)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询（追加BGE中文检索指令前缀）。"""
        queries = [f"{CONFIG.query_instruction}{t}" for t in texts]
        vec = self.model.encode(queries, normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=32)
        return vec.astype(np.float32)


class TfidfEmbedder(BaseEmbedder):
    """离线降级嵌入：字符级2~4gram TF-IDF。"""

    def __init__(self) -> None:
        """初始化TF-IDF向量化器。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb",
                                          ngram_range=(2, 4),
                                          max_features=30000)
        self.dim = 0

    def fit(self, texts: List[str]) -> None:
        """用全部文档块拟合IDF。

        :param texts: 文档文本列表
        """
        matrix = self.vectorizer.fit_transform(texts)
        self.dim = matrix.shape[1]

    def _encode(self, texts: List[str]) -> np.ndarray:
        """统一编码并转稠密归一化矩阵。"""
        vec = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._l2_normalize(vec)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """编码文档块。"""
        return self._encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """编码查询（TF-IDF无需指令前缀）。"""
        return self._encode(texts)


def model_available_locally(model_name: str) -> bool:
    """检测HF模型是否已在本地缓存（离线环境避免联网等待）。

    :param model_name: HF模型ID
    :return: 本地有快照返回True
    """
    import glob
    if os.path.exists(model_name):
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    repo = "models--" + model_name.replace("/", "--")
    return bool(glob.glob(os.path.join(hub, repo, "snapshots", "*")))


def create_embedder(doc_texts: Optional[List[str]] = None) -> BaseEmbedder:
    """嵌入器工厂：本地BGE优先，不可用降级TF-IDF，全程不联网下载。

    :param doc_texts: 文档块文本（TF-IDF拟合用）
    :return: 已就绪嵌入器
    """
    if CONFIG.fallback_embedding != "tfidf" and model_available_locally(
            CONFIG.embedding_model):
        try:
            logger.info("加载本地BGE嵌入模型：%s", CONFIG.embedding_model)
            return BGEEmbedder()
        except Exception as exc:
            logger.warning("BGE加载失败，降级TF-IDF：%s", exc)
    else:
        logger.info("未检测到%s本地缓存，启用TF-IDF离线嵌入（不联网下载）",
                    CONFIG.embedding_model)
    embedder = TfidfEmbedder()
    if doc_texts and any(t.strip() for t in doc_texts):
        embedder.fit(doc_texts)
    logger.info("TF-IDF离线嵌入就绪，维度=%d", embedder.dim)
    return embedder


# ============================ 索引存储 ============================

class DenseVectorStore:
    """内存稠密向量库：归一化矩阵点积即余弦，Top-K毫秒检索。"""

    def __init__(self, matrix: np.ndarray) -> None:
        """保存L2归一化文档向量矩阵。

        :param matrix: shape=(n,dim)
        """
        self.matrix = matrix

    def search(self, query_vec: np.ndarray, top_k: int
               ) -> List[Tuple[int, float]]:
        """向量检索Top-K。

        :param query_vec: 归一化查询向量
        :param top_k: 返回条数
        :return: [(块下标, 余弦分)] 降序
        """
        scores = self.matrix @ query_vec
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx),
                      key=lambda x: x[1], reverse=True)


class BM25Store:
    """纯Python BM25倒排：含IDF与文档长度归一。"""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        """构建词项倒排与文档频率统计。

        :param docs_tokens: 每块分词列表
        :param k1: 词频饱和参数
        :param b: 长度归一参数
        """
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs_tokens]
        self.df: Counter = Counter()
        for doc in self.tf:
            self.df.update(doc.keys())
        self.idf = {
            word: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for word, freq in self.df.items()}

    def search(self, query_tokens: List[str], top_k: int
               ) -> List[Tuple[int, float]]:
        """计算查询对全部文档的BM25分并取Top-K。

        :param query_tokens: 查询分词
        :param top_k: 返回条数
        :return: [(块下标, BM25分)] 降序
        """
        scores = np.zeros(self.n, dtype=np.float32)
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for i, tf in enumerate(self.tf):
                freq = tf.get(word, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1.0)))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        k = min(top_k, self.n)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0),
                      key=lambda x: x[1], reverse=True)


class IndexStore:
    """聚合稠密库、BM25与Chunk元数据，支持构建/保存/加载。"""

    def __init__(self, chunks: List[Chunk], embedder: BaseEmbedder,
                 dense: DenseVectorStore, sparse: BM25Store) -> None:
        """保存索引四要素。

        :param chunks: 检索块（下标对齐向量行/倒排文档）
        :param embedder: 已拟合嵌入器
        :param dense: 稠密向量库
        :param sparse: BM25倒排
        """
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: BaseEmbedder) -> "IndexStore":
        """用全部块文本构建双路索引。

        :param chunks: 检索块
        :param embedder: 嵌入器（TF-IDF先拟合）
        :return: 可检索IndexStore
        """
        texts = [c.text for c in chunks]
        if isinstance(embedder, TfidfEmbedder):
            embedder.fit(texts)
        matrix = embedder.encode_documents(texts)
        return cls(chunks, embedder, DenseVectorStore(matrix),
                   BM25Store([tokenize(t) for t in texts]))

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """稠密召回。"""
        qv = self.embedder.encode_queries([query])[0]
        return self.dense.search(qv, top_k)

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """稀疏关键词召回。"""
        return self.sparse.search(tokenize(query), top_k)

    def save(self, index_dir: str, stats: Optional[dict] = None) -> None:
        """持久化索引（向量+块+BM25+嵌入器+构建统计）。

        :param index_dir: 索引目录
        :param stats: 建库统计（文档数/块数/图片数等）
        """
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "vectors.npy"), self.dense.matrix)
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.sparse, f)
        tfidf_path = os.path.join(index_dir, "tfidf.pkl")
        if isinstance(self.embedder, TfidfEmbedder):
            with open(tfidf_path, "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)
        elif os.path.exists(tfidf_path):
            # 从TF-IDF基线切换为BGE后，移除遗留向量化器，避免加载时误判
            os.remove(tfidf_path)
        meta = {
            "embedding": "tfidf" if isinstance(self.embedder, TfidfEmbedder)
            else CONFIG.embedding_model,
            "chunk_count": len(self.chunks),
            "image_chunk_count": sum(1 for c in self.chunks
                                     if c.chunk_type == "image"),
        }
        if stats:
            meta.update(stats)
        with open(os.path.join(index_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str, embedder: Optional[BaseEmbedder] = None
             ) -> "IndexStore":
        """从目录加载索引（TF-IDF时自动恢复向量化器）。

        :param index_dir: 索引目录
        :param embedder: 查询嵌入器（TF-IDF索引可传None）
        """
        matrix = np.load(os.path.join(index_dir, "vectors.npy"))
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            sparse = pickle.load(f)
        # 仅当索引确为TF-IDF基线构建时才恢复向量化器；BGE索引使用传入的嵌入器
        meta_path = os.path.join(index_dir, "meta.json")
        emb_type = ""
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                emb_type = json.load(f).get("embedding", "")
        tfidf_path = os.path.join(index_dir, "tfidf.pkl")
        if emb_type == "tfidf" and os.path.exists(tfidf_path):
            embedder = TfidfEmbedder()
            with open(tfidf_path, "rb") as f:
                embedder.vectorizer = pickle.load(f)
            embedder.dim = matrix.shape[1]
        return cls(chunks, embedder, DenseVectorStore(matrix), sparse)
