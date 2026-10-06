# -*- coding: utf-8 -*-
# 【索引存储模块 · vector_store.py】稠密向量库 + BM25多字段倒排索引（布尔/短语/模糊匹配），支持持久化
# 工单编号：人工智能NLP-RAG-混合检索任务

"""索引层：

- ``DenseVectorStore``：归一化向量矩阵 + 点积即余弦，全库矩阵化毫秒级语义召回；
- ``BM25Store``：jieba 中文分词 + 英文/数字整词，基于倒排表（postings）的 BM25Okapi；
- ``MultiFieldBM25``：对 标题/正文/表格/公司实体/摘要 五字段分别建倒排，
  按字段配置权重加权求和，并支持工单全文检索要求的布尔查询、短语匹配、模糊匹配；
- ``IndexStore``：聚合稠密库、多字段倒排与 Chunk 元数据，统一构建/保存/加载。
"""
import json
import math
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np

from chunker import Chunk
from config import CONFIG
from embeddings import BaseEmbedder, TfidfEmbedder

# 全文检索支持的布尔运算符（中英文写法）
_BOOL_AND = {"AND", "且", "与"}
_BOOL_OR = {"OR", "或"}
_BOOL_NOT = {"NOT", "非", "不含"}


def tokenize(text: str) -> List[str]:
    """中英文混合分词：jieba 切中文，英文/数字整词保留，过滤纯标点。

    :param text: 原始文本
    :return: 词项列表（已小写化）
    """
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# BM25 召回侧查询停用词：公司组织通用名/地名/公文套话/疑问词等。
# 这些词在同文档数千个块里高频出现（还会命中每块“主体：公司全称”前缀），
# 不过滤会让“释义页/封面长块”靠公司名重复次数霸榜，把事实卡挤出召回池。
# 注意：力源/兴图/新科/赵马克/程家明等实体词绝不能停用（跨文档消歧靠它们）。
RECALL_STOPWORDS = frozenset({
    # 公司组织通用称谓
    "公司", "股份", "有限", "有限公司", "股份有限公司", "集团", "企业", "厂",
    "武汉", "湖北", "深圳市", "深圳",
    # 公文/招股书套话
    "根据", "按照", "招股", "招股书", "招股意向书", "招股说明书", "意向书",
    "说明书", "本次", "本", "该", "报告期", "报告期内", "计划", "使用",
    # 疑问/连接/泛化词
    "哪些", "哪个", "什么样", "什么", "多少", "何时", "谁", "吗", "呢",
    "请问", "一下", "主要", "涉及", "包括", "面向", "已经", "成为", "行业",
    "目标", "相关", "以及", "对于", "根据招股",
    # 英文疑问词与公司组织通用词（中文翻译词不经过此表）
    "the", "of", "is", "a", "an", "what", "does", "do", "and", "or",
    "who", "whom", "whose", "which", "co", "ltd", "inc", "wuhan",
})


def _char_bigrams(word: str) -> set:
    """取词的字符 bigram 集合（模糊匹配相似度用）。

    :param word: 词项
    :return: bigram 集合
    """
    return {word[i:i + 2] for i in range(len(word) - 1)} or {word}


def _fuzzy_similarity(a: str, b: str) -> float:
    """基于字符 bigram 的 Dice 相似度（轻量模糊匹配，容错错别字/英文单复数）。

    :param a: 词项A
    :param b: 词项B
    :return: [0,1] 相似度
    """
    ga, gb = _char_bigrams(a), _char_bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


class DenseVectorStore:
    """内存稠密向量库：矩阵化余弦相似度 Top-K 检索。"""

    def __init__(self, matrix: np.ndarray) -> None:
        """保存已 L2 归一化的文档向量矩阵。

        :param matrix: shape=(n, dim) 归一化矩阵
        """
        self.matrix = matrix

    def search(self, query_vec: np.ndarray, top_k: int,
               allowed_ids: Optional["np.ndarray"] = None
               ) -> List[Tuple[int, float]]:
        """向量检索：点积即余弦相似度。

        :param query_vec: shape=(dim,) 归一化查询向量
        :param top_k: 返回前 K 条
        :param allowed_ids: 文档过滤白名单（多文档库中仅在指定文档块内召回），
            缺省为全库召回
        :return: [(chunk_index, score), ...] 按分数降序
        """
        scores = self.matrix @ query_vec
        if allowed_ids is not None and len(allowed_ids) > 0:
            # 非目标文档块置为负无穷，保证 Top-K 全部来自目标文档（硬过滤）
            masked = np.full(scores.shape[0], -np.inf, dtype=np.float32)
            masked[allowed_ids] = scores[allowed_ids]
            scores = masked
        k = min(top_k, int((scores > -np.inf).sum())
                if allowed_ids is not None else scores.shape[0])
        if k <= 0:
            return []
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        ranked = sorted(((int(i), float(scores[i])) for i in idx
                         if scores[i] > -np.inf),
                        key=lambda x: x[1], reverse=True)
        return ranked


class BM25Store:
    """单字段 BM25 稀疏检索：倒排表 postings 实现，含 IDF 与文档长度归一。"""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        """构建词项倒排表与文档频率统计。

        :param docs_tokens: 该字段下每个文档块的分词列表
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
        self.idf: Dict[str, float] = {
            word: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for word, freq in self.df.items()}
        # 倒排表：词项 -> [(文档id, 词频)]，评分时只遍历命中文档
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for doc_id, counter in enumerate(self.tf):
            for word, freq in counter.items():
                self.postings.setdefault(word, []).append((doc_id, freq))
        # 每文档的词项 id 序列（短语有序命中检测用）
        self.doc_seqs: List[List[int]] = []
        vocab = {w: i for i, w in enumerate(self.df)}
        for toks in docs_tokens:
            self.doc_seqs.append([vocab[t] for t in toks if t in vocab])
        self.vocab = vocab

    def _term_score(self, word: str, doc_ids: Optional[set] = None) -> List[Tuple[int, float]]:
        """计算单个词项在各命中文档上的 BM25 贡献。

        :param word: 查询词项
        :param doc_ids: 布尔 AND/NOT 的候选文档白名单（None 表示不限制）
        :return: [(doc_id, 分数贡献)]
        """
        idf = self.idf.get(word)
        if idf is None:
            return []
        out: List[Tuple[int, float]] = []
        for doc_id, freq in self.postings.get(word, []):
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            denom = freq + self.k1 * (
                1 - self.b + self.b * (self.doc_len[doc_id] / (self.avgdl or 1.0)))
            out.append((doc_id, idf * (freq * (self.k1 + 1)) / denom))
        return out

    def parse_boolean(self, query_tokens: List[str]):
        """解析布尔运算符，返回（普通词项, 必须命中词项, 排除词项）。

        支持 AND/OR/NOT 及中文 且/与/或/非/不含；未显式指定时默认 OR 语义。

        :param query_tokens: 原始查询分词（含运算符词）
        :return: (普通或OR词项, AND必须命中词项, NOT排除词项)
        """
        normal, must_have, must_not = [], [], []
        i = 0
        while i < len(query_tokens):
            tok = query_tokens[i]
            if tok in _BOOL_NOT and i + 1 < len(query_tokens):
                must_not.append(query_tokens[i + 1])
                i += 2
                continue
            if tok in _BOOL_AND and i + 1 < len(query_tokens):
                must_have.append(query_tokens[i + 1])
                i += 2
                continue
            if tok in _BOOL_OR:
                i += 1
                continue
            normal.append(tok)
            i += 1
        return normal, must_have, must_not

    def phrase_bonus(self, doc_id: int, phrase_ids: List[Tuple[int, int]]) -> float:
        """检测查询相邻词项是否在文档中有序相邻出现（短语匹配加分）。

        :param doc_id: 文档 id
        :param phrase_ids: [(词项A的vocab id, 词项B的vocab id), ...]
        :return: 命中的短语对数
        """
        seq = self.doc_seqs[doc_id]
        if len(seq) < 2:
            return 0.0
        seq_set = set(zip(seq[:-1], seq[1:]))
        return float(sum(1 for pair in phrase_ids if pair in seq_set))

    def search(self, query_tokens: List[str],
               top_k: int, fuzzy_threshold: float = 0.75,
               phrase_boost: float = 0.35) -> List[Tuple[int, float]]:
        """单字段 BM25 检索（含布尔过滤、短语加分、模糊匹配）。

        :param query_tokens: 查询分词
        :param top_k: 返回条数
        :param fuzzy_threshold: 模糊匹配相似度阈值
        :param phrase_boost: 每个短语对的加分系数
        :return: [(doc_id, 综合分数)] 降序
        """
        normal, must_have, must_not = self.parse_boolean(query_tokens)
        terms = normal + must_have
        if not terms:
            return []

        # 模糊匹配：索引中不存在的查询词，在词表中找字符 bigram 最相似的词替代（折扣计分）
        expanded: List[Tuple[str, float]] = []
        for t in terms:
            if t in self.idf:
                expanded.append((t, 1.0))
            else:
                best, best_sim = None, 0.0
                for vocab_word in self.idf:
                    sim = _fuzzy_similarity(t, vocab_word)
                    if sim > best_sim:
                        best, best_sim = vocab_word, sim
                if best and best_sim >= fuzzy_threshold:
                    expanded.append((best, 0.6 * best_sim))  # 模糊命中折扣

        # 布尔 AND：先算必须命中词项的文档交集；NOT：收集排除文档
        allowed: Optional[set] = None
        if must_have:
            allowed = set()
            for t in must_have:
                hits = {d for d, _ in self._term_score(t)}
                allowed = hits if allowed is None else allowed & hits
        excluded = set()
        for t in must_not:
            excluded.update(d for d, _ in self._term_score(t))
        if allowed is not None:
            allowed -= excluded
        elif excluded:
            allowed = set(range(self.n)) - excluded

        scores = np.zeros(self.n, dtype=np.float32)
        for term, weight in expanded:
            for doc_id, contribution in self._term_score(term, allowed):
                scores[doc_id] += weight * contribution

        # 短语匹配：查询内容词（去运算符）相邻对，在文档词序列中有序相邻则加分
        content_terms = [t for t, _ in expanded]
        phrase_pairs = [(self.vocab[a], self.vocab[b])
                        for a, b in zip(content_terms[:-1], content_terms[1:])
                        if a in self.vocab and b in self.vocab]
        if phrase_pairs:
            hit_docs = np.nonzero(scores > 0)[0]
            for doc_id in hit_docs:
                scores[doc_id] += phrase_boost * self.phrase_bonus(
                    int(doc_id), phrase_pairs)

        k = min(top_k, self.n)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0),
                      key=lambda x: x[1], reverse=True)


# 多字段名固定顺序
FIELD_NAMES = ("title", "company", "table", "summary", "body")
# 父段落字段不在 Chunk.fields 内（Small-to-Big 专用），单独按 parent_text 建倒排
PARENT_FIELD = "parent"


class MultiFieldBM25:
    """五字段加权 BM25 倒排索引（标题/公司实体/表格/摘要/正文）。

    其中 company（公司实体）字段不做普通线性加权：发行人名在同一文档的
    每个块里近似常量出现，线性叠加会给同文档所有块加大致相同的大常数，
    压垮块间内容相关性。实现改为“文档级实体消歧先验”——Query 命中某文档
    独有的实体词时，仅对该文档的块加一个固定小分（跨文档定序、不影响
    同文档内排序）。
    """

    def __init__(self, field_stores: Dict[str, BM25Store],
                 field_weights: Optional[Dict[str, float]] = None,
                 doc_names: Optional[List[str]] = None,
                 company_fields: Optional[List[str]] = None) -> None:
        """保存各字段 BM25 子索引与权重，并构建文档级实体先验表。

        :param field_stores: {字段名: BM25Store}
        :param field_weights: {字段名: 权重}，缺省取 CONFIG.field_weights
        :param doc_names: 每块来源文档名（用于实体先验）
        :param company_fields: 每块公司实体字段原文
        """
        self.field_stores = field_stores
        self.field_weights = field_weights or dict(CONFIG.field_weights)
        # 每块所属文档名（检索期文档硬过滤用）
        self.chunk_doc_of: List[str] = list(doc_names) if doc_names else []
        self._build_entity_prior(doc_names, company_fields)

    def _build_entity_prior(self, doc_names: Optional[List[str]],
                            company_fields: Optional[List[str]]) -> None:
        """构建 {文档名: 该文档独有实体词集合} 与 {文档名: [块下标]}。

        :param doc_names: 每块来源文档名
        :param company_fields: 每块公司实体字段文本
        """
        self.doc_chunk_ids: Dict[str, List[int]] = {}
        self.doc_entity_tokens: Dict[str, set] = {}
        if not doc_names or not company_fields:
            return
        per_doc_tokens: Dict[str, set] = {}
        for i, (doc, company) in enumerate(zip(doc_names, company_fields)):
            self.doc_chunk_ids.setdefault(doc, []).append(i)
            per_doc_tokens.setdefault(doc, set()).update(tokenize(company))
        # 仅保留“只出现在单一文档实体字段中”的独有词（兴图/新科/力源 等）
        for doc, tokens in per_doc_tokens.items():
            others = set()
            for other, other_tokens in per_doc_tokens.items():
                if other != doc:
                    others |= other_tokens
            self.doc_entity_tokens[doc] = {
                t for t in tokens if t not in others and len(t) >= 2}

    @property
    def n_docs(self) -> int:
        """文档块总数（取正文字段，缺字段时退化为任一子索引长度）。"""
        body = self.field_stores.get("body")
        if body is not None:
            return body.n
        return next(iter(self.field_stores.values())).n

    @property
    def idf(self) -> Dict[str, float]:
        """暴露正文路 IDF（查询词项加权、TF-IDF 重排复用）。"""
        return self.field_stores["body"].idf

    def _apply_entity_prior(self, q_tokens: List[str],
                            scores: np.ndarray) -> None:
        """命中某文档独有实体词时，对该文档全部块加文档级先验分。

        :param q_tokens: 查询分词
        :param scores: 内容得分向量（原地追加）
        """
        if not getattr(self, "doc_entity_tokens", None):
            return
        q_set = set(q_tokens)
        for doc, entities in self.doc_entity_tokens.items():
            if q_set & entities:
                ids = self.doc_chunk_ids.get(doc, [])
                if ids:
                    scores[np.asarray(ids, dtype=np.int64)] += (
                        CONFIG.entity_prior_score)

    def search(self, query: str, top_k: int,
               field_weights: Optional[Dict[str, float]] = None,
               doc_name: Optional[str] = None
               ) -> List[Tuple[int, float]]:
        """多字段加权全文检索：内容字段线性加权 + 公司实体文档级先验。

        内容加权公式：``score(d) = 实体先验(d) + Σ_f∈内容字段 w_f · BM25_f(q,d)``

        :param query: 原始查询
        :param top_k: 返回条数
        :param field_weights: 运行期字段权重覆盖（热配）
        :param doc_name: 文档过滤（多文档库中仅在指定文档块内召回），缺省全库
        :return: [(chunk_index, 加权得分)] 降序
        """
        weights = field_weights or self.field_weights
        # 查询清洗：去除公司通用名/公文套话/疑问词，避免“释义页长块”因公司名
        # 重复次数霸榜；公司实体消歧由文档硬过滤/实体先验（力源、兴图等保留）承担
        raw_tokens = tokenize(query)
        q_tokens = [t for t in raw_tokens if t not in RECALL_STOPWORDS]
        if not q_tokens:  # 极端情况下清洗为空则退回原词，保证可召回
            q_tokens = raw_tokens
        scores = np.zeros(self.n_docs, dtype=np.float32)
        # 文档过滤时子字段先过量召回（否则跨文档命中会占满子字段 Top-K 名额）
        field_k = (min(self.n_docs, max(top_k * 10, 400))
                   if doc_name is not None else top_k)
        for field_name, store in self.field_stores.items():
            if field_name == "company":
                continue  # 公司实体走文档级先验，不做块级线性叠加
            w = float(weights.get(field_name, 0.0))
            if w <= 0:
                continue
            hits = store.search(q_tokens, top_k=field_k,
                                fuzzy_threshold=CONFIG.fuzzy_threshold,
                                phrase_boost=CONFIG.phrase_boost)
            for doc_id, score in hits:
                if doc_name is not None and (
                        self.chunk_doc_of[doc_id] != doc_name):
                    continue  # 文档硬过滤：跨文档块不计分
                scores[doc_id] += w * score
        if doc_name is not None:
            # 已按文档硬过滤，实体先验无需再加（避免影响同文档内排序）
            pass
        else:
            self._apply_entity_prior(q_tokens, scores)
        positive = int((scores > 0).sum())
        k = min(top_k, max(positive, 1))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0),
                      key=lambda x: x[1], reverse=True)


class IndexStore:
    """聚合稠密向量库、多字段 BM25 与块元数据，统一构建/保存/加载。"""

    def __init__(self, chunks: List[Chunk], embedder: BaseEmbedder,
                 dense: DenseVectorStore, sparse: MultiFieldBM25) -> None:
        """保存索引四要素。

        :param chunks: 检索块列表（下标对齐向量行/倒排文档）
        :param embedder: 已拟合的嵌入器（TF-IDF 需随索引持久化）
        :param dense: 稠密向量库
        :param sparse: 多字段 BM25 倒排
        """
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse
        # 文档名 -> 该文档全部块下标（检索期文档过滤白名单）
        self.doc_chunk_idx: Dict[str, np.ndarray] = {}
        doc_buckets: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            doc_buckets.setdefault(c.doc_name, []).append(i)
        for doc, ids in doc_buckets.items():
            self.doc_chunk_idx[doc] = np.asarray(ids, dtype=np.int64)
        # 兼容旧版持久化的多字段索引（补齐每块所属文档名）
        if not getattr(self.sparse, "chunk_doc_of", None):
            self.sparse.chunk_doc_of = [c.doc_name for c in chunks]
        # 兼容旧版持久化索引：补齐父段落（Small-to-Big）BM25 子字段与权重，
        # 无需重建整库；重新执行 build_index 后该字段会随索引持久化
        if PARENT_FIELD not in self.sparse.field_stores:
            parent_tokens = [
                tokenize(c.parent_text if c.parent_text else c.text)
                for c in chunks]
            self.sparse.field_stores[PARENT_FIELD] = BM25Store(parent_tokens)
        self.sparse.field_weights.setdefault(
            PARENT_FIELD, CONFIG.field_weights[PARENT_FIELD])

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: BaseEmbedder) -> "IndexStore":
        """用全部块构建稠密 + 多字段倒排双路索引。

        :param chunks: 检索块列表
        :param embedder: 嵌入器（若为 TF-IDF 会先拟合）
        :return: 可检索的 IndexStore
        """
        if isinstance(embedder, TfidfEmbedder):
            embedder.fit([c.text for c in chunks])
        matrix = embedder.encode_documents([c.text for c in chunks])
        dense = DenseVectorStore(matrix)

        # 五字段分别分词建 BM25；缺失字段用空串占位保证文档对齐
        field_stores: Dict[str, BM25Store] = {}
        for field_name in FIELD_NAMES:
            docs_tokens = [tokenize(c.fields.get(field_name, "")) for c in chunks]
            field_stores[field_name] = BM25Store(docs_tokens)
        # 父段落字段：Small-to-Big——引导句/总述句位于父段落、子块为表格明细时，
        # 父段落词项仍能把明细子块召回（如“前五名客户销售额如下”引导客户表）
        parent_tokens = [
            tokenize(c.parent_text if c.parent_text else c.text) for c in chunks]
        field_stores[PARENT_FIELD] = BM25Store(parent_tokens)
        sparse = MultiFieldBM25(
            field_stores,
            doc_names=[c.doc_name for c in chunks],
            company_fields=[c.fields.get("company", "") for c in chunks])
        return cls(chunks, embedder, dense, sparse)

    def dense_search(self, query: str, top_k: int,
                     doc_name: Optional[str] = None
                     ) -> List[Tuple[int, float]]:
        """稠密语义召回。

        :param query: 用户查询
        :param top_k: 返回条数
        :param doc_name: 文档过滤名（仅在该文档块内召回），缺省全库
        """
        qv = self.embedder.encode_queries([query])[0]
        allowed = self.doc_chunk_idx.get(doc_name) if doc_name else None
        return self.dense.search(qv, top_k, allowed_ids=allowed)

    def fulltext_search(self, query: str, top_k: int,
                        field_weights: Optional[Dict[str, float]] = None,
                        doc_name: Optional[str] = None
                        ) -> List[Tuple[int, float]]:
        """多字段加权全文召回。

        :param query: 用户查询
        :param top_k: 返回条数
        :param field_weights: 字段权重热配覆盖
        :param doc_name: 文档过滤名（仅在该文档块内召回），缺省全库
        """
        return self.sparse.search(query, top_k, field_weights, doc_name=doc_name)

    def save(self, index_dir: str) -> None:
        """持久化索引（向量矩阵 + 块元数据 + 多字段倒排 + 嵌入器）。

        :param index_dir: 索引目录
        """
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "vectors.npy"), self.dense.matrix)
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(index_dir, "bm25_fields.pkl"), "wb") as f:
            pickle.dump(self.sparse, f)
        if isinstance(self.embedder, TfidfEmbedder):
            with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)
        meta = {
            "embedding_backend": getattr(self.embedder, "backend", "bge"),
            "embedding_model": getattr(self.embedder, "model_name",
                                       getattr(self.embedder, "backend", "")),
            "chunk_count": len(self.chunks),
            "field_weights": self.sparse.field_weights,
            "docs": sorted({c.doc_name for c in self.chunks}),
        }
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str,
             embedder: Optional[BaseEmbedder] = None) -> "IndexStore":
        """从目录加载索引；TF-IDF 索引自动恢复 vectorizer，稠密索引按 meta 离线加载。

        :param index_dir: 索引目录
        :param embedder: 查询编码嵌入器（可为空，按 meta 自动构建）
        """
        matrix = np.load(os.path.join(index_dir, "vectors.npy"))
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25_fields.pkl"), "rb") as f:
            sparse = pickle.load(f)

        meta_path = os.path.join(index_dir, "meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(
            meta_path) else {}
        tfidf_path = os.path.join(index_dir, "tfidf.pkl")
        if os.path.exists(tfidf_path):
            embedder = TfidfEmbedder()
            with open(tfidf_path, "rb") as f:
                embedder.vectorizer = pickle.load(f)
            embedder.dim = matrix.shape[1]
        elif embedder is None:
            # 按建库 meta 离线还原同后端模型，保证查询向量与索引同空间
            from embeddings import create_embedder
            backend = meta.get("embedding_backend", CONFIG.embedding_backend)
            embedder = create_embedder(backend)
        return cls(chunks, embedder, DenseVectorStore(matrix), sparse)
