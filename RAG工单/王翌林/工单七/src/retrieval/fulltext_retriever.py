# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval/fulltext_retriever.py —— 工单六 全文检索器

实现要点（任务要求"使用倒排索引技术实现全文检索，支持多字段检索"）：
  1. 数据源：Milvus rag_chunks 全量文本 chunk（约千级，进程内单例懒加载）
  2. 倒排索引：jieba 分词后 token → posting list（chunk 下标 + tf），
     支持 O(命中词表长) 的布尔集合运算
  3. 评分：经典 TF-IDF 向量空间模型 + 余弦相似度
  4. 匹配方式：and（布尔 AND）/ or（布尔 OR）/ phrase（短语连续匹配）/
     fuzzy（字符 bigram Jaccard 模糊匹配）
  5. 多字段：title（doc_id 文档名）/ content（正文）/ summary（chunk 首部摘要），
     字段权重可配；另支持显式 AND/OR/NOT 布尔表达式（+/- 操作符语法）
"""
import math
import re
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from src.retrieval.retrieval_config import (
    FIELD_CONTENT, FIELD_SUMMARY, FIELD_TITLE, MATCH_FUZZY, MATCH_OR,
    MATCH_PHRASE,
)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
_SUMMARY_CHARS = 80

_STOPWORDS = set(
    "的 了 和 是 在 有 我 他 她 它 这 那 就 不 也 都 一 个 上 下 中 到 说 去 "
    "你 好 为 什么 怎么 哪 谁 几 多 少 吗 呢 吧 啊 哦 嗯 会 能 可以 应该 "
    "需要 希望 想 看 听 问 答 知道 了解 多少 以及 及其 关于 对于 报告 期内".split()
)


def tokenize(text: str) -> List[str]:
    """工单六：jieba 分词（去空白/单字功能词，保留业务术语与数字）"""
    import jieba
    out = []
    for t in jieba.lcut(text or ""):
        t = t.strip()
        if not t or t in _STOPWORDS:
            continue
        if re.fullmatch(r"[\d,.\-]*\d[\d,.\-]*", t):  # 数字整体保留（必须含数字）
            out.append(t)
        elif len(t) >= 2 or re.fullmatch(r"[A-Za-z]{2,}", t):
            out.append(t.lower())
    return out


def _bigrams(text: str) -> set:
    """工单六：字符 bigram 集合（模糊匹配用）"""
    text = re.sub(r"\s+", "", text or "")
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else set()


class FulltextRetriever:
    """工单六：倒排索引全文检索器（进程内单例索引，线程安全懒加载）"""

    _global_index: Optional["FulltextRetriever"] = None
    _lock = threading.Lock()

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None,
                 field_weights: Optional[Dict[str, float]] = None):
        """测试可直接注入 chunks；生产从 Milvus rag_chunks 全量加载"""
        self._field_weights = field_weights or {
            FIELD_TITLE: 0.5, FIELD_CONTENT: 1.0, FIELD_SUMMARY: 0.6}
        self.docs: List[Dict[str, Any]] = []
        self._tokens: List[Dict[str, List[str]]] = []   # 每文档每字段的分词
        self._postings: Dict[str, Dict[str, List[int]]] = {
            f: {} for f in (FIELD_TITLE, FIELD_CONTENT, FIELD_SUMMARY)}
        self._idf: Dict[str, Dict[str, float]] = {
            f: {} for f in (FIELD_TITLE, FIELD_CONTENT, FIELD_SUMMARY)}
        if chunks is not None:
            self.build(chunks)

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "FulltextRetriever":
        """工单六：进程级单例（避免每问重建千文档倒排索引）"""
        if cls._global_index is None:
            with cls._lock:
                if cls._global_index is None:
                    retriever = cls()
                    retriever._load_from_milvus()
                    cls._global_index = retriever
        return cls._global_index

    def _load_from_milvus(self) -> None:
        """工单六：从 Milvus rag_chunks 全量拉取构建索引"""
        try:
            from src.vector_store import VectorStore
            vs = VectorStore()
            if not vs.client.has_collection(vs.collection):
                logger.warning("[fulltext_v6] rag_chunks 不存在，索引为空")
                self.build([])
                return
            vs.ensure_collection()
            rows = vs.client.query(
                collection_name=vs.collection, filter="id >= 0",
                output_fields=["doc_id", "chunk_id", "content", "page",
                               "metadata"], limit=16384)
            chunks = [{
                "doc_id": r.get("doc_id", ""), "chunk_id": r.get("chunk_id", ""),
                "content": r.get("content", ""), "page": r.get("page", 0),
                "metadata": r.get("metadata", {}),
            } for r in (rows or [])]
            logger.info(f"[fulltext_v6] 从 Milvus 加载 {len(chunks)} 个 chunk 建倒排索引")
            self.build(chunks)
        except Exception as e:
            logger.warning(f"[fulltext_v6] Milvus 加载失败，索引为空: {e}")
            self.build([])

    # ------------------------------------------------------------------
    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """工单六：构建多字段倒排索引 + IDF 表"""
        self.docs = chunks
        self._tokens = []
        self._postings = {f: {} for f in (FIELD_TITLE, FIELD_CONTENT, FIELD_SUMMARY)}
        for i, c in enumerate(chunks):
            fields = self._doc_fields(c)
            toks = {f: tokenize(fields[f]) for f in fields}
            self._tokens.append(toks)
            for f, ts in toks.items():
                seen = set(ts)
                for tok in seen:
                    self._postings[f].setdefault(tok, []).append(i)
        n = max(len(chunks), 1)
        for f in self._postings:
            self._idf[f] = {tok: math.log((1 + n) / (1 + len(pst)) + 1.0)
                            for tok, pst in self._postings[f].items()}

    @staticmethod
    def _doc_fields(chunk: Dict[str, Any]) -> Dict[str, str]:
        """工单六：抽取多字段（标题=doc_id / 正文=content / 摘要=正文前 80 字）"""
        content = chunk.get("content") or chunk.get("text") or ""
        return {
            FIELD_TITLE: chunk.get("doc_id", "") or "",
            FIELD_CONTENT: content,
            FIELD_SUMMARY: content[:_SUMMARY_CHARS],
        }

    @property
    def size(self) -> int:
        return len(self.docs)

    # ------------------------------------------------------------------
    def _parse_boolean(self, query: str) -> Dict[str, Any]:
        """工单六：解析简单布尔表达式

        语法：词前 '+' 表示必须(AND)、'-' 表示必须不包含(NOT)、
        引号包裹为短语；无操作符按 match 参数决定 AND/OR。
        """
        required, excluded, phrases = [], [], []
        # 先抽取引号短语
        phrases = re.findall(r'[“"]([^”"]+)[”"]', query)
        q = re.sub(r'[“"][^”"]+[”"]', " ", query)
        for raw in q.split():
            if raw.startswith("-"):
                excluded.append(raw[1:])
            elif raw.startswith("+"):
                required.append(raw[1:])
            else:
                required.append(raw)
        return {"required": required, "excluded": excluded, "phrases": phrases}

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 24,
               doc_id: Optional[str] = None,
               match: str = "and",
               fields: Optional[List[str]] = None,
               field_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """工单六：全文检索主入口

        Returns: 与 vector_store.search 同构的 hit 列表
                 [{doc_id, chunk_id, content, page, metadata, score, source, search_path}]
        """
        t0_fields = fields or [FIELD_CONTENT, FIELD_TITLE, FIELD_SUMMARY]
        fw = field_weights or self._field_weights
        bool_expr = self._parse_boolean(query)
        # 工单六：倒排查询词需剔除排除词（-词）与短语词（短语单独走 phrase_hits）
        drop_tokens = set()
        for ex in bool_expr["excluded"]:
            drop_tokens.update(tokenize(ex))
        for ph in bool_expr["phrases"]:
            drop_tokens.update(tokenize(ph))
        q_tokens = [t for t in tokenize(query) if t not in drop_tokens]
        if not self.docs or not q_tokens and not bool_expr["phrases"]:
            return []

        # 工单六：短语匹配（连续子串，跨多字段但主要在正文）
        phrase_hits = set()
        for ph in bool_expr["phrases"]:
            for i, c in enumerate(self.docs):
                ftext = self._doc_fields(c)
                if any(ph in ftext.get(f, "") for f in t0_fields):
                    phrase_hits.add(i)

        scores: Dict[int, float] = {}

        # 工单六：模糊匹配 —— 字符 bigram Jaccard（容忍错字/分词差异）
        if match == MATCH_FUZZY:
            q_bi = _bigrams(query)
            for i, c in enumerate(self.docs):
                ftext = self._doc_fields(c)
                best = 0.0
                for f in t0_fields:
                    d_bi = _bigrams(ftext.get(f, ""))
                    if q_bi and d_bi:
                        jac = len(q_bi & d_bi) / len(q_bi | d_bi)
                        best = max(best, jac * fw.get(f, 1.0))
                if best > 0.05:
                    scores[i] = best
        else:
            # 工单六：倒排索引召回 + 多字段 TF-IDF 评分
            per_field_hits: Dict[str, set] = {}
            for f in t0_fields:
                idxs = self._match_field(f, q_tokens, match, bool_expr)
                per_field_hits[f] = idxs
            candidates = set().union(*per_field_hits.values()) if per_field_hits else set()
            if bool_expr["phrases"]:
                candidates |= phrase_hits
            # 工单六：NOT 排除
            for ex in bool_expr["excluded"]:
                ex_set = set()
                for f in t0_fields:
                    ex_set |= set(self._postings[f].get(ex, set()))
                    if ex:
                        # 子串排除（未登录词）
                        for i, c in enumerate(self.docs):
                            if ex in self._doc_fields(c).get(f, ""):
                                ex_set.add(i)
                candidates -= ex_set
            for i in candidates:
                s = 0.0
                for f in t0_fields:
                    s += self._tfidf_score(i, f, q_tokens) * fw.get(f, 1.0)
                if i in phrase_hits:
                    s += 1.0                      # 工单六：短语精确命中额外加分
                scores[i] = s

        # 工单六：doc_id 过滤（多文档隔离）
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        hits = []
        for i, score in ranked:
            c = self.docs[i]
            if doc_id and c.get("doc_id") != doc_id:
                continue
            if score <= 0:
                continue
            hits.append({
                "doc_id": c.get("doc_id", ""), "chunk_id": c.get("chunk_id", ""),
                "content": c.get("content") or c.get("text") or "",
                "page": c.get("page", 0), "metadata": c.get("metadata", {}),
                "score": round(float(score), 6), "source": "text",
                "search_path": "fulltext",
            })
            if len(hits) >= top_k:
                break
        logger.info(f"[fulltext_v6] match={match} hits={len(hits)}")
        return hits

    # ------------------------------------------------------------------
    def _match_field(self, field: str, q_tokens: List[str],
                     match: str, bool_expr: Dict[str, Any]) -> set:
        """工单六：单字段倒排索引布尔集合运算"""
        postings = self._postings[field]
        if match == MATCH_PHRASE:
            # 工单六：phrase 模式退化为所有查询词在同字段共现
            match = "and"
        lists = [set(postings.get(tok, set())) for tok in q_tokens if tok in postings]
        # 未登录词补子串扫描（招股书专名可能被 jieba 切碎）
        for tok in q_tokens:
            if tok not in postings and len(tok) >= 2:
                sub = set()
                for i, c in enumerate(self.docs):
                    if tok in self._doc_fields(c).get(field, ""):
                        sub.add(i)
                lists.append(sub)
        if not lists:
            return set()
        if match == MATCH_OR:
            return set().union(*lists)
        return set.intersection(*lists)          # 默认 AND（含布尔 + 词）

    def _tfidf_score(self, doc_idx: int, field: str,
                     q_tokens: List[str]) -> float:
        """工单六：查询-文档 TF-IDF 余弦相似度（单字段）"""
        toks = self._tokens[doc_idx].get(field, [])
        if not toks:
            return 0.0
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        # 文档向量 L2 范数（tf-idf 加权）
        norm = 0.0
        dot = 0.0
        q_set = set(q_tokens)
        for tok, f in tf.items():
            idf = self._idf[field].get(tok, 0.0)
            w = (1.0 + math.log(f)) * idf
            norm += w * w
            if tok in q_set:
                qtf = q_tokens.count(tok)
                qw = (1.0 + math.log(qtf)) * idf
                dot += w * qw
        q_norm = 0.0
        q_tf: Dict[str, int] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        for tok, f in q_tf.items():
            idf = self._idf[field].get(tok, math.log(len(self.docs) + 1))
            qw = (1.0 + math.log(f)) * idf
            q_norm += qw * qw
        if norm <= 0 or q_norm <= 0:
            return 0.0
        return dot / math.sqrt(norm * q_norm)
