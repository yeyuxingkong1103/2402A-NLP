# -*- coding: utf-8 -*-
"""
rag_core.py — RAG 通用核心模块
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统

本模块封装了 RAG 问答系统的基础能力，供各任务工单复用：
  - PDF 文本 / 表格 / 图像解析
  - 文本分块（可配置 chunk_size / overlap）
  - 向量嵌入（优先 sentence-transformers，缺省回退到 TF-IDF 离线向量）
  - 向量检索（numpy 余弦相似度）
  - 全文检索（BM25 倒排索引）
  - 混合检索（向量 + 全文 + 加权融合 + 重排）
  - LLM 生成（OpenAI 兼容 API，缺省回退到抽取式回答）
  - 简单 RAGAS 风格评估指标（离线可用）

依赖（离线可用）：numpy、scikit-learn、pymupdf
可选依赖（增强效果）：sentence-transformers、openai
"""
from __future__ import annotations

import os
import re
import math
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

# ============================================================================
# 一、配置
# ============================================================================
class Config:
    """从环境变量 / .env 文件读取配置。"""

    def __init__(self, env_file: Optional[str] = None):
        self._load_env(env_file)
        # LLM 配置（OpenAI 兼容接口，可指向 DeepSeek / Qwen / 智谱 / OpenAI / Ollama 等）
        self.llm_api_base: str = os.getenv("LLM_API_BASE", "")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
        # 嵌入模型（真实向量检索；为空则使用离线 TF-IDF）
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        # 分块参数
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

    @staticmethod
    def _load_env(env_file: Optional[str]):
        path = env_file or os.path.join(os.getcwd(), ".env")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ============================================================================
# 二、PDF 解析
# ============================================================================
def load_pdf_text(path: str) -> str:
    """解析 PDF 并提取纯文本。"""
    import fitz  # pymupdf
    doc = fitz.open(path)
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)


def load_pdf_pages(path: str) -> List[Dict[str, Any]]:
    """解析 PDF，返回每一页的 {page_no, text, tables, images}，供表格/图像工单复用。"""
    import fitz
    doc = fitz.open(path)
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        item: Dict[str, Any] = {
            "page_no": i + 1,
            "text": page.get_text(),
            "tables": _extract_tables(page),
            "images": _extract_images(page),
        }
        pages.append(item)
    doc.close()
    return pages


def _extract_tables(page) -> List[List[List[str]]]:
    """提取单页中的表格（pymupdf 自带表格识别）。

    返回表格列表，每个表格为二维字符串数组（行 × 列）。
    """
    tables: List[List[List[str]]] = []
    try:
        for t in page.find_tables().tables:
            rows: List[List[str]] = []
            for row in t.extract():
                rows.append([("" if c is None else str(c)) for c in row])
            if rows:
                tables.append(rows)
    except Exception:
        pass
    return tables


def _extract_images(page) -> List[Dict[str, Any]]:
    """提取单页中的图像（返回图像字节与坐标，供图像语义解析工单复用）。"""
    import fitz
    images: List[Dict[str, Any]] = []
    for xref, *_ in page.get_images(full=True):
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n - pix.alpha >= 4:  # CMYK 转 RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            images.append({
                "xref": xref,
                "width": pix.width,
                "height": pix.height,
                "bytes": pix.tobytes("png"),
                "samples": pix.samples,
            })
        except Exception:
            continue
    return images


# ============================================================================
# 三、文本分块
# ============================================================================
def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 60) -> List[str]:
    """按字符滑动窗口切分文本，尽量在句号/换行处断句。"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 在窗口内寻找最近的断句点，避免从句子中间切断
            for sep in ("。", "；", "\n", "！", "？", "，", "、"):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


# ============================================================================
# 四、中文分词（字符 + 二元组，离线可用，无需 jieba）
# ============================================================================
def tokenize(text: str) -> List[str]:
    """对中文做「单字 + 二元组」分词，兼顾精确匹配与短语召回。"""
    text = re.sub(r"\s+", " ", text.lower())
    tokens: List[str] = []
    # 英文/数字单词
    for w in re.findall(r"[a-z0-9]+", text):
        tokens.append(w)
    # 中文单字与二元组
    zh = re.sub(r"[a-z0-9\s]", "", text)
    chars = list(zh)
    for i, c in enumerate(chars):
        tokens.append(c)
        if i + 1 < len(chars):
            tokens.append(chars[i] + chars[i + 1])
    return tokens


# ============================================================================
# 五、向量嵌入
# ============================================================================
class Embedder:
    """嵌入器接口：把文本列表编码为归一化向量矩阵 (n, dim)。"""

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TfidfEmbedder(Embedder):
    """离线嵌入：字符级 TF-IDF（无需下载模型，适合演示与快速验证）。"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=20000)
        self._fitted = False
        self._X: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "tfidf-char"

    def fit(self, corpus: List[str]) -> "TfidfEmbedder":
        self._X = self.vec.fit_transform(corpus).toarray().astype(np.float32)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        assert self._fitted, "请先调用 fit(corpus)"
        return self.vec.transform(texts).toarray().astype(np.float32)

    def corpus_matrix(self) -> np.ndarray:
        return self._X


class SentenceEmbedder(Embedder):
    """真实语义嵌入：sentence-transformers（如 bge / m3e）。"""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return "sentence-transformers"

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(model_name: Optional[str], corpus: Optional[List[str]] = None) -> Embedder:
    """工厂：优先使用 sentence-transformers，失败则回退到 TF-IDF。"""
    if model_name:
        try:
            return SentenceEmbedder(model_name)
        except Exception:
            pass
    return TfidfEmbedder().fit(corpus or [])


# ============================================================================
# 六、向量存储与检索
# ============================================================================
class VectorStore:
    """基于 numpy 的向量库，支持余弦相似度检索。"""

    def __init__(self, embedder: Embedder, corpus: Optional[List[str]] = None):
        self.embedder = embedder
        self.vectors: Optional[np.ndarray] = None
        self.corpus: List[str] = list(corpus or [])
        if corpus:
            self.add(corpus)

    def add(self, texts: List[str]):
        self.corpus.extend(texts)
        if self.vectors is None:
            self.vectors = self.embedder.encode(texts)
        else:
            self.vectors = np.vstack([self.vectors, self.embedder.encode(texts)])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.corpus) == 0:
            return []
        qv = self.embedder.encode([query])
        sims = np.dot(self.vectors, qv.T).ravel()  # 已归一化时即余弦
        top_k = min(top_k, len(self.corpus))
        idx = np.argsort(-sims)[:top_k]
        return [
            {"index": int(i), "text": self.corpus[int(i)], "score": float(sims[int(i)])}
            for i in idx
        ]


# ============================================================================
# 七、BM25 全文检索
# ============================================================================
class BM25Index:
    """BM25 倒排索引，支持关键词/短语/模糊匹配。"""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.doc_texts = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = float(np.mean([len(d) for d in self.docs])) if self.docs else 0.0
        self.df: Dict[str, int] = {}
        self._build()

    def _build(self):
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        for t in q_tokens:
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.count(t)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1.0))
                    scores[i] += idf * tf * (self.k1 + 1) / denom
        top_k = min(top_k, self.N)
        idx = np.argsort(-scores)[:top_k]
        return [
            {"index": int(i), "text": self.doc_texts[int(i)], "score": float(scores[int(i)])}
            for i in idx
        ]


# ============================================================================
# 八、LLM 生成
# ============================================================================
class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def answer(self, question: str, context: str) -> str:
        """基于检索上下文回答问题（各实现自行组织 prompt）。"""
        prompt = QA_PROMPT.format(context=context, question=question)
        return self.generate(prompt)


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容 LLM（DeepSeek / Qwen / 智谱 / OpenAI / Ollama / vLLM 等）。"""

    def __init__(self, api_base: str, api_key: str, model: str):
        from openai import OpenAI
        kwargs = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class ExtractiveLLM(BaseLLM):
    """离线抽取式回答：从上下文中挑选与问题最相关的句子作为答案（无 API 时的兜底）。"""

    def answer(self, question: str, context: str) -> str:
        sentences = [s.strip() for s in re.split(r"[。\n；;]", context) if s.strip()]
        if not sentences:
            return "（未检索到相关上下文）"
        q_tokens = set(tokenize(question))
        scored = []
        for s in sentences:
            s_tokens = set(tokenize(s))
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            scored.append((overlap, -abs(len(s) - 120), s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = [x[2] for x in scored[:2] if x[0] > 0]
        return "。".join(best) if best else "（未能从上下文中提取到答案，建议配置 LLM_API_KEY 使用真实大模型）"

    def generate(self, prompt: str) -> str:
        # 通用 generate：若 prompt 内嵌了「上下文：…问题：…」则抽取式回答，否则提示无法回答
        ctx_m = re.search(r"上下文[:：]\s*(.*?)\s*问题[:：]", prompt, re.S)
        if ctx_m:
            q_m = re.search(r"问题[:：]\s*([^\n]+)", prompt)
            question = q_m.group(1).strip() if q_m else ""
            return self.answer(question, ctx_m.group(1))
        return "（未提供检索上下文，无法基于文档回答；请配置 LLM_API_KEY 以使用真实大模型）"


def make_llm(config: Config) -> BaseLLM:
    if config.llm_api_base and config.llm_api_key:
        try:
            return OpenAICompatLLM(config.llm_api_base, config.llm_api_key, config.llm_model)
        except Exception:
            pass
    return ExtractiveLLM()


# ============================================================================
# 九、重排
# ============================================================================
def rerank_by_llm(llm: BaseLLM, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于 LLM 的重排器：让模型对每个候选片段打分排序。"""
    reranked = []
    for h in hits:
        prompt = (
            f"请判断以下文本片段与问题的相关性，仅输出 0 到 1 之间的一个小数。\n"
            f"问题：{query}\n文本片段：{h['text'][:500]}\n相关性分数："
        )
        try:
            score = float(re.search(r"(\d+\.?\d*)", llm.generate(prompt)).group(1))
        except Exception:
            score = h.get("score", 0.0)
        h = dict(h)
        h["score"] = score
        reranked.append(h)
    reranked.sort(key=lambda x: -x["score"])
    return reranked


def rerank_by_tfidf(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于 TF-IDF 的重排器：用词面重合度重排。"""
    q_tokens = set(tokenize(query))
    for h in hits:
        s_tokens = set(tokenize(h["text"]))
        if not s_tokens:
            continue
        h["score"] = len(q_tokens & s_tokens) / len(q_tokens)
    return sorted(hits, key=lambda x: -x["score"])


# ============================================================================
# 十、混合检索
# ============================================================================
class HybridRetriever:
    """向量检索 + 全文检索 + 加权融合（RRF）。"""

    def __init__(self, vector_store: VectorStore, bm25: BM25Index):
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """alpha 为向量检索权重，(1-alpha) 为全文检索权重。"""
        vec_hits = {h["index"]: h for h in self.vector_store.search(query, top_k=top_k * 4)}
        kw_hits = {h["index"]: h for h in self.bm25.search(query, top_k=top_k * 4)}
        # RRF 融合
        rrf: Dict[int, float] = {}
        for rank, i in enumerate(list(vec_hits.keys())):
            rrf[i] = rrf.get(i, 0.0) + alpha * 1.0 / (rank + 60)
        for rank, i in enumerate(list(kw_hits.keys())):
            rrf[i] = rrf.get(i, 0.0) + (1 - alpha) * 1.0 / (rank + 60)
        order = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        merged = []
        for i, s in order:
            h = vec_hits.get(i) or kw_hits.get(i)
            h = dict(h)
            h["rrf"] = s
            merged.append(h)
        return merged


# ============================================================================
# 十一、RAG 问答编排
# ============================================================================
QA_PROMPT = """你是一名金融文档问答助手。请仅依据下面提供的上下文回答用户问题：
- 若上下文包含答案，请给出准确、简洁的回答，并尽量引用关键数字/事实。
- 若上下文不足以回答，请明确说明“根据文档内容无法回答”。

上下文：
{context}

问题：{question}
回答："""


def build_context(hits: List[Dict[str, Any]], top_n: int = 4) -> str:
    return "\n\n".join(h["text"] for h in hits[:top_n])


def answer(question: str, hits: List[Dict[str, Any]], llm: BaseLLM, top_n: int = 4) -> str:
    context = build_context(hits, top_n)
    return llm.answer(question, context)


# ============================================================================
# 十二、简单 RAGAS 风格评估（离线可用）
# ============================================================================
def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def context_precision(question: str, retrieved_text: str, reference: str) -> float:
    """上下文精度：检索到的内容与参考答案的相关程度。"""
    rel = _overlap_ratio(question, retrieved_text)
    ref = _overlap_ratio(reference, retrieved_text)
    return round(max(rel, ref), 4)


def context_recall(question: str, retrieved_text: str, reference: str) -> float:
    """上下文召回：参考答案中的关键信息被检索到的比例。"""
    if not reference:
        return 0.0
    return round(_overlap_ratio(reference, retrieved_text), 4)


def faithfulness(answer: str, retrieved_text: str) -> float:
    """忠实度：回答与检索上下文的覆盖程度。"""
    if not answer:
        return 0.0
    return round(_overlap_ratio(answer, retrieved_text), 4)


def evaluate_rag(question: str, reference: str, retrieved_text: str, answer: str) -> Dict[str, float]:
    return {
        "context_precision": context_precision(question, retrieved_text, reference),
        "context_recall": context_recall(question, retrieved_text, reference),
        "faithfulness": faithfulness(answer, retrieved_text),
    }


def try_ragas_eval(samples: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """若安装 ragas，使用真实 RAGAS 评估（需要 LLM/嵌入配置）。"""
    try:
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall, faithfulness
        from datasets import Dataset
        ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["reference"],
            }
            for s in samples
        ])
        result = evaluate(ds, metrics=[context_precision, context_recall, faithfulness])
        return {k: round(float(v), 4) for k, v in result.items()}
    except Exception:
        return None
