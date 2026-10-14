from __future__ import annotations

import json
import os
import pickle
import re
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPy.*|swigvarlink) has no __module__ attribute",
    category=DeprecationWarning,
)
import faiss
import numpy as np
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from corpus import Chunk


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_MODEL_PATH = Path(
    r"C:\Users\lenovo\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5"
    r"\snapshots\7999e1d3359715c523056ef9478215996d62a620"
)

COMPANY_CODES = {
    "平安银行": "000001",
    "中国平安": "601318",
    "招商银行": "600036",
    "邮储银行": "601658",
    "中国邮政储蓄银行": "601658",
    "中信证券": "600030",
    "中国人寿": "601628",
    "中国太保": "601601",
    "中国太平洋保险": "601601",
    "招商证券": "600999",
    "国泰君安": "601211",
}
METRIC_TERMS = ("营业收入", "净利润", "总资产", "不良贷款率", "保费收入", "偿付能力")


def extract_stock_codes(question: str) -> list[str]:
    return list(dict.fromkeys(code for name, code in COMPANY_CODES.items() if name in question))


def select_candidate_indices(question: str, chunks: list, scores, top_k: int) -> list[int]:
    codes = extract_stock_codes(question)
    terms = [term for term in METRIC_TERMS if term in question]
    adjusted: list[float] = []
    for chunk, raw_score in zip(chunks, scores):
        text = chunk.text
        metric_bonus = sum(0.03 for term in terms if term in text)
        summary_bonus = 0.06 if int(chunk.page) <= 60 else 0.0
        adjusted.append(float(raw_score) + metric_bonus + summary_bonus)

    def ranked_for(code: str | None = None) -> list[int]:
        candidates = [
            index for index, chunk in enumerate(chunks)
            if code is None or chunk.stock_code == code
        ]
        return sorted(candidates, key=lambda index: adjusted[index], reverse=True)

    if len(codes) == 1:
        return ranked_for(codes[0])[:top_k]
    if len(codes) > 1:
        groups = {code: ranked_for(code) for code in codes}
        selected: list[int] = []
        depth = 0
        while len(selected) < top_k and any(depth < len(group) for group in groups.values()):
            for code in codes:
                if depth < len(groups[code]) and len(selected) < top_k:
                    selected.append(groups[code][depth])
            depth += 1
        return selected
    return ranked_for()[:top_k]


def _save_faiss(index, path: Path) -> None:
    path.write_bytes(faiss.serialize_index(index).tobytes())


def _load_faiss(path: Path):
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8").copy())


def build_index(chunks: list[Chunk], data_dir: Path = DATA_DIR) -> None:
    from sentence_transformers import SentenceTransformer

    data_dir.mkdir(parents=True, exist_ok=True)
    texts = [f"{chunk.document} {chunk.title}\n{chunk.text}" for chunk in chunks]
    (data_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False), encoding="utf-8"
    )
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=80000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    with (data_dir / "lexical.pkl").open("wb") as stream:
        pickle.dump((vectorizer, matrix), stream)

    model_path = Path(os.getenv("BGE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(f"本地BGE模型不存在：{model_path}")
    encoder = SentenceTransformer(str(model_path))
    vectors = encoder.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    _save_faiss(index, data_dir / "index.faiss")


class LocalIndex:
    def __init__(self, chunks, vectorizer, matrix, index, encoder):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.index = index
        self.encoder = encoder

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR):
        from sentence_transformers import SentenceTransformer

        required = [data_dir / "chunks.json", data_dir / "lexical.pkl", data_dir / "index.faiss"]
        if not all(path.exists() for path in required):
            raise FileNotFoundError("尚未建立索引，请先运行 prepare.py")
        chunks = [Chunk(**item) for item in json.loads(required[0].read_text(encoding="utf-8"))]
        with required[1].open("rb") as stream:
            vectorizer, matrix = pickle.load(stream)
        model_path = Path(os.getenv("BGE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        encoder = SentenceTransformer(str(model_path))
        return cls(chunks, vectorizer, matrix, _load_faiss(required[2]), encoder)

    def search(self, question: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
        question = re.sub(r"\s+", " ", question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if mode not in {"vector", "fulltext", "hybrid"}:
            raise ValueError("检索模式必须是 vector、fulltext 或 hybrid")
        top_k = max(1, min(int(top_k), 10))
        lexical = np.asarray((self.matrix @ self.vectorizer.transform([question]).T).toarray()).ravel().astype("float32")
        vector = np.zeros(len(self.chunks), dtype="float32")
        if mode != "fulltext":
            query = self.encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            scores, ids = self.index.search(query, min(max(top_k * 8, 40), len(self.chunks)))
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:
                    vector[int(idx)] = float(score)
        if mode == "vector":
            combined = vector
        elif mode == "fulltext":
            combined = lexical
        else:
            combined = vector * 0.65 + lexical * 0.35
        order = select_candidate_indices(question, self.chunks, combined, top_k)
        return [
            {
                **asdict(self.chunks[int(idx)]),
                "score": round(float(combined[int(idx)]), 6),
                "vector_score": round(float(vector[int(idx)]), 6),
                "lexical_score": round(float(lexical[int(idx)]), 6),
                "rank": rank,
            }
            for rank, idx in enumerate(order, 1)
        ]


class RAGSystem:
    def __init__(self, index: LocalIndex | None = None):
        self.index = index or LocalIndex.load()
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com") if key else None

    async def answer(self, question: str, top_k: int = 5, mode: str = "hybrid") -> dict:
        started = time.perf_counter()
        hits = self.index.search(question, top_k=top_k, mode=mode)
        retrieval_ms = (time.perf_counter() - started) * 1000
        evidence = "\n\n".join(f"[{hit['document']} 第{hit['page']}页]\n{hit['text']}" for hit in hits)
        generation_started = time.perf_counter()
        answer_mode = "deepseek" if self.client else "evidence"
        if self.client:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                temperature=0.0,
                max_tokens=450,
                messages=[
                    {"role": "system", "content": "仅根据给定年报证据回答，答案简洁；证据不足时明确说明。"},
                    {"role": "user", "content": f"问题：{question}\n\n年报证据：\n{evidence}"},
                ],
                timeout=20,
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = "未配置DeepSeek API，以下为最相关年报证据摘要：\n" + "\n".join(
                f"- {hit['document']}第{hit['page']}页：{hit['text'][:260]}" for hit in hits
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        return {
            "answer": answer,
            "mode": answer_mode,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
            "hits": hits,
        }
