from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import jieba
import numpy as np
import onnxruntime as ort
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from .config import Settings
from .parsing import ParsedChunk
from .vector_store import SearchHit, VectorStore


def tokenize(text: str) -> list[str]:
    return [token for token in jieba.cut(text, cut_all=False) if token.strip()]


class OnnxEmbedding:
    def __init__(self, model_path: str):
        root = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(root / "onnx" / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        feeds = {item.name: encoded[item.name].astype(np.int64) for item in self.session.get_inputs() if item.name in encoded}
        output = self.session.run(None, feeds)[0]
        mask = encoded["attention_mask"].astype(np.float32)[..., None]
        pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
        return pooled.astype(np.float32).tolist()


class OnnxReranker:
    def __init__(self, model_path: str):
        root = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(root / "onnx" / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        encoded = self.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], padding=True, truncation=True, max_length=512, return_tensors="np")
        feeds = {item.name: encoded[item.name].astype(np.int64) for item in self.session.get_inputs() if item.name in encoded}
        output = self.session.run(None, feeds)[0]
        values = np.asarray(output)
        if values.ndim == 2 and values.shape[1] > 1:
            values = values[:, -1]
        return values.reshape(-1).astype(float).tolist()


class LightweightEmbedding:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimension
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors


class LightweightReranker:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(set(tokenize(question)) & set(tokenize(content)))) for question, content in pairs]


class RetrievalService:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.use_local_models:
            self.embedding = OnnxEmbedding(settings.embedding_model_path)
            self.reranker: OnnxReranker | LightweightReranker | None = None
        else:
            self.embedding = LightweightEmbedding(settings.embedding_dimension)
            self.reranker = LightweightReranker()
        self.store = VectorStore(settings)
        self.bm25_path = settings.parsed_dir / "bm25.pkl"
        self.bm25: BM25Okapi | None = None
        self.bm25_rows: list[SearchHit] = []
        self._load_bm25()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedding.encode(texts)

    def index(self, document_id: str, document_name: str, version: str, chunks: list[ParsedChunk]) -> int:
        vectors = self.embed([chunk.content for chunk in chunks])
        rows = []
        bm25_rows = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{document_id}-{index}"
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "document_name": document_name, "document_version": version, "page_number": chunk.page_number, "section_title": chunk.section_title[:255], "content_type": chunk.content_type, "content": chunk.content, "embedding": vector})
            bm25_rows.append(SearchHit(chunk_id, chunk.content, chunk.page_number, chunk.section_title, document_id, document_name, 0.0, chunk.content_type))
        self.store.upsert(rows)
        self.bm25_rows = [row for row in self.bm25_rows if row.document_id != document_id] + bm25_rows
        self._save_bm25()
        return len(rows)

    def search(self, question: str, document_id: str | None = None) -> list[SearchHit]:
        vector_hits = self.store.search(self.embed([question])[0], document_id, self.settings.retrieval_top_k)
        keyword_hits = []
        if self.bm25:
            scores = self.bm25.get_scores(tokenize(question))
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            keyword_hits = [self.bm25_rows[index] for index, _ in ranked if not document_id or self.bm25_rows[index].document_id == document_id][: self.settings.retrieval_top_k]
        ranked_hits: dict[str, tuple[SearchHit, float]] = {}
        rrf_k = 60
        for rank, hit in enumerate(vector_hits, start=1):
            ranked_hits[hit.chunk_id] = (hit, 1 / (rrf_k + rank))
        for rank, hit in enumerate(keyword_hits, start=1):
            if hit.chunk_id in ranked_hits:
                current, score = ranked_hits[hit.chunk_id]
                ranked_hits[hit.chunk_id] = (current, score + 1 / (rrf_k + rank))
            else:
                ranked_hits[hit.chunk_id] = (hit, 1 / (rrf_k + rank))
        candidates = [item[0] for item in sorted(ranked_hits.values(), key=lambda item: item[1], reverse=True)[: self.settings.retrieval_top_k * 2]]
        if not candidates:
            return []
        if self.reranker is None:
            self.reranker = OnnxReranker(self.settings.reranker_model_path)
        rerank_scores = self.reranker.predict([(question, hit.content) for hit in candidates])
        for hit, score in zip(candidates, rerank_scores):
            hit.score = float(score)
        return sorted(candidates, key=lambda hit: hit.score, reverse=True)[: self.settings.rerank_top_k]

    def _load_bm25(self) -> None:
        if self.bm25_path.exists():
            with self.bm25_path.open("rb") as source:
                self.bm25_rows, tokenized = pickle.load(source)
            if tokenized:
                self.bm25 = BM25Okapi(tokenized)

    def _save_bm25(self) -> None:
        tokenized = [tokenize(hit.content) for hit in self.bm25_rows]
        with self.bm25_path.open("wb") as target:
            pickle.dump((self.bm25_rows, tokenized), target)
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
