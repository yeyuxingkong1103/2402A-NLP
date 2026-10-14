"""工单2：PDF RAG 优化配置。"""

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    chunk_size: int = 760
    chunk_overlap: int = 120
    dense_top_k: int = 20
    lexical_top_k: int = 20
    rerank_top_k: int = 8
    score_threshold: float = 0.15


MODELS = {
    "bge-large": {"provider": "ollama", "model": "bge-large:latest"},
    "nomic-embed-text": {"provider": "ollama", "model": "nomic-embed-text:latest"},
    "m3e-large": {"provider": "ollama", "model": "m3e-large:latest"},
}
