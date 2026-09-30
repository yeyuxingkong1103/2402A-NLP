"""本地模型层：BGE-M3（稠密 + 稀疏）与 Qwen3（生成）。"""

from .embedder import BGEM3Embedder, get_embedder
from .llm import GenerationResult, LocalLLM, get_llm

__all__ = ["BGEM3Embedder", "get_embedder", "LocalLLM", "GenerationResult", "get_llm"]
