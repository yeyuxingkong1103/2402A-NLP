"""Compatibility facade for the long-term Milvus memory API."""

from app.memory.milvus_client import MilvusMemoryClient
from app.memory.milvus_store import EmbeddingQueryAdapter, MilvusMemory

__all__ = ["EmbeddingQueryAdapter", "MilvusMemory", "MilvusMemoryClient"]
