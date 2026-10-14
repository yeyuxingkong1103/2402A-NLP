"""Work order 12: LightRAG ingestion, querying and operational optimizations."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LightRAGConfig:
    working_dir: str = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    chunk_token_size: int = 900
    max_parallel_insert: int = 8


class LightRAGPipeline:
    def __init__(self, config: LightRAGConfig | None = None, rag: Any | None = None):
        self.config = config or LightRAGConfig()
        self.rag = rag
        self.insert_semaphore = asyncio.Semaphore(self.config.max_parallel_insert)

    async def initialize(self):
        if self.rag is None:
            from lightrag import LightRAG

            self.rag = LightRAG(working_dir=self.config.working_dir)
        if hasattr(self.rag, "initialize_storages"):
            await self.rag.initialize_storages()

    async def insert_documents(self, documents: list[str]) -> None:
        await self.initialize()

        async def insert_one(document: str):
            async with self.insert_semaphore:
                result = self.rag.ainsert(document) if hasattr(self.rag, "ainsert") else self.rag.insert(document)
                if asyncio.iscoroutine(result):
                    await result

        await asyncio.gather(*(insert_one(document) for document in documents))

    async def query(self, question: str, mode: str = "hybrid") -> str:
        await self.initialize()
        result = self.rag.aquery(question, param={"mode": mode}) if hasattr(self.rag, "aquery") else self.rag.query(question)
        return await result if asyncio.iscoroutine(result) else result

    async def close(self) -> None:
        if self.rag and hasattr(self.rag, "finalize_storages"):
            result = self.rag.finalize_storages()
            if asyncio.iscoroutine(result):
                await result
