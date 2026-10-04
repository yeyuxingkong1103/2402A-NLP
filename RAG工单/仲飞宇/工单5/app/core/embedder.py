# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
嵌入器：bge-m3（1024 维），走 Ollama 原生 /api/embed。

【为什么分批】548 页切出 ~1100 个 chunk，一次性 POST 会让 Ollama 端
把整批塞进显存做 forward，中途 OOM 风险高（bge-m3 只有 0.66GB，
但激活值随 batch 线性增长）。批大小 16 在实测中稳定。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from app.config import settings
from app.core.ollama_client import OllamaClient, get_client


@dataclass
class EmbedStats:
    n_texts: int = 0
    n_batches: int = 0
    seconds: float = 0.0
    dim: int = 0

    @property
    def throughput(self) -> float:
        return self.n_texts / self.seconds if self.seconds else 0.0


class Embedder:
    """
    bge-m3 向量化。

    【文本前缀】bge-m3 不需要 query/passage 前缀（不同于 bge-large-zh），
    文档和查询用同一套编码即可，这一点与 config.embed_model 的选型一致。
    """

    def __init__(self, client: OllamaClient | None = None,
                 batch_size: int = 16) -> None:
        self.client = client or get_client()
        self.batch_size = batch_size

    async def embed_one(self, text: str, *, on_cpu: bool | None = None) -> list[float]:
        """
        单条向量化（查询路径）。

        默认走 CPU：把显存让给 qwen3，避免"嵌入把聊天模型挤出去 →
        下次问答吃 6 秒冷启动"。实测这么做之后 TTFT 从 6117ms 降到 35ms。
        """
        if on_cpu is None:
            on_cpu = settings.embed_query_on_cpu
        vecs = await self.client.embed(text, num_gpu=0 if on_cpu else None)
        return vecs[0]

    async def embed_many(
        self,
        texts: Sequence[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[list[float]], EmbedStats]:
        """
        分批向量化（入库路径）。

        入库是离线的一次性批量任务，没有并发聊天在抢显存，
        走 GPU 快得多（实测 1116 条 31 秒），所以这里不强制 CPU。
        """
        stats = EmbedStats(n_texts=len(texts))
        out: list[list[float]] = []
        t0 = time.perf_counter()

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            vecs = await self.client.embed(batch)
            out.extend(vecs)
            stats.n_batches += 1
            if progress:
                progress(min(i + self.batch_size, len(texts)), len(texts))
            # 让出事件循环，避免长任务把 FastAPI 饿死（工单01「高并发稳定」）
            await asyncio.sleep(0)

        stats.seconds = time.perf_counter() - t0
        stats.dim = len(out[0]) if out else 0
        return out, stats


async def embed_texts(texts: Sequence[str],
                      progress: Callable[[int, int], None] | None = None
                      ) -> tuple[list[list[float]], EmbedStats]:
    """便捷函数。"""
    return await Embedder().embed_many(texts, progress=progress)


def embed_texts_sync(texts: Sequence[str],
                     progress: Callable[[int, int], None] | None = None
                     ) -> tuple[list[list[float]], EmbedStats]:
    """同步包装，供 scripts/ingest.py 这类命令行脚本使用。"""
    return asyncio.run(embed_texts(texts, progress=progress))


def check_dim(vec: list[float]) -> None:
    if len(vec) != settings.embed_dim:
        raise ValueError(
            f"向量维度不符：实测 {len(vec)}，配置 {settings.embed_dim}。"
            f"Milvus 的 FLOAT_VECTOR 维度建表即定死，必须先对齐 config.embed_dim。"
        )
