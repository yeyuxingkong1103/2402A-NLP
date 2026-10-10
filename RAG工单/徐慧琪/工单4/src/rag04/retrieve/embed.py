# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""bge-m3 向量化（经 Ollama）。批量 16 条/请求，失败自动降批次。"""
from __future__ import annotations

import json
import logging

import requests

from rag04.config import Settings

logger = logging.getLogger("rag04.embed")

DIM = 1024                      # bge-m3 向量维度
BATCH = 16                      # 硬约束：批量 16
_FALLBACK_BATCHES = (16, 8, 4, 2, 1)


class EmbedUnavailableError(Exception):
    """Ollama / bge-m3 不可用。"""


def _post_embed(s: Settings, texts: list[str]) -> list[list[float]]:
    url = f"{s.ollama_url.rstrip('/')}/api/embed"
    payload = json.dumps({"model": s.embed_model, "input": texts})
    r = requests.post(url, data=payload,
                      headers={"Content-Type": "application/json"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    vecs = data.get("embeddings")
    if not vecs:
        raise EmbedUnavailableError(f"Ollama 返回空向量：{str(data)[:200]}")
    return vecs


def embed_texts(texts: list[str], s: Settings) -> list[list[float]]:
    """批量向量化。整批失败时按 8/4/2/1 逐级降批次重试。"""
    if not texts:
        return []

    out: list[list[float]] = []
    i = 0
    batch = min(BATCH, len(texts))
    while i < len(texts):
        chunk = texts[i:i + batch]
        try:
            out.extend(_post_embed(s, chunk))
            i += len(chunk)
        except Exception as e:
            if batch == 1:
                raise EmbedUnavailableError(
                    f"向量化失败：{type(e).__name__}: {e}\n"
                    f"排查：1) 确认 Ollama 已启动（ollama serve）"
                    f" 2) 确认模型已拉取（ollama pull {s.embed_model}）"
                ) from e
            batch = max(1, batch // 2)
            logger.warning("批次 %d 失败，降至 %d 重试：%s", len(chunk), batch, e)

    if len(out) != len(texts):
        raise EmbedUnavailableError(f"向量条数不匹配：期望 {len(texts)}，实得 {len(out)}")
    return out


def embed_one(text: str, s: Settings) -> list[float]:
    """单条向量化。"""
    return embed_texts([text], s)[0]
