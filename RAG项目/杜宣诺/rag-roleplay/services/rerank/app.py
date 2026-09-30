"""BGE-rerank 封装服务。

契约（与 backend/app/core/providers/rerank/bge_rerank_http.py 对齐）：
    POST /rerank  body {"query": str, "passages": [str, ...], "top_m": int}
    -> 200 {"results": [{"index": int, "score": float}, ...]}   # 按 score 降序，最多 top_m 条

模型默认 BAAI/bge-reranker-base（CPU 友好），可用环境变量 RERANK_MODEL 覆盖。
本地加载：模型目录通过 volume 挂载到容器内（默认 /data/bge-reranker-base），
只有 model.safetensors（无 pytorch_model.bin），靠 transformers 自动识别 safetensors 加载。
"""

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bge-rerank")

app = FastAPI(title="bge-rerank")

MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")

# 启动即加载，避免首个请求触发懒加载超过后端 60s 超时
from FlagEmbedding import FlagReranker  # noqa: E402

_safetensors_path = os.path.join(MODEL, "model.safetensors")
if os.path.isdir(MODEL) and not os.path.exists(_safetensors_path):
    logger.warning("未找到 %s（本地仅有 safetensors，无 pytorch_model.bin），加载可能失败", _safetensors_path)

_reranker = FlagReranker(MODEL, use_fp16=True)
logger.info("FlagReranker loaded from %s", MODEL)


class RerankIn(BaseModel):
    query: str
    passages: list[str]
    top_m: int = 5


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rerank")
async def rerank(body: RerankIn):
    if not body.passages:
        return {"results": []}
    pairs = [[body.query, p] for p in body.passages]
    scores = _reranker.compute_score(pairs)
    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(range(len(body.passages)), key=lambda i: -scores[i])[: body.top_m]
    return {"results": [{"index": i, "score": float(scores[i])} for i in ranked]}
