"""BGE-m3 封装服务。

契约（与 backend/app/core/providers/embedding/bge_m3_http.py 对齐）：
    POST /embed  body {"texts": [str, ...]}
    -> 200 {"dense": [[float, ...], ...], "sparse": [{"token_id": float, ...}, ...]}

模型路径用环境变量 EMBED_MODEL 指定（默认 /data/bge-m3），本地加载不联网。
用 FlagEmbedding 的 BGEM3FlagModel 原生输出 dense(1024) + sparse(词权重)。
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="bge-m3")

MODEL = os.environ.get("EMBED_MODEL", "/data/bge-m3")

# 启动即加载，避免首个请求触发懒加载超过后端 60s 超时
from FlagEmbedding import BGEM3FlagModel  # noqa: E402

_model = BGEM3FlagModel(MODEL, use_fp16=True)


class EmbedIn(BaseModel):
    texts: list[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/embed")
async def embed(body: EmbedIn):
    if not body.texts:
        return {"dense": [], "sparse": []}
    out = _model.encode(
        body.texts,
        batch_size=8,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    if dense.ndim == 1:
        dense = dense[None, :]
    # lexical_weights 是 [{token_id:int -> weight:float}, ...]
    sparse = out["lexical_weights"]
    return {
        "dense": dense.tolist(),
        "sparse": [{str(k): float(v) for k, v in w.items()} for w in sparse],
    }
