# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""本机模型服务：嵌入（/v1/embeddings）+ 重排（/rerank），给 RAGFlow 用

## 为什么需要它

RAGFlow 默认用 **TEI**（text-embeddings-inference）做嵌入，那是另一个 Docker
镜像，而且**启动时还要从 HuggingFace 下载模型**（`TEI_MODEL=Qwen/Qwen3-Embedding-0.6B`）
—— 本机 Docker Hub 已经要靠加速器才能拉，HuggingFace 直连更不通，
这一步失败的概率很高。

RAGFlow 的嵌入配置本质上只是一个 **OpenAI 兼容的 base_url**
（`service_conf.yaml.template:57-59` 的 `user_default_llm.default_models.embedding_model`），
并不是绑定 TEI 的。所以这里自己起一个小服务，用**本机已有的模型**：

    · 零下载 —— 模型早就在 D:/Pycharm/yzq/models/ 下
    · 可控 —— 脚本随交付物一起给，评审那边也能复现

## 两个端点

| 端点 | 模型 | 对应 RAGFlow 里怎么填 |
|---|---|---|
| `POST /v1/embeddings` | bge-m3 | embedding 模型的 base_url |
| `POST /rerank` | bge-reranker-v2-m3 | rerank 模型的 base_url |

`/rerank` 的出入参按 RAGFlow `rag/llm/rerank_model.py:195 OpenAI_APIRerank` 来：
入参 `{model, query, documents, top_n}`，出参 `{results:[{index, relevance_score}]}`。

## 两个模型分别放哪，为什么

**重排放 GPU**（fp16, batch 16）：每次提问都要给 66 条候选打分
（`rag/nlp/search.py:388` 的 RERANK_LIMIT）。实测 8 核 CPU 上要 **80 秒**，
直接击穿工单 ≤3s 的验收；GPU 上 **1.0 秒**。

**嵌入放 CPU**：只有提问时编一条查询，0.25s。放 CPU 纯粹是为了给重排腾显存 ——
本机是 6GB 的笔记本 3060，两个模型都塞 GPU 会占满（实测只剩 236MB 空闲），
显存一紧重排那一步就会偶发卡到 **70 秒**。挪走之后重排稳定在 1.0s。

## 用法

    D:/Anaconda/envs/rag_gd/python.exe embedding_server.py --port 9997

然后在 RAGFlow 里把 embedding 的 base_url 指向
`http://host.docker.internal:9997/v1`（容器内访问宿主的写法），
rerank 的 base_url 指向 `http://host.docker.internal:9997/rerank`。
"""
import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = Path(r"D:\Pycharm\yzq\models")
MODEL_NAME = "bge-m3"
RERANK_NAME = "bge-reranker-v2-m3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--model", default=str(MODELS / "bge-m3"))
    ap.add_argument("--rerank-model", default=str(MODELS / "bge-reranker-v2-m3"))
    ap.add_argument("--no-rerank", action="store_true", help="只起嵌入，不加载重排")
    ap.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda"],
                    help="嵌入跑哪（默认 cpu，给重排腾显存）")
    ap.add_argument("--rerank-device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="重排跑哪（auto = 有显卡就用显卡）")
    args = ap.parse_args()

    import torch
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    from sentence_transformers import SentenceTransformer

    print(f"[加载] 嵌入 {args.model}（{args.embed_device}）...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(args.model, device=args.embed_device)
    model.max_seq_length = 8192
    dim = model.get_sentence_embedding_dimension()
    print(f"[就绪] 维度 {dim}，加载耗时 {time.time() - t0:.1f}s", flush=True)

    reranker = None
    if not args.no_rerank:
        dev = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.rerank_device == "auto" \
            else args.rerank_device
        print(f"[加载] 重排 {args.rerank_model}（{dev}）...", flush=True)
        t0 = time.time()
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(args.rerank_model, use_fp16=(dev != "cpu"), devices=dev)
        reranker.compute_score([["warmup", "warmup"]], normalize=True)  # 预热，免得首问慢
        print(f"[就绪] 重排加载耗时 {time.time() - t0:.1f}s", flush=True)

    class EmbedIn(BaseModel):
        input: str | list[str]
        model: str | None = None

    class RerankIn(BaseModel):
        model: str | None = None
        query: str
        documents: list[str]
        top_n: int | None = None

    app = FastAPI(title="local bge models (RAGFlow compatible)")

    @app.get("/health")
    def health():
        return {"status": "ok", "embed": MODEL_NAME, "dim": dim,
                "rerank": RERANK_NAME if reranker else None}

    @app.post("/v1/embeddings")
    def embeddings(inp: EmbedIn):
        texts = [inp.input] if isinstance(inp.input, str) else list(inp.input)
        vecs = model.encode(texts, batch_size=16, normalize_embeddings=True,
                            show_progress_bar=False)
        return {
            "object": "list",
            "model": MODEL_NAME,
            "data": [{"object": "embedding", "index": i,
                      "embedding": [float(x) for x in v]}
                     for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @app.post("/rerank")
    def rerank(inp: RerankIn):
        if reranker is None:
            return {"results": []}
        # RAGFlow 那边已经把文本截断到 500 字符了（rerank_model.py:208），
        # 这里不再重复截断，只把分数算出来
        pairs = [[inp.query, d] for d in inp.documents]
        scores = reranker.compute_score(pairs, normalize=True, batch_size=16)
        if not isinstance(scores, list):
            scores = [scores]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if inp.top_n:
            order = order[: inp.top_n]
        return {"model": RERANK_NAME,
                "results": [{"index": i, "relevance_score": float(scores[i])}
                            for i in order]}

    print(f"[启动] http://{args.host}:{args.port}  /v1/embeddings  /rerank", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
