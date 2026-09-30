"""本地 bge-reranker 重排服务。

对外暴露 SiliconFlow 兼容的 `POST /v1/rerank` 接口，便于本地/在线（硅基流动）
无差别切换：改 .env 里的 RERANK_BASE_URL / RERANK_API_KEY 即可。

启动：rerank_service/.venv/bin/uvicorn rerank_service.server:app --host 127.0.0.1 --port 8001

实际用这个包装脚本起（自动选本地模型目录、重定向日志、写端口专属 pid 文件）：
    bash scripts/run_rerank.sh          # 起服务，默认 127.0.0.1:8001
    bash scripts/verify_rerank.sh       # 起完必须验一次：能返回 200 不等于真在算分
    bash scripts/shutdown.sh            # 收摊

前置条件与副作用：
    - 独立 venv（rerank_service/.venv，装 CPU 版 torch），跟主项目 .venv 不是一套。
    - 首次启动若本地没有模型，会从 HuggingFace 下载约 2.2GB（run_rerank.sh 会把
      HF_ENDPOINT 指到 hf-mirror，并把缓存的本地目录通过 RERANK_MODEL 传进来）。
    - 模块导入时就把权重读进内存，**每个实例常驻约 2.02GB**，冷启动实测约 1m54s。
      .env 里的 RERANK_BASE_URL 只指向一个地址，所以多起的实例纯粹占内存——
      WSL 内存本就只有几 GB，三个就够触发 OOM（见 scripts/run_rerank.sh 的警告）。
    - 无状态、无写入：只读模型、只算分，随便重启，不碰 Milvus / 关系库 / Redis。
"""
from __future__ import annotations

import math
import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# 走环境变量而不是命令行参数/配置类：本服务是个极小的独立进程，没有配置层可用。
# RERANK_MODEL 允许传**本地目录的绝对路径**（run_rerank.sh 就是这么传的），
# 从而绕开联网找模型这一步。
MODEL_NAME = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 1024 而不是模型原生的 8192：交叉编码器是逐对推理，序列越长耗时越接近线性增长，
# 而送进来的候选是几百字的 chunk（入库 chunk_size 默认 500），1024 足够覆盖，
# 调大只会让每轮检索更慢。
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1024"))

# 启动时加载模型（首次会从 HuggingFace 下载，约 2.2GB）
# 刻意放在模块级而不是懒加载：这样"权重没就绪"表现为**进程起不来**（启动即报错、
# 端口不通），而不是服务看着正常、第一个请求才静默变慢或降级。代价是冷启动
# 1~2 分钟里有很长一段窗口服务不可用，这期间应用侧会降级回 score_fusion。
_model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)

app = FastAPI(title="bge-reranker service")


class RerankRequest(BaseModel):
    # 字段名与硅基流动一致（query/documents/top_n），换在线服务时请求体一个字不用改。
    # `model` 声明了但不用：本服务只加载了一个模型，收下就忽略。声明它只是为了
    # 让 OpenAPI 里能看到这个字段——pydantic 默认忽略多余字段，不声明也能跑，
    # 但那样调用方会以为参数没生效是"格式不对"。
    query: str
    documents: list[str]
    model: str | None = None
    top_n: int | None = None


def _sigmoid(x: float) -> float:
    """把 CrossEncoder 的原始 logit 压到 0~1。

    必须压：logit 可正可负（实测同一批候选的 logit 有负值），而硅基流动的
    relevance_score 是 0~1 的语义分。调用方（app/core/reranker.py）正是靠"分数落在
    BGE 区间"来分辨"真精排"和"降级回 RRF 融合分"（后者只有 ~0.016），
    这里不压的话两种量纲就混了。
    """
    return 1.0 / (1.0 + math.exp(-x))


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    """按 query 给每条 document 打语义分，返回按分数降序的 {results:[{index, relevance_score}]}。

    `index` 是**入参 documents 的下标**（不是排名），调用方靠它把分数映射回候选；
    丢了这个字段或改成排名值，app 侧会把分数贴到错误的候选上（它会丢弃越界下标，
    于是表现为"静默没精排"）。`top_n` 为 None 时返回全部——app 侧每次都显式传
    len(documents)，就是不想被服务端默认截断。
    """
    pairs = [(req.query, doc) for doc in req.documents]
    scores = _model.predict(pairs, convert_to_numpy=True)
    # round(..., 6)：分数会原样进前端和日志，6 位小数足够区分名次又不至于刷屏
    results = [
        {"index": i, "relevance_score": round(_sigmoid(float(s)), 6)}
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    if req.top_n is not None:
        results = results[: req.top_n]
    return {"results": results}


@app.get("/health")
def health():
    # 能应答基本等价于权重已就绪（模型是模块级加载的，加载失败进程根本起不来）。
    # 但**"200"不等于"真的在算分"**：分数全等、或压根没接管精排，/health 一样是 ok。
    # 判定精排是否生效要看真实 /v1/rerank 的分数区间，即 scripts/verify_rerank.sh。
    return {"status": "ok", "model": MODEL_NAME}
