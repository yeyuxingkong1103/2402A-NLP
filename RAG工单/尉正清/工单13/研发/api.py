# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""最小 HTTP 接口 —— 给 JMeter 做负载测试用

**为什么必须加这个**：现有 RAG 系统只有 Gradio 界面，JMeter 压不了它
（Gradio 的 API 要先拿 session hash，请求体还是它私有的格式，压出来的数字
反映的是 Gradio 外壳而不是 RAG 本体）。工单要求「使用负载测试工具模拟生产流量」，
所以需要一个标准 REST 端点。

**不引入新依赖**：fastapi / uvicorn 环境里本来就有（gradio 依赖它们），
所以这个接口是零安装成本的。

**关键设计**：知识库和模型在**启动时加载一次**并常驻。这对性能结论是决定性的 ——
「每次请求都重新加载 BGE-M3」和「加载一次复用」差好几个数量级，是压测里
最容易把结论带偏的地方（见 优化/过程问题记录.md）。

用法：
    D:/Anaconda/envs/rag_gd/python.exe api.py            # 默认 0.0.0.0:8000
    JMeter 压 POST http://127.0.0.1:8000/query
"""
import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import perf                                                      # noqa: E402
from config import CACHE_ROOT, LLM_TIMEOUT, PERF_OPT                       # noqa: E402

KB_NAME = "ccf_competition"

# 全局单例：进程启动时建好，请求里只读
_STATE = {"store": None, "engine": None, "ready": False}


def build_state():
    """加载知识库与模型。只调一次。"""
    from rag_engine import RAGEngine
    from vector_store import BGEM3VectorStore

    kb = Path(CACHE_ROOT) / KB_NAME
    if not kb.exists():
        raise SystemExit(f"[错误] 知识库不存在：{kb}")
    t0 = time.time()
    store = BGEM3VectorStore()                 # 加载 BGE-M3（约 1.2GB）
    if not store.load(kb):
        raise SystemExit(f"[错误] 知识库加载失败：{kb}")
    t_model = time.time()
    engine = RAGEngine(store)
    t_index = time.time()
    _STATE.update(store=store, engine=engine, ready=True)
    print(f"[就绪] 知识库 {len(store.chunks):,} 块，加载耗时 {t_model - t0:.1f}s；"
          f"倒排索引另计 {t_index - t_model:.1f}s", flush=True)

    # 优化 D：启动预热。
    # 不打这一枪，**第一个真实用户**要替整个进程付冷启动的账 ——
    # 实测首请求端到端 9.68s，其中检索段 5.43s（CUDA 上下文 + cuDNN 算法选型 +
    # 首次前向的显存分配），而稳态只有 0.4s。预热把它挪到启动阶段，
    # 用户侧看到的就是稳态延迟。
    t_warm = time.time()
    if not PERF_OPT:
        print("[预热] 已按 PERF_OPT=off 跳过", flush=True)
        return
    try:
        engine.answer("预热：平安银行的营业收入是多少？")
        print(f"[预热] 完成，耗时 {time.time() - t_warm:.1f}s —— "
              f"冷启动成本已由启动阶段承担", flush=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"[预热] 失败（不影响服务）：{type(exc).__name__}: {exc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=None,
                    help="单次大模型调用超时；压测时可按需放宽")
    args = ap.parse_args()

    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    class QueryIn(BaseModel):
        question: str
        request_id: str | None = None

    app = FastAPI(title="RAG 性能压测接口")

    @app.on_event("startup")
    def _startup():
        build_state()

    @app.get("/health")
    def health():
        """JMeter 的 setUp 线程组用它确认服务已就绪（模型加载完）再开始压。"""
        return {"ready": _STATE["ready"],
                "chunks": len(_STATE["store"].chunks) if _STATE["store"] else 0}

    @app.post("/query")
    def query(inp: QueryIn):
        if not _STATE["ready"]:
            return {"error": "not ready"}
        engine = _STATE["engine"]
        if args.timeout:
            engine.timeout = args.timeout
        # 整个请求包一层 trace：阶段时间戳 + 请求 ID + 指标，落 perf_log.jsonl
        with perf.trace(question=inp.question, request_id=inp.request_id) as t:
            try:
                out = engine.answer(inp.question)
                status = "ok"
            except Exception as exc:                       # noqa: BLE001
                out, status = {"answer": "", "contexts": [],
                               "retrieve_seconds": 0.0,
                               "elapsed": time.time() - t.t0}, f"error:{type(exc).__name__}"
            rec = t.finish(status=status)
        perf.write(rec)
        # 响应体只回压测关心的量，不回答案全文（JMeter 的聚合报告才是重点）
        return {"request_id": rec["request_id"],
                "retrieve_seconds": round(out.get("retrieve_seconds", 0.0), 4),
                "elapsed": round(out.get("elapsed", 0.0), 4),
                "recall_count": out.get("recall_count", 0),
                "answer_chars": len(out.get("answer", "")),
                "stages": {s["stage"]: s["seconds"] for s in rec["spans"]
                           if s["depth"] == 0},
                "status": status}

    @app.get("/perf_summary")
    def perf_summary():
        """把已经记下的 Trace 汇成阶段耗时表，省得手动跑分析脚本。"""
        return perf.summarize()

    print(f"[启动] http://{args.host}:{args.port}  "
          f"POST /query  |  GET /health  |  GET /perf_summary", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
