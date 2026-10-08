#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
scripts/benchmark_v13.py —— 工单十三 RAG 性能基准测试

用法（env 须在 import src 前设置，本脚本先解析参数再导入）：
  python scripts/benchmark_v13.py --phase before   # 优化前：原始配置
  python scripts/benchmark_v13.py --phase after    # 优化后：召回12+12/截断512/嵌入缓存
  python scripts/benchmark_v13.py --phase profile  # cProfile 剖析 3 条查询
  python scripts/benchmark_v13.py --phase load_before --concurrency 5 --rounds 3
  python scripts/benchmark_v13.py --phase load_after  --concurrency 5 --rounds 3

产出：
  docs/v13_benchmark_before.json / docs/v13_benchmark_after.json
  docs/v13_profile_stats.txt
  docs/v13_load_test.json（load_before/load_after 分别追加）
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 工单十三：加载 .env（DEEPSEEK_API_KEY/MILVUS 等），须在导入 src 模块前
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

ARGS = argparse.ArgumentParser(description="工单十三 性能基准测试")
ARGS.add_argument("--phase", required=True,
                  choices=["before", "after", "profile", "load_before", "load_after"])
ARGS.add_argument("--rounds", type=int, default=2, help="每题重复次数")
ARGS.add_argument("--concurrency", type=int, default=5, help="负载测试并发数")
ARGS.add_argument("--limit", type=int, default=12, help="测试问题数上限")
args = ARGS.parse_args()

# 工单十三：环境变量必须在 src 模块导入前设置（模块导入时读取）
if args.phase in ("before", "load_before"):
    os.environ["RAG_V13_EMBED_CACHE"] = "0"        # 优化前：无查询嵌入缓存
    os.environ["RAG_V13_RERANK_MAX_CHARS"] = "1500"  # 优化前：重排输入 1500 字符
elif args.phase in ("after", "load_after"):
    os.environ["RAG_V13_EMBED_CACHE"] = "1"        # 优化①：查询嵌入 LRU 缓存
    os.environ["RAG_V13_RERANK_MAX_CHARS"] = "512"   # 优化②：重排输入截断 512
    # 优化③：限制 torch/OMP 线程数（本机 32 核，4 并发 × 默认32线程 = 严重超订，
    # torch 计算反而被线程争用拖慢，实测单次重排飙至 133s）
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

from src.perf_v13 import clear_log  # noqa: E402
from src.rag_engine_v6 import RAGEngineV6  # noqa: E402
from src.retrieval.retrieval_config import RetrievalConfig  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-RAG性能瓶颈识别与优化"
OUT_DIR = ROOT / "docs"


def get_questions(limit):
    """工单十三：读取工单十三基准问题集（含 doc_id，模拟真实会话）"""
    d = json.load(open(ROOT / "data" / "benchmark_questions_v13.json",
                       encoding="utf-8"))
    qs = [(q["question"], q.get("doc_id")) for q in d["questions"]]
    return qs[:limit]


def agg(vals):
    """工单十三：均值 / p50 / p95 / max 聚合"""
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    vs = sorted(vals)
    return {"mean": round(statistics.mean(vs), 1),
            "p50": round(vs[len(vs) // 2], 1),
            "p95": round(vs[min(len(vs) - 1, int(round(len(vs) * 0.95)))], 1),
            "max": round(vs[-1], 1), "n": len(vs)}


def build_engine():
    """工单十三：按阶段构建引擎（before=原始配置 / after=优化配置）"""
    if args.phase in ("before", "load_before"):
        cfg = RetrievalConfig(top_k=8)  # 原始：召回 24+24
        label = "优化前(召回24+24,重排截断1500,无嵌入缓存)"
    else:
        cfg = RetrievalConfig(top_k=8, vector_recall_k=12, fulltext_recall_k=12)
        label = "优化后(召回12+12,重排截断512,嵌入缓存)"
    return RAGEngineV6(default_config=cfg), label


def read_stage_records():
    """工单十三：读取本次运行的结构化性能日志并按阶段聚合"""
    log = ROOT / "logs" / "perf_v13.jsonl"
    by_stage = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            by_stage.setdefault(r["stage"], []).append(r["ms"])
    return by_stage


def run_bench():
    """工单十三：串行基准测试（先冷启动一次，再热态循环）"""
    clear_log()
    engine, label = build_engine()
    questions = get_questions(args.limit)
    print(f"[v13] 工单十三 基准测试 phase={args.phase}")
    print(f"[v13] 配置: {label}")
    print(f"[v13] 问题数={len(questions)} rounds={args.rounds}")

    # 冷启动：第一条查询包含模型懒加载（bge-m3/reranker/CLIP/索引构建）
    t0 = time.perf_counter()
    engine.ask(questions[0][0], doc_id=questions[0][1])
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"[v13] 冷启动首查耗时: {cold_ms:.0f}ms")

    per_question = []
    for i, (q, doc_id) in enumerate(questions):
        for r in range(args.rounds):
            t = time.perf_counter()
            res = engine.ask(q, doc_id=doc_id)
            wall = (time.perf_counter() - t) * 1000
            per_question.append({
                "qid": i + 1, "round": r + 1, "query": q[:60],
                "latency_ms": round(wall, 1),
                "retrieve_ms": res["breakdown"]["retrieve_ms"],
                "llm_ms": res["breakdown"]["llm_ms"],
                "stages_ms": res.get("stages_ms", {}),
                "retrieval_elapsed_ms": res["retrieval"]["elapsed_ms"],
            })
            print(f"[{i+1}/{len(questions)}] r{r+1} "
                  f"retrieve={res['breakdown']['retrieve_ms']:.0f}ms "
                  f"llm={res['breakdown']['llm_ms']:.0f}ms "
                  f"e2e={wall:.0f}ms")

    stages = {k: agg(v) for k, v in read_stage_records().items()}
    out = {
        "work_order": WORK_ORDER, "phase": args.phase, "config": label,
        "questions": len(questions), "rounds": args.rounds,
        "cold_start_ms": round(cold_ms, 1),
        "e2e": agg([p["latency_ms"] for p in per_question]),
        "retrieval_wall": agg([p["retrieve_ms"] for p in per_question]),
        "llm": agg([p["llm_ms"] for p in per_question]),
        "stages": stages,
        "per_question": per_question,
    }
    out_path = OUT_DIR / f"v13_benchmark_{args.phase}.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[v13] 已保存 {out_path}")
    print(f"[v13] 检索墙钟 p50={out['retrieval_wall']['p50']}ms "
          f"p95={out['retrieval_wall']['p95']}ms（验收线 3000ms）")
    print(f"[v13] 端到端   p50={out['e2e']['p50']}ms p95={out['e2e']['p95']}ms")


def run_profile():
    """工单十三：cProfile 应用级剖析，定位慢函数"""
    import cProfile
    import pstats
    import io
    engine, label = build_engine()
    questions = get_questions(3)
    engine.ask(questions[0][0], doc_id=questions[0][1])  # 预热，剖析热路径
    profiler = cProfile.Profile()
    profiler.enable()
    for q, doc_id in questions:
        try:
            engine.ask(q, doc_id=doc_id)
        except Exception as e:
            print(f"[v13] profile 跳过异常: {e}")
    profiler.disable()
    buf = io.StringIO()
    st = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    st.print_stats(25)
    text = f"工单十三 cProfile 剖析（phase=profile, config={label}）\n{buf.getvalue()}"
    out_path = OUT_DIR / "v13_profile_stats.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[v13] 已保存 {out_path}")
    print("\n".join(text.splitlines()[:35]))


def run_load():
    """工单十三：并发负载测试（模拟生产流量，瓶颈在压力下显现）"""
    clear_log()
    engine, label = build_engine()
    # 工单十三：优化后阶段先预热（对应生产"服务启动即预热"，消除冷启动抖动；
    # 优化前阶段不预热，保留原系统"上线即压测"的真实表现）
    if args.phase == "load_after":
        print("[v13] 预热模型中（bge-m3/reranker/CLIP/全文索引）...")
        t_w = time.perf_counter()
        engine.warmup()
        print(f"[v13] 预热完成: {time.perf_counter() - t_w:.1f}s")
    questions = get_questions(args.limit)
    tasks = [q for q, _ in questions] * args.rounds
    doc_ids = [d for _, d in questions] * args.rounds
    print(f"[v13] 负载测试 phase={args.phase} 并发={args.concurrency} "
          f"请求数={len(tasks)}")
    latencies, retrieves = [], []
    t_start = time.perf_counter()

    def _one(item):
        q, doc_id = item
        t = time.perf_counter()
        res = engine.ask(q, doc_id=doc_id)
        return (time.perf_counter() - t) * 1000, res["breakdown"]["retrieve_ms"]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for lat, ret in pool.map(_one, list(zip(tasks, doc_ids))):
            latencies.append(lat)
            retrieves.append(ret)
            print(f"  done e2e={lat:.0f}ms retrieve={ret:.0f}ms")
    total_s = time.perf_counter() - t_start
    out = {
        "work_order": WORK_ORDER, "phase": args.phase, "config": label,
        "concurrency": args.concurrency, "requests": len(tasks),
        "total_s": round(total_s, 1),
        "throughput_rps": round(len(tasks) / total_s, 2),
        "e2e": agg(latencies), "retrieval_wall": agg(retrieves),
        "stages": {k: agg(v) for k, v in read_stage_records().items()},
    }
    out_path = OUT_DIR / "v13_load_test.json"
    # load_before / load_after 分别写入同一文件的不同键
    data = {}
    if out_path.exists():
        try:
            data = json.load(open(out_path, encoding="utf-8"))
        except Exception:
            data = {}
    data[args.phase] = out
    json.dump(data, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[v13] 已保存 {out_path}")
    print(f"[v13] 吞吐={out['throughput_rps']}rps "
          f"检索p95={out['retrieval_wall']['p95']}ms "
          f"端到端p95={out['e2e']['p95']}ms")


if __name__ == "__main__":
    if args.phase == "profile":
        run_profile()
    elif args.phase.startswith("load"):
        run_load()
    else:
        run_bench()
