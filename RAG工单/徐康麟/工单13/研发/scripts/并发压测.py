# -*- coding: utf-8 -*-
"""工单13 并发负载压测（标准库自实现）：``研发/scripts/并发压测.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

作用：用 ``concurrent.futures`` 自写压测器（**不引入 k6 / JMeter / Locust**），
按并发 1 / 2 / 4 / 8 依次对**同一个进程内**的 QAEngine 发压，采集：

* 吞吐量（QPS = 完成请求数 / 墙钟秒）；
* 延迟分位（均值 / P50 / P90 / P95 / P99 / 最大）；
* 错误率（异常请求数 / 总请求数，含超时）；
* 排队指标（近似）：``排队+调度 ≈ wall_ms - total_ms``，反映线程池/CPU 争用。

口径与纪律：

* 每题之间不共享会话（``session_id`` 轮转），避免多轮状态互相污染；
* 默认 ``use_cache=True``（贴近真实服务：重复问句命中缓存），但**首次出现的问句**必然
  走完整链路；同时提供 ``--no-cache`` 作为「纯冷链路」并发口径 —— 两种口径都输出，
  报告里必须注明用的是哪一种（否则「并发 8 还能 ≤3 s」这种结论毫无意义）；
* 每档并发前有 ``--settle-s`` 秒静置，避免上一档的 Ollama 队列残留影响下一档；
* 错误一律捕获并计数（不中断压测），错误样本写入 jsonl 供复盘。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/并发压测.py --profile optimized \\
        --concurrency 1,2,4,8 --requests 24 --out 优化/评估结果/并发压测.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import perf as perf_mod                                        # noqa: E402
from app.core.config import get_config                                       # noqa: E402
from app.core.errors import IndexMissingError                                # noqa: E402
from app.core.logging_conf import setup_logging                              # noqa: E402
from app.core.qa_engine import QAEngine                                      # noqa: E402
from app.storage.sqlite_manager import SqliteManager                         # noqa: E402

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"
DEFAULT_QUESTIONS = REPO_ROOT / "测试" / "测试数据" / "测试问题集.jsonl"
SLA_MS = 3000.0

EXIT_OK, EXIT_PARAM, EXIT_INDEX = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="并发压测.py",
                                     description=f"{WORK_ORDER} —— 并发负载压测（标准库实现）")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--concurrency", default="1,2,4,8", help="并发档位，逗号分隔")
    parser.add_argument("--requests", type=int, default=24, help="每档总请求数")
    parser.add_argument("--shots", type=int, default=0, help="复用同一题几次（0=按题库循环，不重复）")
    parser.add_argument("--settle-s", type=float, default=3.0, help="档间静置秒")
    parser.add_argument("--profile", default="config", choices=["config", "baseline", "optimized"])
    parser.add_argument("--no-cache", action="store_true", help="缓存只写不读（冷链路并发口径）")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", default=None, help="结果 JSON 路径")
    parser.add_argument("--jsonl", default=None, help="逐请求明细 jsonl 路径")
    parser.add_argument("--tag", default="")
    return parser


def read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if str(row.get("question") or "").strip():
                rows.append(row)
    return rows


def run_level(engine: QAEngine, questions: list[dict[str, Any]], *, concurrency: int,
              requests: int, use_cache: bool, tag: str) -> dict[str, Any]:
    """跑一档并发，返回该档统计。"""
    counter = {"n": 0}
    lock = threading.Lock()

    def one(index: int) -> dict[str, Any]:
        with lock:
            seq = counter["n"]
            counter["n"] += 1
        row = questions[seq % len(questions)]
        session = f"load-{tag}-{concurrency}-{seq % 8}"
        started = time.perf_counter()
        record: dict[str, Any] = {"seq": seq, "id": row.get("id"), "session": session,
                                  "concurrency": concurrency}
        try:
            result = engine.answer(str(row["question"]), lang=str(row.get("lang") or "auto"),
                                   session_id=session, use_cache=use_cache)
            record.update({"ok": True, "refused": result.refused,
                           "total_ms": float(result.latency_ms.get("total") or 0.0),
                           "retrieve_ms": float(result.latency_ms.get("retrieve") or 0.0),
                           "generate_ms": float(result.latency_ms.get("generate") or 0.0),
                           "first_token_ms": float(result.latency_ms.get("first_token") or 0.0),
                           "cache_hit": bool((result.cache or {}).get("hit"))})
        except Exception as exc:                     # noqa: BLE001 —— 错误必须计数，不中断压测
            record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "total_ms": 0.0, "retrieve_ms": 0.0, "generate_ms": 0.0,
                           "first_token_ms": 0.0, "cache_hit": False})
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["queue_ms"] = round(max(0.0, record["wall_ms"] - record["total_ms"]), 2)
        return record

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = [pool.submit(one, index) for index in range(int(requests))]
        records = [f.result() for f in as_completed(futures)]
    wall_s = time.perf_counter() - started
    records.sort(key=lambda r: r["seq"])
    ok = [r for r in records if r["ok"]]
    totals = [r["total_ms"] for r in ok]
    walls = [r["wall_ms"] for r in ok]
    queues = [r["queue_ms"] for r in ok]
    errors = [r for r in records if not r["ok"]]
    return {
        "并发度": int(concurrency),
        "请求数": len(records),
        "成功数": len(ok),
        "错误数": len(errors),
        "错误率": round(len(errors) / len(records), 4) if records else 0.0,
        "墙钟s": round(wall_s, 3),
        "吞吐_QPS": round(len(records) / wall_s, 4) if wall_s > 0 else 0.0,
        "吞吐_成功QPS": round(len(ok) / wall_s, 4) if wall_s > 0 else 0.0,
        "缓存命中数": sum(1 for r in ok if r["cache_hit"]),
        "延迟_总耗时": perf_mod.summarize(totals),
        "延迟_墙钟": perf_mod.summarize(walls),
        "排队与调度": perf_mod.summarize(queues),
        "SLA": {"阈值ms": SLA_MS, "达标率": perf_mod.sla_rate(totals, SLA_MS),
                "达标题数": sum(1 for v in totals if v <= SLA_MS)},
        "错误样本": [{"seq": r["seq"], "id": r.get("id"), "error": r.get("error")} for r in errors[:5]],
        "_records": records,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flags = perf_mod.apply_profile(args.profile)
    cfg = get_config()
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg, run_id=f"load-{args.profile}")
    questions = read_questions(Path(args.questions))
    if not questions:
        print(f"[参数错误] 测试问题集为空：{args.questions}", file=sys.stderr)
        return EXIT_PARAM
    try:
        storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    except Exception as exc:                       # noqa: BLE001
        log.log_event("load.storage_failed", level="WARNING", error=str(exc))
        storage = None
    try:
        engine = QAEngine(cfg=cfg, logger=log, storage=storage)
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}", file=sys.stderr)
        return EXIT_INDEX
    warm = {}
    if not args.no_warmup:
        warm = engine.warmup(with_generate=True)
        print(f"[预热] {warm.get('elapsed_ms')} ms")

    levels = [int(x) for x in str(args.concurrency).replace("，", ",").split(",") if x.strip()]
    use_cache = not args.no_cache
    all_records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, level in enumerate(levels):
        if index > 0 and args.settle_s > 0:
            time.sleep(float(args.settle_s))
        got = run_level(engine, questions, concurrency=level, requests=int(args.requests),
                        use_cache=use_cache, tag=args.tag or args.profile)
        all_records.extend(got.pop("_records"))
        results.append(got)
        print(f"[并发 {level:>2}] QPS={got['吞吐_QPS']:>7.3f}  "
              f"P50={got['延迟_总耗时']['p50']:>8.1f}ms  P95={got['延迟_总耗时']['p95']:>8.1f}ms  "
              f"最大={got['延迟_总耗时']['max']:>8.1f}ms  错误率={got['错误率'] * 100:.1f}%  "
              f"SLA={got['SLA']['达标率'] * 100:.1f}%")
    payload = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "脚本": str(Path(__file__).resolve()),
        "标签": args.tag or None,
        "档位": args.profile,
        "口径": {"缓存": "读写缓存" if use_cache else "只写不读（--no-cache）",
                 "每档请求数": int(args.requests), "档间静置s": args.settle_s,
                 "压测器": "标准库 concurrent.futures.ThreadPoolExecutor（未引入 k6/JMeter/Locust）",
                 "SLA_ms": SLA_MS,
                 "共享CPU声明": "Ollama 为本机 CPU 单实例，压测与其它任务共享 CPU；档间静置以减少串扰。"},
        "优化开关": {k: v for k, v in flags.items() if k != "adaptive_top_k_values"},
        "预热": warm,
        "各档结果": results,
    }
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 压测汇总 → {path}")
    if args.jsonl:
        path = Path(args.jsonl)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[OK] 逐请求明细 → {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
