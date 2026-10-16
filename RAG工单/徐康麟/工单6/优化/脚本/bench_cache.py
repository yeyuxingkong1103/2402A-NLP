# -*- coding: utf-8 -*-
"""T7 缓存基准：同一批问题跑两遍，量出「四级缓存」的命中率与省下的耗时。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_cache.py --questions 优化/基线/golden16.txt

产出：
    * ``研发/data/cache/cache_stats.json``（四级缓存命中率/条目数/省下的毫秒，可读 JSON）
    * ``优化/评估结果/缓存基准.json``（第一遍冷跑 vs 第二遍热跑的 P50/P95/最大）

口径说明：
    * 「缓存命中」= T7 的 ``研发/app/core/cache.py`` 四级缓存（归一化 / 检索 / 生成 / 改写）
      **加** T5 已有的答案级缓存（SQLite）；两者都在**同进程**内生效；
    * 冷跑（第一遍）**每问都必须走完整链路**；热跑（第二遍）应当全部命中；
    * 报告里两档数字**分列**，不得用热跑数字冒充全量性能。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

from app.core.cache import get_cache                      # noqa: E402
from app.core.config import get_config                    # noqa: E402
from app.core.qa_engine import QAEngine                   # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def load_questions(path: Path | None) -> list[str]:
    if path is not None and path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    golden = ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
    return [json.loads(line)["question"] for line in golden.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run_pass(engine: QAEngine, questions: list[str], *, use_cache: bool, tag: str) -> dict:
    latencies: list[float] = []
    first_tokens: list[float] = []
    refusals = 0
    for question in questions:
        started = time.perf_counter()
        result = engine.answer(question, lang="zh", use_cache=use_cache)
        latencies.append(round((time.perf_counter() - started) * 1000, 1))
        first_tokens.append(float(result.latency_ms.get("first_token") or 0.0))
        refusals += int(result.refused)
    return {"tag": tag, "use_cache": use_cache, "n": len(questions),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
            "latency_min_ms": round(min(latencies), 1) if latencies else 0.0,
            "first_token_min_ms": round(min(first_tokens), 2) if first_tokens else 0.0,
            "first_token_max_ms": round(max(first_tokens), 2) if first_tokens else 0.0,
            "refusals": refusals, "latencies": latencies}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T7 多级缓存基准（冷跑 vs 热跑）")
    parser.add_argument("--questions", default=None, help="问题清单文件（每行一题；默认用 golden 16 题）")
    parser.add_argument("--out", default=str(ROOT / "优化" / "评估结果" / "缓存基准.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / "研发" / "data" / "cache"))
    parser.add_argument("--cold-cache", action="store_true",
                        help="冷跑前清空进程内缓存统计（新进程默认就是冷的；本开关用于同进程复跑）")
    args = parser.parse_args(argv)

    cfg = get_config()
    cache = get_cache(cfg)
    questions = load_questions(Path(args.questions) if args.questions else None)
    engine = QAEngine(cfg=cfg, warmup=True)

    cold = run_pass(engine, questions, use_cache=True, tag="cold")
    warm = run_pass(engine, questions, use_cache=True, tag="warm")
    # 第三遍：**关掉答案级缓存**、但 T7 的检索/生成缓存已热 —— 用于单独量出
    # 「检索结果缓存 + 生成结果缓存」的收益（否则会被答案级缓存短路，看不到这两层命中）
    layers = run_pass(engine, questions, use_cache=False, tag="warm_t7_layers")
    stats = cache.stats()
    cache_path = cache.persist(args.cache_dir)

    improvement = round(cold["latency_p50_ms"] / warm["latency_p50_ms"], 2) if warm["latency_p50_ms"] else None
    layer_speedup = round(cold["latency_p50_ms"] / layers["latency_p50_ms"], 2) if layers["latency_p50_ms"] else None
    payload = {
        "work_order": "人工智能NLP-RAG-图像内容解析及检索优化",
        "protocol": ("同进程三遍：①冷跑（完整链路）②热跑（答案级缓存）"
                     "③关答案缓存但 T7 检索/生成缓存已热（单独量这两层）"),
        "questions": len(questions),
        "cold": cold, "warm": warm, "warm_t7_layers": layers,
        "p50_speedup_x": improvement,
        "p50_speedup_t7_layers_x": layer_speedup,
        "cache_layers": stats["layers"],
        "cache_fingerprint": stats["fingerprint"],
        "cache_stats_file": str(cache_path),
        "note": ("缓存命中档只能代表「重复提问」的性能；单次问答性能见 优化/评估结果/run1_full"
                 "（--no-cache 全量）"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[T7 缓存基准] 题数={len(questions)}")
    print(f"  ①冷跑（完整链路）：P50 {cold['latency_p50_ms']} / P95 {cold['latency_p95_ms']} / "
          f"max {cold['latency_max_ms']} ms")
    print(f"  ②热跑（答案级缓存）：P50 {warm['latency_p50_ms']} / P95 {warm['latency_p95_ms']} / "
          f"max {warm['latency_max_ms']} ms（P50 加速 {improvement}×）")
    print(f"  ③热跑（仅 T7 检索/生成缓存，答案缓存关闭）：P50 {layers['latency_p50_ms']} / "
          f"P95 {layers['latency_p95_ms']} / max {layers['latency_max_ms']} ms"
          f"（P50 加速 {layer_speedup}×）")
    for name, layer in stats["layers"].items():
        print(f"  [{name:<10}] 命中 {layer['hits']:>3} / 未命中 {layer['misses']:>3} "
              f"命中率 {layer['hit_rate']:.2%} 条目 {layer['entries']:>3} 省时 {layer['saved_ms'] / 1000:.2f} s")
    print(f"  落盘 → {out} / {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
