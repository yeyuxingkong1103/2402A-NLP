# -*- coding: utf-8 -*-
"""工单13 五阶段分段计时剖析：``研发/scripts/分段计时剖析.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

作用：在同一进程内对**同一测试集**逐题跑完整问答链路，采集
「查询处理与增强 / 检索 / 上下文组装与提示工程 / LLM 生成 / 后处理与响应格式化」
五阶段耗时 + 若干子阶段（查询嵌入 / 向量检索 / BM25 检索 / RRF 融合+重排 / 证据页补全 /
LLM 调用 / 护栏复核 …），输出 jsonl + 汇总 json，作为「优化前 / 优化后」对比与瀑布图的唯一数据源。

口径（**必须与验收口径一致，不许含糊**）：

* ``--no-cache`` 语义：``engine.answer(..., use_cache=False)`` —— 答案级、检索级、生成级
  三级缓存**只写不读**，量到的是真实链路耗时（不是缓存命中的 ~1 ms）；
* 默认跑 **2 轮**（``--rounds``）：第 1 轮当预热轮丢弃，**只统计第 2 轮**，
  避免首题偶发抖动污染统计；
* 题间 ``--gap-s`` 秒休眠（默认 0.3）：Ollama 与评测脚本共享 CPU，留出冷却间隙，
  与 工单7「响应时间基准」的批间休眠做法一致；
* 耗时一律 ``time.perf_counter()``（单调时钟）；总耗时同时记「引擎内部 total」与
  「脚本墙钟 wall_ms」，两者差值即进程内调度与序列化开销（如实呈现，不隐藏）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/分段计时剖析.py \\
        --profile optimized --out 优化/评估结果/优化后.jsonl --summary 优化/评估结果/优化后_汇总.json
    pwsh -NoProfile -File run_py.ps1 研发/scripts/分段计时剖析.py \\
        --profile baseline --out 优化/评估结果/优化前.jsonl --summary 优化/评估结果/优化前_汇总.json

退出码：``0`` 正常 / ``2`` 参数错误 / ``3`` 索引缺失 / ``4`` 生成后端不可用。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import perf as perf_mod                                        # noqa: E402
from app.core.config import describe_runtime, get_config                     # noqa: E402
from app.core.errors import IndexMissingError, LLMProbeError                 # noqa: E402
from app.core.logging_conf import setup_logging                              # noqa: E402
from app.core.qa_engine import QAEngine                                      # noqa: E402
from app.core.retriever import clear_bundle_cache                            # noqa: E402
from app.storage.sqlite_manager import SqliteManager                         # noqa: E402

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"
DEFAULT_QUESTIONS = REPO_ROOT / "测试" / "测试数据" / "测试问题集.jsonl"
SLA_MS = 3000.0

EXIT_OK, EXIT_PARAM, EXIT_INDEX, EXIT_BACKEND = 0, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="分段计时剖析.py", description=f"{WORK_ORDER} —— 五阶段分段计时剖析")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="测试问题集 jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    parser.add_argument("--rounds", type=int, default=2, help="轮数（默认 2；第 1 轮丢弃当预热）")
    parser.add_argument("--keep-rounds", type=int, default=0,
                        help="保留前 N 轮不计入统计（默认 0 ⇒ 等价 rounds-1）")
    parser.add_argument("--gap-s", type=float, default=0.3, help="题间休眠秒（默认 0.3）")
    parser.add_argument("--profile", default="config", choices=["config", "baseline", "optimized"],
                        help="性能档位（写进报告，便于复核对照口径）")
    parser.add_argument("--top-k", type=int, default=None, help="显式检索 K（不传=引擎默认/自适应）")
    parser.add_argument("--out", default=None, help="逐题结果 jsonl 路径")
    parser.add_argument("--summary", default=None, help="汇总 json 路径")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（仅冷启动实验用）")
    parser.add_argument("--use-cache", action="store_true",
                        help="读取缓存（默认不读 ⇒ --no-cache 口径）")
    parser.add_argument("--tag", default="", help="结果标签（写进输出，便于区分 run）")
    return parser


def read_questions(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("question") or "").strip():
            rows.append(row)
    return rows[:limit] if limit and limit > 0 else rows


def _stage_row(perf: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {}
    for key, value in (perf.get("stages") or {}).items():
        row[f"stage::{key}"] = round(float(value), 3)
    for key, value in (perf.get("spans") or {}).items():
        row[f"span::{key}"] = round(float(value), 3)
    return row


def run_once(engine: QAEngine, rows: list[dict[str, Any]], args: argparse.Namespace,
             *, round_no: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        question = str(row["question"])
        wall0 = time.perf_counter()
        result = engine.answer(question, lang=str(row.get("lang") or "auto"), top_k=args.top_k,
                               session_id=None, return_evidence=False,
                               use_cache=bool(args.use_cache))
        wall_ms = round((time.perf_counter() - wall0) * 1000, 2)
        perf = dict((result.diagnostics or {}).get("perf") or {})
        latency = dict(result.latency_ms or {})
        item = {
            "round": round_no, "seq": index, "id": row.get("id"), "question": question,
            "answer": result.answer, "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "cited_pages": result.cited_pages, "language": result.language,
            "backend": result.backend, "model": result.model,
            "generation_method": result.generation_method,
            "cache": result.cache,
            "wall_ms": wall_ms,
            "total_ms": float(latency.get("total") or 0.0),
            "retrieve_ms": float(latency.get("retrieve") or 0.0),
            "generate_ms": float(latency.get("generate") or 0.0),
            "first_token_ms": float(latency.get("first_token") or 0.0),
            "perf": perf,
            "hit_ids": [str(h.get("chunk_id")) for h in ((result.diagnostics or {}).get("hits") or [])],
            "answer_chars": len(result.answer or ""),
        }
        item.update(_stage_row(perf))
        out.append(item)
        if args.gap_s > 0 and index < len(rows):
            time.sleep(float(args.gap_s))
    return out


def summarize(items: list[dict[str, Any]], *, stage_keys: list[str]) -> dict[str, Any]:
    def col(name: str) -> list[float]:
        return [float(it.get(name) or 0.0) for it in items]

    total = col("total_ms")
    stats: dict[str, Any] = {
        "样本数": len(items),
        "拒答数": sum(1 for it in items if it.get("refused")),
        "总耗时": perf_mod.summarize(total),
        "检索": perf_mod.summarize(col("retrieve_ms")),
        "生成": perf_mod.summarize(col("generate_ms")),
        "首字": perf_mod.summarize(col("first_token_ms")),
        "墙钟": perf_mod.summarize(col("wall_ms")),
        "SLA": {"阈值ms": SLA_MS, "达标率": perf_mod.sla_rate(total, SLA_MS),
                "达标题数": sum(1 for v in total if v <= SLA_MS),
                "超时题号": [it.get("id") for it in items if float(it.get("total_ms") or 0) > SLA_MS]},
        "SLA_首字": {"阈值ms": SLA_MS, "达标率": perf_mod.sla_rate(col("first_token_ms"), SLA_MS)},
        "SLA_墙钟": {"阈值ms": SLA_MS, "达标率": perf_mod.sla_rate(col("wall_ms"), SLA_MS)},
    }
    stage_means: dict[str, float] = {}
    stage_stats: dict[str, Any] = {}
    for key in stage_keys:
        values = col(key)
        if not values:
            continue
        stage_means[key] = round(statistics.fmean(values), 2)
        stage_stats[key] = perf_mod.summarize(values)
    stats["阶段"] = stage_stats
    stats["阶段均值"] = stage_means
    # 占比按「裸阶段名」计算（stage_keys 带 ``stage::`` 前缀，需还原）
    stats["阶段占比"] = perf_mod.stage_share(
        {key.split("::", 1)[-1]: value for key, value in stage_means.items()})
    span_keys = sorted({k for it in items for k in it if str(k).startswith("span::")})
    stats["子阶段均值"] = {k: round(statistics.fmean(col(k)), 2) for k in span_keys} if items else {}
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = perf_mod.apply_profile(args.profile)
    cfg = get_config()
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg, run_id=f"profile-{args.profile}")
    log.log_event("profiling.start", argv=sys.argv[1:], profile=args.profile,
                  flags={k: v for k, v in profile.items() if k != "adaptive_top_k_values"},
                  runtime=describe_runtime())

    questions = read_questions(Path(args.questions), args.limit)
    if not questions:
        print(f"[参数错误] 测试问题集为空：{args.questions}", file=sys.stderr)
        return EXIT_PARAM
    try:
        storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    except Exception as exc:                       # noqa: BLE001 —— 落盘失败不阻断测量
        log.log_event("profiling.storage_failed", level="WARNING", error=str(exc))
        storage = None
    try:
        engine = QAEngine(cfg=cfg, logger=log, storage=storage)
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}", file=sys.stderr)
        return EXIT_INDEX
    warm: dict[str, Any] = {}
    if not args.no_warmup:
        warm = engine.warmup(with_generate=True)
        print(f"[预热] {warm.get('elapsed_ms')} ms → 明细 {json.dumps({k: v for k, v in warm.items() if k.startswith('ms_') or k == 'index_chunks'}, ensure_ascii=False)}")
    try:
        engine.health()
    except Exception as exc:                        # noqa: BLE001
        print(f"[健康检查失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_BACKEND if isinstance(exc, LLMProbeError) else EXIT_OK

    rounds = max(1, int(args.rounds))
    drop = int(args.keep_rounds) if args.keep_rounds else max(0, rounds - 1)
    all_items: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for index in range(1, rounds + 1):
        started = time.perf_counter()
        items = run_once(engine, questions, args, round_no=index)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        all_items.extend(items)
        if index > drop:
            kept.extend(items)
        print(f"[第 {index}/{rounds} 轮] {len(items)} 题，{elapsed} ms，"
              f"均值 {round(statistics.fmean([i['total_ms'] for i in items]), 1)} ms"
              f"{'（丢弃，当预热轮）' if index <= drop else '（计入统计）'}")
        log.log_event("profiling.round", round=index, questions=len(items), elapsed_ms=elapsed,
                      mean_total_ms=round(statistics.fmean([i["total_ms"] for i in items]), 2))

    stage_keys = [f"stage::{name}" for name in perf_mod.STAGES]
    stats = summarize(kept, stage_keys=stage_keys)
    payload = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "脚本": str(Path(__file__).resolve()),
        "标签": args.tag or None,
        "档位": args.profile,
        "口径": {
            "缓存": "只写不读（use_cache=False）" if not args.use_cache else "读写缓存",
            "轮数": rounds, "丢弃轮数": drop, "计入统计题数": len(kept),
            "题间休眠s": args.gap_s, "top_k参数": args.top_k,
            "SLA_ms": SLA_MS,
            "耗时口径": "perf_counter 单调时钟；total_ms 为引擎内部统计，wall_ms 为脚本墙钟",
        },
        "环境": {"解释器": sys.executable, "python": sys.version.split()[0],
                 "runtime": describe_runtime()},
        "预热": warm,
        "优化开关": {k: v for k, v in profile.items() if k != "adaptive_top_k_values"},
        "自适应top_k配置": profile.get("adaptive_top_k_values"),
        "统计": stats,
        "逐题": kept,
    }
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[OK] 逐题结果 → {out_path}")
    if args.summary:
        sum_path = Path(args.summary)
        if not sum_path.is_absolute():
            sum_path = REPO_ROOT / sum_path
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        sum_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 汇总 → {sum_path}")
    print("\n=== 五阶段耗时均值（ms）===")
    for name in perf_mod.STAGES:
        key = f"stage::{name}"
        value = stats["阶段均值"].get(key, 0.0)
        share = stats["阶段占比"].get(name, 0.0)
        print(f"  {name:<22} {value:>10.2f}   {share * 100:>6.1f}%")
    print(f"\n总耗时：均值 {stats['总耗时']['mean']} ms / P50 {stats['总耗时']['p50']} ms / "
          f"P90 {stats['总耗时']['p90']} ms / P95 {stats['总耗时']['p95']} ms / "
          f"最大 {stats['总耗时']['max']} ms")
    print(f"SLA ≤{SLA_MS:.0f} ms 达标率：总耗时 {stats['SLA']['达标率'] * 100:.1f}%"
          f"（{stats['SLA']['达标题数']}/{stats['样本数']}），"
          f"首字 {stats['SLA_首字']['达标率'] * 100:.1f}%")
    clear_bundle_cache()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
