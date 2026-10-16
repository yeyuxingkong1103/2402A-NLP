# -*- coding: utf-8 -*-
"""工单13 cProfile 热点剖析：``研发/scripts/cProfile剖析.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

作用：用标准库 ``cProfile`` 对**同一条 RAG 链路**跑 ≥20 次查询，输出：

* ``Top-N 热点函数表``（按 ``cumtime`` 与 ``tottime`` 两个口径各一份，含 ``ncalls`` / ``percall``）；
* 按模块聚合的耗时表（``app.core.retriever`` / ``generator`` / ``bm25_index`` …）；
* ``pstats`` 文本转储 + 结构化 JSON（供截图与报告直接引用）。

工具选型与替代口径（**工单要求如实注明**）：

* 任务书建议 ``snakeviz``（cProfile 可视化）——本机**未安装**，且 ``pip`` 落点红线不允许裸装
  （见 ``_共享约定.md`` §2），故改用 ``pstats`` 文本 + 本仓库自绘 PNG（``研发/scripts/生成截图.py``）
  呈现同一份 cProfile 数据，**口径不变**（同一 ``cProfile.Profile`` 对象）；
* 任务书建议 ``py-spy``（采样剖析器，可挂到运行中进程）——本机**未安装**，同样不 pip 安装，
  改用标准库 ``sys.setprofile`` 写**自采样剖析器**作为替代口径：``sys.setprofile`` 在每个
  Python 函数调用/返回时触发回调，按「记录当前调用栈 + 时间戳」的方式得到**采样式**热点分布
  （见 ``采样剖析.py`` 与本文件 ``--sample`` 开关）。两者差异必须写清：
  ``cProfile`` 是**确定性**（每次调用都计数，含完整调用次数，但开销大、放大了小函数占比）；
  ``sys.setprofile`` 是**概率式**（按时间采样，更接近真实 CPU 占用，但样本量受查询数限制）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/cProfile剖析.py \\
        --profile baseline --rounds 2 --top 20 --out 研发/profiles/cProfile_优化前.txt \\
        --json 研发/profiles/cProfile_优化前.json
    pwsh -NoProfile -File run_py.ps1 研发/scripts/cProfile剖析.py --profile optimized --rounds 2 ...

退出码：``0`` 正常 / ``2`` 参数错误 / ``3`` 索引缺失。
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
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

EXIT_OK, EXIT_PARAM, EXIT_INDEX = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cProfile剖析.py",
                                     description=f"{WORK_ORDER} —— cProfile 热点剖析（≥20 次查询）")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--rounds", type=int, default=2, help="跑几轮（17 题 × 2 轮 = 34 次查询）")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top", type=int, default=20, help="Top-N 热点函数")
    parser.add_argument("--sort", default="cumulative", choices=["cumulative", "tottime"],
                        help="主排序口径（另一口径也会一并输出）")
    parser.add_argument("--profile", default="config", choices=["config", "baseline", "optimized"])
    parser.add_argument("--out", default=None, help="pstats 文本输出路径")
    parser.add_argument("--json", dest="json_out", default=None, help="结构化热点 JSON 输出路径")
    parser.add_argument("--gap-s", type=float, default=0.2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def read_questions(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if str(row.get("question") or "").strip():
                rows.append(row)
    return rows[:limit] if limit and limit > 0 else rows


def _rows(profile: cProfile.Profile, sort_key: str, top: int) -> list[dict[str, Any]]:
    """从 cProfile 取 Top-N（``pstats.Stats`` 同时给 cumtime 与 tottime）。"""
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.sort_stats(sort_key)
    out: list[dict[str, Any]] = []
    for func, (cc, nc, tt, ct, callers) in sorted(
            stats.stats.items(), key=lambda kv: -kv[1][3 if sort_key == "cumulative" else 2])[:top]:
        name = func[2]
        out.append({
            "rank": len(out) + 1,
            "function": name,
            "file": func[0],
            "line": func[1],
            "ncalls": nc,
            "primitive_calls": cc,
            "tottime_s": round(tt, 6),
            "cumtime_s": round(ct, 6),
            "tottime_ms": round(tt * 1000, 3),
            "cumtime_ms": round(ct * 1000, 3),
            "percall_tottime_ms": round((tt / nc) * 1000, 4) if nc else 0.0,
            "percall_cumtime_ms": round((ct / nc) * 1000, 4) if nc else 0.0,
            "callers": len(callers),
        })
    return out


def _module_totals(profile: cProfile.Profile) -> list[dict[str, Any]]:
    """按「函数所属文件」聚合 tottime（判断时间花在哪个模块/依赖上）。"""
    stats = pstats.Stats(profile)
    bucket: dict[str, dict[str, Any]] = {}
    for func, (_cc, nc, tt, ct, _callers) in stats.stats.items():
        path = func[0]
        # 归一到「包.模块」级别，便于读：.../app/core/retriever.py → app.core.retriever
        marker = "/app/" if "/app/" in path else ("\\app\\" if "\\app\\" in path else None)
        if marker:
            tail = path.split(marker, 1)[1].replace("\\", "/")
            key = "app." + tail[:-3].replace("/", ".") if tail.endswith(".py") else "app." + tail
        else:
            key = Path(path).name
        item = bucket.setdefault(key, {"module": key, "tottime_ms": 0.0, "cumtime_ms": 0.0, "calls": 0})
        item["tottime_ms"] += tt * 1000
        item["cumtime_ms"] += ct * 1000
        item["calls"] += nc
    rows = sorted(bucket.values(), key=lambda r: -r["tottime_ms"])
    for row in rows:
        row["tottime_ms"] = round(row["tottime_ms"], 2)
        row["cumtime_ms"] = round(row["cumtime_ms"], 2)
    return rows[:25]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_flags = perf_mod.apply_profile(args.profile)
    cfg = get_config()
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg, run_id=f"cprofile-{args.profile}")
    questions = read_questions(Path(args.questions), args.limit)
    if not questions:
        print(f"[参数错误] 测试问题集为空：{args.questions}", file=sys.stderr)
        return EXIT_PARAM
    try:
        storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    except Exception as exc:                       # noqa: BLE001
        log.log_event("cprofile.storage_failed", level="WARNING", error=str(exc))
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

    profiler = cProfile.Profile()
    runs = 0
    total_ms = 0.0
    wall0 = time.perf_counter()
    try:
        profiler.enable()
        for _round in range(1, max(1, int(args.rounds)) + 1):
            for index, row in enumerate(questions, 1):
                t0 = time.perf_counter()
                result = engine.answer(str(row["question"]), lang=str(row.get("lang") or "auto"),
                                       use_cache=False)     # 缓存只写不读：量真实链路
                total_ms += (time.perf_counter() - t0) * 1000
                runs += 1
                if args.gap_s > 0 and index < len(questions):
                    profiler.disable()                       # 休眠不计入剖析
                    time.sleep(float(args.gap_s))
                    profiler.enable()
    finally:
        profiler.disable()
    wall_ms = round((time.perf_counter() - wall0) * 1000, 2)
    print(f"[剖析完成] {runs} 次查询，链路累计 {round(total_ms, 1)} ms，墙钟 {wall_ms} ms，"
          f"单次均值 {round(total_ms / max(runs, 1), 1)} ms")

    by_cum = _rows(profiler, "cumulative", int(args.top))
    by_tot = _rows(profiler, "tottime", int(args.top))
    modules = _module_totals(profiler)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(args.sort)
    stats.print_stats(int(args.top) * 3)
    stats.print_callers(int(args.top))
    text = stream.getvalue()
    payload = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "脚本": str(Path(__file__).resolve()),
        "标签": args.tag or None,
        "档位": args.profile,
        "优化开关": {k: v for k, v in profile_flags.items() if k != "adaptive_top_k_values"},
        "口径": {
            "剖析器": "cProfile（标准库，确定性剖析：每次函数调用都计数）",
            "替代说明": ("任务书建议的 snakeviz（可视化）与 py-spy（采样）本机均未安装；"
                       "按 _共享约定.md §2 的 pip 落点红线不许裸装，故用 pstats 文本 + 自绘 PNG "
                       "呈现同一份 cProfile 数据；采样口径由 研发/scripts/采样剖析.py（sys.setprofile）提供。"),
            "查询次数": runs,
            "题数": len(questions),
            "轮数": int(args.rounds),
            "缓存": "只写不读（use_cache=False）",
            "平均单次链路耗时ms": round(total_ms / max(runs, 1), 2),
            "剖析墙钟ms_不含题间休眠": wall_ms,
            "注意": "cProfile 会放大纯 Python 小函数的占比（Ollama HTTP 等待期间几乎不上采样），因此 tottime 口径需与 分段计时剖析.py 的墙钟数据交叉验证。",
        },
        "环境": {"解释器": sys.executable, "python": sys.version.split()[0]},
        "预热": warm,
        "Top20_cumtime": by_cum,
        "Top20_tottime": by_tot,
        "模块聚合_tottime": modules,
    }
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# {WORK_ORDER}\n# cProfile pstats 转储｜档位={args.profile}｜"
                  f"{runs} 次查询｜生成时间 {payload['生成时间']}\n# "
                  f"替代口径：snakeviz/py-spy 未安装且不许 pip 安装，改用 pstats 文本 + 自绘 PNG\n\n")
        path.write_text(header + text, encoding="utf-8")
        print(f"[OK] pstats 文本 → {path}")
    if args.json_out:
        path = Path(args.json_out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 热点 JSON → {path}")

    print(f"\n=== Top-{args.top} 热点函数（cumtime 口径）===")
    print(f"{'#':>3} {'函数':<46} {'调用':>8} {'cumtime_ms':>12} {'tottime_ms':>12}")
    for row in by_cum:
        print(f"{row['rank']:>3} {row['function'][:46]:<46} {row['ncalls']:>8} "
              f"{row['cumtime_ms']:>12.2f} {row['tottime_ms']:>12.2f}")
    print(f"\n=== Top-{args.top} 热点函数（tottime 口径，前 10）===")
    for row in by_tot[:10]:
        print(f"{row['rank']:>3} {row['function'][:46]:<46} {row['ncalls']:>8} "
              f"{row['cumtime_ms']:>12.2f} {row['tottime_ms']:>12.2f}")
    print("\n=== 模块聚合（tottime 前 10）===")
    for row in modules[:10]:
        print(f"  {row['module']:<46} {row['tottime_ms']:>12.2f} ms  calls={row['calls']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
