# -*- coding: utf-8 -*-
"""工单13 采样剖析（``sys.setprofile`` 替代 py-spy）：``研发/scripts/采样剖析.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

**为什么不用 py-spy**：任务书建议的采样剖析器 ``py-spy`` 本机未安装；``_共享约定.md`` §2
明确「需要新依赖一律 --target 第三方库，不做裸 pip 安装」，而 ``py-spy`` 是 Rust 二进制
（不是纯 Python wheel），无法用 ``--target`` 方式旁装 ⇒ **不安装**，改用标准库替代口径。

**替代口径（本文件实现）**：``sys.setprofile`` 在每个 Python 函数 **调用/返回** 时触发回调，
在回调里记录「(当前栈帧函数) → 时间戳」，按 **两次事件之间的时间差** 归到「当时正在执行的函数」
头上，得到**概率式**（采样）热点分布。与 ``cProfile`` 的差异必须写进报告：

======================  ==========================  ==============================
项目                    cProfile（确定性）           本文件 sys.setprofile（采样）
======================  ==========================  ==============================
触发时机                每次函数调用                    每次函数调用/返回（记录时间差）
统计量                  调用次数 + 精确耗时             时间差归因（近似 CPU 时间）
小函数偏差              放大（纯 Python 小函数占比虚高）  接近真实占比
C 扩展/IO 等待          不计入（Ollama HTTP 等待不体现）  同样不体现（等待期无 Python 事件）
适用结论                调用次数、调用链                        CPU 热点占比、长尾函数
======================  ==========================  ==============================

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/采样剖析.py \\
        --profile optimized --rounds 2 --top 20 --json=研发/profiles/采样热点_优化后.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import FrameType
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


class ProfileSampler:
    """``sys.setprofile`` 采样器：把两次事件之间的时间差归给「当时栈顶的 Python 函数」。

    实现要点：
        * 只在**主线程**启用（``threading.setprofile`` 会污染子线程，本剖析只关心主流程）；
        * 用 ``time.perf_counter()`` 取时间差，累积到「函数 → 累计秒 + 命中次数」；
        * 单个时间差超过 ``max_slice_s``（默认 0.5 s）视为「进入了 C 扩展 / 网络等待」，
          归入 ``<C扩展或IO等待>`` 桶 —— **不把它算到任何 Python 函数头上**，
          这样得到的 Python 侧分布才有意义。
    """

    def __init__(self, *, max_slice_s: float = 0.5) -> None:
        self.max_slice_s = float(max_slice_s)
        self.hits: dict[str, dict[str, Any]] = {}
        self.last_ts: float | None = None
        self.last_func: str | None = None
        self.waits: dict[str, float] = {}
        self.enabled = False

    @staticmethod
    def _name(frame: FrameType, event: str) -> str:
        code = frame.f_code
        return f"{code.co_filename.split('app')[-1] if 'app' in code.co_filename else code.co_filename}" \
               f":{code.co_name}"

    def _hook(self, frame: FrameType, event: str, arg: Any) -> None:
        if not self.enabled or event not in {"call", "return", "c_call", "c_return"}:
            return
        now = time.perf_counter()
        if self.last_ts is not None and self.last_func is not None:
            delta = now - self.last_ts
            if delta >= 0.0:
                if delta > self.max_slice_s:
                    self.waits["<C扩展或IO等待>"] = self.waits.get("<C扩展或IO等待>", 0.0) + delta
                else:
                    item = self.hits.setdefault(self.last_func, {"hits": 0, "seconds": 0.0})
                    item["hits"] += 1
                    item["seconds"] += delta
        self.last_ts = now
        self.last_func = self._name(frame, event) if event in {"call", "c_call"} else self.last_func

    def start(self) -> None:
        self.last_ts = time.perf_counter()
        self.last_func = "<root>"
        self.enabled = True
        sys.setprofile(self._hook)

    def stop(self) -> None:
        self.enabled = False
        sys.setprofile(None)

    def report(self, top: int = 20) -> dict[str, Any]:
        total = sum(item["seconds"] for item in self.hits.values()) + sum(self.waits.values())
        rows = sorted(self.hits.items(), key=lambda kv: -kv[1]["seconds"])[:top]
        return {
            "采样总时长s": round(total, 4),
            "Python侧占比": round(sum(i["seconds"] for i in self.hits.values()) / total, 4) if total else 0.0,
            "等待占比": round(sum(self.waits.values()) / total, 4) if total else 0.0,
            "Top热点": [{"rank": i + 1, "function": name, "hits": item["hits"],
                        "seconds": round(item["seconds"], 6),
                        "ms": round(item["seconds"] * 1000, 3),
                        "占比": round(item["seconds"] / total, 4) if total else 0.0}
                       for i, (name, item) in enumerate(rows)],
            "等待项": {k: round(v, 4) for k, v in sorted(self.waits.items(), key=lambda kv: -kv[1])},
        }


def read_questions(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if str(row.get("question") or "").strip():
                rows.append(row)
    return rows[:limit] if limit and limit > 0 else rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="采样剖析.py",
                                     description=f"{WORK_ORDER} —— sys.setprofile 采样剖析（py-spy 替代口径）")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-slice", type=float, default=0.5, help="超过该秒数的单段归为 IO/等待")
    parser.add_argument("--profile", default="config", choices=["config", "baseline", "optimized"])
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--gap-s", type=float, default=0.2)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flags = perf_mod.apply_profile(args.profile)
    cfg = get_config()
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg, run_id=f"sample-{args.profile}")
    questions = read_questions(Path(args.questions), args.limit)
    if not questions:
        print(f"[参数错误] 测试问题集为空：{args.questions}", file=sys.stderr)
        return EXIT_PARAM
    try:
        storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    except Exception as exc:                       # noqa: BLE001
        log.log_event("sample.storage_failed", level="WARNING", error=str(exc))
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

    sampler = ProfileSampler(max_slice_s=float(args.max_slice))
    runs = 0
    total_ms = 0.0
    sampler.start()
    try:
        for _round in range(1, max(1, int(args.rounds)) + 1):
            for index, row in enumerate(questions, 1):
                t0 = time.perf_counter()
                engine.answer(str(row["question"]), lang=str(row.get("lang") or "auto"), use_cache=False)
                total_ms += (time.perf_counter() - t0) * 1000
                runs += 1
                if args.gap_s > 0 and index < len(questions):
                    sampler.stop()
                    time.sleep(float(args.gap_s))
                    sampler.start()
    finally:
        sampler.stop()
    report = sampler.report(int(args.top))
    payload = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "脚本": str(Path(__file__).resolve()),
        "档位": args.profile,
        "口径": {
            "剖析器": "sys.setprofile（标准库，事件时间差采样）",
            "为什么替代": "py-spy 未安装且为 Rust 二进制（无法用 pip --target 旁装），按 _共享约定.md §2 不安装；改本采样器",
            "与 cProfile 的差异": "cProfile 确定性计数（放大纯 Python 小函数）；本采样器按时间差归因，更接近真实 CPU 占比；两者都不体现 Ollama HTTP 等待",
            "单段阈值s": args.max_slice,
            "查询次数": runs,
            "缓存": "只写不读（use_cache=False）",
            "平均单次链路耗时ms": round(total_ms / max(runs, 1), 2),
        },
        "优化开关": {k: v for k, v in flags.items() if k != "adaptive_top_k_values"},
        "预热": warm,
        "采样结果": report,
    }
    if args.json_out:
        path = Path(args.json_out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 采样热点 → {path}")
    print(f"\n=== sys.setprofile 采样 Top-{args.top}（{runs} 次查询，"
          f"Python 侧占比 {report['Python侧占比'] * 100:.1f}%）===")
    for row in report["Top热点"]:
        print(f"{row['rank']:>3} {row['function'][:60]:<60} {row['ms']:>10.2f} ms  "
              f"{row['占比'] * 100:>6.2f}%  hits={row['hits']}")
    print("\n=== 等待/非 Python 段 ===")
    for key, value in report["等待项"].items():
        print(f"  {key}: {value:.3f} s")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
