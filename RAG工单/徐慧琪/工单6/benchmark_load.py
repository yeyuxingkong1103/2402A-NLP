# 工单编号：人工智能NLP-RAG-混合检索任务
# -*- coding: utf-8 -*-
"""并发压测：10 / 50 / 100 并发档位下的响应时间、错误率、吞吐与资源占用。

用途
====
本工单要求混合检索系统「端到端 <=3 秒」并给出并发能力证据。本脚本**直接调用
Python 函数** ``rag.ask(question, use_cache=False)``（不经过 FastAPI/HTTP，与
线上接口共用同一条 rag 代码路径），按 ``--levels`` 指定的并发档位逐档施压，
统计 P50/P90/P95/mean/max 时延、错误率、<=3 秒占比、吞吐（req/s），并记录
进程 RSS 与系统内存的峰值/均值；CUDA 可用时另记显存峰值。

与上游工单4 benchmarks/loadtest.py 的关系：借鉴其分档结构与统计口径
（线性插值分位数、失败请求不参与 <=3 秒达标计数），但**不走 HTTP** ——
本工单不要求起服务，直接调用函数更贴近真实链路且无网络开销。

用法
====
    python benchmark_load.py                                  # 默认 10,50,100 并发
    python benchmark_load.py --levels 10,50 --per-level 20
    python benchmark_load.py --strategy vector --no-warmup
    python benchmark_load.py --out data/eval/loadtest_result.json

输出：JSON 结果写入 ``--out``；中文 markdown 摘要打印到 stdout。

口径说明
========
- 响应时间 = ``time.perf_counter()`` 包住一次 ``rag.ask`` 的墙钟（毫秒）。
- 分位数样本**含失败请求**（快速失败会拉低分位数，不能单独当成功耗时读）；
  <=3 秒达标的分子只数**成功且 <=3000ms** 的请求，失败请求无答案产出、不计达标。
- 资源占用由后台采样线程每 0.2s 采一次；显存峰值取
  ``torch.cuda.max_memory_allocated()``，CUDA 不可用时如实标注「未测得」，
  绝不填 0 冒充。
- 预热失败不中断压测：以 ``warmup.ok=false`` 记入结果，首题冷启动会体现在
  第一档的时延里，读报告时需知情。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import bootstrap  # noqa: F401,E402  —— 必须最先导入（离线环境 + 大栈）

WORKORDER = "人工智能NLP-RAG-混合检索任务"
BUDGET_MS = 3000.0                     # 工单硬指标：端到端 <=3 秒
DEFAULT_QUESTIONS = ROOT / "data" / "test_questions.json"
DEFAULT_OUT = ROOT / "data" / "eval" / "loadtest_result.json"
SAMPLE_INTERVAL_S = 0.2                # 资源采样周期


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def _percentile(data: list[float], p: float) -> float:
    """线性插值分位数（与工单4 loadtest 同口径）；空列表返回 0.0。"""
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _mean(data: list[float]) -> float:
    return float(statistics.mean(data)) if data else 0.0


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> list[dict]:
    """读题库；每一项需含 ``id`` 与 ``question``。空/损坏直接抛错（不静默跑空压测）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"题库格式应为 JSON 数组：{path}")
    questions: list[dict] = []
    for i, item in enumerate(raw):
        if isinstance(item, str) and item.strip():
            questions.append({"id": i + 1, "question": item.strip()})
        elif isinstance(item, dict) and str(item.get("question", "")).strip():
            questions.append({"id": item.get("id", i + 1),
                              "question": str(item["question"]).strip()})
    if not questions:
        raise ValueError(f"题库为空或无有效问题：{path}")
    return questions


# ---------------------------------------------------------------------------
# 运行环境与显存（测不到就如实标注，不编造）
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 —— torch 缺失/CUDA 不可用
        return False


def _cuda_reset_peak() -> bool:
    """压测前清零显存峰值计数；失败返回 False（不影响压测继续）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


def _cuda_peak_mb() -> float | None:
    """本进程 CUDA 峰值显存（MB）；测不到返回 None（不填 0 冒充）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return None


def collect_env() -> dict:
    """运行环境快照（os / CPU / 内存 / GPU）。"""
    env: dict = {
        "workorder": WORKORDER,
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": None,
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_note": None,
    }
    try:
        import psutil
        env["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 * 1024), 1)
    except Exception as exc:  # noqa: BLE001
        env["ram_note"] = f"psutil 不可用，内存未测得：{type(exc).__name__}"
    env["cuda_available"] = _cuda_available()
    if env["cuda_available"]:
        try:
            import torch
            env["torch"] = torch.__version__
            env["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            env["gpu_note"] = f"torch 查询失败：{type(exc).__name__}: {exc}"
    else:
        env["gpu_note"] = "CUDA 不可用或未安装 torch，显存占用未测得"
    return env


class ResourceSampler:
    """后台线程轮询进程 RSS 与系统内存，给出均值/峰值。

    采样线程只读 psutil，不影响压测线程；psutil 不可用时 ``available=False``，
    汇总结果里如实标注「未测得」。
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self._interval = max(0.01, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss: list[float] = []
        self._sys_used: list[float] = []
        self._sys_pct: list[float] = []
        self._lock = threading.Lock()
        self.available = False
        self._psutil = None
        self._proc = None

    def _resolve(self) -> None:
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self.available = True
        except Exception:  # noqa: BLE001
            self.available = False

    def _sample(self) -> None:
        if not self.available:
            return
        try:
            rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            vm = self._psutil.virtual_memory()
            with self._lock:
                self._rss.append(float(rss_mb))
                self._sys_used.append(float(vm.used) / (1024 * 1024))
                self._sys_pct.append(float(vm.percent))
        except Exception:  # noqa: BLE001 —— 单次采样失败不致命
            pass

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> None:
        self._resolve()
        self._sample()
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, name="resource-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """停止采样并返回汇总（mean/peak）；采不到时全部 None。"""
        if self.available:
            self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            rss, used, pct = list(self._rss), list(self._sys_used), list(self._sys_pct)
        if not self.available or not rss:
            return {"measured": False, "rss_mb_mean": None, "rss_mb_peak": None,
                    "sys_mem_used_mb_peak": None, "sys_mem_percent_mean": None,
                    "sys_mem_percent_peak": None, "n_samples": 0,
                    "note": "psutil 不可用，资源占用未测得"}
        return {
            "measured": True,
            "rss_mb_mean": round(_mean(rss), 1),
            "rss_mb_peak": round(max(rss), 1),
            "sys_mem_used_mb_peak": round(max(used), 1),
            "sys_mem_percent_mean": round(_mean(pct), 1),
            "sys_mem_percent_peak": round(max(pct), 1),
            "n_samples": len(rss),
            "note": None,
        }


# ---------------------------------------------------------------------------
# 单请求与分档执行
# ---------------------------------------------------------------------------
@dataclass
class RequestRecord:
    """单次 ``rag.ask`` 的原始样本（每题明细，写入 JSON）。"""

    seq: int                     # 提交序号（稳定排序用）
    question_id: object
    question: str
    ok: bool
    latency_ms: float            # 墙钟，成功/失败都记
    error_type: str = ""         # 异常类型名，或 rag 内部降级失败标记
    mode: str = ""               # RagResult.mode
    total_ms: float | None = None  # RagResult.timings["total_ms"]（自报口径，仅对照）
    answer_chars: int = 0
    from_cache: bool = False


def _one_request(seq: int, item: dict, use_cache: bool = False) -> RequestRecord:
    """执行一次提问。任何异常都在此捕获并计入错误率，绝不让压测整体中断。

    通过 ``rag.ask`` **属性查找**调用（而非 from ... import ask），
    保证测试里的 monkeypatch 生效、且不误用缓存。
    """
    from src import rag  # 延迟导入：import 本模块不触发任何模型加载

    question = item["question"]
    t0 = time.perf_counter()
    try:
        result = rag.ask(question, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 —— 单请求异常计入错误率
        return RequestRecord(seq=seq, question_id=item.get("id"), question=question,
                             ok=False, latency_ms=(time.perf_counter() - t0) * 1000.0,
                             error_type=type(exc).__name__)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    timings = getattr(result, "timings", None) or {}
    mode = getattr(result, "mode", "") or ""
    ok = mode != "error"          # rag 内部检索失败会以 mode="error" 返回而非抛出
    return RequestRecord(
        seq=seq, question_id=item.get("id"), question=question, ok=ok,
        latency_ms=latency_ms,
        error_type="" if ok else "rag_error",
        mode=mode,
        total_ms=timings.get("total_ms") if isinstance(timings, dict) else None,
        answer_chars=len(getattr(result, "answer", "") or ""),
        from_cache=bool(getattr(result, "from_cache", False)),
    )


def summarize_level(concurrency: int, records: list[RequestRecord], elapsed_s: float,
                    resources: dict | None = None, gpu_peak_mb: float | None = None,
                    gpu_note: str | None = None) -> dict:
    """把原始样本汇总成该档的统计字典（纯函数，便于单测）。"""
    n = len(records)
    ok_records = [r for r in records if r.ok]
    errors = n - len(ok_records)
    lat_all = [r.latency_ms for r in records]
    error_kinds: dict[str, int] = {}
    for r in records:
        if not r.ok:
            kind = r.error_type or "unknown"
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
    # 达标分子：成功且 <= 3s；分母为该档总请求数（失败请求无答案，不算达标）
    within_3s = sum(1 for r in ok_records if r.latency_ms <= BUDGET_MS)
    error_samples = [
        {"seq": r.seq, "question_id": r.question_id,
         "question": r.question[:80], "error_type": r.error_type,
         "latency_ms": round(r.latency_ms, 1), "mode": r.mode}
        for r in records if not r.ok
    ][:5]
    return {
        "concurrency": int(concurrency),
        "n_requests": n,
        "ok": len(ok_records),
        "errors": errors,
        "error_rate": round(errors / n, 4) if n else 0.0,
        "elapsed_s": round(elapsed_s, 3),
        "qps": round(n / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "p50_ms": round(_percentile(lat_all, 50), 1),
        "p90_ms": round(_percentile(lat_all, 90), 1),
        "p95_ms": round(_percentile(lat_all, 95), 1),
        "mean_ms": round(_mean(lat_all), 1),
        "max_ms": round(max(lat_all), 1) if lat_all else 0.0,
        "budget_ms": BUDGET_MS,
        "within_3s": within_3s,
        "within_3s_rate": round(within_3s / n, 4) if n else 0.0,
        "error_kinds": error_kinds,
        "error_samples": error_samples,
        "resources": dict(resources or {}),
        "gpu_peak_mb": gpu_peak_mb,
        "gpu_note": gpu_note,
        "requests": [asdict(r) for r in sorted(records, key=lambda r: r.seq)],
    }


def run_level(concurrency: int, per_level: int, questions: list[dict]) -> dict:
    """在指定并发下施压 per_level 个请求（线程池 = 并发数，题目循环取）。

    每档前清零显存峰值、起资源采样线程；结束后汇总。
    """
    if not questions:
        raise ValueError("questions 不能为空：压测必须带真实问题")
    concurrency = max(1, int(concurrency))
    n = max(1, int(per_level))
    tasks = [(i, questions[i % len(questions)]) for i in range(n)]

    gpu_used = _cuda_available()
    if gpu_used:
        _cuda_reset_peak()

    sampler = ResourceSampler()
    sampler.start()

    records: list[RequestRecord] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency,
                            thread_name_prefix="load") as pool:
        futures = [pool.submit(_one_request, seq, item) for seq, item in tasks]
        for fut in as_completed(futures):
            records.append(fut.result())
    elapsed_s = time.perf_counter() - t0

    resources = sampler.stop()
    gpu_peak = _cuda_peak_mb() if gpu_used else None
    gpu_note = None
    if not gpu_used:
        gpu_note = "CUDA 不可用，显存峰值未测得"
    elif gpu_peak is None:
        gpu_note = "torch 查询显存失败，显存峰值未测得"
    return summarize_level(concurrency, records, elapsed_s, resources,
                           gpu_peak_mb=gpu_peak, gpu_note=gpu_note)


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def _safe_print(text: str = "") -> None:
    """打印中文时容错 GBK 控制台（编码不支持时用 replace，避免脚本中途崩溃）。"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(text.encode(enc, errors="replace") + b"\n")
        sys.stdout.buffer.flush()


def _fmt(value, nd: int = 1, dash: str = "未测得") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def build_markdown(payload: dict) -> str:
    """中文 markdown 摘要：方案、分档结果、错误类型、资源占用。"""
    env = payload.get("os", {})
    lines = [
        "# RAG 混合检索并发压测摘要",
        "",
        f"- 工单编号：{payload.get('workorder', WORKORDER)}",
        f"- 生成时间：{payload.get('generated_at', '')}",
        f"- 运行环境：{env.get('os', '-')}｜Python {env.get('python', '-')}"
        f"｜CPU {env.get('cpu_count', '-')} 核｜内存 "
        f"{_fmt(env.get('ram_total_mb'), 0)} MB",
        f"- 推理设备：{'CUDA（' + str(env.get('gpu_name')) + '）' if env.get('cuda_available') else 'CPU（CUDA 不可用，显存峰值未测得）'}",
        f"- 检索策略：{payload.get('runtime', {}).get('strategy', '-')}"
        f"（{payload.get('runtime', {}).get('strategy_label', '-')}）"
        f"｜融合算法：{payload.get('runtime', {}).get('fusion', '-')}",
        f"- 预热：{_warmup_line(payload.get('warmup', {}))}",
        f"- 时延口径：客户端墙钟（一次 rag.ask 全程，含排队）；分位数样本含失败请求，"
        f"≤{BUDGET_MS:g}ms 达标只数成功的请求。",
        "",
        "## 分档结果",
        "",
        "| 并发 | 请求数 | 成功 | 失败 | 错误率 | P50(ms) | P90(ms) | P95(ms) "
        "| 均值(ms) | 最大(ms) | ≤3s 占比 | 吞吐(req/s) | RSS均值(MB) | RSS峰值(MB) "
        "| 系统内存峰值(%) | 显存峰值(MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- |",
    ]
    for lv in payload.get("levels", []):
        res = lv.get("resources", {}) or {}
        gpu = lv.get("gpu_peak_mb")
        lines.append(
            f"| {lv['concurrency']} | {lv['n_requests']} | {lv['ok']} | {lv['errors']} "
            f"| {lv['error_rate']:.2%} | {lv['p50_ms']:.1f} | {lv['p90_ms']:.1f} "
            f"| {lv['p95_ms']:.1f} | {lv['mean_ms']:.1f} | {lv['max_ms']:.1f} "
            f"| {lv['within_3s_rate']:.2%} | {lv['qps']:.2f} "
            f"| {_fmt(res.get('rss_mb_mean'))} | {_fmt(res.get('rss_mb_peak'))} "
            f"| {_fmt(res.get('sys_mem_percent_peak'))} | {_fmt(gpu)} |"
        )
    lines += ["", "### 错误类型样例", ""]
    any_error = False
    for lv in payload.get("levels", []):
        if lv.get("error_kinds"):
            any_error = True
            kinds = "、".join(f"{k}×{v}" for k, v in sorted(lv["error_kinds"].items()))
            lines.append(f"- 并发 {lv['concurrency']}：{kinds}")
    if not any_error:
        lines.append("- 无")
    notes: list[str] = []
    for lv in payload.get("levels", []):
        res = lv.get("resources", {}) or {}
        if res and not res.get("measured", True):
            notes.append(f"- 并发 {lv['concurrency']}：资源占用未测得"
                         f"（{res.get('note', '原因未知')}）")
        if lv.get("gpu_note"):
            notes.append(f"- 并发 {lv['concurrency']}：{lv['gpu_note']}")
    if payload.get("notes"):
        notes += [f"- {n}" for n in payload["notes"]]
    if notes:
        lines += ["", "### 备注", ""] + notes
    lines += ["", f"> JSON 明细：{payload.get('out_path', '-')}", ""]
    return "\n".join(lines)


def _warmup_line(warmup: dict) -> str:
    if not warmup.get("enabled", False):
        return "未启用（--no-warmup，首题含冷启动）"
    if warmup.get("ok"):
        detail = warmup.get("result") or {}
        failed = {k: v for k, v in detail.items() if isinstance(v, str)}
        parts = "、".join(f"{k}={v}" for k, v in detail.items() if k != "elapsed_s")
        return (f"成功（{parts}，耗时 {detail.get('elapsed_s', '-')}s）"
                + (f"；注意有子项失败：{failed}" if failed else ""))
    return f"失败（{warmup.get('error', '未知错误')}）—— 已按要求不中断压测"


def write_json(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def _parse_levels(raw: str) -> list[int]:
    try:
        levels = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--levels 需为逗号分隔整数，如 10,50,100：{raw}") from exc
    if not levels or any(v <= 0 for v in levels):
        raise argparse.ArgumentTypeError(f"--levels 必须为正整数：{raw}")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_load.py",
        description="RAG 混合检索并发压测（直接调用 rag.ask，分档统计时延/错误率/资源占用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--levels", type=_parse_levels, default=[10, 50, 100],
                        help="并发档位，逗号分隔，如 10,50,100")
    parser.add_argument("--per-level", type=int, default=20,
                        help="每个档位的总请求数（题目从 data/test_questions.json 循环取）")
    parser.add_argument("--strategy", default="hybrid",
                        help="检索策略（vector/fulltext/hybrid，经 runtime.update 切换）")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="结果 JSON 输出路径")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True,
                        help="压测前预热全部模型（默认开启）")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="跳过预热（冒烟测试/快速验证用）")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                        help=argparse.SUPPRESS)   # 题库路径，一般无需改
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src import config, runtime

    if args.strategy not in config.RETRIEVAL_STRATEGIES:
        _safe_print(f"[压测] 未知检索策略 {args.strategy!r}，可选："
                    f"{sorted(config.RETRIEVAL_STRATEGIES)}")
        return 2
    if args.per_level <= 0:
        _safe_print("[压测] --per-level 必须为正整数")
        return 2

    try:
        questions = load_questions(args.questions)
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"[压测] 题库加载失败：{type(exc).__name__}: {exc}")
        return 2
    levels = sorted(set(args.levels))

    try:
        runtime.update(strategy=args.strategy)
        runtime_info = runtime.describe()
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"[压测] 策略切换失败：{type(exc).__name__}: {exc}")
        return 2

    warmup_info: dict = {"enabled": bool(args.warmup), "ok": None, "result": None,
                         "error": None}
    if args.warmup:
        _safe_print("[压测] 预热模型（首次约 30-40s）…")
        try:
            from src import rag
            warmup_info["result"] = rag.warmup()
            warmup_info["ok"] = True
        except Exception as exc:  # noqa: BLE001 —— 预热失败不中断压测
            warmup_info["ok"] = False
            warmup_info["error"] = f"{type(exc).__name__}: {exc}"
            _safe_print(f"[压测] 预热失败（不中断）：{warmup_info['error']}")
        else:
            _safe_print(f"[压测] 预热完成：{warmup_info['result']}")

    payload: dict = {
        "workorder": WORKORDER,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "os": collect_env(),
        "args": {"levels": levels, "per_level": args.per_level,
                 "strategy": args.strategy, "out": str(args.out),
                 "warmup": bool(args.warmup), "questions": str(args.questions)},
        "runtime": runtime_info,
        "warmup": warmup_info,
        "levels": [],
        "notes": [],
    }
    if not args.warmup:
        payload["notes"].append("--no-warmup：未预热，首题冷启动会抬高第一档时延。")
    if args.per_level < max(levels):
        payload["notes"].append(
            f"--per-level={args.per_level} < 最大并发 {max(levels)}："
            f"最大档的实际在飞并发受请求数限制，QPS 不能代表满并发能力。")

    for concurrency in levels:
        _safe_print(f"[压测] 并发 {concurrency}：{args.per_level} 个请求 …")
        level = run_level(concurrency, args.per_level, questions)
        payload["levels"].append(level)
        _safe_print(json.dumps({k: v for k, v in level.items() if k != "requests"},
                               ensure_ascii=False))

    out_path = Path(args.out)
    payload["out_path"] = str(out_path)
    write_json(payload, out_path)
    _safe_print()
    _safe_print(build_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(bootstrap.run_with_large_stack(main))
