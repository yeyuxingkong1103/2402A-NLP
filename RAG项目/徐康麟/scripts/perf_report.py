# -*- coding: utf-8 -*-
"""耗时评估报告：按 p95 排出「哪些函数/代码块在拖慢系统」——P0.5 的落地入口。

对应需求（用户口径）：「要有大量的日志输入，详细的输出函数前后、总体状态，可以打上时间戳，
来方便以后对于模块、函数等进行速度评估。」本脚本就是其中**"速度评估"那一步的入口**。

数据来源（三选一，自动判定）
---------------------------
1. ``--url http://127.0.0.1:8000/metrics`` —— 直接抓 **运行中服务** 的 ``/metrics``，
   解析 ``func_seconds`` 直方图。**最推荐**：拿到的是真实流量的耗时分布。
2. ``--log logs/app.log`` —— 解析日志里的 ``METRIC {json}`` 行（由 ``metrics.emit_metrics``
   写出，``METRICS_LOG_ENABLED=true`` 时默认开）。**离线可用**：把生产日志拷回来就能复盘。
3. ``--json snapshot.json`` —— 读任意一份 ``MetricsSnapshot`` 的 JSON（``to_json()`` 产物）。

为什么不能"直接 import 后读注册表"
----------------------------------
耗时数据是**进程内内存态**（``metrics.MetricRegistry``），新起一个 Python 进程读到的
一定是空表。所以本脚本刻意只做"从外部来源读"，不假装能读别人的内存。

用法示例
--------
::

    # 看本机服务最慢的 20 个函数（默认按 p95 排）
    python scripts/perf_report.py --url http://127.0.0.1:8000/metrics --top 20

    # 离线复盘一份生产日志，只看样本数 >= 30 的
    python scripts/perf_report.py --log logs/app.log --min-count 30

    # 导出成 markdown 表格贴进报告
    python scripts/perf_report.py --log logs/app.log --format md

配套的环境变量（都写在 ``legal_rag/observability.py`` 顶部）
----------------------------------------------------------
* ``TRACE_LEVEL``      —— L3 全量函数前后日志的级别（``DEBUG`` 看得最全；不设则回落 ``LOG_LEVEL``）
* ``SLOW_CALL_MS``     —— L2 慢调用阈值（默认 500ms）；``[SLOW]`` 行只在超过它时出现
* ``SLOW_CALL_SAMPLE`` —— L2 日志采样率（默认 1.0；**不影响** L1 直方图）
* ``TRACE_SAMPLE``     —— L3 前后日志采样率（默认 1.0）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 目标指标名（由 ``observability.timed`` / ``timed_scope`` 采集）
FUNC_METRIC = "func_seconds"

#: 解析 ``METRIC {json}`` 日志行用的前缀
_LOG_PREFIX = "METRIC "

#: 解析 Prometheus 文本里的直方图：``func_seconds_bucket{func="x",le="0.01"} 3``
_PROM_BUCKET_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{(?P<labels>.*)\}\s+(?P<value>[0-9.eE+-]+)$'
)
_PROM_COUNT_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_count\{(?P<labels>.*)\}\s+(?P<value>[0-9.eE+-]+)$'
)
_PROM_SUM_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_sum\{(?P<labels>.*)\}\s+(?P<value>[0-9.eE+-]+)$'
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def setup_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文输出会乱码 —— 与 ``tests/run_tests.py`` 同一处理。"""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# --------------------------------------------------------------------------
# 来源 1：Prometheus 文本
# --------------------------------------------------------------------------

def parse_prometheus(text: str, metric: str = FUNC_METRIC) -> dict[str, dict]:
    """把 Prometheus 文本里的直方图解析成 ``{标签串: {count, sum, p50, p95, buckets}}``。

    ``p50``/``p95`` 由**桶的线性插值**估计（Prometheus 的 ``histogram_quantile`` 同口径），
    因为直方图只保留桶累计值、不保留原始样本。
    """
    series: dict[str, dict] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for regex, kind in ((_PROM_BUCKET_RE, "bucket"), (_PROM_COUNT_RE, "count"),
                            (_PROM_SUM_RE, "sum")):
            match = regex.match(line)
            if not match:
                continue
            if match.group("name") != metric:
                break
            labels = dict(_LABEL_RE.findall(match.group("labels")))
            le = labels.pop("le", None)
            key = _label_key(labels)
            entry = series.setdefault(key, {"labels": labels, "buckets": [], "count": 0.0,
                                            "sum": 0.0})
            value = float(match.group("value"))
            if kind == "bucket" and le is not None:
                if le != "+Inf":
                    entry["buckets"].append((float(le), value))
            elif kind == "count":
                entry["count"] += value
            elif kind == "sum":
                entry["sum"] += value
            break
    for entry in series.values():
        entry["buckets"].sort()
        entry["p50"] = _bucket_quantile(entry, 0.50)
        entry["p95"] = _bucket_quantile(entry, 0.95)
        entry["avg"] = (entry["sum"] / entry["count"]) if entry["count"] else 0.0
    return series


def _bucket_quantile(entry: dict, q: float) -> float:
    """按 Prometheus ``histogram_quantile`` 的线性插值口径估算分位数。

    ⚠️ **必须理解的边界**：直方图只保留桶累计值，不保留原始样本，所以
    Prometheus 侧的分位数**永远是近似值**，精度受分桶密度限制：

    * 分布跨越多个桶时，线性插值的误差在"单桶宽度"内，够用；
    * 分布**全部落在第一个桶**内时（典型例子：12µs 的纯计算函数，首桶 10µs 起），
      插值只能给出"≤ 首桶上界"的值，属于**上限**而非真值。

    ⇒ 需要**精确**分位数时请看内存快照（``MetricsSnapshot`` / ``METRIC {json}``
    日志行，含真实 ``p50``/``p95``），本函数只负责从 ``/metrics`` 这样的外部来源估计。
    """
    total = float(entry.get("count") or 0.0)
    buckets: list[tuple[float, float]] = entry.get("buckets") or []
    if not total or not buckets:
        return 0.0
    rank = q * total
    prev_bound, prev_cum = 0.0, 0.0
    for bound, cum in buckets:
        if cum >= rank:
            if cum == prev_cum:
                return bound
            ratio = (rank - prev_cum) / (cum - prev_cum)
            return prev_bound + (bound - prev_bound) * ratio
        prev_bound, prev_cum = bound, cum
    return buckets[-1][0]


def _label_key(labels: dict) -> str:
    """标签字典 → 稳定字符串键（排序后拼接，便于比对与打印）。"""
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


# --------------------------------------------------------------------------
# 来源 2：日志里的 METRIC {json} 行
# --------------------------------------------------------------------------

def parse_log_file(path: str | Path, metric: str = FUNC_METRIC) -> dict[str, dict]:
    """扫描日志，取**最后一条**含目标指标的 ``METRIC {json}`` 行。

    取最后一条的理由：``MetricsSnapshot`` 是**进程累计**快照（``metrics.py`` 的
    「累计 vs 存量」口径），越晚的快照覆盖的样本越多。
    """
    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            index = line.find(_LOG_PREFIX)
            if index < 0:
                continue
            payload = line[index + len(_LOG_PREFIX):].strip()
            if not payload.startswith("{"):
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            metrics = data.get("metrics") or {}
            target = metrics.get(metric)
            if not isinstance(target, dict):
                continue
            series: dict[str, dict] = {}
            for sample in target.get("samples") or []:
                labels = sample.get("labels") or {}
                key = _label_key(labels)
                series[key] = {
                    "labels": labels,
                    "count": float(sample.get("count") or 0.0),
                    "sum": float(sample.get("sum") or 0.0),
                    "p50": float(sample.get("p50") or 0.0),
                    "p95": float(sample.get("p95") or 0.0),
                    "avg": float(sample.get("avg") or 0.0),
                    "buckets": [(float(b), float(c)) for b, c in (sample.get("buckets") or [])],
                }
            if series:
                latest = series
    return latest


# --------------------------------------------------------------------------
# 来源 3：MetricsSnapshot JSON
# --------------------------------------------------------------------------

def parse_snapshot_json(path: str | Path, metric: str = FUNC_METRIC) -> dict[str, dict]:
    """读 ``MetricsSnapshot.to_json()`` 的产物（``{"ts":..., "metrics":{...}}``）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    target = (data.get("metrics") or {}).get(metric)
    if not isinstance(target, dict):
        return {}
    series: dict[str, dict] = {}
    for sample in target.get("samples") or []:
        labels = sample.get("labels") or {}
        series[_label_key(labels)] = {
            "labels": labels,
            "count": float(sample.get("count") or 0.0),
            "sum": float(sample.get("sum") or 0.0),
            "p50": float(sample.get("p50") or 0.0),
            "p95": float(sample.get("p95") or 0.0),
            "avg": float(sample.get("avg") or 0.0),
            "buckets": [(float(b), float(c)) for b, c in (sample.get("buckets") or [])],
        }
    return series


def fetch_metrics(url: str, timeout: float = 10.0) -> str:
    """抓 ``/metrics`` 文本。失败时抛出带可读原因的异常（不静默返回空表）。"""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - 用户显式传入
        return response.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# 汇总与输出
# --------------------------------------------------------------------------

def _fmt_ms(seconds: float) -> str:
    """秒 → 便于人读的耗时串（<1s 用 ms，否则用 s）。"""
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def rank_series(series: dict[str, dict], *, sort_by: str = "p95",
                min_count: float = 1.0) -> list[dict]:
    """把序列按指定字段降序排，滤掉样本数不足的（噪声）。"""
    rows: list[dict] = []
    for key, entry in series.items():
        if float(entry.get("count") or 0.0) < min_count:
            continue
        rows.append({
            "key": key,
            "func": (entry.get("labels") or {}).get("func", key),
            "count": float(entry.get("count") or 0.0),
            "avg": float(entry.get("avg") or 0.0),
            "p50": float(entry.get("p50") or 0.0),
            "p95": float(entry.get("p95") or 0.0),
        })
    rows.sort(key=lambda r: (-r.get(sort_by, 0.0), r["func"]))
    return rows


def render_table(rows: list[dict], *, sort_by: str = "p95", top: int = 20,
                 fmt: str = "text") -> str:
    """渲染成 text / md 表格。

    列固定为「次数 / avg / p50 / p95」，``sort_by`` **只决定排序、不改列名**——
    否则按 ``count`` 排序时表头会出现一个叫 ``count`` 的耗时列，容易误读。
    """
    shown = rows[:top] if top > 0 else rows
    if not shown:
        return (f"（没有 {FUNC_METRIC} 样本：确认服务已经跑过流量，"
                f"且已用 @timed / timed_scope 装饰目标函数）")
    if fmt == "md":
        out = ["| 函数 | 次数 | avg | p50 | p95 |", "|---|---:|---:|---:|---:|"]
        out += [f"| `{r['func']}` | {int(r['count'])} | {_fmt_ms(r['avg'])} | "
                f"{_fmt_ms(r['p50'])} | {_fmt_ms(r['p95'])} |" for r in shown]
        return "\n".join(out)
    width = max(len(r["func"]) for r in shown)
    header = f"{'函数'.ljust(width)}  {'次数':>8}  {'avg':>10}  {'p50':>10}  {'p95':>10}"
    lines = [header, "-" * len(header)]
    for r in shown:
        lines.append(f"{r['func'].ljust(width)}  {int(r['count']):>8}  "
                     f"{_fmt_ms(r['avg']):>10}  {_fmt_ms(r['p50']):>10}  {_fmt_ms(r['p95']):>10}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 p95 排出最慢的函数/代码块（数据来自 /metrics、日志或快照 JSON）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", default="", help="运行中服务的 /metrics 地址（推荐）")
    source.add_argument("--log", default="", help="日志文件路径（解析 METRIC {json} 行）")
    source.add_argument("--json", dest="json_path", default="", help="MetricsSnapshot JSON 路径")
    parser.add_argument("--metric", default=FUNC_METRIC, help=f"指标名（默认 {FUNC_METRIC}）")
    parser.add_argument("--top", type=int, default=20, help="显示前 N 个（默认 20；0=全部）")
    parser.add_argument("--sort-by", default="p95", choices=("p95", "p50", "avg", "count"),
                        help="排序字段（默认 p95）")
    parser.add_argument("--min-count", type=float, default=1.0,
                        help="样本数下限，滤掉噪声（默认 1）")
    parser.add_argument("--format", default="text", choices=("text", "md"), help="输出格式")
    parser.add_argument("--timeout", type=float, default=10.0, help="抓 /metrics 的超时秒数")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_utf8_stdout()
    args = build_parser().parse_args(argv)

    try:
        if args.url:
            text = fetch_metrics(args.url, timeout=args.timeout)
            series = parse_prometheus(text, metric=args.metric)
            origin = f"/metrics @ {args.url}"
        elif args.log:
            series = parse_log_file(args.log, metric=args.metric)
            origin = f"日志 {args.log}"
        elif args.json_path:
            series = parse_snapshot_json(args.json_path, metric=args.metric)
            origin = f"快照 {args.json_path}"
        else:
            print("请指定数据来源：--url / --log / --json（三者选一）", file=sys.stderr)
            return 2
    except FileNotFoundError as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI 顶层要给人读的原因
        print(f"读取失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    rows = rank_series(series, sort_by=args.sort_by, min_count=args.min_count)
    print(f"# 耗时评估（来源：{origin}；指标：{args.metric}；排序：{args.sort_by}）")
    print(f"# 序列数 {len(series)}，满足样本下限 {args.min_count:g} 的 {len(rows)} 个")
    if args.url:
        # 直方图只有桶累计值 ⇒ 分位数为插值近似；日志/快照来源带真实 p50/p95。
        # 实测量级（见 docs/REFACTOR-PLAN.md §3.7）：分布跨多个桶时误差约 20%，
        # 全部落在首桶内时误差可达数倍 —— 所以这里明确提示，避免把近似值当真值。
        # 注意：不用 emoji（⚠️ 等）—— 本项目的 GBK 控制台会把它们打成乱码，
        # 约定写 `[注意]`，回归锁见 tests/test_console_encoding.py。
        print("# [注意] 该来源为**桶插值近似**（Prometheus 直方图不含原始样本）："
              "实测跨桶分布误差约 20%，全部落在首桶时更大")
        print("#        需要精确分位数请用 --log / --json（那两者带真实 p50/p95）")
    print()
    print(render_table(rows, sort_by=args.sort_by, top=args.top, fmt=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
