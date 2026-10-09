# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""分阶段计时埋点：请求 ID + 阶段出入口时间戳 + 结构化落盘

工单把「日志记录」列为识别瓶颈的手段之一，并明确要求：

  · 在每个主要处理阶段和子阶段的进入和退出处，实施带有详细时间戳的结构化日志记录
  · 包含标识符（例如，请求 ID）以追踪单个请求在流程中的执行情况
  · 记录相关指标，如检索到的文档数量、上下文长度和生成的 token

这三条分别对应本模块的 `Trace.stage()` / `Trace.id` / `Trace.metric()`。

**为什么不用分布式追踪（OpenTelemetry/Jaeger）**：工单把它列为手段之一，但那一节
写的是「对于构建为微服务集合的 RAG 系统」。本系统是单体进程，请求不出进程边界，
跨服务链路追踪没有用武之地 —— 进程内阶段计时就能回答「时间花在哪」。这个取舍
写在 `设计/设计说明.md` 里。

**开销**：`time.perf_counter()` 是纳秒级计数器，埋点本身的开销远小于被测量的阶段
（毫秒到秒级），不会扭曲结论。这也是为什么没有用 cProfile 的绝对耗时下结论 ——
cProfile 会给每个函数调用插桩，开销可达数倍（见 优化/过程问题记录.md）。
"""
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE.parent / "测试" / "results" / "perf_log.jsonl"

_local = threading.local()


def current():
    """取当前线程正在记录的 Trace，没有就返回 None。"""
    return getattr(_local, "trace", None)


def new_id():
    return uuid.uuid4().hex[:12]


class Trace:
    """一次请求的全过程记录。"""

    def __init__(self, request_id=None, question=""):
        self.id = request_id or new_id()
        self.question = question
        self.spans = []          # [{"stage":..., "seconds":..., "depth":...}]
        self.metrics = {}        # 召回数 / 上下文长度 / token 数 …
        self.t0 = time.perf_counter()
        self._stack = []

    @contextmanager
    def stage(self, name):
        """给一个阶段计时。支持嵌套 —— 嵌套深度会记下来，便于区分
        「上下文组装」和它内部的子阶段（比如逐块截断）。"""
        depth = len(self._stack)
        self._stack.append(name)
        t = time.perf_counter()
        try:
            yield
        finally:
            self._stack.pop()
            self.spans.append({"stage": name, "depth": depth,
                               "seconds": round(time.perf_counter() - t, 4)})

    def metric(self, **kw):
        """记录阶段产出（召回文档数、上下文长度、token 数…）。"""
        self.metrics.update(kw)

    def seconds_of(self, stage):
        """某阶段的总耗时（同名阶段出现多次时累加）。"""
        return round(sum(s["seconds"] for s in self.spans if s["stage"] == stage), 4)

    def finish(self, status="ok"):
        total = round(time.perf_counter() - self.t0, 4)
        rec = {"request_id": self.id, "question": self.question[:60],
               "status": status, "total_seconds": total,
               "spans": self.spans, "metrics": self.metrics}
        # 未归入任何子阶段的时间 —— 「时间花在哪」最关键的一项：
        # 各阶段加起来远小于总耗时，说明还有没埋到的地方
        rec["unaccounted_seconds"] = round(
            total - sum(s["seconds"] for s in self.spans if s["depth"] == 0), 4)
        return rec


@contextmanager
def trace(question="", request_id=None):
    """在入口处包一层：自动把记录写进 perf_log.jsonl 并恢复上一个 trace。"""
    parent = current()
    t = Trace(request_id, question)
    _local.trace = t
    try:
        yield t
    finally:
        _local.trace = parent


@contextmanager
def stage(name):
    """阶段计时。没有活跃 trace 时是空操作（比如被别的脚本单独调用时）。"""
    t = current()
    if t is None:
        yield
        return
    with t.stage(name):
        yield


def metric(**kw):
    t = current()
    if t is not None:
        t.metric(**kw)


def write(rec):
    """追加一行 JSON。失败不抛异常 —— 性能日志不该拖垮业务请求。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_all():
    if not LOG_PATH.exists():
        return []
    out = []
    with open(LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def clear():
    LOG_PATH.unlink(missing_ok=True)


def summarize(records=None):
    """把一堆 Trace 汇总成阶段耗时表 —— 报告里的数字全部出自这里。"""
    import statistics
    records = records if records is not None else read_all()
    if not records:
        return {}
    stages = {}
    for r in records:
        for s in r["spans"]:
            if s["depth"] != 0:          # 只统计顶层阶段，子阶段另看
                continue
            stages.setdefault(s["stage"], []).append(s["seconds"])
    total = [r["total_seconds"] for r in records]
    out = {"_total": {"mean": statistics.mean(total),
                      "median": statistics.median(total),
                      "p95": _pct(total, 95), "n": len(total)}}
    for name, vals in stages.items():
        out[name] = {"mean": statistics.mean(vals),
                     "median": statistics.median(vals),
                     "p95": _pct(vals, 95), "n": len(vals),
                     "share": statistics.mean(vals) / statistics.mean(total)}
    return out


def _pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = max(0, min(len(vals) - 1, int(round((p / 100) * len(vals) + 0.5)) - 1))
    return vals[k]
