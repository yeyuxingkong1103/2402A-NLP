"""进程内运行指标：请求数 / 延迟 / 错误数的最小 JSON（设计 §二 第 7 条、§九 第 8 步）。

存在的理由：本轮**不引 `prometheus_client`**（依赖锁得紧，且当前没有拉取端）——
`/metrics` 给一个自描述的 JSON 快照；将来上 Prometheus 时换格式，换的就是本模块的
`endpoint()` 一个函数（中间件与 Registry 不动），这一条写在这里当交接。

**500 的计入点不在中间件**（本任务的核心裁决）：中间件只能看见「用户中间件栈之内
产生的响应」，而 `ServerErrorMiddleware` 在所有用户中间件**外面**（任务 3 实测：500
不经过任何用户中间件）——「把计数放最外层」也数不到它。故 500 只有一个入口：兜底
处理器 `main._on_unexpected` 调 `record_unhandled()`。若照直写成「最外层中间件兜
500」，那是个数不到任何东西的假接线（变异刀：删掉那一行，500 分桶用例当场红）。

**延迟口径**：滚动窗口（最近 `LATENCY_WINDOW` 个样本）+ 最近排队法 P95。不用「保留
全部样本」—— 那是长期运行进程里的无界内存增长（本模块的全部状态 = 三个计数器 +
一个定长 deque）；也不用分位数估计（要引依赖或自造一套误差口径，超出本轮「最小」
定位）。**代价写明**：P95 是窗口内的，长尾样本滑出后不再出现在 p95 里，而
`requests_total` 是全生命周期计数 —— 两个数的时间基准不同，读的时候要知道。

**并发**：`def` 端点进线程池 → `record` 会从多个线程被调；`self._total += 1` 与
`dict[k] += 1` 都是读-改-写（不是原子操作），全部更新收在一把锁里。锁内只有内存
操作（无 I/O、无等待），与 ratelimit.RateLimiter 同一纪律。
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse

# 延迟窗口的样本数：定长 deque（内存上界 = 这个数 × 一个 float）。取 256 的理由：
# 够让 P95 落在十余个样本上（不至于被单个样本左右），又小到可忽略排序与内存成本
LATENCY_WINDOW = 256
# 状态码分桶只分到「类」。理由：①设计 §二 第 7 条要的是「错误数」，4xx/5xx 才是
# 运维分流粒度（4xx 是调用方的问题、5xx 是我们的）；②逐码计数要在进程里存一张
# 无界的键表（外部能打出任意整数状态码），没必要
_STATUS_CLASSES = ("2xx", "3xx", "4xx", "5xx")
# 不计入的路径：`/metrics` 是自观测（数它等于让刷这个端点的频率改变 QPS 读数）、
# `/healthz` 是探针流量（编排层按秒打）—— 二者都不是业务量，混进去会让「公众侧
# QPS」随探针频率浮动。口径以 `excluded_paths` 字段出现在响应体里，运维读到的
# 是同一份声明，不必读代码
EXCLUDED_PATHS = frozenset(("/metrics", "/healthz"))
# scope["state"] 里记请求开始时刻的键：500 的延迟只能在兜底处理器里补记，而那时
# 中间件已经出栈（时刻拿不到）—— 故开始时刻必须先放进 scope（与 request_id 同法）
STARTED_KEY = "metrics_started"


def _milliseconds(started: float) -> float:
    """从 monotonic 起点到现在的毫秒数（三位小数）。

    用 `time.monotonic` 而不是 wall clock：延迟是「间隔」，系统时钟被 NTP 校正时
    wall clock 会跳（跳出来的是负数或几小时），monotonic 保证单调不减。
    """
    return round((time.monotonic() - started) * 1000.0, 3)


def _percentile(ordered: list[float], q: float) -> float:
    """最近排队法（nearest-rank）分位数；`ordered` 必须已升序且非空。

    n=1 时给那个样本本身；q=0.95、n=100 时给第 95 个（下标 94）。不用线性插值：
    插值出来的值不是任何一次真实请求的延迟，而本轮的读者是人不是监控系统。
    """
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


class Registry:
    """三项指标的家：请求总数、状态类分桶、延迟滚动窗口。**每 app 一个实例**。

    每 app 而不是模块级单例：①`create_app` 会被测试调用很多次，单例会让两次用例的
    计数互相污染（用例只能改成断言差值，而差值断言在「上一次多记了」时照样绿）；
    ②多 app 场景（测试夹具、将来的灰度实例）本来就该各看各的。生产只有 uvicorn
    那一个 app，行为与单例无异。

    公开面只有 `record()` 与 `snapshot()`：写数方（中间件、兜底处理器）与读数方
    （`/metrics`）都不碰内部结构，形状的改动收在这一处。
    """

    def __init__(self, window: int = LATENCY_WINDOW) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._by_class = {name: 0 for name in _STATUS_CLASSES}
        self._latencies: deque[float] = deque(maxlen=window)

    def record(self, status_code: int, latency_ms: float) -> None:
        """记一次请求（状态码 + 一个延迟样本）。更新全在锁内（见模块 docstring）。"""
        bucket = f"{status_code // 100}xx"
        with self._lock:
            # 下面三行都是读-改-写：`+=` 在 int 与 dict 上都不是原子操作，线程池里
            # 两个请求同时走到这里就会丢掉一次计数（丢的正是「请求量」—— 容量口径
            # 的依据），故必须在锁内
            self._total += 1
            if bucket in self._by_class:
                self._by_class[bucket] += 1
            self._latencies.append(float(latency_ms))

    def snapshot(self) -> dict:
        """当前读数（JSON 可序列化）。锁内做拷贝、快照与排序（256 个样本可忽略）。"""
        with self._lock:
            total = self._total
            buckets = dict(self._by_class)
            samples = sorted(self._latencies)
        return {
            "requests_total": total,
            "errors": {"4xx": buckets["4xx"], "5xx": buckets["5xx"],
                       "total": buckets["4xx"] + buckets["5xx"]},
            "latency_ms": {
                "samples": len(samples),
                "avg": round(sum(samples) / len(samples), 3) if samples else 0.0,
                "p95": _percentile(samples, 0.95) if samples else 0.0,
                "max": samples[-1] if samples else 0.0,
            },
            "excluded_paths": sorted(EXCLUDED_PATHS),
            "window": {"latency_samples_max": self._latencies.maxlen},
        }


class MetricsMiddleware:
    """给每条经过的响应计数（状态类 + 延迟）。装配位置与理由见 main.create_app。

    只数「经过它的响应」：`http.response.start` 一到就记。异常**直接穿透不吞** ——
    未处理异常的 500 由 ServerErrorMiddleware 出，那条路径不经过本中间件，计入点在
    兜底处理器（`record_unhandled`）。若这里在 except 里也补记一笔，同一发 500 会
    记两次 —— 而总数虚高这种事没有任何读数会报错，只能靠这条注释与用例挡住。
    """

    def __init__(self, app, registry: Registry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") in EXCLUDED_PATHS:
            # lifespan / websocket 与两个不计入的路径原样放行（连时刻都不放）
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        # 开始时刻记进 scope：兜底处理器补记 500 的延迟时中间件已出栈，只能从
        # scope 拿（request_id 的 STATE_KEY 同法，两边都是「跨出栈的接缝」）
        scope.setdefault("state", {})[STARTED_KEY] = started

        async def send_and_count(message) -> None:
            # 只认响应头那一条消息：状态码在这里、且它恰好出现一次（响应体可能
            # 分多块，按块计数会让一个响应记成多次）
            if message["type"] == "http.response.start":
                self.registry.record(message["status"], _milliseconds(started))
            await send(message)

        await self.app(scope, receive, send_and_count)


def registry_of(app) -> Registry | None:
    """取 app 上的注册表（装配时放在 `app.state.metrics`）；没有给 None。

    不抛：兜底处理器与端点都可能被**没经过 create_app** 的调用触发（单测直接调、
    将来的嵌入式用法），那时少记一行指标不该把请求变成另一个 500。
    """
    return getattr(getattr(app, "state", None), "metrics", None)


def record_unhandled(request: Request) -> None:
    """500 的计入口（`main._on_unexpected` 调用）。见模块 docstring 的「500 计入点」。

    延迟从 scope 里的开始时刻算；没有那个键（请求没经过计数中间件）就记 0 ——
    0 是一个诚实的「不知道」，而伪造一个值会让 p95 里混进编的数字。
    """
    registry = registry_of(request.app)
    if registry is None:
        return
    started = (request.scope.get("state") or {}).get(STARTED_KEY)
    registry.record(500, _milliseconds(started) if started is not None else 0.0)


def endpoint(request: Request) -> JSONResponse:
    """`/metrics` 的响应。内网端点的鉴权处置（本轮不鉴权、记为部署项）见任务 8 报告。

    未装配时给一份**形状完全相同**的空读数（读的人不必分两种形状），而不是 500：
    监控拉不到数据时会告警，让它拿到「零流量」比拿到 500 更容易被误读成故障。
    """
    registry = registry_of(request.app)
    body = registry.snapshot() if registry is not None else Registry().snapshot()
    return JSONResponse(status_code=200, content=body)
