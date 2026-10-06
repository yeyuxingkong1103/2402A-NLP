# /metrics 的三项指标：计数分桶、延迟聚合、排除口径与 500 的计入点（设计 §二 第 7 条）。
#
# 判据走**真 create_app**（中间件、异常映射、路由都来自真装配），重资源换轻替身；
# 另挂几条只产状态码的探针路由（不属于交付面）。为什么不用真业务端点产 400/503：
# 那要么真加载模型、花 API 钱，要么把判据挂在业务分支上 —— 本文件判的是「接线与
# 分桶」，某个状态码由哪条业务逻辑产生不是判据。
import sys
import threading
import time

from fastapi.testclient import TestClient

from app import main
from app.core import config, errors, metrics
from tests._fakes_http import fake_factory

# 探针路由：`/_probe/*` 不在限流清单里（计数用例不该先撞上 429）；
# 带 public 前缀的那条在清单里 —— 429 用例专门打它，用来判「中间件自己出的
# 响应也算进分桶」
PROBE_OK = "/_probe/ok"
PROBE_SLOW = "/_probe/slow"
PROBE_BAD = "/_probe/bad"
PROBE_UNAVAILABLE = "/_probe/unavailable"
PROBE_BOOM = "/_probe/boom"
PROBE_SLOW_BOOM = "/_probe/slow-boom"
PROBE_LIMITED = "/api/v1/public/_probe/limited"
SLOW_SECONDS = 0.03


def _probe_ok() -> dict:
    return {"ok": True}


def _probe_slow() -> dict:
    """故意慢的探针：延迟聚合用例要一个**下界确定**的样本（30ms）。

    没有它，把 `_milliseconds` 改成常量 0 的变异只会在断言「avg > 0」上抖 ——
    而真实请求可能恰好亚毫秒；有了确定的慢请求，判据是 `p95 >= 10`，不靠运气。
    """
    time.sleep(SLOW_SECONDS)
    return {"ok": True}


def _probe_bad() -> dict:
    raise errors.BadRequest("探针：参数不合法")


def _probe_unavailable() -> dict:
    raise errors.ServiceUnavailable("探针：上游不可用")


def _probe_boom() -> dict:
    raise RuntimeError("BOOM-指标探针")


def _probe_slow_boom() -> dict:
    """慢**失败**探针：先睡再抛（500 的「延迟起点」判据要一个下界确定的样本）。

    与 `_probe_slow` 的区别是它抛异常：走的正是「中间件已出栈、兜底处理器
    从 scope 补记延迟」那条路径 —— 也就是 `STARTED_KEY` 那根接缝。
    """
    time.sleep(SLOW_SECONDS)
    raise RuntimeError("BOOM-慢失败探针")


def _install_probes(app) -> None:
    """把探针路由挂到真 app 上（只加路径，不碰中间件与异常映射）。"""
    app.add_api_route(PROBE_OK, _probe_ok, methods=["GET", "POST"])
    app.add_api_route(PROBE_SLOW, _probe_slow, methods=["GET"])
    app.add_api_route(PROBE_BAD, _probe_bad, methods=["GET"])
    app.add_api_route(PROBE_UNAVAILABLE, _probe_unavailable, methods=["GET"])
    app.add_api_route(PROBE_BOOM, _probe_boom, methods=["GET"])
    app.add_api_route(PROBE_SLOW_BOOM, _probe_slow_boom, methods=["GET"])
    app.add_api_route(PROBE_LIMITED, _probe_ok, methods=["GET"])


def _client(monkeypatch, *, rate_limit: int | None = None) -> TestClient:
    """真 app + 假 Services + 探针路由；`raise_server_exceptions=False` 让 500 以
    响应的形态出现（否则 BOOM 会抛进用例，兜底处理器那条计入路径反而判不到）。

    `rate_limit` 给值就覆盖限流阈值（env）：429 用例把窗口压到 1 次，其余用例
    走默认值（30 次/60s，本文件远打不满）。
    """
    if rate_limit is not None:
        monkeypatch.setenv(config.ENV_RATE_LIMIT_MAX, str(rate_limit))
    app = main.create_app(services_factory=fake_factory()[0])
    _install_probes(app)
    return TestClient(app, raise_server_exceptions=False)


def _read(client: TestClient) -> dict:
    response = client.get("/metrics")
    assert response.status_code == 200
    return response.json()


def _traffic(client: TestClient) -> None:
    """打七发覆盖各分桶的请求（2xx×2、4xx×3、5xx×2）。次序固定、逐发核对状态码，
    否则下面「逐个对上」的断言会变成「随便对上几个」。"""
    assert client.get(PROBE_OK).status_code == 200
    assert client.get(PROBE_SLOW).status_code == 200
    assert client.get(PROBE_BAD).status_code == 400
    assert client.get("/api/v1/no-such-path").status_code == 404
    assert client.get(PROBE_BOOM).status_code == 500
    assert client.get(PROBE_UNAVAILABLE).status_code == 503
    oversized = client.post(PROBE_OK, content=b"x" * (main.MAX_BODY_BYTES + 1))
    assert oversized.status_code == 413


# ---- 计数与分桶逐发对上 ----


def test_the_counters_match_the_traffic_bucket_by_bucket(monkeypatch):
    """七发请求 → 总数 7、4xx 三发（400/404/413）、5xx 两发（500/503）、2xx 两发。

    逐桶断言而不是断言总和：bucket 的分界（4xx 与 5xx 之间、2xx 与 4xx 之间）
    正是运维分流「调用方的问题 vs 我们的问题」的依据，总和相同而分桶漂了
    （比如 4xx 记进 5xx）会让告警指错方向，而总数根本看不出来。
    """
    client = _client(monkeypatch)
    _traffic(client)
    body = _read(client)
    assert body["requests_total"] == 7
    assert body["errors"] == {"4xx": 3, "5xx": 2, "total": 5}


def test_the_latency_sample_count_follows_the_counted_requests(monkeypatch):
    """延迟样本数 = 被计数的请求数（含 400/404/413/500/503 —— 失败请求也耗时，
    把它们排除出去会让 p95 只反映顺利路径，恰好掩盖最慢的一类）。"""
    client = _client(monkeypatch)
    _traffic(client)
    body = _read(client)
    assert body["latency_ms"]["samples"] == 7


def test_the_slow_request_shows_up_in_p95_and_max(monkeypatch):
    """30ms 的慢请求必须出现在 p95/max 里（延迟聚合真的在量、不是常量）。

    下界取 10ms（实测值 30ms 量级）：给慢机器留余量，而把聚合改成常量 0 时
    它必红 —— 这条就是简报变异 3 的判据。
    """
    client = _client(monkeypatch)
    _traffic(client)
    latency = _read(client)["latency_ms"]
    assert latency["p95"] >= 10, latency
    assert latency["max"] >= 10, latency
    assert 0 < latency["avg"] <= 5000, latency


def test_a_middleware_generated_429_is_still_counted(monkeypatch):
    """429 由限流中间件**自己**出（不经过异常映射），计数中间件在它之外才数得到。

    这条同时是「计数中间件不能挪到限流器里面」的判据：挪进去后第二次请求返回
    429 而不经过计数，4xx 分桶当场少一（简报变异 2 的同一形态）。
    """
    client = _client(monkeypatch, rate_limit=1)
    assert client.get(PROBE_LIMITED).status_code == 200
    assert client.get(PROBE_LIMITED).status_code == 429
    body = _read(client)
    assert body["requests_total"] == 2
    assert body["errors"]["4xx"] == 1


def test_an_unhandled_exception_is_counted_in_the_5xx_bucket(monkeypatch):
    """500 的计入点在兜底处理器（`metrics.record_unhandled`），不在中间件。

    形态判据用**差值**：500 那条路径由 ServerErrorMiddleware 下发、不经过任何用户
    中间件（任务 3 实测），若只断言「总量变了」会分不清是谁记的 —— 差值 + 样本数
    同时 +1 才证明兜底处理器既记了状态码也补了延迟样本（scope 里那个开始时刻）。
    """
    client = _client(monkeypatch)
    before = _read(client)
    assert client.get(PROBE_BOOM).status_code == 500
    after = _read(client)
    assert after["requests_total"] - before["requests_total"] == 1
    assert after["errors"]["5xx"] - before["errors"]["5xx"] == 1
    assert (after["latency_ms"]["samples"]
            - before["latency_ms"]["samples"]) == 1


def test_a_slow_failure_reports_the_real_latency_recorded_in_the_scope(monkeypatch):
    """500 的延迟起点（中间件写进 scope 的 started）真的被兜底处理器用上。

    设计成**窗口里只有这一发样本**：别的请求一个不打、`/metrics` 自身不计入，
    `samples == 1` 先钉住这个前提 —— 于是 `max` 只可能由这一发的延迟抬起来，
    判据才承重。删掉 `MetricsMiddleware` 里写 `STARTED_KEY` 的那行，补记的
    延迟静默变 0.0，本条在 `max >= 10` 上红（N-A 刀实测）。
    """
    client = _client(monkeypatch)
    assert client.get(PROBE_SLOW_BOOM).status_code == 500
    latency = _read(client)["latency_ms"]
    assert latency["samples"] == 1, latency
    assert latency["max"] >= 10, latency


# ---- 排除口径：/metrics 与 /healthz 不计入 ----


def test_metrics_and_healthz_are_not_counted(monkeypatch):
    """探针与自观测都不进账：否则运维看到的 QPS 会随探针频率浮动。

    口径写进响应体（excluded_paths）—— 读的人不必去读代码；这条断言同时钉住
    「声明与行为一致」（只改声明不改行为，或反之，都会红）。
    """
    client = _client(monkeypatch)
    assert client.get("/healthz").status_code in (200, 503)
    assert client.get("/metrics").status_code == 200
    body = _read(client)
    assert body["requests_total"] == 0, "探针与 /metrics 自身都不该进账"
    assert body["excluded_paths"] == ["/healthz", "/metrics"]


# ---- 聚合口径与窗口上界（单元层，样本可控） ----


def test_the_latency_aggregate_is_the_documented_nearest_rank_percentile():
    """1..100 的样本：p95 必须恰好是 95（最近排队法），max 100，avg 50.5。

    这条把聚合口径钉成字面量：改成 max（100）、均值（50.5）、常量 0 都当场红。
    用真实服务打的样本做不到这种精度（数量与取值都不受控）。
    """
    registry = metrics.Registry(window=100)
    for value in range(1, 101):
        registry.record(200, float(value))
    latency = registry.snapshot()["latency_ms"]
    assert latency == {"samples": 100, "avg": 50.5, "p95": 95.0, "max": 100.0}


def test_the_latency_window_is_bounded_while_the_total_keeps_counting(monkeypatch):
    """300 发请求 → 总数 300，而延迟样本停在窗口上界（256）。

    「保留全部样本」的病是长期运行进程的无界内存增长，而它在短用例里一点看不出来
    —— 这条把上界钉成行为：窗口满了以后样本数不再涨，总数继续涨。两个数的时间
    基准不同（模块 docstring 已写明代价），这里同时把那份代价钉成事实。
    """
    client = _client(monkeypatch)
    shots = metrics.LATENCY_WINDOW + 44
    for _ in range(shots):
        assert client.get(PROBE_OK).status_code == 200
    body = _read(client)
    assert body["requests_total"] == shots
    assert body["latency_ms"]["samples"] == metrics.LATENCY_WINDOW


def test_each_app_has_its_own_registry():
    """两个 app 的计数互不污染（注册表是每 app 一个，不是模块级单例）。

    单例形态的病：测试之间、将来多 app 之间读到的数混在一起，用例只能改成断言
    差值 —— 而差值在「上一次多记了」时照样绿。这条用两个独立 app 直接判。
    """
    first = main.create_app(services_factory=fake_factory()[0])
    _install_probes(first)
    second = main.create_app(services_factory=fake_factory()[0])
    _install_probes(second)
    with TestClient(first, raise_server_exceptions=False) as client:
        assert client.get(PROBE_OK).status_code == 200
    with TestClient(second, raise_server_exceptions=False) as client:
        assert _read(client)["requests_total"] == 0


def test_metrics_answers_with_the_same_shape_before_any_traffic(monkeypatch):
    """零流量时形状不变（未装配 services 也 200）：读的人不必分两种形状。

    不进 lifespan（不起 services）：/metrics 是观测端点，它自己不能依赖于重资源
    是否就绪 —— 那会让「服务起不来」时监控连一眼都看不到。
    """
    app = main.create_app(services_factory=fake_factory()[0])
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "requests_total": 0,
        "errors": {"4xx": 0, "5xx": 0, "total": 0},
        "latency_ms": {"samples": 0, "avg": 0.0, "p95": 0.0, "max": 0.0},
        "excluded_paths": ["/healthz", "/metrics"],
        "window": {"latency_samples_max": metrics.LATENCY_WINDOW},
    }


# ---- 并发：record 的锁 ----


class _CountingLock:
    """包住真锁并记 `__enter__` 次数（与 test_ratelimit 的同名件同法、同理由）：
    「临界区有没有被进入」不需要调度配合 —— 删掉 `with self._lock:` 时**确定性
    变红**，而丢计数的竞态在默认 GIL 切换间隔下几乎不可观测。"""

    def __init__(self, real) -> None:
        self._real = real
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self._real.__enter__()

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)


def test_record_never_loses_an_update_and_really_enters_the_lock():
    """两条判据合取：①多线程并发下总数与分桶**逐次对上**（不丢更新）；
    ②record 真的进过那把锁（确定性判据，见 _CountingLock）。

    切换间隔压到 1e-6 并在 finally 恢复（压间隔是全局状态，泄漏会污染整轮）——
    与 test_ratelimit 的并发用例同一纪律。
    """
    registry = metrics.Registry()
    probe = _CountingLock(registry._lock)
    registry._lock = probe
    workers, attempts = 8, 500
    expected = workers * attempts
    original_interval = sys.getswitchinterval()

    def worker() -> None:
        for _ in range(attempts):
            registry.record(503, 1.0)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    try:
        sys.setswitchinterval(1e-6)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(original_interval)
    assert sys.getswitchinterval() == original_interval, "切换间隔没恢复原值"
    # 先核对进锁次数再读快照：snapshot() 自己也进同一把锁，判据要在它之前取
    assert probe.enters == expected, "record 的每次调用都必须进出一次 self._lock"
    snapshot = registry.snapshot()
    assert snapshot["requests_total"] == expected, "并发下丢了计数"
    assert snapshot["errors"]["5xx"] == expected
