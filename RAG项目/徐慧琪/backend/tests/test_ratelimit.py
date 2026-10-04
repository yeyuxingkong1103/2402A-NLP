# 进程内限流组件的判据（core/ratelimit.py）：滑动窗口、计数、淘汰、闸门、键、路径谓词。
#
# 为什么这些用例不经过 HTTP：中间件那层要判的是**接线**（顺序、哪些路径、状态码形状），
# 在 test_main_ratelimit.py；这里的对象是「窗口本身算得对不对」，而窗口的边界（恰好滑出、
# 拒绝不计入、淘汰谁）用真 HTTP 打不出来 —— 那要打几十发请求且时间不可控。
#
# **时钟一律注入**（FakeClock）：本项目反复吃的教训是 sleep 出来的用例在慢机器上必红，
# 而「窗口滑过之后该放行」正是最需要 sleep 的判据。
#
# 并发那条是个例外：它要判的恰恰是「锁有没有挡住竞态」，而 GIL 默认切换间隔下这个竞态
# 几乎不可观测（审查实测：默认间隔 4000 轮 × 32 线程对无锁实现也是 0 轮超发）。故它显式
# 把 sys.setswitchinterval 压到激进值再跑多轮，并在 finally 里恢复原值 —— 这是「不用 sleep
# 也能让竞态可观测」的替代手段，不是把 sleep 换了个名字。
import sys
import threading

import pytest

from app.core import ratelimit


class FakeClock:
    """可注入的时钟：`advance()` 手动推时间，用例里一次 sleep 都没有。"""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---- RateLimiter：窗口与计数 ----

def test_a_request_within_the_limit_passes_and_the_one_over_it_is_refused():
    """阈值内放行、超阈值拒绝（简报 Step 5 的第一个双向判据）。"""
    rate = ratelimit.RateLimiter(limit=3, window_s=60, clock=FakeClock())
    assert [rate.allow("ip-a") for _ in range(3)] == [True, True, True]
    assert rate.allow("ip-a") is False


def test_the_window_slides_so_the_slot_comes_back_after_it_passes():
    """窗口滑过后恢复（注入时钟，不 sleep）。

    判据取**边界两侧**：窗口差一点没过时仍拒（说明它真在等窗口而不是「拒绝一次就
    永久放行」），满窗口时放行（说明它真会恢复）。只断言「后来能过」的话，把窗口
    实现成「拒绝后立刻清零」也会绿 —— 而那是另一种坏法。
    """
    clock = FakeClock()
    rate = ratelimit.RateLimiter(limit=1, window_s=60, clock=clock)
    assert rate.allow("ip-a") is True
    clock.advance(59.999)
    assert rate.allow("ip-a") is False
    clock.advance(0.001)
    assert rate.allow("ip-a") is True


def test_a_refused_request_is_not_recorded_so_hammering_does_not_extend_the_block():
    """被拒的请求不记进窗口 —— 否则「持续重试」会把自己永久锁死。

    形态：limit=1、t=0 放行一次，t=59/60 各拒一次。若拒绝也计数，t=60 时窗口里
    还剩 t=59.9 那次命中，仍然拒绝；而正确实现只认 t=0 那次，它刚好滑出去。
    """
    clock = FakeClock()
    rate = ratelimit.RateLimiter(limit=1, window_s=60, clock=clock)
    assert rate.allow("ip-a") is True
    clock.advance(59.9)
    assert rate.allow("ip-a") is False
    clock.advance(0.1)
    assert rate.allow("ip-a") is True


def test_retry_after_counts_down_the_oldest_hit_leaving_the_window():
    """Retry-After = 最老那次命中滑出窗口还要几秒（向上取整，且 ≥1）。

    没有它，客户端只能盲等或立刻重试（后者把限流变成放大器）；取最老那次而不是
    窗口全长，是因为额度是按次恢复的 —— 报 60 会让客户端白等。
    """
    clock = FakeClock()
    rate = ratelimit.RateLimiter(limit=2, window_s=60, clock=clock)
    assert rate.retry_after("ip-a") == 0, "没满时不该建议退避"
    assert rate.allow("ip-a") and rate.allow("ip-a")
    assert rate.allow("ip-a") is False
    assert rate.retry_after("ip-a") == 60, "最老一次在 t=0，滑出窗口还要 60 秒"
    clock.advance(39.5)
    assert rate.retry_after("ip-a") == 21, "向上取整：剩余 20.5 秒报 21"


def test_the_clock_can_come_from_the_call_as_well_as_the_constructor():
    """`now` 参数覆盖构造时的时钟（两种注入形态都可用，接口面见模块 docstring）。"""
    clock = FakeClock(now=0.0)
    rate = ratelimit.RateLimiter(limit=1, window_s=10, clock=clock)
    assert rate.allow("ip-a", now=0.0) is True
    assert rate.allow("ip-a", now=9.0) is False
    assert rate.allow("ip-a", now=10.0) is True, "显式 now 推过了窗口，与构造函数无关"


def test_the_limit_is_per_key_not_global():
    """配额按 key 分开算：一个人的配额用完不该把别人挡在门外（这正是限流的意义）。"""
    rate = ratelimit.RateLimiter(limit=1, window_s=60, clock=FakeClock())
    assert rate.allow("ip-a") is True
    assert rate.allow("ip-b") is True
    assert rate.allow("ip-a") is False


def test_the_key_table_stays_bounded_when_new_keys_keep_arriving():
    """键数有上限（匿名端点人人可造键，无界 = 一条内存耗尽通道）。

    判据是 key_count() 而不是内存读数：淘汰有没有真发生，只有它看得见。
    """
    rate = ratelimit.RateLimiter(limit=1, window_s=60, max_keys=10,
                                 clock=FakeClock())
    for i in range(100):
        rate.allow(f"ip-{i}")
    assert rate.key_count() <= 10


def test_eviction_drops_the_oldest_key_and_keeps_the_newest():
    """按插入序淘汰最老的键：新键进得来（可用性优先），代价是旧键的计数精度。"""
    rate = ratelimit.RateLimiter(limit=1, window_s=60, max_keys=2,
                                 clock=FakeClock())
    assert rate.allow("old") is True
    assert rate.allow("new") is True
    assert rate.allow("newest") is True, "满了也要能进新键（淘汰而不是拒绝）"
    assert rate.key_count() == 2
    # old 被淘汰后计数归零：再打一次应当放行；若淘汰的是别人，这里会拒
    assert rate.allow("old") is True


def test_the_sweep_forgets_keys_whose_window_has_passed():
    """过期的键会被清扫掉（不然 max_keys 个僵尸键会永远占着表）。"""
    clock = FakeClock()
    rate = ratelimit.RateLimiter(limit=1, window_s=60, max_keys=2, clock=clock)
    assert rate.allow("a") and rate.allow("b")
    clock.advance(61)
    assert rate.allow("c") is True
    assert rate.key_count() == 1, "a/b 的命中都已滑出窗口，应被清扫只留 c"


def test_concurrent_allows_never_admit_more_than_the_limit():
    """并发下计数不丢：limit 是硬上限，不能因为竞态多放。

    用真线程（不是顺序调用）：allow 会在线程池的多个线程里被调（def 端点），
    没有锁的实现会在这里丢计数 —— 而丢计数的形态是「限流偶尔不生效」，
    单线程用例永远看不见。

    **判别力边界（第 12 次「不可能失败的测试」修的就是它）**：GIL 默认切换间隔
    （5ms）下这个竞态几乎不可观测 —— 审查实测：默认间隔 4000 轮 × 32 线程的压力
    对无锁实现也是 0 轮超发，删掉 `with self._lock:` 后旧形状 25/25 全绿。故本用例
    显式把切换间隔压到 1e-6（激进抢占），并跑「多轮 × 每轮新建限流器」：无锁实现
    在这个形状下实测稳定超发（审查者 502/3000 轮；本仓库修复时以真变异复测
    200 轮 × 16 线程 × 10 次尝试，删锁后 5 次连跑分别超发 77/39/54/19/78 次），
    锁删掉后当场变红。间隔用 try/finally 恢复，
    并在最后断言它真的回到了原值 —— 压间隔是全局状态，泄漏出去会污染整轮测试。
    """
    original_interval = sys.getswitchinterval()
    rounds, workers, attempts, limit = 200, 16, 10, 100
    over_admits = 0
    affected_rounds = 0
    try:
        sys.setswitchinterval(1e-6)
        for _ in range(rounds):
            rate = ratelimit.RateLimiter(limit=limit, window_s=600,
                                         clock=FakeClock())
            barrier = threading.Barrier(workers)
            results: list[bool] = []
            collector = threading.Lock()

            def worker() -> None:
                barrier.wait()
                for _ in range(attempts):
                    allowed = rate.allow("shared")
                    with collector:
                        results.append(allowed)

            threads = [threading.Thread(target=worker) for _ in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            excess = sum(results) - limit  # 只多不少：拒的前提是数到了 limit 次放行
            over_admits += max(0, excess)
            affected_rounds += 1 if excess > 0 else 0
    finally:
        sys.setswitchinterval(original_interval)
    assert sys.getswitchinterval() == original_interval, "切换间隔没恢复原值"
    assert over_admits == 0, (
        f"{rounds} 轮里有 {affected_rounds} 轮超发（共 {over_admits} 次）：锁没挡住竞态"
        f"（每轮 {workers} 线程 × {attempts} 次尝试，上限 {limit}）")


class _CountingLock:
    """包住真锁并记 `__enter__` 次数：用来判「allow() 真的进过这把锁」。

    锁删除的竞态在默认调度下不可观测（见上一条的判别力边界），但「临界区有没有
    被进入」不需要调度配合 —— 换成本探针后，`with self._lock:` 一被删就确定性变红。
    """

    def __init__(self, real) -> None:
        self._real = real
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self._real.__enter__()

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)


def test_allow_really_enters_the_lock_that_makes_it_thread_safe():
    """确定性判据：allow() 的每次调用都必须进出一次 `self._lock`。

    与上一条互补：上一条判「锁在并发下真的挡住了多放」（要激进切换间隔才看得见），
    这一条判「锁真的被用了」—— 把 `limiter._lock` 换成记 `__enter__` 次数的探针，
    删掉 `with self._lock:` 时**确定性变红**，不依赖 GIL 调度。合起来：既不接受
    「有锁但没人用」，也不接受「用了锁但竞态仍超发」。
    """
    rate = ratelimit.RateLimiter(limit=2, window_s=60, clock=FakeClock())
    probe = _CountingLock(rate._lock)
    rate._lock = probe
    assert rate.allow("ip-a") is True
    assert rate.allow("ip-a") is True
    assert rate.allow("ip-a") is False
    assert probe.enters == 3, "allow() 的每次调用都必须进出一次 self._lock"


# ---- ConcurrencyGate：公众侧并发上限 ----

def test_the_gate_admits_exactly_the_limit_and_refuses_the_next():
    """第 21 个被拒（简报 Step 5 的硬要求，limit=20 对齐 AC-14）。"""
    gate = ratelimit.ConcurrencyGate(limit=20)
    assert [gate.try_acquire() for _ in range(20)] == [True] * 20
    assert gate.try_acquire() is False
    assert gate.in_flight() == 20, "被拒的那次不能占槽"


def test_release_returns_the_slot_and_a_refused_acquire_consumes_nothing():
    """释放后槽位回来（finally 里 release 的意义），被拒的一次不消耗槽位。"""
    gate = ratelimit.ConcurrencyGate(limit=2)
    assert gate.try_acquire() and gate.try_acquire()
    assert gate.try_acquire() is False
    gate.release()
    assert gate.try_acquire() is True
    assert gate.try_acquire() is False


def test_releasing_more_than_acquired_does_not_create_slots():
    """多还的槽不许把计数减成负数 —— 那等于闸门凭空多放行（比漏释放更隐蔽）。"""
    gate = ratelimit.ConcurrencyGate(limit=1)
    gate.release()
    gate.release()
    assert gate.in_flight() == 0
    assert gate.try_acquire() is True
    assert gate.try_acquire() is False


def test_the_gate_is_thread_safe():
    """并发抢槽：恰好 limit 个成功，不多不少。"""
    gate = ratelimit.ConcurrencyGate(limit=5)
    barrier = threading.Barrier(32)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        acquired = gate.try_acquire()
        with lock:
            results.append(acquired)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(results) == 5
    assert gate.in_flight() == 5


def test_the_gate_really_enters_the_lock_that_makes_it_thread_safe():
    """确定性判据：try_acquire / release 每次都进出一次 `self._lock`。

    与上一条互补：上一条判「并发下恰好 limit 个成功」，而 GIL 下丢更新是概率
    事件 —— 终审把两把锁分别删掉，test_ratelimit 30 条与 test_main_ratelimit
    11 条全绿（第 14 次「不可能失败的测试」）。探针形态抄同文件 RateLimiter 的
    那条（闸门这把此前没有），不依赖 GIL 调度。
    """
    gate = ratelimit.ConcurrencyGate(limit=1)
    probe = _CountingLock(gate._lock)
    gate._lock = probe
    assert gate.try_acquire() is True
    assert gate.try_acquire() is False
    gate.release()
    assert probe.enters == 3, "try_acquire/release 每次调用都必须进出一次锁"


# ---- client_key：IP 的加盐哈希 ----

def _scope(host: str | None = "10.0.0.7") -> dict:
    """造一个最小的 ASGI scope（client_key 只读 scope["client"]）。"""
    return {"type": "http", "client": None if host is None else (host, 51234)}


def test_the_same_client_always_maps_to_the_same_key_and_different_clients_differ():
    salt = "salt-0123456789abcdef"
    assert ratelimit.client_key(_scope("10.0.0.7"), salt) == \
        ratelimit.client_key(_scope("10.0.0.7"), salt)
    assert ratelimit.client_key(_scope("10.0.0.7"), salt) != \
        ratelimit.client_key(_scope("10.0.0.8"), salt)


def test_the_key_hides_the_raw_ip_and_really_uses_the_salt():
    """键里不许有 IP 原文（设计 §六），且换一个盐必须换一副键。

    第二个断言防的是「盐被忽略、退化成裸 sha256(ip)」——那种实现的键照样稳定，
    单看第一个断言分辨不出来，而它的形态是「拿 IPv4 全空间枚举即可还原 IP」。
    """
    key = ratelimit.client_key(_scope("10.0.0.7"), "salt-0123456789abcdef")
    assert "10.0.0.7" not in key
    assert len(key) == ratelimit.KEY_HEX_CHARS
    assert set(key) <= set("0123456789abcdef"), "键是十六进制摘要，不含分隔符与原文"
    other = ratelimit.client_key(_scope("10.0.0.7"), "another-salt-0123456789")
    assert key != other


def test_a_request_without_a_client_shares_one_fallback_key():
    """缺 client 的请求共用一个固定键：共享一份配额，而不是各自新键（灌键）。"""
    salt = "salt-0123456789abcdef"
    assert ratelimit.client_key(_scope(None), salt) == \
        ratelimit.client_key({"type": "http"}, salt)


# ---- 路径谓词：哪些路径受限、哪些计入并发槽 ----

@pytest.mark.parametrize("path", [
    "/api/v1/public/qa",
    "/api/v1/lawyers/recommend",
    "/api/v1/law/nav",
    "/api/v1/law/民法典/articles/584",
    "/api/v1/auth/login",
])
def test_the_public_read_paths_and_the_login_endpoint_are_limited(path):
    """公众可读的三条（含 /law/* 的动态段）与登录（终审 I-3）过限流窗口。"""
    assert ratelimit.is_limited_path(path) is True


@pytest.mark.parametrize("path", [
    "/healthz",
    "/api/v1/qa",
    "/api/v1/search",
    "/api/v1/cases/search",
    "/api/v1/admin/audit/export",
    "/api/v1/no-such-path",
])
def test_protected_and_unknown_paths_are_not_limited(path):
    """律师侧、健康检查与未知路径**不**过限流窗口。

    这不是省事：/qa 这类路径若先经过限流，「未认证 + 高频」会拿到 429 而不是 404
    —— 429 等于承认这条路径存在（设计 §二 第 9 条）；/healthz 被 429 则与「实例
    不健康」在监控上不可区分，编排层会摘掉一个健康的实例。
    """
    assert ratelimit.is_limited_path(path) is False


def test_only_the_two_inference_endpoints_occupy_a_concurrency_slot():
    """/law/* 在窗口里但**不**占并发槽（成本模型不同：索引查询 vs 秒级推理+GPU）。

    这条是任务 6 的裁决点之一（报告里单列）：浏览器加载一次导航树不该占掉一份
    推理配额；但匿名爬取仍按每 IP 计入窗口。两个方向都钉：该计的计、不该占的不占。
    """
    assert ratelimit.is_capped_path("/api/v1/public/qa") is True
    assert ratelimit.is_capped_path("/api/v1/lawyers/recommend") is True
    assert ratelimit.is_capped_path("/api/v1/law/nav") is False
    assert ratelimit.is_capped_path("/api/v1/law/民法典/articles/584") is False
    assert ratelimit.is_capped_path("/api/v1/auth/login") is False, "登录不占并发槽"
