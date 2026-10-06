# 限流与并发上限**接进真装配**后的判据（main.create_app 里的中间件顺序、路径清单、
# 429 形状、并发释放、缺盐的失败面）。
#
# 与 test_ratelimit.py 的分工：那边判「窗口算得对不对」（注入时钟、直接调组件），
# 这边判「接线」—— 阈值从环境变量来、受限的到底是哪几条路径、429 是不是统一形状、
# 被拒时业务有没有被跑到。所有 app 都走真 create_app（中间件顺序来自真装配）。
#
# 并发那条**真起 20 个线程**把请求压在飞行中，而不是直接调计数器：直接调计数器
# 只能证明「闸门类会拒第 21 个」，证明不了「中间件在 finally 里还了槽」—— 而漏还
# 槽的形态是几次之后公众侧永久 429，是本项目最怕的静默失效。
import json
import threading

from fastapi.testclient import TestClient

from app import main
from app.core import config, errors, ratelimit, request_id
from app.generation.answer import QAResult
from tests._fakes_http import FakeConn, FakeServices, fake_factory, without_request_id

URL = "/api/v1/public/qa"
LOGIN = "/api/v1/auth/login"
BODY = {"question": "房东不退押金怎么办"}
# 超过 main.MAX_BODY_BYTES（64 KiB）的问句：用来触发体上限中间件的 413
HUGE = {"question": "x" * 70000}


class _OkAnswerer:
    """假 Answerer：直接返回 ok 终态（不花钱、不连库）。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def answer(self, question: str, side: str) -> QAResult:
        self.calls.append((question, side))
        return QAResult(status="ok", answer="依据《民法典》第七百三十三条……")


class _RaisingAnswerer:
    """假 Answerer：每次都炸（用来判「异常路径也必须还槽」）。"""

    def answer(self, question: str, side: str) -> QAResult:
        raise RuntimeError("业务炸了")


class _GatedAnswerer:
    """假 Answerer：`hold` 个请求进入后就放行一个事件，供用例把并发压满。

    计数与事件都收在锁里；`release.wait` 带超时 —— 卡死的测试比失败的测试难查，
    超时后照常返回，让用例在断言上红而不是把闸门挂住。
    """

    def __init__(self, hold: int) -> None:
        self.hold = hold
        self.entered = 0
        self.reached = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def answer(self, question: str, side: str) -> QAResult:
        with self._lock:
            self.entered += 1
            if self.entered >= self.hold:
                self.reached.set()
        self.release.wait(timeout=30)
        return QAResult(status="ok", answer="ok")


def _env(monkeypatch, *, limit=None, window=None, concurrency=None) -> None:
    """把阈值写进环境变量（限流器懒建，故在第一个受限请求之前设置即生效）。

    只设用例关心的那几个：其余取 config 里的默认值，这条链路的默认值本身
    由 test_core_config 的用例钉住。
    """
    if limit is not None:
        monkeypatch.setenv(config.ENV_RATE_LIMIT_MAX, str(limit))
    if window is not None:
        monkeypatch.setenv(config.ENV_RATE_LIMIT_WINDOW_S, str(window))
    if concurrency is not None:
        monkeypatch.setenv(config.ENV_PUBLIC_CONCURRENCY, str(concurrency))


def _app(answerer=None, *, services=None) -> "main.FastAPI":
    """真 create_app + 假 Services；raise_server_exceptions=False 让 500 以响应出现。"""
    built = services if services is not None else FakeServices(answerer=answerer)
    factory_fn, _ = fake_factory(built)
    return main.create_app(services_factory=factory_fn)


def _client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _noop() -> dict:
    """探针端点：只回一个最短的 200。"""
    return {"ok": True}


def _raising(exc: Exception):
    """造一个抛指定异常的探针端点（429 形状对照用）。"""
    def endpoint() -> None:
        raise exc

    return endpoint


def _shape(response) -> str:
    """响应体的对外形状（挖掉 request_id，逐字节化的可比较形式）。"""
    return json.dumps(without_request_id(response), sort_keys=True,
                      ensure_ascii=False)


# ---- 窗口限流：429 的形状、Retry-After、业务没被跑到 ----

def test_a_request_within_the_threshold_passes_and_the_one_over_is_429(monkeypatch):
    """阈值内 200、超阈值 429，且 429 走统一错误形状 + Retry-After 的**真倒计时**。

    Retry-After 取**量级**判据：窗口 60s、被拒时最老一次命中刚发生，倒计时应当
    接近窗口全长（落在 [30, 60]）。只断言「≥1」的话，把接线里
    `rate.retry_after(key)` 钉死成常量 1 也全绿（审查 N10 实测 11 passed）——
    而常量 1 会让客户端每秒重试一次，把限流本身变成放大器。
    业务那一侧用 calls 记账判「被拒的请求没进业务」：只断言状态码的话，把中间件
    挂到业务**之后**（先答完再 429）照样绿 —— 那会白烧一次 GPU 推理。
    """
    _env(monkeypatch, limit=2, window=60)
    answerer = _OkAnswerer()
    with _client(_app(answerer)) as client:
        assert client.post(URL, json=BODY).status_code == 200
        assert client.post(URL, json=BODY).status_code == 200
        refused = client.post(URL, json=BODY)
    assert refused.status_code == 429
    retry_after = int(refused.headers["Retry-After"])
    assert 30 <= retry_after <= 60, (
        f"窗口 60s 的倒计时应落在 [30, 60]（最老命中刚发生），实际 {retry_after}："
        "常量 1 之类的假倒计时会让客户端每秒重试")
    assert refused.headers[request_id.HEADER] == refused.json()["request_id"]
    assert refused.json()["error"]["code"] == "rate_limited"
    assert len(answerer.calls) == 2, "第 3 发必须没被送进业务层"


def test_the_middleware_429_body_is_byte_identical_to_the_api_error_429(monkeypatch):
    """两条 429 路径（中间件直出 / ApiError 经异常处理器）的响应体必须同形。

    中间件里那份信封是同一形状的**第二处**实现（core 不能 import main，见
    ratelimit._limited_body 的注释），故用这条用例把两处缝在一起：文案或字段
    漂了，它当场红。后一发的 detail 带标记，顺带断言异常原文不外泄。
    """
    _env(monkeypatch, limit=1)
    app = _app(_OkAnswerer())
    app.add_api_route("/_probe/limited",
                      _raising(errors.RateLimited("SECRET-原文", retry_after=7)),
                      methods=["GET"])
    with _client(app) as client:
        assert client.post(URL, json=BODY).status_code == 200
        from_middleware = client.post(URL, json=BODY)
        from_api_error = client.get("/_probe/limited")
    assert from_middleware.status_code == from_api_error.status_code == 429
    assert _shape(from_middleware) == _shape(from_api_error)
    assert "SECRET-原文" not in from_middleware.text


def test_healthz_is_never_rate_limited_even_with_the_quota_spent(monkeypatch):
    """/healthz 不受限流（编排层的探针被 429 会与「实例不健康」不可区分）。

    先把配额打光（第二发已经是 429）再打 healthz：这样这条用例同时证明了
    「限流器在那个实例上真的武装着」—— 否则「healthz 免费」可能只是因为
    整个限流根本没生效（把中间件删掉也绿）。
    """
    _env(monkeypatch, limit=1)
    with _client(_app(_OkAnswerer())) as client:
        assert client.post(URL, json=BODY).status_code == 200
        assert client.post(URL, json=BODY).status_code == 429
        for _ in range(5):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


def test_a_body_size_rejection_still_counts_toward_the_window(monkeypatch):
    """中间件顺序判据：限流在体上限**之外**，被 413 打回的请求也计数。

    不然灌大包那条路不计数（一份 64KiB 的包换一次免罚），而限流要计的是所有
    打到公开端点的尝试。顺序若反了（体上限在外），第二发就不是 429。
    """
    _env(monkeypatch, limit=1)
    with _client(_app(_OkAnswerer())) as client:
        too_big = client.post(URL, json=HUGE)
        assert too_big.status_code == 413
        assert too_big.headers[request_id.HEADER] == too_big.json()["request_id"]
        assert client.post(URL, json=BODY).status_code == 429


# ---- Step 4：越权探测的矩阵（该公开的没被挡、该藏的没漏、藏得没有缝）----

def test_the_matrix_of_hidden_and_public_paths(monkeypatch):
    """五组探针：公开的照常答，藏起来的四组**逐字节同形**。

    组：/public/qa 匿名 200；/qa 匿名 404；/qa 匿名高频仍 404（不是 429）；/qa 错方法
    404 且无 Allow 头；未知路径 404。后四组挖掉 request_id 后必须完全一致 —— 只比
    状态码会漏掉「code/message 不同」这条缝（那正是把路径存在性漏出去的形态）。

    最后三发 /public/qa 是**武装证明**：配额只有 2，第三发 429 说明限流器在这个
    实例上真开着 —— 否则上面 7 发 /qa 全是 404 可能只是因为整个限流没生效。
    这条也是「未认证 + 高频 → 404 而不是 429」的判据（限流若先于鉴权对 /qa 生效，
    第 3 发起就是 429，与其余 404 组不同形）。
    """
    _env(monkeypatch, limit=2)
    with _client(_app(_OkAnswerer())) as client:
        anonymous = [client.post("/api/v1/qa", json=BODY) for _ in range(5)]
        wrong_method = client.get("/api/v1/qa")
        unknown = client.get("/api/v1/no-such-path")
        hidden = [*anonymous, wrong_method, unknown]
        assert [r.status_code for r in hidden] == [404] * len(hidden)
        assert len({_shape(r) for r in hidden}) == 1, "四组必须逐字节同形"
        assert "allow" not in wrong_method.headers, "Allow 头会把正确方法奉上"
        assert client.post(URL, json=BODY).status_code == 200
        assert client.post(URL, json=BODY).status_code == 200
        assert client.post(URL, json=BODY).status_code == 429, "限流器没在武装状态"


def test_the_law_prefix_is_counted_in_the_window(monkeypatch):
    """/law/* 计入限流窗口（裁决点之一，报告里单列）。

    它是匿名可读的公开路径，按 IP 计数才能挡住「一个人把整棵导航树刷成自己的
    缓存」；代价是正常用户在窗口内多翻几页法条也会计入 —— 故它的窗口配额与
    问答共用（值见报告里的算账）。
    """
    _env(monkeypatch, limit=2)
    app = _app()
    app.add_api_route("/api/v1/law/_probe", _noop, methods=["GET"])
    with _client(app) as client:
        assert client.get("/api/v1/law/_probe").status_code == 200
        assert client.get("/api/v1/law/_probe").status_code == 200
        assert client.get("/api/v1/law/_probe").status_code == 429


def test_the_law_prefix_does_not_occupy_a_concurrency_slot(monkeypatch):
    """/law/* **不**占公众侧并发槽（裁决点之二）。

    它是索引查询，与「秒级推理 + GPU」不是同一个成本模型：把浏览器加载导航树
    算进 20 个槽，等于拿贵资源给便宜端点买单（一次导航就把一份推理配额用掉）。
    判据：并发上限设为 1 并被一个问答请求占满时，/law/* 仍应 200。
    """
    _env(monkeypatch, concurrency=1, limit=100)
    answerer = _GatedAnswerer(hold=1)
    app = _app(answerer)
    app.add_api_route("/api/v1/law/_probe", _noop, methods=["GET"])
    with _client(app) as client:
        holder: list = []
        thread = threading.Thread(
            target=lambda: holder.append(client.post(URL, json=BODY)))
        thread.start()
        assert answerer.reached.wait(30), "占槽的那个请求没进业务层"
        probe = client.get("/api/v1/law/_probe")
        answerer.release.set()
        thread.join(30)
    assert probe.status_code == 200, "闸门已满，但 /law/* 不占槽，必须仍放行"
    assert holder[0].status_code == 200


# ---- 并发上限：第 21 个被拒、前 20 个释放后可再用 ----

def test_the_21st_concurrent_request_is_refused_and_the_slots_come_back(monkeypatch):
    """第 21 个并发请求被拒；一轮打完槽位全还回来，第二轮照样通过。

    只测「会拒」的话，把 release 整个删掉也绿（拒了正好，槽永远占着）—— 故判据
    必须包含第二轮。这里真起 20 个线程把请求压在业务层里，第 21 发由主线程打：
    它会在**事件循环里**（中间件那层）被拒，不需要线程池，故不会与 20 个阻塞中
    的请求抢线程。
    """
    _env(monkeypatch, concurrency=20, limit=1000)
    answerer = _GatedAnswerer(hold=20)
    with _client(_app(answerer)) as client:
        first: list = []
        threads = [threading.Thread(
            target=lambda: first.append(client.post(URL, json=BODY).status_code))
            for _ in range(20)]
        for thread in threads:
            thread.start()
        assert answerer.reached.wait(30), "20 个请求没有全部进入业务层"
        overflow = client.post(URL, json=BODY)
        answerer.release.set()
        for thread in threads:
            thread.join(30)
        assert overflow.status_code == 429
        assert overflow.json()["error"]["code"] == "rate_limited"
        assert first == [200] * 20
        second = [client.post(URL, json=BODY).status_code for _ in range(20)]
    assert second == [200] * 20, "槽位没还回来：第二轮会被自己的上一轮挡住"


def test_a_failed_public_request_does_not_leak_its_concurrency_slot(monkeypatch):
    """异常路径也要还槽（finally 而非正常出口）—— 泄漏的形态是永久 429。

    并发上限设为 1：第一发在业务里炸出 500（走异常路径），第二发必须还能进得来
    （仍是 500）。若 release 只在正常路径上做，第二发会变成 429 —— 而那种坏法
    不会报错、不会有日志，几十次之后公众侧就再也答不了。
    """
    _env(monkeypatch, concurrency=1, limit=100)
    with _client(_app(_RaisingAnswerer())) as client:
        assert client.post(URL, json=BODY).status_code == 500
        second = client.post(URL, json=BODY)
    assert second.status_code == 500, "槽位被第一发的异常路径泄漏了"


# ---- 缺盐：配置故障不许伪装成业务拒绝 ----

def test_a_missing_rate_salt_is_a_500_and_not_a_429(monkeypatch):
    """缺加盐值时是 500（配置故障），不是 429（业务拒绝）。

    三个方向都要钉：状态码不是 429（不能拿假业务理由骗调用方）、不是 200
    （不能悄悄不限流 —— fail-open 是本项目最怕的静默失效）、响应里没有
    变量名/异常原文（§七 通用文案）。
    """
    monkeypatch.delenv(config.ENV_RATE_LIMIT_SALT)
    with _client(_app(_OkAnswerer())) as client:
        response = client.post(URL, json=BODY)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert config.ENV_RATE_LIMIT_SALT not in response.text
    assert "MissingRateSaltError" not in response.text


# ---- 清单防漂移：受限的路径集合恰好是公众可读的那几条 ----

def test_only_the_public_read_routes_and_login_of_the_real_route_table_are_limited():
    """路由表里的受限路径**恰好**是公众可读的四条 + 登录（终审 I-3 起）。

    写成字面量集合而不是「照着 PUBLIC_READ_PREFIXES 算一遍」：后者在有人新增
    一条 /api/v1/public/* 路由时恒绿，而「新端点默不默认受限」正是要有人看一眼
    的事（写端点上若不设防，形态是静默敞开）。
    """
    app = main.create_app(services_factory=fake_factory()[0])
    limited = {route.path for route in app.routes
               if ratelimit.is_limited_path(route.path)}
    assert limited == {"/api/v1/public/qa", "/api/v1/lawyers/recommend",
                       "/api/v1/law/nav",
                       "/api/v1/law/{law_id}/articles/{article_no}",
                       LOGIN}
    # 律师侧不在清单里：内网自己人，卡它等于让一个人的长问句把同事挡在门外
    for path in ("/api/v1/qa", "/healthz"):
        assert ratelimit.is_limited_path(path) is False


def test_login_attempts_are_rate_limited_with_retry_after(monkeypatch):
    """登录进限流窗口（终审 I-3）：超阈值 429 + Retry-After，不再是无限次 401。

    登录匿名可达，每次 scrypt ~40ms/16MiB（DUMMY_HASH 让查无此人照付），且与
    律师侧共用 40 个 worker —— 不限流时既能无上限撞库，也能靠它把线程池打满、
    饿住律师侧全部 def 端点。判据三向：前两发 401（真走了业务）、第三发 429、
    倒计时是窗口量级（常量 1 之类的假值会让客户端每秒重试）。
    """
    _env(monkeypatch, limit=2, window=60)
    services = FakeServices(account_conn=FakeConn(row=None))  # 查无此人 → 401
    body = {"username": "nobody", "password": "pw-123456"}
    with _client(_app(services=services)) as client:
        first = [client.post(LOGIN, json=body) for _ in range(2)]
        blocked = client.post(LOGIN, json=body)
    assert [r.status_code for r in first] == [401, 401]
    assert blocked.status_code == 429
    assert 30 <= int(blocked.headers["Retry-After"]) <= 60
    assert blocked.json()["error"]["code"] == "rate_limited"
