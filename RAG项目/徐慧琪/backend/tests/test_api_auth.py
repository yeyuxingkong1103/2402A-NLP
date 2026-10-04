# 登录端点（`POST /api/v1/auth/login`）的判据：设计 §四 的响应契约，
# 以及「失败一律 401 且三种失败不可区分」这条红线。
#
# 鉴权依赖（token 校验、越权 404、停用即时生效、连接策略）在 test_api_deps.py ——
# 那是另一件事：一个是「换取身份」，一个是「每个受保护请求怎么核身份」。
# 共用件（假密钥、账号行、token 助手）在 _fakes_http.py，不在这里各写一份。
#
# 判据一律走真 create_app（中间件、异常映射、路由都来自真装配），重资源换替身。
import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import auth
from app.core import security
from tests._fakes_http import ACCOUNT_PASSWORD, JWT_SECRET, account_row, fake_factory
from tests._fakes_http import FakeConn, FakeServices, without_request_id

# 响应体与日志里都不许出现的口令/标记：断言「不在」时越独特越可靠
LEAK = "hunter2-secret"


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """签名密钥走环境变量（security.load_secret 的唯一来源）。

    autouse：几乎每条用例都要签发或校验 token，漏设的那几条会以
    MissingSecretError 的形态失败，而那不是它们要测的东西。I-5 起 conftest
    也全局设同一把密钥（lifespan 启动自检要求它）；密钥值仍是一处。
    """
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


def _app(services: FakeServices):
    """起一个真装配的应用（登录用例不需要探针路由：它不校验 token）。"""
    factory_fn, _ = fake_factory(services)
    return main.create_app(services_factory=factory_fn)


def _login(client: TestClient, username: str = "wangsan",
           password: str = ACCOUNT_PASSWORD):
    return client.post("/api/v1/auth/login",
                       json={"username": username, "password": password})


def _accounts(row) -> FakeServices:
    """一套假资源：账号连接的登录查询按 row 回答（None = 查无此人）。"""
    return FakeServices(account_conn=FakeConn(row=row))


# ---- 成功 ----


def test_a_correct_password_returns_a_token_and_the_identity():
    """登录契约（设计 §四）：四个键，且 token 真能还原出同一个身份。

    只断言「拿到一个非空串」是不够的：把 sign_token 换成返回固定串、或把
    team_id 写成空串，那种断言都照样绿，而后者会让三层隔离的过滤依据在客户端
    就断了。故这里用真 security.verify_token 把 token 解回来逐字段比。
    """
    with TestClient(_app(_accounts(account_row()))) as client:
        response = _login(client)
    assert response.status_code == 200
    body = response.json()
    # 键集合当字面量锚点（§四 的响应字段）：多一个键（比如顺手把 exp 也带出来）
    # 是契约变更，应当有人看一眼
    assert set(body) == {"access_token", "token_type", "role", "team_id"}
    assert body["token_type"] == "bearer"
    assert body["role"] == "lawyer"
    assert body["team_id"] == "team-a"
    user = security.verify_token(body["access_token"], secret=JWT_SECRET)
    assert (user.id, user.role, user.team_id) == (7, security.Role.LAWYER, "team-a")


def test_login_reads_the_account_by_username_with_bound_parameters():
    """查账号的 SQL 必须参数化下发：拼串会让一个含引号的用户名改写整条语句，
    而「注入下也登录成功」与正常路径在同一条用例里难以分辨 —— 故直接钉下发方式
    （语句含 %s 占位符、用户名是独立参数）。"""
    account = FakeConn(row=account_row())
    with TestClient(_app(FakeServices(account_conn=account))) as client:
        assert _login(client).status_code == 200
    assert account.statements, "登录没有查库"
    assert all("%s" in sql and "users" in sql for sql in account.statements)


# ---- 失败：三种一律 401，且逐字节不可区分 ----


def test_wrong_password_and_unknown_user_are_indistinguishable():
    """第一红线：口令不对与查无此人**必须**是同一种响应。

    只断言「都是 401」不够 —— 两个响应的 error code / message 只要有一处不同，
    探测者就能靠它枚举出哪些用户名存在（撞库的第一步）。故逐字段比对两个响应体
    （只挖掉 request_id：它每次都不一样，且与账号无关）。
    """
    with TestClient(_app(_accounts(account_row()))) as client:
        wrong = _login(client, password="not-the-password")
    with TestClient(_app(_accounts(None))) as client:
        unknown = _login(client, username="nobody-here")
    assert wrong.status_code == unknown.status_code == 401
    assert without_request_id(wrong) == without_request_id(unknown)
    assert set(wrong.json()) == {"request_id", "error"}
    assert wrong.json()["error"]["code"] == "unauthorized"
    # 文案写字面量而不是引用 errors.PUBLIC_ERRORS[401]：引用常量时，把两句文案
    # 改成分开的两句也不会红（两边同源，一起漂）
    assert wrong.json()["error"]["message"] == "用户名或密码错误"
    # 两个响应都不回显输入的用户名（回显本身就是一份可枚举的信号）
    assert "wangsan" not in wrong.text and "nobody-here" not in unknown.text


def test_a_disabled_account_fails_login_the_same_way():
    """停用账号在**登录**这一层也拒，且与口令错不可区分。

    这一条同时钉住判据是**读行里的 is_active**，而不是靠 security.from_row 抛
    PermissionDeniedError —— 后者会变成 404，与另外两种失败（401）区分开，等于
    把「这个账号存在且被停用」白送给对方。
    """
    with TestClient(_app(_accounts(account_row(active=0)))) as client:
        disabled = _login(client)
    with TestClient(_app(_accounts(account_row()))) as client:
        wrong = _login(client, password="not-the-password")
    assert disabled.status_code == 401
    assert without_request_id(disabled) == without_request_id(wrong)


@pytest.mark.parametrize("body", [
    {"username": "wangsan"},                      # 缺口令
    {"password": ACCOUNT_PASSWORD},               # 缺用户名
    {"username": "wangsan", "password": 12345},   # 口令不是字符串
    {"username": 7, "password": ACCOUNT_PASSWORD},  # 用户名不是字符串
    {"username": "wangsan", "password": ""},      # 空口令
    {"username": "  ", "password": ACCOUNT_PASSWORD},  # 空白用户名（strip 后为空）
])
def test_malformed_bodies_are_400_and_never_500(body):
    """畸形请求体一律 400（校验层），不许变成 500。

    `password` 给数字那一条是本任务点名的：pydantic v2 不做 int→str 隐式转换，
    故它在校验层被拒；少了这层，security.verify_password 会在 `password.encode()`
    上抛 AttributeError —— 一个「客户端发了畸形 JSON」表现成「服务端内部错误」。
    """
    with TestClient(_app(_accounts(account_row()))) as client:
        response = client.post("/api/v1/auth/login", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


# ---- 时间侧信道：未知用户名也要陪跑一次 scrypt ----


def test_the_unknown_user_path_still_pays_for_a_scrypt(monkeypatch):
    """未知用户名也要跑一次真 scrypt，否则账号存在性会从**耗时**漏出去。

    查无此人时直接返回比「存在则算一次哈希」快约 40ms（一次 scrypt 的量级）。
    钉法是替身而不是计时（计时用例会因机器负载偶发红）：断言 verify_password
    在未知用户名这条路上被调用过，且喂进去的正是那个陪跑哈希 —— 少任何一半
    （不调用 / 调用但喂了别的）都说明陪跑失效。
    """
    seen: list[str] = []

    def spy(password, stored):
        seen.append(stored)
        return False

    # 打在 security 模块上：auth 用 `security.verify_password(...)` 取值，
    # 与实现取的是同一处，故这条替身不会替掉别的东西
    monkeypatch.setattr(security, "verify_password", spy)
    with TestClient(_app(_accounts(None))) as client:
        assert _login(client, username="nobody-here").status_code == 401
    assert seen == [auth.DUMMY_HASH]


def test_the_decoy_hash_is_a_legal_scrypt_string():
    """陪跑哈希必须**能过 _parse**：格式非法时 verify_password 立刻返回 False，
    陪跑变成空转 —— 而上面那条替身用例照样绿（它只数调用次数与取值）。

    用私有的 _parse 是因为那正是决定「要不要真算」的那一步：这是本判据的定义，
    不是实现细节。换成按值断言（「以 scrypt$ 开头」）挡不住参数超限那类非法串
    —— 它们同样以 scrypt$ 开头，却会在 _parse 里被判 None。
    """
    assert security._parse(auth.DUMMY_HASH) is not None


def test_the_decoy_costs_exactly_what_a_real_hash_costs():
    """陪跑哈希的**代价参数**必须与真实哈希逐项相同 —— 上面两条合起来不够。

    上面两条只钉「调过 + 能解析」：任何 `SCRYPT_*` 的限内漂移都会让陪跑变便宜，
    而套件全绿。实测（2026-09-30 任务 4 审查 M5）：把 `SCRYPT_P` 从 1 改成 2
    （仍在 `_SCRYPT_LIMITS` 内，不违规）→ `test_api_auth + test_security`
    **48 passed 全绿**，而登录端点端到端耗时变成「查无此人 50.7 ms / 口令错
    98.5 ms」—— 账号存在性从**耗时**重新漏了出去，正是上面那条用例要关的门。
    形态属本项目「不可能失败的测试」家族：判据是「调用过」，性质是「同代价」。

    断言写成与 `security.SCRYPT_*` **同一个来源**会不会恒真？不会：一边漂、
    另一边不漂时它必红（漂移的形态正是这样，两处一起改等于两处一起重新对齐，
    那是合法的改动）。故这里读常量，不写死字面量 —— 写死的话，合法地调整
    scrypt 参数会连陪跑一起被判红，那是把「护栏」变成「锁死参数」。
    """
    parsed = security._parse(auth.DUMMY_HASH)
    assert parsed is not None, "先要能解析，见上一条用例"
    # _parse 的返回顺序是 (n, r, p, salt, expected)，见它的签名
    n, r, p, _salt, expected = parsed
    assert (n, r, p) == (security.SCRYPT_N, security.SCRYPT_R, security.SCRYPT_P)
    assert len(expected) == security.SCRYPT_DKLEN


# ---- 日志侧 ----


def test_a_failed_login_keeps_the_password_out_of_logs_and_body(caplog):
    """失败细节只进日志（error.detail），且**不含口令**。

    两个方向都断言：响应里没有口令（只断言「日志里没有」时，把日志整段删掉也绿）；
    日志里有那次尝试的用户名（只断言「没泄露」时，把 detail 删成空串也绿）——
    排查撞库需要它，而它既不含口令也不含哈希。
    """
    with caplog.at_level(logging.WARNING):
        with TestClient(_app(_accounts(account_row()))) as client:
            response = _login(client, password=LEAK)
    assert LEAK not in response.text
    assert LEAK not in caplog.text
    assert "wangsan" in caplog.text


# ---- 启动自检：缺密钥起不来（终审 I-5，任务 2 交接的承接）----


def test_a_missing_signing_secret_makes_startup_fail(monkeypatch):
    """缺密钥时应用**起不来** —— 不是「起得来但鉴权全 500」。

    修复前实测：healthz 200 ok、公众侧 200、登录与律师侧全 500，编排层读到的是
    「healthy」。现在 lifespan 在装配前先 load_secret()：缺配置当场抛
    MissingSecretError（不伪装成 401/500），且**重资源没被加载** —— 缺配置不该
    先花几十秒起模型。
    """
    monkeypatch.delenv(security.JWT_SECRET_ENV, raising=False)
    factory_fn, built = fake_factory()
    app = main.create_app(services_factory=factory_fn)
    with pytest.raises(security.MissingSecretError, match=security.JWT_SECRET_ENV):
        with TestClient(app):
            pass
    assert built == [], "密钥都没配就把重资源装起来了"


def test_a_configured_signing_secret_starts_normally():
    """反向：密钥在时启动必须正常（只钉「缺密钥必抛」的话，恒抛也全绿）。"""
    factory_fn, built = fake_factory()
    with TestClient(main.create_app(services_factory=factory_fn)) as client:
        assert client.get("/healthz").status_code == 200
    assert len(built) == 1
