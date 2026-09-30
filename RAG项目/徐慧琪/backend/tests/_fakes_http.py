# HTTP 层测试共用的假体（test_main.py 与 test_main_errors.py 都用）。
#
# 为什么拆成独立模块：两个文件的用例都要「一个能过 healthz 探针的假 Services」，
# 各写一份的话，改探针契约时只会改到其中一份，另一份继续用旧形状通过 —— 那正是
# 本项目反复吃的「两处各存一份口径」。本模块名不带 test_ 前缀，pytest 不收集它。
#
# 这里全部是**轻**替身：不连库、不加载模型、不调 DeepSeek。接口层的用例测的是
# 「接线与映射」，真依赖那一路由 test_main.py 的 healthz 用例覆盖。
from __future__ import annotations

from app.core import security
from app.core.security import Role

# ---- 鉴权用例共用件（test_api_auth 与 test_api_deps 都要，故放这里）----

# 测试用的签名密钥：长度必须 ≥ security.MIN_SECRET_LEN，否则 load_secret 会拒，
# 而那时用例的失败原因会指向「密钥太短」这个与判据无关的地方
JWT_SECRET = "test-secret-0123456789abcdef-0123456789"
# 限流加盐值（任务 6）：同理，必须 ≥ config.MIN_RATE_SALT_LEN。它没有默认值 ——
# 打公开端点（限流清单里的 /public/qa、/lawyers/recommend、/law/*）的用例都得先设它，
# 缺了会以 MissingRateSaltError → 500 的形态失败（设计如此：配置故障不许伪装成业务拒绝）
RATE_SALT = "test-rate-salt-0123456789abcdef"
# 账号的口令（建号与登录用它）。与 security 的口令策略无关 —— 策略在运维脚本
ACCOUNT_PASSWORD = "pw-123456"


def account_row(user_id: int = 7, username: str = "wangsan",
                password: str = ACCOUNT_PASSWORD, role: str = "lawyer",
                team_id: str = "team-a", active: int = 1) -> tuple:
    """造一行 users（列序 = db.users._COLUMNS，供 FakeConn.row 用）。

    列序写死在这里而不是从 db.users._COLUMNS 取：引用它的话，改列序时用例会
    跟着一起改而不会红 —— 而「按位置造行」正是要被暴露的那件事。
    """
    return (user_id, username, security.hash_password(password), role, team_id,
            active)


def jwt_token(*, role: Role = Role.LAWYER, user_id: int = 7, team_id: str = "team-a",
              secret: str = JWT_SECRET, ttl_s: int = security.TOKEN_TTL_S) -> str:
    """签一个真 token。用手拼载荷的替身会在实现换字段名时照样绿（而换掉的正是
    鉴权要读的那个字段），故一律走真 security.sign_token。"""
    user = security.CurrentUser(id=user_id, role=role, team_id=team_id)
    return security.sign_token(user, secret=secret, ttl_s=ttl_s)


def bearer(token: str) -> dict:
    """Authorization 头。七个鉴权用例都要拼它，拼错一次就是一条假用例。"""
    return {"Authorization": f"Bearer {token}"}


def without_request_id(response) -> dict:
    """响应体的对外形状（去掉 request_id 再比）。

    用于「两种失败必须不可区分」的判据：只比状态码会漏掉「code/message 不同」——
    而那正是把账号存在性漏出去的那条缝。request_id 每次都不一样，且与账号无关。
    """
    body = response.json()
    body.pop("request_id", None)
    return body


class FakeConn:
    """假 MySQL 连接：默认过一次只读探针；fail=True 时在 execute 上抛。

    抛的异常文本刻意写成真实故障的原话（"MySQL server has gone away"）——
    用例拿它来断言「异常原文没进响应体」，比自造的 BOOM 更能说明问题。

    row 是 fetchone 的固定答案（任务 4 起可定制）：鉴权与登录要按行内容分支
    （is_active=0、查无此人、账号那一行的六列），而 healthz 的探针只关心
    「有没有取到行」。默认 (1,) 保持任务 3 的既有用例一字不变。
    带 params 的 execute 签名是给 db/users 那些带占位符的查询用的（它们都走
    参数化下发），只记录不解释 —— 替身不模拟 SQL 语义，只看调用序列。
    """

    def __init__(self, fail: bool = False, row=(1,)) -> None:
        self.fail = fail
        self.row = row
        self.statements: list[str] = []

    def cursor(self) -> "FakeConn":
        return self

    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.statements.append(sql)
        if self.fail:
            raise RuntimeError("MySQL server has gone away")

    def fetchone(self):
        return self.row


class FakeMilvus:
    """假 Milvus 客户端：alive=False 时 has_collection 抛（模拟服务掉了）。"""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.asked: list[str] = []

    def has_collection(self, name: str) -> bool:
        self.asked.append(name)
        if not self.alive:
            raise RuntimeError("milvus down: connection refused")
        return True


class FakeServices:
    """假的一套重资源：形状与真 Services 一致（conn / client / close）。

    encoder / reranker / answerer 默认给 None：healthz 与中间件都不碰它们，
    真给了反而暗示这些用例覆盖到了模型装配（那是 CLI 与 test_core_factory 的事）。
    answerer 与 conn 可注入（任务 4 的问答与鉴权用例要按行内容/回答分支），
    注入的仍是**轻**替身：不连库、不加载模型、不调 DeepSeek。

    account_conn 默认与 conn **同一个对象**：真 Services 里它们是两条不同的连接
    （业务连接 autocommit=False、账号连接 autocommit=True，见 factory.Services
    的字段注释），但替身不模拟事务，而这个默认值让「没传 account_conn 的用例」
    不必改写。要钉「账号路径走的是账号连接」的用例必须显式给两个不同的对象。
    """

    def __init__(self, *, mysql_ok: bool = True, milvus_ok: bool = True,
                 answerer=None, conn=None, account_conn=None) -> None:
        self.conn = conn if conn is not None else FakeConn(fail=not mysql_ok)
        self.account_conn = account_conn if account_conn is not None else self.conn
        self.client = FakeMilvus(alive=milvus_ok)
        self.encoder = None
        self.reranker = None
        self.answerer = answerer
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def fake_factory(services: FakeServices | None = None):
    """造一个零参装配器（create_app 的注入缝），可选带一个现成的 Services。

    返回 (装配器, 记装配次数的列表)：次数是「重资源绝不能被每请求装配」这条的判据。
    """
    built: list[FakeServices] = []
    fixed = services

    def factory() -> FakeServices:
        services_obj = fixed if fixed is not None else FakeServices()
        built.append(services_obj)
        return services_obj

    return factory, built


class ResultStub:
    """QAResult 的最小替身：接口层只读 status 与 answer 两个字段。"""

    def __init__(self, status: str, answer: str = "") -> None:
        self.status = status
        self.answer = answer
