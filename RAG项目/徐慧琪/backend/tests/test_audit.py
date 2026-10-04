# 审计表与写入的判据（设计 §五、§六审计段；FR-6.3、AC-11、AC-19）。
#
# 真连 MySQL（项目惯例：能用真依赖就不用替身）—— 它同时是「建表语句与库里的表一致」
# 的集成测，以及 AC-19 硬边界的**落库检查**：走公众侧端点真发一次请求，再查库里那
# 一行。每个用例自建行、按 request_id 精确清理，重复跑不互相干扰、也不留垃圾。
import datetime as dt
import json
import logging
import pathlib
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import lawyer as lawyer_mod
from app.core import audit, config, security
from app.db import audit as audit_db
from app.db.mysql import SCHEMA_PATH, connect
from app.generation.answer import QAResult
from app.retrieval.pipeline import RetrievalResult
from tests._fakes_http import (JWT_SECRET, FakeConn, FakeServices, account_row,
                               bearer, fake_factory, jwt_token)

# 问句原文：AC-19 要断言它**没有**落进公众侧那一行，故必须是个一眼认得出的串
QUESTION = "房东不退押金怎么办-审计标记"
# 设计 §五 的列，一个不多一个不少。白名单而不是黑名单：黑名单天然漏项，而这张表
# 是要装身份与问句的（AC-19 的落库检查全靠它）
EXPECTED_COLUMNS = {"id", "ts", "request_id", "user_id", "role", "team_id", "action",
                    "query_text", "recall_path", "verify_result", "status_code",
                    "latency_ms"}
DESIGN_ACTIONS = {"login", "qa", "search", "article", "nav", "recommend", "export"}


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """签名密钥走环境变量：漏设的用例会以 MissingSecretError 的形态变成 500，
    而那不是它们要测的东西（与 test_api_lawyer / test_api_auth 同形）。"""
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


# 本用例写过的 request_id（_rid 登记）。清理走 conn 夹具的收尾，而不是各用例
# 末尾的 _forget：**断言失败的用例也要被清干净**，否则留下的行会在下一次跑时冒充
# 「本次插入」（test_fee_log 吃过这个亏，本任务早期一次失败运行就留下过两行）
_WRITTEN: list[str] = []


@pytest.fixture()
def conn():
    """真连接（扮演业务连接的角色）；建表、收尾清理、关闭都在这里。

    清理放夹具而不是各用例末尾：断言失败时用例末尾的 _forget 根本走不到，而留下的
    行（本表的 request_id 是唯一标记）会在下一次跑时让「查得到」来自陈货。
    """
    connection = connect()
    audit_db.ensure_table(connection)
    _WRITTEN.clear()
    yield connection
    _forget(connection, *_WRITTEN)
    _WRITTEN.clear()
    connection.close()


def _rid(prefix: str) -> str:
    """本次运行唯一的请求 ID。**用例自带 X-Request-ID 头**而不是读响应里的 id：
    按一个事先不知道的值删行，写不出「精确清理」这四个字。登记进 _WRITTEN，
    由 conn 夹具在收尾（含失败路径）统一删。"""
    rid = f"t7-{prefix}-{uuid.uuid4().hex[:12]}"
    _WRITTEN.append(rid)
    return rid


def _row(conn, request_id: str) -> dict | None:
    """按唯一 request_id 取本用例写的那一行（列序 = db/audit._COLUMNS）。"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(audit_db._COLUMNS)} FROM audit_log "
                    "WHERE request_id = %s ORDER BY id DESC LIMIT 1", (request_id,))
        row = cur.fetchone()
    return None if row is None else dict(zip(audit_db._COLUMNS, row))


def _count(conn, request_id: str) -> int:
    """这一 id 落了几行。判「一次请求恰好一行」用它（只查一行会漏掉重复写）。"""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audit_log WHERE request_id = %s",
                    (request_id,))
        return int(cur.fetchone()[0])


def _forget(conn, *request_ids: str) -> None:
    """删掉本用例写的行（只删自己那几个，绝不动别的行）。"""
    if not request_ids:
        return
    with conn.cursor() as cur:
        cur.executemany("DELETE FROM audit_log WHERE request_id = %s",
                        [(rid,) for rid in request_ids])
    conn.commit()


class _Answerer:
    """假 Answerer：返回预置的真 QAResult（审计从响应契约读它的字段，替身会让
    字段改名漂过去）。"""

    def __init__(self, result: QAResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def answer(self, question: str, side: str) -> QAResult:
        self.calls.append((question, side))
        return self.result


def _qa_result() -> QAResult:
    """ok 终态 + 两条命中块：recall_path（sources 的条号）与 verify_result 的原料。"""
    return QAResult(status="ok", answer="依《民法典》第五百八十四条……",
                    sources=[{"article_no": 584}, {"article_no": 585}])


class _Retriever:
    """假 retrieve：返回预置的 RetrievalResult（判的是审计取材，不是检索本身）。"""

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    def __call__(self, question: str, **kwargs) -> RetrievalResult:
        return self.result


def _search_result() -> RetrievalResult:
    return RetrievalResult(question=QUESTION, blocks=[{"article_no": 584}],
                           exact_nos=[584])


def _app(conn, *, answerer=None, account=None):
    """真 create_app + 假 Services（conn 是真连接：审计要真落库）。"""
    services = FakeServices(conn=conn, answerer=answerer,
                            account_conn=account if account is not None
                            else FakeConn(row=(1,)))
    factory_fn, _ = fake_factory(services)
    return main.create_app(services_factory=factory_fn)


# ---- AC-19：公众侧那一行不许有问句、不许有身份（两个方向都测）----

def test_the_public_side_row_has_no_question_and_no_identity(conn):
    """AC-19 的硬判据：**真发一次公众侧请求**，再查库里那一行。

    为什么必须真连库：这条边界的失败形态是「行照写、只是多带了两格」，而替身里
    那两格本来就是 None —— 判据落在**库里的行**上，才与 AC-19 的「落库检查」同一。
    """
    rid = _rid("public")
    app = _app(conn, answerer=_Answerer(_qa_result()))
    with TestClient(app) as client:
        response = client.post("/api/v1/public/qa", json={"question": QUESTION},
                               headers={"X-Request-ID": rid})
    assert response.status_code == 200
    row = _row(conn, rid)
    assert row is not None, "公众侧请求没有留下审计行"
    assert row["query_text"] is None, f"公众侧记了提问原文：{row['query_text']!r}"
    assert (row["user_id"], row["role"], row["team_id"]) == (None, None, None), \
        "公众侧行带了身份"
    assert row["action"] == "qa" and row["status_code"] == 200
    assert _count(conn, rid) == 1, "一次请求写了不止一行"
    _forget(conn, rid)


def test_a_public_read_path_stays_anonymous_even_with_a_valid_token(conn):
    """AC-19 的公开侧方向：公开路径 + 有效 token 仍必须三格全 NULL。

    终审探针实测：没有路径 gate 时，同一路径带 token 就落 user_id=7/role/
    team_id（验收行 1698/1699 即实例）——「公众侧恒 NULL」只靠「公众通常不带
    token」这个偶然而成立。反向（律师侧三格仍在）由下一条用例承重：两向合起来，
    「gate 写反」与「两边都不记」两类退化都挡得住。
    """
    rid = _rid("public-token")
    app = _app(conn, answerer=_Answerer(_qa_result()))
    with TestClient(app) as client:
        headers = {**bearer(jwt_token()), "X-Request-ID": rid}
        assert client.post("/api/v1/public/qa", json={"question": QUESTION},
                           headers=headers).status_code == 200
    row = _row(conn, rid)
    assert row is not None, "公众侧请求没有留下审计行"
    assert (row["user_id"], row["role"], row["team_id"]) == (None, None, None), \
        f"公开路径带 token 时落了身份：{row}"
    _forget(conn, rid)


def test_the_lawyer_side_row_carries_the_question_and_the_identity(conn):
    """反例方向：**只测公众侧那一向挡不住「两边都不写了」这种退化**（把取问句的
    整支删掉，AC-19 那条照样绿）。同一串再走一次律师侧，断言问句、身份、action、
    召回路径与校验结果都落进了各自那一格（I-2 的新用例与本条互为反向）。
    """
    rid = _rid("lawyer")
    app = _app(conn, answerer=_Answerer(_qa_result()))
    with TestClient(app) as client:
        response = client.post("/api/v1/qa", json={"question": QUESTION},
                               headers={**bearer(jwt_token()), "X-Request-ID": rid})
    assert response.status_code == 200
    row = _row(conn, rid)
    assert row is not None, "律师侧请求没有留下审计行"
    assert row["query_text"] == QUESTION, "律师侧的提问原文没有落库"
    # 身份来自 token 载荷（验签后还原），三格逐格比：只查 user_id 时把 role 写串
    # 到 team_id 那一格照样绿，而这两格是三层隔离的依据
    assert (row["user_id"], row["role"], row["team_id"]) == (7, "lawyer", "team-a")
    assert row["action"] == "qa" and row["status_code"] == 200
    # 召回路径 = QAResult.sources 里的条号（FR-6.3）；verify_result = QAResult.status
    # 原值（ok = 引用全过）。两格都不是编的：改坏 sources 的条号它们立刻跟着变
    assert row["recall_path"] == "584,585", f"召回路径不对：{row['recall_path']!r}"
    assert row["verify_result"] == "ok"
    assert row["latency_ms"] >= 0 and row["ts"] is not None
    _forget(conn, rid)


def test_the_search_row_carries_the_question_too(conn, monkeypatch):
    """`/search` 也是律师侧的**取问句**路径（`_QUESTION_PATHS` 两条之一），而它
    落库的 `query_text` 此前没有任何判据：把 `/api/v1/search` 从 `_QUESTION_PATHS`
    里删掉时，三份审计用例 27 条**全绿**（审查 N11）—— 那条路径的问句静默变 NULL，
    看起来与「律师从不提问」一模一样。判据落在库里的行上（替身那格本来就是
    None）；检索用预置结果的 retrieve 替身（打桩点与 test_api_lawyer 同），
    真链路要 2GB 模型 + 真 Milvus，不该为「审计取不取问句」付费。
    """
    rid = _rid("search")
    monkeypatch.setattr(lawyer_mod, "retrieve", _Retriever(_search_result()))
    app = _app(conn, answerer=_Answerer(_qa_result()))
    with TestClient(app) as client:
        response = client.post("/api/v1/search", json={"question": QUESTION},
                               headers={**bearer(jwt_token()), "X-Request-ID": rid})
    assert response.status_code == 200, response.text
    row = _row(conn, rid)
    assert row is not None, "/search 没有留下审计行"
    assert row["action"] == "search"
    assert row["query_text"] == QUESTION, \
        f"/search 的问句没有落库（_QUESTION_PATHS 里少了这条路径？）：{row['query_text']!r}"
    _forget(conn, rid)


def test_the_login_row_gets_its_identity_from_the_token_it_just_issued(conn):
    """login 请求上没有 token（正是来换 token 的），身份只能从**响应里刚签发的
    access_token** 还原 —— 这条钉住那半边。成功登录那一行必须能回答「谁在什么时候
    登录了」（FR-6.3 的「谁」）；失败登录没有 token，记 NULL 是诚实的。
    """
    row7 = account_row(user_id=7, username="wangsan")
    rid = _rid("login")
    app = _app(conn, account=FakeConn(row=row7))
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/login",
                               json={"username": "wangsan", "password": "pw-123456"},
                               headers={"X-Request-ID": rid})
    assert response.status_code == 200, response.text
    row = _row(conn, rid)
    assert row is not None, "登录没有留下审计行"
    assert row["action"] == "login"
    assert (row["user_id"], row["role"], row["team_id"]) == (7, "lawyer", "team-a")
    assert row["query_text"] is None, "登录行不该有问句"
    _forget(conn, rid)


# ---- 哪些请求入账（以及被拒的也要入账）----

def test_paths_outside_the_action_list_are_not_audited(conn):
    """action 列只有设计 §五 那七种取值，/healthz 与未知路径都不在内 —— 给它们
    「先编一个 action」等于往审计里掺假数据（AC-11 要的是完整率，不是数量）。
    同时钉住「条文路径」这一条动态形状真的被认出来了（前缀 + 三段）。
    """
    health = _rid("health")
    unknown = _rid("unknown")
    article = _rid("article")
    app = _app(conn)
    with TestClient(app) as client:
        assert client.get("/healthz",
                          headers={"X-Request-ID": health}).status_code == 200
        assert client.get("/api/v1/nope",
                          headers={"X-Request-ID": unknown}).status_code == 404
        got = client.get("/api/v1/law/minfadian/articles/1",
                         headers={"X-Request-ID": article})
    assert got.status_code == 200
    assert _count(conn, health) == 0, "healthz 不该入账"
    assert _count(conn, unknown) == 0, "未知路径不该入账"
    assert _row(conn, article)["action"] == "article", "动态条文路径没被认出来"
    _forget(conn, article)


def test_a_rate_limited_request_is_still_audited(conn, monkeypatch):
    """被 429 拦下的请求**也要入账** —— 它是 FR-9.1 里「拦截率」的唯一来源，而
    审计中间件因此装在限流之外（顺序见 main.create_app）。放进限流里面时，被拦的
    请求一行都不落，指标恒为 0，而 0 看起来最正常。
    """
    monkeypatch.setenv(config.ENV_RATE_LIMIT_MAX, "1")
    monkeypatch.setenv(config.ENV_RATE_LIMIT_WINDOW_S, "60")
    first, second = _rid("rl-1"), _rid("rl-2")
    app = _app(conn, answerer=_Answerer(_qa_result()))
    with TestClient(app) as client:
        assert client.post("/api/v1/public/qa", json={"question": QUESTION},
                           headers={"X-Request-ID": first}).status_code == 200
        blocked = client.post("/api/v1/public/qa", json={"question": QUESTION},
                              headers={"X-Request-ID": second})
    assert blocked.status_code == 429, blocked.text
    row = _row(conn, second)
    assert row is not None, "被限流的请求没有入账（拦截率就没有来源了）"
    assert row["status_code"] == 429
    assert row["query_text"] is None, "公众侧被拦的那一行也不许带问句"
    _forget(conn, first, second)


def test_a_failing_write_keeps_the_request_and_logs_a_warning(conn, caplog):
    """留痕失败**不阻断请求**，但必须留一条 warning（③b-1 的教训原话：静默吞掉的
    故障信号会让证据链连续几小时写不进去而全绿）。

    判据是**日志记录**而不是响应：只断言响应的写法，把 warning 删掉照样绿（响应
    本来就不受写失败影响）。故断言记录的三样属性 —— 等级、日志器名、request_id ——
    再拿注入异常里的唯一标记确认「这一条说的是这次失败」，而不是别处飘来的一条
    warning 冒充。
    """
    app = _app(conn, answerer=_Answerer(_qa_result()))
    marker = "MARK-审计表炸了"

    def boom(connection, fields):
        raise RuntimeError(marker)

    app.state.audit_writer = boom
    rid = _rid("boom")
    with caplog.at_level(logging.WARNING, logger="app.core.audit"):
        with TestClient(app) as client:
            response = client.post("/api/v1/public/qa", json={"question": QUESTION},
                                   headers={"X-Request-ID": rid})
    assert response.status_code == 200, "留痕失败把请求带崩了"
    records = [r for r in caplog.records
               if r.levelno == logging.WARNING and r.name == "app.core.audit"]
    assert records, "留痕失败没有留下任何 warning"
    assert any(r.request_id == rid and marker in r.getMessage() for r in records), \
        f"warning 与这次失败对不上号：{[(r.request_id, r.getMessage()) for r in records]}"
    assert _count(conn, rid) == 0, "写入替身抛了异常，却仍有行落库"


def test_every_design_action_maps_to_a_real_route_and_back():
    """七种 action 与路由表**双向**对齐：每个 action 都有一条真实路由（漏一条的
    表现是那条端点静默不入账），而映射表里的每条路径也必须是真路由（路由改名后
    忘了改这里，映射就永远匹配不上 —— 同样是静默不入账）。
    """
    app = main.create_app(services_factory=fake_factory()[0])
    routes = {(method, route.path) for route in app.routes
              for method in getattr(route, "methods", ()) or ()}
    covered = {audit.action_for(method, path) for method, path in routes}
    assert DESIGN_ACTIONS <= covered, f"有 action 没有对应的真路由：{DESIGN_ACTIONS - covered}"
    for method, path in audit._ACTION_PATHS:
        assert (method, path) in routes, f"映射表里的 {method} {path} 不是真路由"


def test_action_for_takes_only_the_exact_shape():
    """形状判据的边角：方法不对、路径多一段/少一段、少前缀都不算那条 action
    （记错 action 比不记更糟 —— 它会让按 action 统计出来的账目对不上）。
    """
    assert audit.action_for("GET", "/api/v1/qa") is None       # 方法不对（这条只收 POST）
    assert audit.action_for("POST", "/api/v1/law/nav") is None  # 路径对、方法不对
    assert audit.action_for("GET", "/api/v1/law/x/articles/1/extra") is None
    assert audit.action_for("GET", "/api/v1/law//articles/1") is None
    assert audit.action_for("GET", "/law/nav") is None          # 少了前缀
    assert audit.action_for("GET", "/api/v1/law/x/articles/1") == "article"
