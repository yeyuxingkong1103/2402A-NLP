# 鉴权依赖（api/deps.py）的判据：设计 §二 第 9 条（未认证/越权一律 404、不暴露
# 路径存在性）、§七 第 2 行，以及任务 2 复审交接的「停用即时生效」与连接策略。
#
# 判据挂在**探针路由**上（本项目既有的做法，见 test_main_errors.py 的 _app）：
# 律师侧真路由是任务 5 的交付，本任务没有它们，而鉴权依赖现在就要求被钉住。
# 探针路由走的是真 Depends(deps.current_user)，不是测试另写的一份鉴权。
#
# 登录端点（换 token 那一步）的判据在 test_api_auth.py；共用件在 _fakes_http.py。
import logging
import uuid

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app import main
from app.api import deps
from app.core import factory, request_id, security
from app.core.security import CurrentUser, Role
from app.db import users
from app.db.mysql import connect, connect_autocommit
from tests._fakes_http import ACCOUNT_PASSWORD, JWT_SECRET, bearer, fake_factory
from tests._fakes_http import jwt_token, without_request_id, FakeConn, FakeServices


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """签名密钥走环境变量（security.load_secret 的唯一来源）。

    autouse 的理由与 test_api_auth.py 那份相同：漏设的用例会以 MissingSecretError
    的形态变成 500，而那不是它们要测的东西；密钥值仍是 _fakes_http.JWT_SECRET 一处。
    """
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)


def _whoami(user: CurrentUser = Depends(deps.current_user)) -> dict:
    """已认证探针：把上下文回显，用例据此断言 token 里的身份真的到了处理函数。"""
    return {"id": user.id, "role": user.role.value, "team_id": user.team_id}


def _partner_only(user: CurrentUser = Depends(deps.require_roles(Role.PARTNER))) -> dict:
    """角色门探针：只有 partner 能过；角色不够与没带 token 对外必须同一种响应。"""
    return {"id": user.id}


def _probe_app(services: FakeServices):
    """真 app + 两个探针路由（鉴权语义来自真装配的异常映射，不是测试自己挂的）。"""
    factory_fn, _ = fake_factory(services)
    app = main.create_app(services_factory=factory_fn)
    app.add_api_route("/_probe/whoami", _whoami, methods=["GET"])
    app.add_api_route("/_probe/partner", _partner_only, methods=["GET"])
    return app


def test_a_valid_token_reaches_the_handler_with_its_identity():
    """有效 token 要能走到处理函数，并把**载荷里的**身份带过去。

    只断言 200 不够：把 deps 改成返回一个写死的 CurrentUser 也照样 200，而那时
    「谁在调接口」这件事已经错了（审计与数据层隔离都建立在它上面）。
    """
    app = _probe_app(FakeServices(account_conn=FakeConn(row=(1,))))
    with TestClient(app) as client:
        response = client.get("/_probe/whoami", headers=bearer(jwt_token()))
    assert response.status_code == 200
    assert response.json() == {"id": 7, "role": "lawyer", "team_id": "team-a"}


def test_the_account_status_is_checked_per_request_by_primary_key():
    """三条判据一起钉：①**逐请求**查一次（不是签发时查一次）；②查的是主键 id
    （用户名可变，这条判据不该随它漂）；③走专用账号连接，业务连接一个语句都不发。

    反例（改坏的方向）：去掉这次查询，token 在有效期内一直有效 —— 停用只对新登录
    生效，离职的人手里那个 token 还能用满 8 小时。
    """
    account = FakeConn(row=(1,))
    business = FakeConn(row=(1,))
    app = _probe_app(FakeServices(account_conn=account, conn=business))
    with TestClient(app) as client:
        for _ in range(3):
            headers = bearer(jwt_token())
            assert client.get("/_probe/whoami", headers=headers).status_code == 200
    assert len(account.statements) == 3, "账号状态不是逐请求查的"
    assert all("is_active" in sql for sql in account.statements)
    assert all("WHERE id" in sql for sql in account.statements), \
        "查的不是主键 id（按 username 查会随改名漂）"
    assert all("username" not in sql for sql in account.statements)
    assert business.statements == [], "账号路径用了业务连接"


def test_every_token_failure_is_404_and_looks_like_a_missing_path():
    """过期 / 伪造 / 根本不是 token：三者一律 404，且与「路径不存在」逐字段同形。

    为什么要跟「路径不存在」比：§二 第 9 条要的是**不暴露路径存在性**。若鉴权的
    404 与「没有这个路径」的 404 在文案或 error code 上有一点差别，扫描器照样能
    枚举出哪些路径存在。过期与伪造在实现里是两种异常（内部要分开计数），对外必须合流。
    """
    app = _probe_app(FakeServices(account_conn=FakeConn(row=(1,))))
    cases = {
        "expired": jwt_token(ttl_s=-10),
        "forged": jwt_token(secret="another-secret-0123456789abcdef-0123"),
        "garbage": "not-a-token-at-all",
    }
    with TestClient(app) as client:
        missing = client.get("/api/v1/no-such-path")
        for name, token in cases.items():
            response = client.get("/_probe/whoami", headers=bearer(token))
            assert response.status_code == 404, name
            assert without_request_id(response) == without_request_id(missing), name
            # §四：每个响应带 request_id（404 也要，便于按 id 捞这次探测）
            assert response.headers[request_id.HEADER] == response.json()["request_id"]


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Token abc",
                                    "Basic dXNlcjpwdw==", "abc"])
def test_a_missing_or_malformed_authorization_header_is_404(header):
    """缺头与畸形头都是 404（不是 401）：告诉调用方「你少写了 Bearer」等于承认这个
    路径存在。7 种形态一次列全 —— 只测「没有头」会漏掉「有头但形状不对」这一整类
    （后者在真浏览器/网关后面更常见）。"""
    app = _probe_app(FakeServices(account_conn=FakeConn(row=(1,))))
    headers = {} if header is None else {"Authorization": header}
    with TestClient(app) as client:
        response = client.get("/_probe/whoami", headers=headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_an_insufficient_role_is_404_and_indistinguishable_from_no_token():
    """角色不够也是 404，且与「没带 token」同形：403 会把「路径存在且我角色不够」
    从 404 里分出来 —— 比「路径存在」更细的一份情报。"""
    app = _probe_app(FakeServices(account_conn=FakeConn(row=(1,))))
    with TestClient(app) as client:
        allowed = client.get("/_probe/partner",
                             headers=bearer(jwt_token(role=Role.PARTNER)))
        denied = client.get("/_probe/partner",
                            headers=bearer(jwt_token(role=Role.LAWYER)))
        none = client.get("/_probe/partner")
    assert allowed.status_code == 200
    assert denied.status_code == none.status_code == 404
    assert without_request_id(denied) == without_request_id(none)


def test_disabling_an_account_kills_a_live_token_on_the_next_request():
    """**停用即时生效**（任务 2 复审的交接、本任务的验收点）。

    同一个 token 连打四次，中间只改库里的状态（运维脚本做的事）：停用后必须立刻
    404，启用回来又必须 200 —— 后者证明拒绝是「按当前库状态判的」而不是一次性闩锁
    （闩锁会把一次误停用变成永久拒绝，且没有红灯）。
    """
    account = FakeConn(row=(1,))
    app = _probe_app(FakeServices(account_conn=account))
    headers = bearer(jwt_token())
    with TestClient(app) as client:
        assert client.get("/_probe/whoami", headers=headers).status_code == 200
        account.row = (0,)
        assert client.get("/_probe/whoami", headers=headers).status_code == 404, \
            "停用没有立刻生效（token 仍在有效期内）"
        account.row = None
        assert client.get("/_probe/whoami", headers=headers).status_code == 404, \
            "账号已不存在却仍放行"
        account.row = (1,)
        assert client.get("/_probe/whoami", headers=headers).status_code == 200


def test_without_assembled_services_a_valid_token_gets_503_not_404():
    """装配没起来（lifespan 没跑 / 启动失败）时判 503：那是「我们坏了」，不是
    「你没登录」。压成 404 会让运维顺着鉴权查下去，而真因在启动日志里。"""
    factory_fn, _ = fake_factory()
    app = main.create_app(services_factory=factory_fn)
    app.add_api_route("/_probe/whoami", _whoami, methods=["GET"])
    # 不进 with：lifespan 不跑，app.state 上就没有 services
    probe = TestClient(app, raise_server_exceptions=False)
    response = probe.get("/_probe/whoami", headers=bearer(jwt_token()))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_a_missing_signing_secret_is_a_loud_500_not_a_disguised_404(monkeypatch, caplog):
    """没配签名密钥 → 500，且日志里**响亮**地说出是哪个环境变量。

    任务 2 的裁决：MissingSecretError 刻意不继承 AuthError，就是为了这里 —— 伪装
    成 404 的话，一台配错的机器会像「所有人的密码都错了」一样地跑着。两个方向都
    断言：响应里没有变量名（那是运维看日志时该知道的），日志里有（否则「为什么
    全是 500」要人去翻代码）。
    """
    monkeypatch.delenv(security.JWT_SECRET_ENV, raising=False)
    app = _probe_app(FakeServices(account_conn=FakeConn(row=(1,))))
    with caplog.at_level(logging.ERROR):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/_probe/whoami", headers=bearer(jwt_token()))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert security.JWT_SECRET_ENV not in response.text
    assert security.JWT_SECRET_ENV in caplog.text


# ---- 真连库：停用即时生效的端到端验收（含连接策略的验收）----


def _mysql_online() -> bool:
    """MySQL 在线才跑真依赖用例（离线时跳过，而不是假装通过）。"""
    try:
        connect().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mysql_online(), reason="MySQL 未在线")
def test_disabling_a_real_account_blocks_a_live_token_immediately():
    """项目惯例：能用真依赖就真连 —— 这条判据的价值全在真库那一列上。

    全程走真 MySQL：运维连接建账号 → 服务登录拿 token → 运维连接停用并提交 →
    用**同一个** token 再打一次必须 404。

    为什么替身答不了：替身里的 is_active 是测试自己摆的，它证明不了「服务那条连接
    真的看得见别的会话提交的 UPDATE」。而这里恰好有一个实测过的陷阱 ——
    autocommit=False 的业务连接会停在启动时的 REPEATABLE READ 快照上，停用永远
    读不到（见 db/mysql.connect_autocommit 的 docstring）。故这条用例同时在验收
    账号连接这个设计：把它换回业务连接（下面那个 business）时，第二枪会读到旧
    快照、返回 200，本用例当场变红。
    """
    admin = connect()
    business = connect()
    account = connect_autocommit()
    username = f"t4-{uuid.uuid4().hex[:12]}"
    try:
        users.ensure_table(admin)
        users.create(admin, username, security.hash_password(ACCOUNT_PASSWORD),
                     "lawyer", "team-a")
        # 复现真服务的启动状态：业务连接先读过一次（真服务里是启动时的 healthz
        # 探针），于是它从此停在那一刻的快照上。这一步不是装饰：少了它，「拿业务
        # 连接查 is_active」的旧写法反而会读对（它的第一次读就是新鲜读），上面
        # 说的那个变红也就不成立了
        assert users.get_by_username(business, username) is not None

        def services_fn():
            return factory.Services(conn=business, client=None, encoder=None,
                                    reranker=None, answerer=None,
                                    account_conn=account)

        app = main.create_app(services_factory=services_fn)
        app.add_api_route("/_probe/whoami", _whoami, methods=["GET"])
        with TestClient(app) as client:
            login = client.post("/api/v1/auth/login",
                                json={"username": username,
                                      "password": ACCOUNT_PASSWORD})
            assert login.status_code == 200, login.text
            headers = bearer(login.json()["access_token"])
            assert client.get("/_probe/whoami", headers=headers).status_code == 200
            # 停用发生在**另一个会话**上（运维脚本就是这么干的），提交后服务
            # 手里那条连接必须立刻看得见
            assert users.set_active(admin, username, False) is True
            assert client.get("/_probe/whoami", headers=headers).status_code == 404
            assert users.set_active(admin, username, True) is True
            assert client.get("/_probe/whoami", headers=headers).status_code == 200
    finally:
        # 只清本用例建的这一行（users 表跑完留 0 行）；删除走 admin 那条连接，
        # 另外两条已由 lifespan 的 close 关掉，这里再关一次会抛「Already closed」
        with admin.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
        admin.commit()
        admin.close()
