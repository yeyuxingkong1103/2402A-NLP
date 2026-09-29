"""tests/integration/test_api_flow.py —— 新架构的接口集成测试。

在链路中的位置：
    集成测试层（tests/integration/），与 tests/unit/ 那层相对。
    它用 FastAPI 的 TestClient 把整个应用（src/api/main.py 的 app）跑起来，
    走真实的 HTTP 路由 → 依赖注入 → 数据库，验证多个组件串起来能工作。

覆盖的流程（一条完整的主干路径）：
    注册 → （必要时登录）→ 列角色 → 建会话 → 查历史

与单元测试的分工：
    单元测试验"每个零件对不对"，本文件验"零件组装起来能不能跑"。
    比如"JWT 签发"和"JWT 解析"各自单测都过，但两边用的算法名不一致时，
    只有集成测试能发现这个问题。
"""
from fastapi.testclient import TestClient

from src.api.main import app


def test_register_role_session_flow():
    """走通"注册-取角色-建会话-查历史"这条主干路径。

    为什么 register 允许返回 409（`status_code in (200, 409)`）：
        本测试用的是固定用户名 "test-user"，而关系库是持久化的
        （默认 SQLite 文件或 MySQL）。第二次运行这个测试时该用户已存在，
        注册会返回 409 —— 这不是故障，而是"重复运行"的正常结果。

        所以代码显式处理两种情况：
            200 -> 注册成功，直接用返回的令牌
            409 -> 用户已存在，改走登录拿令牌

        这个"允许两种状态"的写法让测试**可重复执行**，
        同时也顺带验证了登录路径 —— 一举两得。
        反过来说，如果硬要求 200，这个测试就只能在一个全新的数据库上跑一次。

    with TestClient(app) as client 用上下文管理器：
        这会触发 lifespan 钩子（建表 init_db、同步角色 sync_roles、配日志），
        和真实启动服务时的行为一致。
        不用 with 的话 lifespan 不会执行，测试就会因为"表不存在"而失败。

    最后一步查 /messages 而不只查会话 id：
        验证会话确实可用（能读到它名下的消息列表）。
        只断言建会话返回 200 的话，一个"建得出来但读不出来"的 bug 会漏过去。
    """
    with TestClient(app) as client:
        resp = client.post("/api/v1/auth/register", json={"username": "test-user", "password": "password123"})
        assert resp.status_code in (200, 409)
        if resp.status_code == 409:
            # 用户已存在（重复运行时的情况）：改走登录拿令牌
            resp = client.post("/api/v1/auth/login", json={"username": "test-user", "password": "password123"})
        token = resp.json()["access_token"]
        # 后续所有请求都要带这个 Bearer 令牌，由 current_user 依赖解析出身份
        headers = {"Authorization": f"Bearer {token}"}
        roles = client.get("/api/v1/roles", headers=headers)
        assert roles.status_code == 200
        session = client.post("/api/v1/sessions", headers=headers, json={"role_id": "lawyer", "title": "测试"})
        assert session.status_code == 200
        sid = session.json()["session_id"]
        history = client.get(f"/api/v1/sessions/{sid}/messages", headers=headers)
        assert history.status_code == 200
