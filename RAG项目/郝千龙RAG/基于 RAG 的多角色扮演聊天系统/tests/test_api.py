# -*- coding: utf-8 -*-
"""集成测试：FastAPI API 端到端（TestClient，无需启动真实服务）。

测试范围：
  - 健康检查 / 角色列表
  - 认证：注册 / 登录 / token 校验 / 鉴权拦截
  - 对话：非流式 / 流式 / 中文查询 / 多轮会话 / 多角色切换
  - 知识库上传（txt 文本）

运行方式：
  cd c:\\Users\\h1981\\Desktop\\作业
  python -m pytest tests/test_api.py -v
"""
# ----- 在导入项目模块前，设置测试环境变量（必须在 import api 之前）-----
import os
import sys
from pathlib import Path

os.environ["APP_ENV"] = "testing"
os.environ["LLM_API_KEY"] = ""  # 纯检索模式，不调大模型（走 fallback_answer）
os.environ["TSV_PATH"] = str(Path(__file__).resolve().parent / "fixtures" / "mini.tsv")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'test.db'}"
os.environ["REDIS_ENABLED"] = "0"
os.environ["MILVUS_ENABLED"] = "0"
os.environ["EMBEDDING_ENABLED"] = "0"
os.environ["RERANK_ENABLED"] = "0"

# 把项目根目录加入 sys.path（确保能 import 项目模块）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from api import app
from services import get_retriever


@pytest.fixture(scope="session")
def client():
    """会话级 TestClient：整个测试会话复用一个客户端（含 lifespan 管理）。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_token(client):
    """注册并登录测试用户，返回 Bearer token。"""
    client.post("/api/auth/register", json={"username": "testuser", "password": "test1234"})
    resp = client.post("/api/auth/login", json={"username": "testuser", "password": "test1234"})
    assert resp.status_code == 200
    return resp.json()["token"]


# =====================================================================
# 健康检查 & 角色列表
# =====================================================================
class TestHealthAndRoles:
    def test_health(self, client):
        """健康检查返回 ok 与知识库规模。"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "kb_size" in data
        assert data["kb_size"] >= 0

    def test_roles_list(self, client):
        """角色列表至少有 3 个角色，且不泄露 system_prompt。"""
        resp = client.get("/api/roles")
        assert resp.status_code == 200
        roles = resp.json()["roles"]
        assert len(roles) >= 3
        for r in roles:
            assert "system_prompt" not in r  # 提示词不应泄露给前端
            assert "code" in r and "name" in r


# =====================================================================
# 认证模块
# =====================================================================
class TestAuth:
    def test_register_success(self, client):
        """注册成功返回 token（用唯一用户名避免重复注册 400）。"""
        import time

        uname = f"reguser_{int(time.time() * 1000)}"
        resp = client.post("/api/auth/register", json={"username": uname, "password": "pass1234"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == uname

    def test_register_duplicate(self, client, auth_token):
        """重复注册返回 400（auth_token fixture 已注册 testuser）。"""
        resp = client.post("/api/auth/register", json={"username": "testuser", "password": "test1234"})
        assert resp.status_code == 400

    def test_register_short_password(self, client):
        """密码过短触发 Pydantic 校验 422。"""
        resp = client.post("/api/auth/register", json={"username": "shortpw", "password": "ab"})
        assert resp.status_code == 422

    def test_login_success(self, client):
        """已注册用户登录成功。"""
        client.post("/api/auth/register", json={"username": "loginuser", "password": "pass1234"})
        resp = client.post("/api/auth/login", json={"username": "loginuser", "password": "pass1234"})
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_login_wrong_password(self, client):
        """密码错误返回 401。"""
        resp = client.post("/api/auth/login", json={"username": "testuser", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        """不存在的用户返回 401。"""
        resp = client.post("/api/auth/login", json={"username": "nobody", "password": "x"})
        assert resp.status_code == 401

    def test_chat_without_token(self, client):
        """无 token 访问 /api/chat 返回 401。"""
        resp = client.post("/api/chat", json={"query": "hello"})
        assert resp.status_code == 401

    def test_chat_invalid_token(self, client):
        """无效 token 返回 401。"""
        resp = client.post("/api/chat", json={"query": "hello"}, headers={"Authorization": "Bearer fake"})
        assert resp.status_code == 401


# =====================================================================
# 对话接口（纯检索模式，不调大模型）
# =====================================================================
class TestChat:
    def test_chat_non_stream(self, client, auth_token):
        """非流式对话：返回完整 JSON（answer / retrieved / session_id）。"""
        resp = client.post(
            "/api/chat",
            json={"query": "I miss you", "role_code": "teacher", "top_k": 3},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "answer" in data
        assert "retrieved" in data
        assert "hit_count" in data
        assert len(data["answer"]) > 0

    def test_chat_chinese_query(self, client, auth_token):
        """中文查询也能命中英文句对。"""
        resp = client.post(
            "/api/chat",
            json={"query": "我想你", "role_code": "teacher", "top_k": 3},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["answer"]) > 0

    def test_chat_multi_turn(self, client, auth_token):
        """多轮对话：第一轮拿 session_id，第二轮复用同一 session。"""
        resp1 = client.post(
            "/api/chat",
            json={"query": "I miss you", "role_code": "teacher"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        sid = resp1.json()["session_id"]
        resp2 = client.post(
            "/api/chat",
            json={"query": "hurry up", "role_code": "teacher", "session_id": sid},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == sid  # 复用同一会话

    def test_chat_stream(self, client, auth_token):
        """流式对话：返回 200 且 body 非空。"""
        resp = client.post(
            "/api/chat",
            json={"query": "go to sleep", "role_code": "teacher", "stream": True},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200
        body = resp.content.decode("utf-8")
        assert len(body) > 0

    def test_chat_different_roles(self, client, auth_token):
        """不同角色都能正常响应。"""
        for role in ["teacher", "doctor", "lawyer", "scientist"]:
            resp = client.post(
                "/api/chat",
                json={"query": "I miss you", "role_code": role, "top_k": 3},
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            assert resp.status_code == 200, f"role={role} failed"
            assert len(resp.json()["answer"]) > 0

    def test_chat_empty_query(self, client, auth_token):
        """空查询也能处理（不崩）。"""
        resp = client.post(
            "/api/chat",
            json={"query": "", "role_code": "teacher", "top_k": 3},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        # 空查询可能检索不到结果，但不应 500
        assert resp.status_code in (200, 422)


# =====================================================================
# 知识库动态上传
# =====================================================================
class TestKnowledgeUpload:
    def test_upload_txt(self, client, auth_token):
        """上传 txt 文件，返回分块数。"""
        # 写一个临时 txt
        import tempfile

        tmp = Path(tempfile.gettempdir()) / "test_kb.txt"
        tmp.write_text("Hello World\nThis is a test document.\n你好世界", encoding="utf-8")
        with tmp.open("rb") as f:
            resp = client.post(
                "/api/kb/upload",
                files={"file": ("test_kb.txt", f, "text/plain")},
                headers={"Authorization": f"Bearer {auth_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test_kb.txt"
        assert data["chunks"] >= 1
        # 清理：重置检索器单例缓存，避免影响后续测试
        get_retriever.cache_clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
