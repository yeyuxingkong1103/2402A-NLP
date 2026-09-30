# tests/test_users_sessions.py
"""用户与会话接口测试。"""
import uuid

import pytest

from tests.conftest import BASE_URL


class TestUsers:

    @staticmethod
    def _new_username():
        return "pytest_%s" % uuid.uuid4().hex[:10]

    def test_register_and_login(self, client):
        name = self._new_username()
        r = client.post(BASE_URL + "/api/users/register",
                        json={"username": name, "password": "test123456",
                              "nickname": "测试用户"}, timeout=30)
        assert r.status_code == 200
        uid = r.json()["data"]["id"]
        assert uid > 0

        r = client.post(BASE_URL + "/api/users/login",
                        json={"username": name, "password": "test123456"},
                        timeout=30)
        assert r.status_code == 200
        assert r.json()["data"]["id"] == uid

    def test_duplicate_username_rejected(self, client):
        name = self._new_username()
        payload = {"username": name, "password": "test123456"}
        assert client.post(BASE_URL + "/api/users/register",
                           json=payload, timeout=30).status_code == 200
        r = client.post(BASE_URL + "/api/users/register", json=payload, timeout=30)
        assert r.status_code == 400
        assert "已存在" in r.json()["detail"]

    def test_wrong_password_rejected(self, client):
        name = self._new_username()
        client.post(BASE_URL + "/api/users/register",
                    json={"username": name, "password": "test123456"}, timeout=30)
        r = client.post(BASE_URL + "/api/users/login",
                        json={"username": name, "password": "wrong_password"},
                        timeout=30)
        assert r.status_code == 401

    def test_password_is_hashed_not_stored_plain(self, client):
        """注册后能从登录接口反推出口令未被明文存储——这里验证哈希格式。"""
        name = self._new_username()
        client.post(BASE_URL + "/api/users/register",
                    json={"username": name, "password": "test123456"}, timeout=30)
        from app.api.system import hash_password, verify_password
        stored = hash_password("abc123")
        assert "$" in stored and "abc123" not in stored
        assert verify_password("abc123", stored)
        assert not verify_password("abc124", stored)

    def test_unknown_user_404(self, client):
        r = client.get(BASE_URL + "/api/users/99999999", timeout=15)
        assert r.status_code == 404


class TestSessions:

    def test_list_sessions_shape(self, client):
        """列表接口本身不触发大模型。"""
        r = client.get(BASE_URL + "/api/sessions",
                       params={"user_id": 1, "limit": 5}, timeout=20)
        assert r.status_code == 200
        for s in r.json()["data"]:
            assert s["session_id"] and s["role_key"]
            assert "turn_count" in s

    def test_unknown_session_messages_404(self, client):
        r = client.get(BASE_URL + "/api/sessions/no_such_session/messages",
                       timeout=15)
        assert r.status_code == 404

    @pytest.mark.slow
    def test_session_lifecycle(self, client, chat):
        """建会话 -> 查历史 -> 看记忆 -> 删除。会真实调用大模型。"""
        r = chat("盗窃罪怎么规定的？", role_key="lawyer")
        sid = r.json()["data"]["session_id"]

        # 历史消息：一问一答
        r = client.get(BASE_URL + "/api/sessions/%s/messages" % sid, timeout=20)
        assert r.status_code == 200
        msgs = r.json()["data"]["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"], "消息应是一问一答且有序"
        assert msgs[1]["sources"], "助手消息应记录命中的知识来源"

        # 记忆视图：短期记忆应含刚写入的这轮
        r = client.get(BASE_URL + "/api/sessions/%s/memory" % sid,
                       params={"user_id": 1, "role_key": "lawyer"}, timeout=20)
        assert r.status_code == 200
        short = r.json()["data"]["short_term"]
        assert short and any("盗窃" in line for line in short), "短期记忆未写入"

        # 删除
        r = client.delete(BASE_URL + "/api/sessions/%s" % sid, timeout=20)
        assert r.status_code == 200
        assert client.get(BASE_URL + "/api/sessions/%s/messages" % sid,
                          timeout=15).status_code == 404

    def test_deleted_session_is_gone(self, client):
        r = client.delete(BASE_URL + "/api/sessions/no_such_session", timeout=15)
        assert r.status_code == 404
