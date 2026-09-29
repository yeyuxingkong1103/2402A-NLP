# tests/test_chat.py
"""问答接口测试（集成测试，会真实调用大模型，耗时较长）。"""
import pytest

from tests.conftest import BASE_URL


class TestChatValidation:
    """参数校验分支——这些不触发大模型，跑得很快。"""

    def test_empty_question_rejected(self, client):
        r = client.post(BASE_URL + "/api/chat/ask",
                        json={"question": "", "role_key": "lawyer", "user_id": 1},
                        timeout=30)
        # Pydantic 的 min_length 校验
        assert r.status_code == 422

    def test_missing_role_rejected(self, client):
        r = client.post(BASE_URL + "/api/chat/ask",
                        json={"question": "测试", "user_id": 1}, timeout=30)
        assert r.status_code == 422

    def test_unknown_role_rejected(self, client):
        r = client.post(BASE_URL + "/api/chat/ask",
                        json={"question": "测试问题", "role_key": "no_such_role",
                              "user_id": 1},
                        timeout=60)
        assert r.status_code == 400
        assert "角色" in r.json()["detail"]


@pytest.mark.slow
class TestChatFlow:

    def test_ask_lawyer_returns_answer_with_sources(self, chat):
        r = chat("酒驾撞人要判多久？", role_key="lawyer")
        assert r.status_code == 200
        data = r.json()["data"]

        assert data["answer"].strip(), "回答为空"
        assert data["session_id"]
        assert data["role"]["role_key"] == "lawyer"
        # 应当检索到知识，而不是空手让模型自由发挥
        assert len(data["sources"]) > 0, "没有检索到任何资料"
        for s in data["sources"]:
            assert s["title"]

    def test_disclaimer_appended(self, chat, client):
        """每个角色的免责声明必须自动附在回答末尾。"""
        role = client.get(BASE_URL + "/api/roles/lawyer", timeout=15).json()["data"]
        r = chat("借钱不还怎么办？", role_key="lawyer")
        answer = r.json()["data"]["answer"]
        assert role["disclaimer"][:20] in answer, "回答末尾缺少该角色的免责声明"

    def test_multi_turn_session_continuity(self, chat):
        """带 session_id 追问应落回同一会话。"""
        r1 = chat("交通事故致人死亡怎么判？", role_key="lawyer")
        sid = r1.json()["data"]["session_id"]

        r2 = chat("那要是逃逸了呢？", role_key="lawyer", session_id=sid)
        assert r2.status_code == 200
        d2 = r2.json()["data"]
        assert d2["session_id"] == sid, "追问没有落到同一个会话"
        # 追问改写应把指代补全，检索用的问题不该还是「那要是逃逸了呢」
        assert d2["search_query"] != "那要是逃逸了呢？", "追问未被改写"

    def test_metadata_route_hits_statute(self, chat):
        """带法条号的问题应能精确命中法条原文。"""
        r = chat("刑法第一百三十三条是怎么规定的？", role_key="lawyer")
        data = r.json()["data"]
        titles = " ".join(s["title"] for s in data["sources"])
        assert "刑法" in titles
        # 回答里应出现条文原文的关键表述
        assert "交通运输管理法规" in data["answer"] or "交通肇事" in data["answer"]

    def test_out_of_domain_uses_fallback_not_fabrication(self, chat):
        """知识库外的问题应走兜底，而不是编造。"""
        r = chat("请预测一下明天上证指数的收盘点位。", role_key="financial_advisor")
        assert r.status_code == 200
        answer = r.json()["data"]["answer"]
        assert answer.strip()
        # 明确不应当给出具体点位预测
        assert "不构成具体投资建议" in answer or "风险" in answer

    def test_session_ownership_enforced(self, chat, client):
        """用别人的 session_id 提问应被拒绝。"""
        r1 = chat("公司不签劳动合同怎么办？", role_key="lawyer", user_id=1)
        sid = r1.json()["data"]["session_id"]

        r2 = client.post(BASE_URL + "/api/chat/ask",
                         json={"question": "继续问", "role_key": "lawyer",
                               "user_id": 99999, "session_id": sid},
                         timeout=120)
        assert r2.status_code == 403


@pytest.mark.slow
class TestChatStream:

    def test_stream_emits_meta_delta_done(self, base_url, timeout):
        """SSE 流式：先 meta（含来源），再 delta 增量，最后 done。"""
        import json as _json
        import requests

        with requests.post(base_url + "/api/chat/stream",
                           json={"question": "盗窃罪怎么判？", "role_key": "lawyer",
                                 "user_id": 1},
                           stream=True, timeout=timeout) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            kinds, text = [], []
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                item = _json.loads(raw[6:])
                kinds.append(item["type"])
                if item["type"] == "delta":
                    text.append(item.get("content", ""))
                if item["type"] == "done":
                    break

        assert kinds and kinds[0] == "meta", "首个事件应为 meta"
        assert "delta" in kinds, "没有收到正文增量"
        assert kinds[-1] == "done", "没有收到结束事件"
        assert "".join(text).strip(), "流式正文为空"
