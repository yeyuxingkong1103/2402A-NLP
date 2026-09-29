"""HTTP 接口（前端）测试：根路由、健康检查、对话、多轮记忆、角色管理、知识库上传/列表。

沿用 conftest 里的 fake_pipeline / client 夹具，全离线运行，不依赖 Ollama / Milvus Lite。

原先是两份文件（test_api.py + test_api_extended.py），前者 5 个用例里有 3 个被后者
完整覆盖、另 2 个（根路由、session_id 回显）并进来后删掉了——留着两份会让人以为
其中一份是过期未整理的。
"""
from __future__ import annotations

from app.main import app


# ---- 根路由 ----
def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "rag-roleplay"


# ---- 健康检查 ----
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["components"] == {
        "llm": "ok",
        "embedding": "ok",
        "milvus": "ok",
        "sql": "ok",
        "memory": "ok",
    }


# ---- 对话（非流式）----
def test_chat_contract(client):
    resp = client.post("/chat", json={"question": "高血压饮食注意什么", "role_id": "psychologist"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["answer"], str) and data["answer"]
    assert isinstance(data["sources"], list)
    assert data["session_id"] == "default"

    # 传入的 session_id 必须原样回显（前端靠它把回答对回本地会话）
    echoed = client.post(
        "/chat",
        json={"question": "高血压饮食注意什么", "role_id": "psychologist", "session_id": "t1"},
    )
    assert echoed.json()["session_id"] == "t1"


def test_chat_empty_question_rejected(client):
    resp = client.post("/chat", json={"question": "", "role_id": "psychologist"})
    assert resp.status_code == 422  # Pydantic 校验：question 至少 1 个字符


def test_chat_multi_turn_memory(client):
    session_id = "mem-test"
    for q in ("第一个问题", "第二个问题"):
        resp = client.post("/chat", json={"question": q, "role_id": "psychologist", "session_id": session_id})
        assert resp.status_code == 200

    # 通过 HTTP 两轮对话后，短期记忆应累计 2 问 2 答共 4 条（按「会话+角色」键）
    key = app.state.pipeline._memory_key(session_id, "psychologist")
    history = app.state.pipeline.memory.get_history(key)
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]


def test_memory_is_role_scoped(client):
    """切换角色不串上下文：同一 session 内不同角色的记忆相互隔离。"""
    sid = "role-scope"
    client.post("/chat", json={"question": "你好", "role_id": "psychologist", "session_id": sid})

    doc_key = app.state.pipeline._memory_key(sid, "psychologist")
    nut_key = app.state.pipeline._memory_key(sid, "nutritionist")
    assert [h["role"] for h in app.state.pipeline.memory.get_history(doc_key)] == ["user", "assistant"]
    assert app.state.pipeline.memory.get_history(nut_key) == []


# ---- 对话（流式 SSE）----
def test_chat_stream_events(client):
    """事件名是前端唯一的分支依据（前端按 sources/delta/done 各接各的处理），改名即破坏前端。"""
    resp = client.post("/chat/stream", json={"question": "高血压？", "role_id": "psychologist"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: sources" in body
    assert "event: delta" in body
    assert "event: done" in body


# ---- 角色管理 ----
def test_role_list(client):
    # 内置角色首次被访问时自动登记入库，先触发登记再断言列表包含它
    client.get("/role/psychologist")
    resp = client.get("/role/list")
    assert resp.status_code == 200
    roles = resp.json()
    ids = {r["role_id"] for r in roles}
    assert "psychologist" in ids


def test_role_get(client):
    resp = client.get("/role/psychologist")
    assert resp.status_code == 200
    assert resp.json()["name"] == "心理咨询师"


def test_role_create_roundtrip(client):
    payload = {
        "role_id": "tutor",
        "name": "学习辅导老师",
        "description": "测试用角色",
        "system_prompt": "你是一位耐心的学习辅导老师。",
    }
    resp = client.post("/role", json=payload)
    assert resp.status_code == 200
    assert resp.json()["role_id"] == "tutor"

    resp = client.get("/role/tutor")
    assert resp.status_code == 200
    assert resp.json()["name"] == "学习辅导老师"

    resp = client.get("/role/list")
    assert "tutor" in {r["role_id"] for r in resp.json()}


# ---- 知识库 ----
def test_knowledge_upload_and_list(client):
    """上传后能在列表里查到：登记关系库这一步出过两次事故（孤儿向量、重复登记），这里锁一头一尾。

    set(d) 的等值断言锁的是 /knowledge/list 的字段契约（前端列表按这几个键渲染），
    字段增删或改名都会红——动响应结构时得先想清楚前端跟不跟得上。
    """
    files = {"file": ("nutrition.md", "# 营养\n\n每日蔬菜应摄入 300～500 克。", "text/markdown")}
    resp = client.post("/knowledge/upload", files=files, params={"role_id": "psychologist"})
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] >= 1

    resp = client.get("/knowledge/list")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1
    for d in resp.json()["documents"]:
        assert set(d) == {"id", "role_id", "source", "title", "chunk_count"}


def test_knowledge_list_filter_by_role(client):
    files = {"file": ("demo.md", "# 标题\n\n高血压患者应限制钠盐摄入。", "text/markdown")}
    client.post("/knowledge/upload", files=files, params={"role_id": "psychologist"})

    resp = client.get("/knowledge/list", params={"role_id": "psychologist"})
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert docs and all(d["role_id"] == "psychologist" for d in docs)


def test_knowledge_upload_empty_rejected(client):
    files = {"file": ("empty.md", "", "text/markdown")}
    resp = client.post("/knowledge/upload", files=files, params={"role_id": "psychologist"})
    assert resp.status_code == 400  # 无有效文本


# ---- 2026-09-21 审计：三处「有意图但没接完」的代码接线后的行为 ----
def test_health_reports_degraded_reason(client):
    """只报 degraded 不够——降级的原因必须能看见（否则重启丢上下文的代价无从得知）。"""
    from app.core.store.memory import DegradedMemoryStore

    app.state.pipeline.memory = DegradedMemoryStore(10, "Redis 不可达: redis://127.0.0.1:6379/0")
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["components"]["memory"] == "degraded"
    assert "Redis 不可达" in body["details"]["memory"]


def test_health_omits_details_when_ok(client):
    """正常时不多出空 details；components 的值仍是三档之一，只做等值判断的调用方不受影响。"""
    body = client.get("/health").json()

    assert "details" not in body
    assert set(body["components"].values()) == {"ok"}


def test_summary_upload_follows_summary_enabled(client, monkeypatch):
    """上传不传 summary 时跟随 .env 的 SUMMARY_ENABLED；显式传值时以传的为准。"""
    import app.core.document.ingest as ingest_stage

    calls: list[int] = []
    # patch 在真正调用它的地方（入库链路的公共实现），不是在接口模块——那里已经不再 import 它
    monkeypatch.setattr(
        ingest_stage,
        "summarize_chunks",
        lambda llm, texts: calls.append(len(texts)) or ["摘要"] * len(texts),
    )
    files = {"file": ("s.md", "# 标题\n\n每日食盐摄入量宜控制在 5 克以内。", "text/markdown")}
    app.state.pipeline.settings.summary_enabled = True

    ok = client.post("/knowledge/upload", files=files, params={"role_id": "psychologist"})
    assert ok.status_code == 200
    assert calls, "SUMMARY_ENABLED=true 时，不传 summary 也应生成摘要"

    calls.clear()
    off = client.post(
        "/knowledge/upload",
        files=files,
        params={"role_id": "psychologist", "summary": "false"},
    )
    assert off.status_code == 200
    assert not calls, "显式 summary=false 应覆盖 SUMMARY_ENABLED=true"


def test_ingest_cli_summary_follows_settings(monkeypatch):
    """CLI 的 --summary 不传时同样跟随 SUMMARY_ENABLED（与上传接口同一套语义）。"""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    ingest = importlib.import_module("ingest")

    seen: dict = {}
    monkeypatch.setattr(ingest, "setup_logging", lambda *a, **kw: None)  # 别动全局日志配置
    monkeypatch.setattr(ingest.settings, "summary_enabled", True)
    monkeypatch.setattr(ingest, "ingest_path", lambda *a, **kw: seen.setdefault("summary", a[6]) or 1)
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--dir", "data/corpus/lawyer", "--role", "lawyer"])

    ingest.main()

    assert seen["summary"] is True


# ---- 错误码口径（2026-09-21 审计：接口之间不一致）----
def test_unknown_role_returns_404_not_500(client):
    """未知角色在 /role 与 /chat 上必须是同一个口径（404），不能一个 404 一个 500。"""
    assert client.get("/role/no_such_role").status_code == 404
    assert client.post("/chat", json={"question": "你好", "role_id": "no_such_role"}).status_code == 404


def test_too_long_question_rejected(client):
    """提问有上界：不然一条几万字的提问就把 prompt 预算吃光、历史被整段裁掉还静默超窗。"""
    resp = client.post("/chat", json={"question": "啊" * 2001, "role_id": "psychologist"})
    assert resp.status_code == 422


def test_upload_dependency_failure_maps_to_503(client, monkeypatch):
    """上传时依赖掉线要给 503（与 /chat 一致），不能裸 500。"""
    import app.api.knowledge as knowledge_api

    def boom(*_a, **_kw):
        raise RuntimeError("Embedding 掉线")

    monkeypatch.setattr(knowledge_api, "build_chunks", boom)
    files = {"file": ("a.md", "# 标题\n\n正文内容足够长，能过低质量过滤。", "text/markdown")}
    resp = client.post("/knowledge/upload", files=files, params={"role_id": "psychologist"})

    assert resp.status_code == 503
    assert "服务暂时不可用" in resp.json()["detail"]
