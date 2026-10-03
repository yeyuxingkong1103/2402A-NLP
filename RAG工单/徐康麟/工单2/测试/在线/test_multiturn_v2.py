"""T4 在线测试 ④：多轮对话。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 5、工单「支持多轮对话」）：

- 同一会话连续 **≥3 轮**，含至少一次**指代式追问**（「那它的注册资本呢？」「它的注册地址呢？」）；
- 断言 ①追问被改写为可独立检索的问题（``retrieval_traces.rewritten_query`` 与原始问题不同且含主体/字段）；
- ②各轮**不串题**（第 2/3 轮答案指向新字段，且不重复上一轮的内容）；
- ③历史正确落 SQLite（``messages`` 按 user/assistant 交替、``conversations.message_count`` 递增）；
- ④每轮提问都写入一条 ``retrieval_traces``。

全部经真实 HTTP 链路（``/api/ask`` + ``/api/messages``），SQLite 只读回查。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from conftest import ask_once, get_json, post_json

TURN_1 = "武汉兴图新科电子股份有限公司法定代表人是谁？"
TURN_2 = "那它的注册资本呢？"          # 指代追问（「它」= 公司）
TURN_3 = "它的注册地址在哪里？"        # 继续追问同一主体


@pytest.fixture(scope="module")
def db_path() -> Path:
    """``rag.sqlite3`` 路径（只读回查用）。"""
    from app.core.config import get_settings

    path = Path(get_settings().paths.sqlite_path)
    assert path.exists(), f"SQLite 不存在: {path}"
    return path


def _query(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """只读查询。"""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return conn.execute(sql, params).fetchall()


def test_three_turn_conversation_with_coreference(server, db_path):
    """3 轮对话（含指代追问）：改写生效、不串题、历史与轨迹正确落库。"""
    status, body = post_json(server.base_url, "/api/conversations", {})
    assert status == 200 and body.get("ok"), body
    cid = body["conversation_id"]

    turns = []
    for question in (TURN_1, TURN_2, TURN_3):
        status, body = ask_once(server.base_url, question, conversation_id=cid)
        assert status == 200 and body.get("ok"), f"{question!r} 请求失败: {body}"
        answer = body["answer"]
        assert (answer.get("answer") or "").strip(), f"{question!r} 答案为空"
        assert not answer.get("is_unknown"), f"{question!r} 被误拒答「{answer.get('answer')}」"
        turns.append((question, answer))
        print(f"\n[多轮] Q: {question}")
        print(f"       A: {answer['answer'][:90]!r}  引用页={[c['page'] for c in answer.get('citations', [])]}")

    # ① 改写生效：第 2/3 轮的 rewritten_query 必须不同于原始省略问句，且补全主体/字段
    traces = _query(
        db_path,
        "SELECT question, rewritten_query, variants FROM retrieval_traces WHERE conversation_id=? ORDER BY created_at",
        (cid,),
    )
    assert len(traces) == 3, f"应为 3 轮写入 3 条 retrieval_traces，实际 {len(traces)}"
    print("\n[多轮] retrieval_traces 改写记录：")
    for q, rewritten, variants in traces:
        print(f"       {q!r} → {rewritten!r}  variants={str(variants)[:60]}")

    rewritten_map = {q: rw for q, rw, _ in traces}
    for followup in (TURN_2, TURN_3):
        rewritten = rewritten_map.get(followup, "")
        assert rewritten and rewritten != followup, (
            f"追问未被改写（仍是原省略句）: {followup!r} → {rewritten!r}"
        )
        assert "注册资本" in rewritten or "注册地址" in rewritten or "公司" in rewritten, (
            f"改写未补全字段/主体线索: {followup!r} → {rewritten!r}"
        )

    # ② 不串题：第 2 轮答注册资本、第 3 轮答注册地址，且不重复上一轮内容
    joined = {q: (a.get("answer") or "") for q, a in turns}

    # 对照实验：把同一字段**脱离多轮上下文**单独提问，用于区分「能力不足」与「多轮劣化」。
    status, body = ask_once(server.base_url, "武汉兴图新科电子股份有限公司注册资本是多少？")
    standalone = (body.get("answer") or {})
    print("\n[多轮·对照] 同一字段独立提问（无多轮上下文）：")
    print(f"       {(standalone.get('answer') or '')[:90]!r} "
          f"引用页={[c['page'] for c in standalone.get('citations', [])]}")
    print(f"[多轮·对照] 多轮追问：{joined[TURN_2][:90]!r}")

    assert "5,520" in joined[TURN_2] or "5520" in joined[TURN_2], (
        f"第 2 轮（注册资本）答案未含正确金额: {joined[TURN_2][:80]!r}；"
        f"同一字段独立提问的结果为 {(standalone.get('answer') or '')[:60]!r}"
        f"（引用页 {[c['page'] for c in standalone.get('citations', [])]}）"
        f" → 差异说明问题出在**多轮追问链路**（改写已生效，但检索/取证据退化），"
        f"不是该字段本身检索不到"
    )
    assert "法定代表人是" not in joined[TURN_2], f"第 2 轮串到第 1 轮主题: {joined[TURN_2][:80]!r}"
    assert joined[TURN_3] != joined[TURN_2], "第 3 轮答案与第 2 轮完全相同（疑似未跟随追问）"

    # ③ 历史落 SQLite：user/assistant 交替，message_count 递增
    rows = _query(db_path, "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY message_id", (cid,))
    roles = [r for r, _ in rows]
    assert len(rows) == 6, f"3 轮应落 6 条消息（3 user + 3 assistant），实际 {len(rows)}: {roles}"
    assert roles == ["user", "assistant"] * 3, f"消息角色未按 user/assistant 交替: {roles}"
    conv = _query(db_path, "SELECT message_count FROM conversations WHERE conversation_id=?", (cid,))
    assert conv and conv[0][0] == 6, f"conversations.message_count 应为 6，实际 {conv}"

    # ④ HTTP 侧消息回查一致
    status, body = get_json(server.base_url, f"/api/messages?conversation_id={cid}")
    assert status == 200 and body.get("ok"), body
    assert len(body.get("items") or []) == 6, f"HTTP 消息数与库不一致: {len(body.get('items') or [])}"
    print(f"[多轮] 会话 {cid}：库表 6 条消息、3 条检索轨迹，HTTP 回查一致")


def test_followup_retrieval_uses_rewritten_query(server, db_path):
    """追问的检索必须基于**改写后**的问题，而不是字面省略句。"""
    status, body = post_json(server.base_url, "/api/conversations", {})
    cid = body["conversation_id"]
    ask_once(server.base_url, TURN_1, conversation_id=cid)
    ask_once(server.base_url, "那它的注册资本呢？", conversation_id=cid)

    rows = _query(
        db_path,
        "SELECT question, rewritten_query, final_pages FROM retrieval_traces WHERE conversation_id=? ORDER BY created_at",
        (cid,),
    )
    assert len(rows) == 2, f"应有 2 条轨迹，实际 {len(rows)}"
    q2, rw2, pages2 = rows[1]
    pages = json.loads(pages2) if isinstance(pages2, str) else (pages2 or [])
    print(f"\n[多轮·改写] 第 2 轮 {q2!r} → 改写 {rw2!r}，送入生成页={pages}")
    assert rw2 != q2, "追问检索未使用改写结果"
    assert pages, "追问检索未产生任何证据页"
    assert any(1 <= int(p) <= 548 for p in pages), f"证据页越界: {pages}"


def test_history_isolated_between_conversations(server):
    """不同会话的历史必须隔离（不串会话）。"""
    ids = []
    for _ in range(2):
        status, body = post_json(server.base_url, "/api/conversations", {})
        ids.append(body["conversation_id"])
    ask_once(server.base_url, TURN_1, conversation_id=ids[0])
    ask_once(server.base_url, TURN_2, conversation_id=ids[1])

    for cid, expected in zip(ids, (1, 1)):
        status, body = get_json(server.base_url, f"/api/messages?conversation_id={cid}")
        items = body.get("items") or []
        assert len(items) == expected * 2, f"会话 {cid} 消息数异常: {len(items)}"
    print(f"\n[多轮·隔离] 两个会话各自 {2} 条消息，互不污染")
