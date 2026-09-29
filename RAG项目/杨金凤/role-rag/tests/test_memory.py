"""memory.py 单测：mock DeepSeek / Milvus / embedder，覆盖抽取、去重、召回与触发。"""
import json
from unittest.mock import MagicMock

import numpy as np

import memory
import rag


def _make_client(text):
    """构造 fake DeepSeek client，返回固定 message content。"""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    client.chat.completions.create.return_value = resp
    return client


def _patch_deepseek(monkeypatch, text):
    """把 rag._get_client 换成返回固定文本的 fake client。"""
    monkeypatch.setattr(rag, "_get_client", lambda: _make_client(text))


# ---- extract_facts ----

def test_extract_facts_returns_facts(monkeypatch):
    _patch_deepseek(monkeypatch, '["用户有高血压", "用户对青霉素过敏"]')
    history = [
        {"role": "user", "content": "我有高血压"},
        {"role": "assistant", "content": "注意饮食"},
    ]
    assert memory.extract_facts(history) == ["用户有高血压", "用户对青霉素过敏"]


def test_extract_facts_empty_array(monkeypatch):
    _patch_deepseek(monkeypatch, "[]")
    assert memory.extract_facts([{"role": "user", "content": "你好"}]) == []


def test_extract_facts_strips_code_fence(monkeypatch):
    _patch_deepseek(monkeypatch, '```json\n["用户有高血压"]\n```')
    assert memory.extract_facts([{"role": "user", "content": "我有高血压"}]) == ["用户有高血压"]


def test_extract_facts_invalid_json_returns_empty(monkeypatch):
    _patch_deepseek(monkeypatch, "这不是 JSON")
    assert memory.extract_facts([{"role": "user", "content": "hi"}]) == []


def test_extract_facts_deepseek_failure_returns_empty(monkeypatch):
    client = _make_client("")
    client.chat.completions.create.side_effect = RuntimeError("boom")
    monkeypatch.setattr(rag, "_get_client", lambda: client)
    assert memory.extract_facts([{"role": "user", "content": "hi"}]) == []


def test_extract_facts_empty_history(monkeypatch):
    assert memory.extract_facts([]) == []


def test_extract_facts_truncates_to_max(monkeypatch):
    _patch_deepseek(monkeypatch, json.dumps([f"事实{i}" for i in range(7)]))
    assert len(memory.extract_facts([{"role": "user", "content": "hi"}])) == 5


# ---- save_facts / _fact_id ----

def test_fact_id_dedups_by_user_and_content():
    assert memory._fact_id("u1", "有高血压") == memory._fact_id("u1", "有高血压")
    assert memory._fact_id("u1", "有高血压") != memory._fact_id("u1", "有糖尿病")
    assert memory._fact_id("u1", "有高血压") != memory._fact_id("u2", "有高血压")


def test_save_facts_inserts_records(monkeypatch):
    insert = MagicMock()
    monkeypatch.setattr(memory, "create_memory_collection", MagicMock())
    monkeypatch.setattr(memory, "insert", insert)
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 1024, [0.6] * 1024])
    monkeypatch.setattr(memory, "load_embedder", lambda: embedder)

    memory.save_facts("u1", ["事实A", "事实B"])

    insert.assert_called_once()
    collection, records = insert.call_args[0]
    assert collection == memory.MEMORY_COLLECTION
    assert [r["user_id"] for r in records] == ["u1", "u1"]
    assert [r["content"] for r in records] == ["事实A", "事实B"]
    assert [r["id"] for r in records] == [
        memory._fact_id("u1", "事实A"),
        memory._fact_id("u1", "事实B"),
    ]
    assert all("embedding" in r for r in records)


def test_save_facts_empty_skips(monkeypatch):
    insert = MagicMock()
    monkeypatch.setattr(memory, "insert", insert)
    memory.save_facts("u1", [])
    insert.assert_not_called()


def test_save_facts_failure_swallowed(monkeypatch):
    monkeypatch.setattr(memory, "create_memory_collection", MagicMock())
    monkeypatch.setattr(memory, "insert", MagicMock(side_effect=RuntimeError("boom")))
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 1024])
    monkeypatch.setattr(memory, "load_embedder", lambda: embedder)
    memory.save_facts("u1", ["事实A"])  # 不抛异常


# ---- recall_facts ----

def test_recall_facts_returns_contents(monkeypatch):
    search = MagicMock(return_value=["事实A", "事实B"])
    monkeypatch.setattr(memory, "search_memory", search)
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 1024])
    monkeypatch.setattr(memory, "load_embedder", lambda: embedder)

    facts = memory.recall_facts("u1", "血压多少")
    assert facts == ["事实A", "事实B"]
    search.assert_called_once()
    assert search.call_args[0][2] == 3  # 默认 top_k


def test_recall_facts_failure_returns_empty(monkeypatch):
    embedder = MagicMock()
    embedder.encode.side_effect = RuntimeError("embed fail")
    monkeypatch.setattr(memory, "load_embedder", lambda: embedder)
    assert memory.recall_facts("u1", "血压") == []


# ---- rag._maybe_extract_memory 触发 ----

def test_maybe_extract_triggers_every_5_rounds(monkeypatch):
    schedule = MagicMock()
    monkeypatch.setattr(memory, "schedule_extract", schedule)
    history = []
    for i in range(4):  # 已有 4 轮，本轮为第 5 轮
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    rag._maybe_extract_memory("u1", history, "q5", "a5")
    schedule.assert_called_once()
    mem_key, recent = schedule.call_args[0]
    assert mem_key == "u1"
    assert len(recent) == 10  # 最近 5 轮 = 10 条
    assert recent[-2:] == [
        {"role": "user", "content": "q5"},
        {"role": "assistant", "content": "a5"},
    ]


def test_maybe_extract_skips_non_5(monkeypatch):
    schedule = MagicMock()
    monkeypatch.setattr(memory, "schedule_extract", schedule)
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]  # 1 轮，本轮第 2 轮
    rag._maybe_extract_memory("u1", history, "q2", "a2")
    schedule.assert_not_called()
