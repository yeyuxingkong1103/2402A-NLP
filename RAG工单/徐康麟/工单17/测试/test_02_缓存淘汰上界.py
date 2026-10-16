# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 02：**全局缓存/会话的容量上界与淘汰**（修复项 F5 / F6 / F8）。

对应工单原文泄漏点一「全局缓存或会话未释放」。本组用例验证四条上界都存在且真的生效：

* ``app/core/cache.py::TTLCache`` —— LRU + TTL（基座已有，回归保护）；
* ``app/core/qa_engine.py::_memory_cache`` —— **原来无锁、无 TTL**（本工单修掉），
  现在必须：① 条目数不超过上限；② TTL 到期即失效；③ 并发读写不炸；
* ``app/storage/sqlite_manager.py::answer_cache`` 表 —— 条数上限清理；
* ``app/core/dialogue_cache.py::DialogueCache`` —— LRU + TTL（回归保护）。
"""

from __future__ import annotations

import threading
import time

from conftest import reset_singletons


# ----------------------------------------------------------------------
# 基座已有的缓存：回归保护（防止修复时把它们改坏）
# ----------------------------------------------------------------------
def test_多层缓存的LRU与TTL上界(perf_env):
    from app.core.cache import TTLCache

    cache = TTLCache("测试层", max_entries=5, ttl_s=0.3)
    for i in range(20):
        cache.put(f"k{i}", i)
    assert len(cache) == 5, f"LRU 上限失效：{len(cache)}"
    assert cache.stats.evictions >= 15
    # 剩下的应该是最后写入的 5 个
    assert cache.get("k19")[0] is True
    assert cache.get("k0")[0] is False

    time.sleep(0.35)
    hit, _ = cache.get("k19")
    assert hit is False, "TTL 过期失效"
    assert cache.stats.expired >= 1


def test_对话缓存的LRU与TTL上界(perf_env):
    from app.core.dialogue_cache import DialogueCache

    cache = DialogueCache(max_size=4, ttl_s=0.3)
    for i in range(10):
        cache.put(DialogueCache.make_key(f"问句{i}", "zh", "s1"),
                  {"session_id": "s1", "n": i})
    assert cache.size() <= 4, f"对话缓存上限失效：{cache.size()}"
    assert cache.stats()["evictions"] >= 6
    # 清空会话
    removed = cache.invalidate_session("s1")
    assert removed >= 0 and cache.size() <= 4


# ----------------------------------------------------------------------
# 工单17 新修：qa_engine._memory_cache
# ----------------------------------------------------------------------
def _make_engine(perf_env):
    """构造一个**不触碰索引/Ollama** 的 QAEngine（只测缓存层本身）。"""
    from app.core.qa_engine import Answer, QAEngine

    class _NullLog:
        def log_event(self, *_a, **_kw):
            return None

        def bind(self, **_kw):
            return self

        class _Span:
            def set_output(self, *_a, **_kw):
                return None

            def __enter__(self):
                return self

            def __exit__(self, *_e):
                return False

        def enter(self, *_a, **_kw):
            return _NullLog._Span()

    engine = QAEngine.__new__(QAEngine)      # 绕过 __init__（不加载索引/模型）
    from collections import OrderedDict

    engine.cfg = type("C", (), {"retrieval": type("R", (), {"cache_size": 512})()})()
    engine.log = _NullLog()
    engine.storage = None
    engine._memory_cache = OrderedDict()
    engine._cache_lock = threading.RLock()
    engine._cache_evictions = 0
    engine._cache_expired = 0
    engine._stats = {"asked": 0, "refused": 0, "cache_hit": 0}
    return engine, Answer


def test_答案缓存条目数不超过上限(perf_env):
    """PERF_ANSWER_CACHE_MAX=8（conftest 设定）→ 写入 50 条后仍 ≤ 8。"""
    engine, Answer = _make_engine(perf_env)
    for i in range(50):
        engine._cache_put(f"key{i}", Answer(question=f"q{i}", answer=f"a{i}"))
    assert len(engine._memory_cache) <= 8, f"上限失效：{len(engine._memory_cache)}"
    assert engine._cache_evictions >= 42
    stats = engine.stats()
    assert stats["memory_cache"] <= 8
    assert stats["memory_cache_max"] == 8
    assert stats["memory_cache_ttl_s"] == 0.4


def test_答案缓存TTL到期即未命中(perf_env):
    engine, Answer = _make_engine(perf_env)
    engine._cache_put("k", Answer(question="q", answer="a"))
    assert engine._cache_get("k") is not None
    time.sleep(0.45)
    assert engine._cache_get("k") is None, "TTL 到期后仍命中（无过期逻辑）"
    assert engine._cache_expired >= 1


def test_答案缓存在基座口径下不设TTL(base_env):
    """``PERF_ENABLED=0``：TTL=0 = 永不过期、上限=cfg.retrieval.cache_size（512）＝基座行为。"""
    engine, Answer = _make_engine(base_env)
    engine._cache_put("k", Answer(question="q", answer="a"))
    time.sleep(0.45)
    assert engine._cache_get("k") is not None, "基座口径不应过期"
    assert engine.stats()["memory_cache_ttl_s"] == 0.0
    assert engine.stats()["memory_cache_max"] == 512


def test_答案缓存并发读写不炸(perf_env):
    """**核心断言**：多线程并发 ``_cache_get`` / ``_cache_put`` 不得抛异常。

    基座的 ``_memory_cache`` 是裸 ``OrderedDict``，并发 ``move_to_end`` / ``popitem``
    会撞坏内部结构（实测表现为偶发 ``KeyError`` / ``RuntimeError``）。
    """
    engine, Answer = _make_engine(perf_env)
    errors: list[str] = []

    def worker(wid: int) -> None:
        try:
            for i in range(300):
                key = f"k{(wid * 7 + i) % 40}"
                engine._cache_put(key, Answer(question=key, answer="a" * 50))
                engine._cache_get(f"k{(wid + i) % 40}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发读写抛异常：{errors[:5]}"
    assert len(engine._memory_cache) <= 8


def test_缓存条目瘦身有界(perf_env):
    """F5 瘦身：缓存条目里的 ``chunks[].text`` 必须被截断到 PERF_CACHE_CHUNK_CHARS。"""
    engine, Answer = _make_engine(perf_env)
    long_text = "字" * 5000
    answer = Answer(question="q", answer="a",
                    chunks=[{"chunk_id": "c1", "text": long_text}],
                    evidence=[{"chunk_id": "c1", "text": long_text}])
    engine._cache_put("k", answer)
    _, cached = engine._memory_cache["k"]
    assert len(cached.chunks[0]["text"]) == 400
    assert len(cached.evidence[0]["text"]) == 400
    # 原对象不被改动（瘦身只作用于缓存副本）
    assert len(answer.chunks[0]["text"]) == 5000


# ----------------------------------------------------------------------
# sqlite answer_cache 表上限
# ----------------------------------------------------------------------
def test_sqlite答案缓存表有条数上限(perf_env, tmp_path):
    from app.storage.sqlite_manager import SqliteManager

    db = tmp_path / "cache_limit.sqlite3"
    storage = SqliteManager(db)
    try:
        for i in range(80):
            storage.save_answer_cache(f"key{i}", f"q{i}", {"answer": "a"})
        info = storage.stats()
        assert info["cache_table_max"] == 50
        assert info["cache_entries"] <= 60, f"表未清理：{info['cache_entries']}"
        assert info["cache_pruned_total"] > 0
    finally:
        storage.close()


def test_sqlite在基座口径下不清理(base_env, tmp_path):
    """``PERF_ENABLED=0`` 时不做表清理（＝基座行为：只增不减）。"""
    from app.storage.sqlite_manager import SqliteManager

    storage = SqliteManager(tmp_path / "cache_base.sqlite3")
    try:
        for i in range(80):
            storage.save_answer_cache(f"key{i}", f"q{i}", {"answer": "a"})
        info = storage.stats()
        assert info["cache_table_max"] is None
        assert info["cache_entries"] == 80
        assert info["cache_pruned_total"] == 0
    finally:
        storage.close()


# ----------------------------------------------------------------------
# 会话卷上限
# ----------------------------------------------------------------------
def test_会话文件卷有条数上限(perf_env, tmp_path):
    from service.session_store import FileSessionStore, Turn

    store = FileSessionStore(tmp_path, ttl_s=0, max_turns=5, max_files=10, sweep_s=0.0)
    for i in range(60):
        store.append(f"用户{i}", Turn(role="assistant", question=f"q{i}", answer="a"))
    assert store.count() <= 11, f"会话文件未清理：{store.count()}"
    assert store.swept_files > 0
    assert store.stats()["max_files"] == 10


def test_会话文件卷在基座口径下无上限(base_env, tmp_path):
    """``PERF_ENABLED=0``：max_files=0 = 不清理（＝基座行为）。"""
    from service.session_store import FileSessionStore, Turn

    store = FileSessionStore(tmp_path, ttl_s=0, max_turns=5, max_files=0, sweep_s=0.0)
    for i in range(30):
        store.append(f"用户{i}", Turn(role="assistant", question=f"q{i}", answer="a"))
    assert store.count() == 30
    assert store.swept_files == 0
