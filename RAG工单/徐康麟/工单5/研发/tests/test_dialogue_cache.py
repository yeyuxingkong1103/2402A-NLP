# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

``DialogueCache`` 真单元测试（非烟雾）：命中/未命中、TTL 过期、LRU 淘汰顺序、
按会话失效、深拷贝隔离、session_id 敏感键、落盘 + 回填、缓存故障不抛异常。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_dialogue_cache.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import dialogue_cache as dc  # noqa: E402
from app.core.dialogue_cache import DialogueCache  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"
SID_A = "sess-A"
SID_B = "sess-B"


@pytest.fixture()
def tmp_dir():
    """本测试专用的临时目录（**在工作区内**）。

    **不用 pytest 的 ``tmp_path`` / ``tempfile.mkdtemp``**：DSH 沙箱下系统临时目录
    （``C:\\Users\\...\\AppData\\Local\\Temp\\dsh-*``）不可读、且其子目录不可再建目录，
    ``tmp_path`` 与 ``mkdtemp`` 都会撞上 ``PermissionError: [WinError 5]``。
    这里改为在**工作区**内建一个进程唯一目录（工作区是可写的），跑完删掉。
    """
    path = REPO_ROOT / ".pytest_runtime" / f"dialogue_cache_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)



class _FakeClock:
    """可手动推进的假时钟（monkeypatch 掉模块级 ``time.time``）。"""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class _RecordingLogger:
    """记录全部 ``log_event`` 调用（用于「日志不可用/落盘失败」断言）。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.warnings: list[tuple[str, dict]] = []

    def log_event(self, event: str = "", **fields) -> None:
        self.events.append((event, fields))
        if str(fields.get("level", "")).upper() == "WARNING":
            self.warnings.append((event, fields))

    def bind(self, **_kw) -> "_RecordingLogger":
        return self


def _value(session_id: str = SID_A, answer: str = "答案是 68 页") -> dict:
    return {"session_id": session_id, "answer": answer, "rewritten_query": "武汉兴图…的法定代表人是谁？",
            "strategy": "pronoun", "cited_pages": [68],
            "rewrite": {"resolved_refs": [{"surface": "这个公司"}]}, "latency_ms": {"total": 700.0}}


# --------------------------------------------------------------------- 基本命中
def test_miss_then_hit_counts_and_hit_rate() -> None:
    cache = DialogueCache(max_size=8, ttl_s=60.0, logger=None)
    key = cache.make_key("武汉兴图新科电子股份有限公司的法定代表人是谁？", "zh", SID_A)

    assert cache.get(key) is None, "空缓存必须是 miss"
    cache.put(key, _value())
    got = cache.get(key)
    assert got is not None and got["answer"] == "答案是 68 页"

    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1, "命中/未命中必须分别计数"
    assert stats["hit_rate"] == 0.5, f"命中率应为 1/2，实际 {stats['hit_rate']}"
    assert stats["size"] == 1
    assert stats["max_size"] == 8 and stats["ttl_s"] == 60.0
    for field in ("hits", "misses", "hit_rate", "size", "evictions", "expired",
                  "max_size", "ttl_s"):
        assert field in stats, f"stats() 缺少契约字段 {field}"


def test_make_key_is_session_sensitive_and_stable() -> None:
    q, lang = "这个公司的法定代表人是谁？", "zh"
    key_a = DialogueCache.make_key(q, lang, SID_A)
    key_b = DialogueCache.make_key(q, lang, SID_B)

    assert key_a != key_b, "同一问句在不同会话必须得到不同 key（不得跨会话复用）"
    assert key_a == DialogueCache.make_key(q, lang, SID_A), "同输入必须稳定"
    assert len(key_a) == 16, "契约要求 sha1 前 16 位"
    assert DialogueCache.make_key(q, "en", SID_A) != key_a, "语言必须进 key"
    assert DialogueCache.make_key(q + "？", lang, SID_A) != key_a, "问句必须进 key"


# --------------------------------------------------------------------- TTL
def test_ttl_expiry_treated_as_miss_and_removed(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(dc.time, "time", clock)
    cache = DialogueCache(max_size=8, ttl_s=10.0, logger=None)
    key = cache.make_key("法定代表人是谁？", "zh", SID_A)
    cache.put(key, _value())

    clock.advance(9.0)
    assert cache.get(key) is not None, "TTL 内必须命中"

    clock.advance(2.0)                       # 距写入 11s > ttl 10s
    assert cache.get(key) is None, "过期必须按 miss 处理"
    stats = cache.stats()
    assert stats["expired"] == 1, f"expired 应计 1，实际 {stats['expired']}"
    assert stats["size"] == 0, "过期条目必须被删除"


def test_ttl_zero_means_never_expire(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(dc.time, "time", clock)
    cache = DialogueCache(max_size=8, ttl_s=0, logger=None)
    key = cache.make_key("注册资本是多少？", "zh", SID_A)
    cache.put(key, _value())
    clock.advance(10_000_000.0)
    assert cache.get(key) is not None, "ttl_s<=0 表示永不过期"


# --------------------------------------------------------------------- LRU
def test_lru_eviction_order_with_max_size_two() -> None:
    cache = DialogueCache(max_size=2, ttl_s=0, logger=None)
    keys = [cache.make_key(f"第{i}问？", "zh", SID_A) for i in range(3)]
    for i, key in enumerate(keys):
        cache.put(key, _value(answer=f"答案{i}"))

    stats = cache.stats()
    assert stats["size"] == 2, "max_size=2 时只能留 2 条"
    assert stats["evictions"] == 1, f"应淘汰 1 条，实际 {stats['evictions']}"
    assert cache.get(keys[0]) is None, "最先插入的第 1 条必须被淘汰"
    assert cache.get(keys[1]) is not None and cache.get(keys[2]) is not None


def test_lru_hit_refreshes_recency() -> None:
    cache = DialogueCache(max_size=2, ttl_s=0, logger=None)
    k1 = cache.make_key("第1问？", "zh", SID_A)
    k2 = cache.make_key("第2问？", "zh", SID_A)
    k3 = cache.make_key("第3问？", "zh", SID_A)
    cache.put(k1, _value(answer="a1"))
    cache.put(k2, _value(answer="a2"))
    assert cache.get(k1) is not None, "命中最旧的 k1，把它刷新为最新"
    cache.put(k3, _value(answer="a3"))       # 此时应淘汰 k2（最久未使用）

    assert cache.get(k1) is not None, "命中过的条目不应被淘汰"
    assert cache.get(k2) is None, "未命中过的 k2 才该被淘汰"
    assert cache.get(k3) is not None


# --------------------------------------------------------------------- 会话失效
def test_invalidate_session_only_removes_that_session() -> None:
    cache = DialogueCache(max_size=16, ttl_s=0, logger=None)
    key_a = cache.make_key("这个公司的法定代表人是谁？", "zh", SID_A)
    key_b = cache.make_key("这个公司的法定代表人是谁？", "zh", SID_B)
    cache.put(key_a, _value(session_id=SID_A, answer="A 的答案"))
    cache.put(key_b, _value(session_id=SID_B, answer="B 的答案"))
    assert cache.size() == 2

    removed = cache.invalidate_session(SID_A)
    assert removed == 1, f"应只删 1 条，实际 {removed}"
    assert cache.get(key_a) is None, "A 会话条目必须已清"
    assert cache.get(key_b) is not None, "B 会话条目必须保留（不得串台）"
    assert cache.size() == 1
    assert cache.invalidate_session("不存在的会话") == 0


def test_invalidate_session_falls_back_to_value_scan() -> None:
    """value 里 session_id 存在、但内部索引缺失时也必须删干净。"""
    cache = DialogueCache(max_size=4, ttl_s=0, logger=None)
    key = cache.make_key("法定代表人是谁？", "zh", SID_A)
    cache.put(key, _value(session_id=SID_A))
    cache._session_keys.clear()              # 人为破坏内部索引
    assert cache.invalidate_session(SID_A) == 1
    assert cache.get(key) is None


# --------------------------------------------------------------------- 深拷贝
def test_get_returns_deep_copy() -> None:
    cache = DialogueCache(max_size=4, ttl_s=0, logger=None)
    key = cache.make_key("销售部有哪些？", "zh", SID_A)
    cache.put(key, _value())

    first = cache.get(key)
    assert first is not None
    first["answer"] = "被调用方改坏了"
    first["cited_pages"].append(999)
    first["rewrite"]["resolved_refs"].append({"surface": "篡改"})

    second = cache.get(key)
    assert second is not None
    assert second["answer"] == "答案是 68 页", "改返回值不得影响缓存内部"
    assert second["cited_pages"] == [68], "嵌套 list 也必须深拷贝"
    assert len(second["rewrite"]["resolved_refs"]) == 1, "嵌套 dict 也必须深拷贝"


def test_put_stores_deep_copy() -> None:
    """写入方后续修改源对象，也不应污染缓存。"""
    cache = DialogueCache(max_size=4, ttl_s=0, logger=None)
    key = cache.make_key("组织结构？", "zh", SID_A)
    payload = _value()
    cache.put(key, payload)
    payload["answer"] = "外部改坏了"
    payload["cited_pages"].append(1)
    got = cache.get(key)
    assert got is not None
    assert got["answer"] == "答案是 68 页"
    assert got["cited_pages"] == [68]


# --------------------------------------------------------------------- 落盘
def test_persist_and_reload(tmp_dir: Path) -> None:
    path = tmp_dir / "对话缓存.json"
    cache = DialogueCache(max_size=8, ttl_s=1800.0, persist_path=path, logger=None)
    key = cache.make_key("军用领域收入是多少？", "zh", SID_A)
    cache.put(key, _value(session_id=SID_A, answer="军用领域收入 1.2 亿元"))

    assert path.is_file(), "put 后必须落盘"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [e["key"] for e in raw["entries"]] == [key], "落盘格式必须含 key/value/ts"
    assert not list(tmp_dir.glob("*.tmp")), "原子写不得残留 .tmp 文件"

    fresh = DialogueCache(max_size=8, ttl_s=1800.0, persist_path=path, logger=None)
    assert fresh.size() == 1, "新进程构造时必须回填"
    got = fresh.get(key)
    assert got is not None and got["answer"] == "军用领域收入 1.2 亿元"
    assert fresh.stats()["loaded"] == 1
    assert fresh.invalidate_session(SID_A) == 1, "回填后的会话索引也必须可用"


def test_corrupt_persist_file_degrades_to_memory(tmp_dir: Path) -> None:
    path = tmp_dir / "坏缓存.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")
    logger = _RecordingLogger()
    cache = DialogueCache(max_size=4, ttl_s=60.0, persist_path=path, logger=logger)

    assert cache.size() == 0, "损坏文件必须降级为空缓存"
    assert cache.stats()["load_errors"] == 1
    assert logger.warnings and logger.warnings[0][0] == "dialogue_cache.load_failed"
    key = cache.make_key("法定代表人是谁？", "zh", SID_A)
    cache.put(key, _value())
    assert cache.get(key) is not None, "降级后缓存仍必须可用"


def test_unwritable_persist_path_never_raises(tmp_dir: Path) -> None:
    """落盘路径指向一个「目录」→ 写盘必然失败，但不得抛异常。"""
    bad = tmp_dir / "这是目录不是文件"
    bad.mkdir()
    logger = _RecordingLogger()
    cache = DialogueCache(max_size=4, ttl_s=60.0, persist_path=bad, logger=logger)
    key = cache.make_key("法定代表人是谁？", "zh", SID_A)
    cache.put(key, _value())          # 不抛
    assert cache.get(key) is not None, "写盘失败不得影响内存命中"
    assert cache.stats()["save_errors"] >= 1
    assert any(ev == "dialogue_cache.save_failed" for ev, _ in logger.warnings)


# --------------------------------------------------------------------- 兜底与线程
def test_logger_none_and_missing_log_event_never_raise() -> None:
    cache = DialogueCache(max_size=2, ttl_s=0, logger=None)
    k1 = cache.make_key("第1问？", "zh", SID_A)
    for i in range(3):                 # 触发淘汰（会调 log_event）
        cache.put(cache.make_key(f"第{i}问？", "zh", SID_A), _value())
    cache.clear()
    assert cache.invalidate_session(SID_A) == 0
    assert cache.get(k1) is None

    class _BrokenLogger:
        def log_event(self, *_a, **_kw):
            raise RuntimeError("日志后端炸了")

    broken = DialogueCache(max_size=1, ttl_s=0, logger=_BrokenLogger())
    try:
        broken.put(broken.make_key("问？", "zh", SID_A), _value())
        broken.put(broken.make_key("问2？", "zh", SID_A), _value())
        broken.clear()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"日志故障不得向上抛：{type(exc).__name__}: {exc}")


def test_concurrent_put_get_stays_consistent() -> None:
    cache = DialogueCache(max_size=64, ttl_s=0, logger=None)
    keys = [cache.make_key(f"并发第{i}问？", "zh", f"sess-{i}") for i in range(40)]
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for i, key in enumerate(keys):
                if (i + offset) % 2 == 0:
                    cache.put(key, _value(session_id=f"sess-{i}", answer=f"答案{i}"))
                else:
                    cache.get(key)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发下不得抛异常：{errors[:2]}"
    stats = cache.stats()
    assert stats["size"] <= stats["max_size"], "并发后条目数不得超上限"
    assert stats["hits"] + stats["misses"] + stats["evictions"] >= 0
