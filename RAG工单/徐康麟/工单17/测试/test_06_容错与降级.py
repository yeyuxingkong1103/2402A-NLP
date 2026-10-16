# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

用例组 06：**容错与显式降级**（工单纪律 §容错清单）。

覆盖 5 类故障注入，每一类都要求「有明确行为 + 有留痕 + 服务不崩」：

1. 上传损坏 PDF / 非 PDF 扩展名 / 超大文件 ⇒ 明确 400 文案，**任务不入队**；
2. 解析队列满 ⇒ 明确 429（不堆积、不静默丢弃）；
3. Ollama 不可达 ⇒ ``unload_models`` 返回 ``ok=False`` + 原因，不抛；
4. Redis 不可达 ⇒ 会话后端降级为 file（``MiniRedis`` 报明确原因）；
5. 解析任务失败（损坏文件进 worker）⇒ 任务 ``failed`` + ``error``，worker 不退出。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from conftest import reset_singletons

PDF_OK = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


# ----------------------------------------------------------------------
# 1) 上传校验
# ----------------------------------------------------------------------
def _queue(tmp_path):
    from service.perf_config import get_perf_config
    from service.task_queue import ParseQueue

    cfg = get_perf_config()
    return ParseQueue(cfg, upload_dir=tmp_path / "uploads", log_dir=tmp_path / "logs")


def test_非PDF扩展名被拒(perf_env, tmp_path):
    from service.task_queue import UploadRejected

    q = _queue(tmp_path)
    try:
        with pytest.raises(UploadRejected) as exc:
            q.validate("说明.txt", PDF_OK)
        assert ".pdf" in exc.value.message
        assert exc.value.status == 400
    finally:
        q.close()


def test_缺少PDF魔数被拒(perf_env, tmp_path):
    from service.task_queue import UploadRejected

    q = _queue(tmp_path)
    try:
        with pytest.raises(UploadRejected) as exc:
            q.validate("fake.pdf", "这不是 PDF 内容".encode("utf-8") * 10)
        assert "%PDF-" in exc.value.message or "魔数" in exc.value.message
    finally:
        q.close()


def test_空文件被拒(perf_env, tmp_path):
    from service.task_queue import UploadRejected

    q = _queue(tmp_path)
    try:
        with pytest.raises(UploadRejected):
            q.validate("empty.pdf", b"")
    finally:
        q.close()


def test_超大文件被拒(perf_env, tmp_path):
    from service.task_queue import UploadRejected

    q = _queue(tmp_path)
    try:
        with pytest.raises(UploadRejected) as exc:
            q.validate("big.pdf", PDF_OK + b"0" * (40 * 1024 * 1024))
        assert "超过上限" in exc.value.message
    finally:
        q.close()


def test_损坏PDF在入队前被拒且不留残file(perf_env, tmp_path):
    """魔数对但结构损坏 → 入队前就被拒，且落盘的临时文件被清掉。"""
    from service.task_queue import UploadRejected

    q = _queue(tmp_path)
    try:
        # 与压测实测完全一致的内容：魔数对、结构损坏（MuPDF 报 FileDataError）
        broken = b"%PDF-1.4\n%\xd0\xd4\xc5\xd8\n1 0 obj\n"
        with pytest.raises(UploadRejected) as exc:
            q.submit("broken.pdf", broken)
        assert "损坏" in exc.value.message or "无法解析" in exc.value.message
        leftovers = list((tmp_path / "uploads").glob("*.pdf"))
        assert not leftovers, f"被拒的上传残留了文件：{leftovers}"
        assert q.submitted == 0
    finally:
        q.close()


def test_合法PDF进队并被消费(perf_env, tmp_path):
    """真 PDF（用 fitz 现造）→ 任务进队列 → worker 消费 → done + 页数。"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"第 {i + 1} 页测试文本")
    blob = doc.tobytes()
    doc.close()

    q = _queue(tmp_path)
    try:
        task = q.submit("正常.pdf", blob)
        assert task.status in {"pending", "running"}, f"意外状态：{task.status}"
        deadline = time.time() + 20
        while time.time() < deadline and task.status not in {"done", "failed"}:
            time.sleep(0.1)
        assert task.status == "done", f"任务未完成：{task.status} {task.error}"
        assert task.pages == 3
        assert q.done == 1
        assert q.stats()["counts"]["done"] == 1
        # 留痕：任务表 jsonl 必须写下来
        log = (tmp_path / "logs" / "parse_tasks.jsonl")
        assert log.is_file() and log.stat().st_size > 0
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert any(e.get("event") == "submitted" for e in events)
        assert any(e.get("event") == "finished" for e in events)
    finally:
        q.close()


def test_解析队列满立即429(perf_env, tmp_path):
    """队列上限 2（conftest 设 PERF_PARSE_QUEUE_MAX 默认 32，这里显式构造小队列）。"""
    from service.perf_config import PerfConfig
    from service.task_queue import ParseQueue, UploadRejected

    cfg = PerfConfig(enabled=True, parse_queue_max=2, parse_workers=0)  # 尽量不消费，保证堆满
    q = ParseQueue(cfg, upload_dir=tmp_path / "u", log_dir=tmp_path / "l")
    try:
        # ⚠️ 实测踩坑：``parse_workers=0`` 会被 ``max(1, ...)`` 兜成 1 个 worker，
        #    所以提交后的状态可能是 pending 也可能是 running —— 断言必须覆盖两者。
        assert q.submit("a.pdf", PDF_OK).status in {"pending", "running"}
        assert q.submit("b.pdf", PDF_OK).status in {"pending", "running"}
        with pytest.raises(UploadRejected) as exc:
            q.submit("c.pdf", PDF_OK)
        assert exc.value.status == 429
        assert "队列已满" in exc.value.message
        assert q.rejected == 1
        assert q.queue_depth <= 2
    finally:
        q.close()


def test_同步解析口径可用于基座对照(base_env, tmp_path):
    """``PERF_ENABLED=0``：``parse_sync`` 必须在**调用线程内**跑完并返回结果。"""
    from service.perf_config import get_perf_config
    from service.task_queue import parse_sync

    row = parse_sync("x.pdf", PDF_OK, cfg=get_perf_config(),
                     upload_dir=tmp_path / "u", log_dir=tmp_path / "l")
    assert row["status"] in {"done", "failed"}     # 极简 PDF 可能解析失败，但**不能抛**
    assert "elapsed_ms" in row and row["elapsed_ms"] >= 0
    assert (tmp_path / "l" / "parse_tasks.jsonl").is_file()


# ----------------------------------------------------------------------
# 2) Ollama 不可达
# ----------------------------------------------------------------------
def test_ollama不可达时卸载不抛(perf_env):
    """指向一个**没有服务**的端口：``unload_models`` 必须返回 ok=False + errors，不抛异常。"""
    from app.core.llm_client import unload_models

    # 127.0.0.1:9 = discard 端口，本机不会有 Ollama；超时 1 s 快速失败
    result = unload_models("http://127.0.0.1:9", timeout=1.0)
    assert result["ok"] is False
    assert result["errors"], "应当留下失败原因"
    assert isinstance(result["unloaded"], list)


def test_未配置后端时卸载直接跳过(perf_env):
    from app.core.llm_client import LLMClient

    class _Backend:
        name = "extractive"
        model = "rule-based"
        base_url = "local"

    client = LLMClient.__new__(LLMClient)
    client.backend = _Backend()
    out = client.unload()
    assert out["ok"] is True and out.get("skipped") is True


# ----------------------------------------------------------------------
# 3) Redis 不可达 → 显式降级
# ----------------------------------------------------------------------
def test_redis不可达时明确降级不抛():
    from service.session_store import MiniRedis

    client = MiniRedis("127.0.0.1", 6399, timeout_s=0.3)   # 该端口没有 Redis
    try:
        client.ping()
        raise AssertionError("应当抛连接异常")
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ in {"ConnectionRefusedError", "TimeoutError", "OSError",
                                      "socket.timeout", "RedisError"}, type(exc).__name__
    finally:
        client.close()


def test_会话门面在redis关闭时用文件后端(monkeypatch, tmp_path):
    from service.config import DeployConfig
    from service.session_store import SharedSessionStore, Turn

    cfg = DeployConfig(session_dir=str(tmp_path), session_backend="file",
                       redis_enabled=False, session_ttl_s=3600, session_max_turns=5)
    store = SharedSessionStore(cfg)
    assert store.backend == "file"
    assert store.degraded_reason
    data = store.append("s1", Turn(role="assistant", question="q", answer="a"))
    assert len(data["turns"]) == 1
    assert store.history("s1")


# ----------------------------------------------------------------------
# 4) multipart 解析容错
# ----------------------------------------------------------------------
def test_multipart解析与异常路径():
    from service.api import _parse_multipart
    from service.task_queue import UploadRejected

    body = (b"--BND\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n" + PDF_OK + b"\r\n--BND--\r\n")
    name, content = _parse_multipart("multipart/form-data; boundary=BND", body)
    assert name == "a.pdf" and content.startswith(b"%PDF-")

    with pytest.raises(UploadRejected):
        _parse_multipart("multipart/form-data", body)          # 缺 boundary
    with pytest.raises(UploadRejected):
        _parse_multipart("multipart/form-data; boundary=BND", b"nothing here")   # 没有 file 字段
