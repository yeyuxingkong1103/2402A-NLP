# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""FastAPI 接口层单测。

全部用 MagicMock 替身注入流水线：本文件绝不构造真实 RAGPipeline ——
那会打开 data/qdrant（嵌入式客户端独占锁）并加载模型，属于集成范畴。
"""
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag04.api import server
from rag04.api.server import MAX_QUESTION_LEN, _get_app, create_app
from rag04.config import get_settings
from rag04.pipeline import IngestStats
from rag04.schema import Answer, Hit

# 模块首次导入时的 app 值：必须为 None，证明导入不构造流水线（不锁索引/不载模型）
_APP_AT_IMPORT = server.app


@pytest.fixture
def pipeline():
    pipe = MagicMock()
    pipe.health.return_value = {"mode": "full_04", "qdrant": "ok", "clip": "ok"}
    pipe.ask.return_value = Answer(
        question="本次发行股数是多少",
        answer="本次发行 2,000 万股，占发行后总股本 25%",
        lang="zh",
        citations=[{"page": 10, "block_type": "text", "source_id": "text#10#0"}],
        hits=[Hit(chunk_id="c", doc_id="d", page=10, block_type="text",
                  source_id="s", text="t", score=1.0)],
        latency_ms=123.4, llm_backend="deepseek",
    )
    return pipe


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline))


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "full_04"


def test_ask_endpoint_returns_answer_and_citations(client):
    r = client.post("/api/ask", json={"question": "本次发行股数是多少"})
    assert r.status_code == 200
    body = r.json()
    assert "2,000" in body["answer"]
    assert body["citations"][0]["page"] == 10
    assert body["lang"] == "zh"
    assert body["latency_ms"] > 0


def test_ask_rejects_empty_question(client):
    r = client.post("/api/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_rejects_missing_field(client):
    r = client.post("/api/ask", json={})
    assert r.status_code == 422


def test_ask_rejects_overlong_question(client):
    r = client.post("/api/ask", json={"question": "长" * 3000})
    assert r.status_code == 422


def test_ask_handles_pipeline_exception(client):
    client.app.state.pipeline.ask.side_effect = RuntimeError("boom")
    r = client.post("/api/ask", json={"question": "本次发行股数是多少"})
    assert r.status_code == 500
    assert "error" in r.json()


def test_stats_endpoint(client):
    client.app.state.pipeline.store = MagicMock()
    client.app.state.pipeline.store.counts.return_value = {"text_chunks": 10}
    r = client.get("/api/stats")
    assert r.status_code == 200


# --- 补充：入参边界、真实时延上报、启动预热、建库端点 ---


def test_ask_rejects_whitespace_only_question(client):
    """全空白串 strip 后为空，同样 422（长度校验拦不住它）。"""
    r = client.post("/api/ask", json={"question": "   \n\t "})
    assert r.status_code == 422


def test_ask_accepts_question_at_max_length(client):
    r = client.post("/api/ask", json={"question": "长" * MAX_QUESTION_LEN})
    assert r.status_code == 200


def test_ask_latency_is_pipeline_measurement_not_server_clock(client, pipeline):
    """latency_ms 必须原样取 Answer.latency_ms：服务侧另测会掩盖流水线真实时延。"""
    pipeline.ask.return_value = replace(pipeline.ask.return_value, latency_ms=3931.7)
    body = client.post("/api/ask", json={"question": "本次发行股数是多少"}).json()
    assert body["latency_ms"] == 3931.7


@pytest.mark.parametrize("latency,shown,expected", [
    (2500.0, 2500.0, True),      # 预算内
    (3000.0, 3000.0, True),      # 恰好等于预算：含边界
    (3000.04, 3000.0, True),     # 先取 1 位小数再比较：显示 3000.0 就必须算达标
    (3900.0, 3900.0, False),     # 实测 p50 档位：如实标 False，不截断不美化
])
def test_ask_surfaces_latency_budget_verdict(client, pipeline, latency, shown, expected):
    pipeline.ask.return_value = replace(pipeline.ask.return_value, latency_ms=latency)
    body = client.post("/api/ask", json={"question": "本次发行股数是多少"}).json()
    assert body["latency_ms"] == shown
    assert body["within_budget"] is expected
    assert body["latency_budget_ms"] == get_settings().latency_budget_ms


@pytest.mark.parametrize("latency", [3000.04, 3000.06, 2999.99, 3000.0, 3300.4])
def test_ask_verdict_uses_the_shared_config_helper(client, pipeline, latency):
    """接口判定 = ``rag04.config.latency_verdict``（界面与压测报告同一函数）。

    终审修正B：三处曾各算各的（接口先取整再比、界面与压测用原值比），同一份
    测量能得出不同结论。这里直接拿共享实现做对账。
    """
    from rag04.config import latency_verdict

    pipeline.ask.return_value = replace(pipeline.ask.return_value, latency_ms=latency)
    body = client.post("/api/ask", json={"question": "本次发行股数是多少"}).json()
    shown, within = latency_verdict(latency, body["latency_budget_ms"])
    assert body["latency_ms"] == shown
    assert body["within_budget"] is within


def test_health_exposes_latency_budget_and_warmup_state(client):
    body = client.get("/health").json()
    assert body["latency_budget_ms"] == get_settings().latency_budget_ms
    assert body["latency_budget_ms"] > 0
    assert body["warmup"]["ran"] is False           # 注入流水线：不预热


def test_health_degrades_when_pipeline_health_raises(client):
    client.app.state.pipeline.health.side_effect = RuntimeError("qdrant-lock")
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded" and "qdrant-lock" in body["error"]
    assert body["latency_budget_ms"] > 0            # 降级也不缺预算字段


def test_ask_ignores_mode_field_but_echoes_effective_mode(client, pipeline):
    """服务进程内只有一套索引：请求里的 mode 不切换链路，响应回显真实口径。

    断言直接落在日志记录与调用参数上（不依赖 propagate/收集顺序）。
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect()
    logger = logging.getLogger("rag04.api")
    logger.addHandler(handler)
    try:
        r = client.post("/api/ask",
                        json={"question": "本次发行股数是多少", "mode": "baseline_03"})
    finally:
        logger.removeHandler(handler)

    assert r.status_code == 200
    assert r.json()["mode"] == "full_04"                 # 实际生效模式
    assert pipeline.ask.call_args.args[0] == "本次发行股数是多少"
    assert any("baseline_03" in rec.getMessage() for rec in records)


def test_ask_rejects_unknown_mode(client):
    r = client.post("/api/ask", json={"question": "本次发行股数是多少", "mode": "nope"})
    assert r.status_code == 422


def test_ask_reports_displayed_latency_and_verdict_consistently(client, pipeline):
    """3000.04ms 显示成 3000.0 时必须同为「预算内」，否则界面自相矛盾。"""
    pipeline.ask.return_value = replace(pipeline.ask.return_value, latency_ms=3000.04)
    body = client.post("/api/ask", json={"question": "本次发行股数是多少"}).json()
    assert body["latency_ms"] == 3000.0
    assert body["within_budget"] is True


def test_ask_over_budget_after_rounding_stays_over(client, pipeline):
    """反向也守：取整后仍超预算（3000.06 → 3000.1）不得被判定达标。"""
    pipeline.ask.return_value = replace(pipeline.ask.return_value, latency_ms=3000.06)
    body = client.post("/api/ask", json={"question": "本次发行股数是多少"}).json()
    assert body["latency_ms"] == 3000.1
    assert body["within_budget"] is False


def test_stats_marks_not_loaded_store_explicitly(client):
    """未打开的存储绝不能被报成 {"counts": {}}（与空索引无法区分）。"""
    client.app.state.pipeline.store = None
    body = client.get("/api/stats").json()
    assert body["store"] == "not_loaded"
    assert body["counts"] is None




def test_ask_refused_flag_is_passed_through(client, pipeline):
    pipeline.ask.return_value = replace(pipeline.ask.return_value,
                                        answer="资料中未提及", refused=True)
    body = client.post("/api/ask", json={"question": "无关问题"}).json()
    assert body["refused"] is True


def test_stats_returns_mode_and_counts(client):
    client.app.state.pipeline.store = MagicMock()
    client.app.state.pipeline.store.counts.return_value = {"text_chunks": 10}
    body = client.get("/api/stats").json()
    assert body["mode"] == "full_04"
    assert body["store"] == "ok"
    assert body["counts"] == {"text_chunks": 10}


def test_stats_reports_error_without_crashing(client):
    client.app.state.pipeline.store.counts.side_effect = RuntimeError("no-index")
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["store"] == "error" and body["counts"] is None
    assert "no-index" in body["error"]


def test_ingest_endpoint_returns_stats(client):
    client.app.state.pipeline.build.return_value = [
        IngestStats(doc_id="d", n_text=3, n_chunk=4, seconds=1.5)]
    body = client.post("/api/ingest", json={"confirm": True}).json()
    assert body["ok"] is True
    assert body["stats"][0]["n_chunk"] == 4


def test_ingest_is_a_real_full_rebuild_with_reset(client, pipeline):
    """docstring 说「全量重建」就必须真清库：build(reset=True)，否则旧块与新块共存。"""
    pipeline.build.return_value = [IngestStats(doc_id="d", n_chunk=4)]
    client.post("/api/ingest", json={"confirm": True})
    pipeline.build.assert_called_once_with(reset=True)


def test_ingest_endpoint_reports_failure(client):
    client.app.state.pipeline.build.side_effect = RuntimeError("build-boom")
    r = client.post("/api/ingest", json={"confirm": True})
    assert r.status_code == 500
    assert "build-boom" in r.json()["error"]


class _FakePipeline:
    """真实构造路径的替身：只记录构建/预热，绝不碰索引与模型。"""

    instances = 0
    warmup_error: Exception | None = None

    def __init__(self, settings=None):
        type(self).instances += 1
        self.s = settings or get_settings()
        self.warmed = 0

    def warmup(self):
        if type(self).warmup_error is not None:
            raise type(self).warmup_error
        self.warmed += 1

    def health(self):
        return {"mode": self.s.pipeline_mode, "qdrant": "ok"}

    def ask(self, question):
        return Answer(question=question, answer="替身回答", lang="zh",
                      latency_ms=1.0, llm_backend="fake")

    def build(self, names=None, reset=True):
        return [IngestStats(doc_id="d", n_chunk=1)]


@pytest.fixture
def fake_factory(monkeypatch):
    """把真实 RAGPipeline 换成替身，只走 create_app 的构造/预热分支。"""
    _FakePipeline.instances = 0
    _FakePipeline.warmup_error = None
    monkeypatch.setattr("rag04.pipeline.RAGPipeline", _FakePipeline)
    monkeypatch.setattr(server, "app", None)
    return _FakePipeline


def test_app_is_not_built_at_import():
    """导入即构造会加载模型并锁索引：必须延迟到工厂被调用时。"""
    assert _APP_AT_IMPORT is None


def test_factory_path_warms_up_pipeline(fake_factory):
    app = create_app()
    assert isinstance(app.state.pipeline, _FakePipeline)
    assert app.state.pipeline.warmed == 1            # 构造真实流水线后立即预热
    assert app.state.warmup["ran"] is True
    assert app.state.warmup["ok"] is True
    assert TestClient(app).get("/health").json()["warmup"]["ok"] is True


def test_warmup_failure_is_tolerated(fake_factory):
    """预热失败只记录不抛出：它是优化，不是正确性前提。"""
    _FakePipeline.warmup_error = RuntimeError("reranker 缺失")
    app = create_app()
    assert app.state.warmup["ran"] is True
    assert app.state.warmup["ok"] is False
    assert "reranker 缺失" in app.state.warmup["error"]

    c = TestClient(app)                              # 服务照常可用
    assert c.get("/health").json()["warmup"]["ok"] is False
    r = c.post("/api/ask", json={"question": "本次发行股数是多少"})
    assert r.status_code == 200 and r.json()["answer"] == "替身回答"


def test_injected_pipeline_is_not_warmed_up(client, pipeline):
    assert client.app.state.warmup["ran"] is False
    assert client.app.state.warmup["reason"] == "pipeline_injected"
    pipeline.warmup.assert_not_called()


def test_get_app_builds_and_warms_once(fake_factory):
    """uvicorn --factory 可能被调用多次：app 必须复用，避免重复预热。"""
    a1 = _get_app()
    a2 = _get_app()
    assert a1 is a2
    assert _FakePipeline.instances == 1
    assert a1.state.pipeline.warmed == 1


# --- 复核修正：重建端点不再撞 Qdrant 独占锁 + 并发首查只开一次存储 ---


def test_ingest_refuses_without_explicit_confirm(client):
    """无鉴权 + 分钟级阻塞：无 body / confirm=false 一律拒绝并给出可执行提示。"""
    for body in (None, {}, {"confirm": False}):
        r = (client.post("/api/ingest") if body is None
             else client.post("/api/ingest", json=body))
        assert r.status_code == 400, body
        assert "confirm" in r.json()["error"]
        # 提示必须是真的清空重建：CLI 建议带 --reset，否则照着敲仍会留下旧块
        assert "--reset" in r.json()["error"]
        client.app.state.pipeline.build.assert_not_called()


def test_ingest_rebuilds_even_though_warmup_holds_store(client, pipeline):
    """复核一缺口：启动预热后 store 常驻，重建仍必须可达。

    pipeline.build() 自己会先释放句柄再让 build_all 独占索引，故不得以
    「存储已打开」为由 409。
    """
    pipeline.store = SimpleNamespace(mode="embedded", close=lambda: None)
    pipeline.build.return_value = [IngestStats(doc_id="d", n_chunk=7)]
    r = client.post("/api/ingest", json={"confirm": True})
    assert r.status_code == 200
    assert r.json()["stats"][0]["n_chunk"] == 7
    pipeline.build.assert_called_once_with(reset=True)      # 全量重建 = 先清三库


def test_ingest_serializes_concurrent_rebuilds(client, pipeline):
    """同一时刻只允许一个重建：第二个并发请求 409，且不得触发第二次 build。"""
    assert client.app.state.rebuild_lock.acquire(blocking=False)   # 模拟重建进行中
    try:
        r = client.post("/api/ingest", json={"confirm": True})
    finally:
        client.app.state.rebuild_lock.release()
    assert r.status_code == 409
    assert "重建进行中" in r.json()["error"]
    pipeline.build.assert_not_called()


def test_ingest_clears_rebuilding_state_after_failure(client, pipeline):
    """失败必须解除重建标记与锁，否则服务永久卡在「重建中」。"""
    pipeline.build.side_effect = RuntimeError("build-boom")
    r = client.post("/api/ingest", json={"confirm": True})
    assert r.status_code == 500 and "build-boom" in r.json()["error"]
    assert client.app.state.rebuilding is False
    assert client.app.state.rebuild_lock.acquire(blocking=False)   # 锁已释放
    client.app.state.rebuild_lock.release()

    pipeline.build.side_effect = None
    pipeline.build.return_value = [IngestStats(doc_id="d", n_chunk=1)]
    assert client.post("/api/ingest", json={"confirm": True}).status_code == 200


def test_ingest_gate_is_inside_try_so_prebuild_errors_unlock(client, pipeline,
                                                            monkeypatch):
    """rebuilding/锁必须在 try 内设置：写日志若在 try 外抛错会永久卡住服务。

    旧实现把 ``rebuilding = True`` 与 warning 放在 try 之前：那里任何异常都会绕过
    finally，此后所有问答 503、所有重建 409，且没有任何路径能恢复（终审修正B）。
    """
    def boom(*_a, **_k):
        raise RuntimeError("logging-boom")

    monkeypatch.setattr(server.logger, "warning", boom)
    # 未修复时异常在 try 之外冒泡（TestClient 直接把它抛回测试）；修复后是 500
    r = client.post("/api/ingest", json={"confirm": True})
    assert r.status_code == 500
    assert "logging-boom" in r.json()["error"]

    pipeline.build.assert_not_called()
    assert client.app.state.rebuilding is False               # 闸门已放下
    assert client.app.state.rebuild_lock.acquire(blocking=False)   # 锁已释放
    client.app.state.rebuild_lock.release()
    assert client.post("/api/ask",
                       json={"question": "本次发行股数是多少"}).status_code == 200


def test_ask_returns_503_while_rebuilding(client, pipeline):
    """重建窗口内不得给出「像拒答的 200」或裸 500，必须 503 + 可执行文案。"""
    client.app.state.rebuilding = True
    r = client.post("/api/ask", json={"question": "本次发行股数是多少"})
    assert r.status_code == 503
    assert "重建" in r.json()["error"] and "重试" in r.json()["error"]
    pipeline.ask.assert_not_called()

    client.app.state.rebuilding = False
    assert client.post("/api/ask",
                       json={"question": "本次发行股数是多少"}).status_code == 200


def test_ask_maps_qdrant_lock_error_to_503(client, pipeline):
    """跨进程占用索引（如 CLI 重建）：503 + 建议，原始错误只作诊断附注。"""
    pipeline.ask.side_effect = RuntimeError(
        "Storage folder D:\\x\\data\\qdrant is already accessed by another "
        "instance of Qdrant client.")
    r = client.post("/api/ask", json={"question": "本次发行股数是多少"})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err.startswith("索引被其它进程占用")
    assert "重试" in err


def test_health_reports_rebuilding_flag(client):
    assert client.get("/health").json()["rebuilding"] is False
    client.app.state.rebuilding = True
    assert client.get("/health").json()["rebuilding"] is True


def test_ask_accepts_empty_mode_string(client):
    """空串等同「未指定」（旧行为）；只有拼错的非空值才 422。"""
    r = client.post("/api/ask", json={"question": "本次发行股数是多少", "mode": ""})
    assert r.status_code == 200


def test_ingest_maps_qdrant_lock_runtime_error_to_409(client, pipeline):
    """跨进程占用（别的进程握着 data/qdrant）时，500 必须降级成可执行的 409。"""
    pipeline.store = None                      # 本进程未持有：冲突来自外部
    pipeline.build.side_effect = RuntimeError(
        "Storage folder D:\\x\\data\\qdrant is already accessed by another "
        "instance of Qdrant client.")
    r = client.post("/api/ingest", json={"confirm": True})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err.startswith("索引已被占用")       # 主文案是可执行建议，原始详情在后
    assert "build_index.py" in err
    assert "--reset" in err                     # 建议的 CLI 必须带清库参数


class _FakeClient:
    """样板判据只用到 client.scroll：返回空库即可（不触发任何真实索引访问）。"""

    def scroll(self, coll, limit=100_000, with_payload=True):
        return [], None


class _FakeStore:
    """替身：计数与关闭留档，绝不打开真实 Qdrant。"""

    mode = "embedded"

    def __init__(self):
        self.closed = False
        self.client = _FakeClient()

    def counts(self):
        return {"text_chunks": 3}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_store_factory(monkeypatch):
    """把 VectorStore 换成「第二次打开即抛 Qdrant 锁错误」的工厂。"""
    from rag04.pipeline import RAGPipeline

    made: list[_FakeStore] = []

    def factory(settings):
        if made:                               # 复现真实行为：不能开第二个客户端
            raise RuntimeError(
                "Storage folder ... is already accessed by another instance "
                "of Qdrant client.")
        st = _FakeStore()
        made.append(st)
        return st

    monkeypatch.setattr("rag04.ingest.store.VectorStore", factory)
    monkeypatch.setattr("rag04.retrieve.embed.embed_one", lambda text, s: [0.0] * 1024)
    return RAGPipeline, made


def _pipeline(tmp_path, mode="baseline_03", **overrides):
    """baseline_03：不触发 hybrid/样板判据，聚焦存储锁本身。

    full_04 档必须用 ``**overrides`` 关掉重模型（``use_rerank`` / ``use_clip_retrieval``）：
    本文件的铁律是「绝不构造会加载模型/锁真实索引的流水线」，而 full_04 的
    ``__post_init__`` 只对 baseline_03 翻转开关，所以得在 replace 里显式关。
    """
    from rag04.config import Settings
    from rag04.pipeline import RAGPipeline
    s = Settings(pipeline_mode=mode, data_dir=tmp_path)
    if overrides:
        s = replace(s, **overrides)
    return RAGPipeline(s)


def test_full_mode_test_pipeline_keeps_models_off(tmp_path):
    """守住上一条约束：full_04 + overrides 仍保留 hybrid（断言需要样板判据），
    但 rerank/CLIP 必须为 False —— 否则 build()→warmup() 会在默认套件里加载
    1.1GB reranker（约 10 秒、内存 >1GB）。"""
    p = _pipeline(tmp_path, mode="full_04", use_rerank=False,
                  use_clip_retrieval=False)
    assert p.s.pipeline_mode == "full_04"
    assert p.s.use_hybrid is True
    assert p.s.use_rerank is False and p.s.use_clip_retrieval is False


def test_ensure_loaded_opens_store_once_under_concurrency(tmp_path, fake_store_factory):
    """并发首查只能开一个客户端：第二次 Open 会抛锁异常（压测爬坡期 500 的根因）。"""
    import threading

    _, made = fake_store_factory
    p = _pipeline(tmp_path)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def first_request():
        try:
            barrier.wait(timeout=5)
            p._ensure_loaded()
        except BaseException as e:             # noqa: BLE001 - 测试要收集全部异常
            errors.append(e)

    threads = [threading.Thread(target=first_request) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(made) == 1, "并发首查重复打开了存储"
    assert p.store is made[0]
    for _ in range(3):                          # 已加载后重复调用同样不得再开
        p._ensure_loaded()
    assert len(made) == 1


def test_warmup_preloads_store(tmp_path, fake_store_factory):
    """服务启动预热必须把存储一并打开，首次问答不再付开库成本。"""
    _, made = fake_store_factory
    p = _pipeline(tmp_path)
    p.warmup()
    assert len(made) == 1 and p.store is made[0]
    p.warmup()                                  # 幂等：重复预热不开第二个客户端
    assert len(made) == 1


def test_warmup_survives_store_failure(tmp_path, monkeypatch):
    """存储预热失败只记录不抛出：预热是优化，不是正确性前提（Task 20 也用）。"""
    def boom(settings):
        raise RuntimeError("Storage folder is already accessed by another instance")

    monkeypatch.setattr("rag04.ingest.store.VectorStore", boom)
    monkeypatch.setattr("rag04.retrieve.embed.embed_one", lambda text, s: [0.0] * 1024)
    p = _pipeline(tmp_path)
    p.warmup()
    assert p.store is None                      # 未加载，问答时再重试


def test_build_releases_store_before_reingest(tmp_path, monkeypatch):
    """重建必须先让路：同进程持有句柄时再开客户端同样会撞锁（UI 重建按钮同样受益）。"""
    from rag04.pipeline import IngestStats

    made: list[_FakeStore] = []

    def factory(settings):
        st = _FakeStore()
        made.append(st)
        return st

    monkeypatch.setattr("rag04.ingest.store.VectorStore", factory)
    monkeypatch.setattr("rag04.retrieve.embed.embed_one", lambda text, s: [0.0] * 1024)
    seen = {}

    def fake_build_all(s, reset=False, names=None):
        seen["store_at_build"] = p.store        # build_all 自己开客户端：此处必须已释放
        return [IngestStats(doc_id="d", n_chunk=1)]

    monkeypatch.setattr("rag04.pipeline.build_all", fake_build_all)
    p = _pipeline(tmp_path)
    p._ensure_loaded()
    first = p.store
    stats = p.build()
    assert seen["store_at_build"] is None and first.closed is True
    assert stats[0].n_chunk == 1
    assert p.store is made[-1] and p.store is not first   # 重建后重新载入新句柄
    assert p.bm25 is not None                             # BM25 也随新索引重新载入


def test_build_forwards_names_and_reset_to_build_all(tmp_path, monkeypatch):
    """接口契约（UI 上传建库依赖）：``build(names=[...], reset=True)`` 必须原样下传。

    ``build_all`` 默认只遍历 ``Settings.corpus``；这一层若把 ``names`` 吞掉，
    上传的文件就永远不进索引，而界面还会显示成功（终审修正B 的根因）。
    """
    from rag04.pipeline import IngestStats

    seen: dict = {}
    monkeypatch.setattr("rag04.ingest.store.VectorStore", lambda s: _FakeStore())
    monkeypatch.setattr("rag04.retrieve.embed.embed_one", lambda text, s: [0.0] * 1024)

    def fake_build_all(s, reset=False, names=None):
        seen.update(reset=reset, names=names)
        return [IngestStats(doc_id="d", n_chunk=1)]

    monkeypatch.setattr("rag04.pipeline.build_all", fake_build_all)
    p = _pipeline(tmp_path)
    p.build(names=["新文档.pdf"], reset=True)
    assert seen == {"reset": True, "names": ["新文档.pdf"]}


def test_build_reloads_bm25_and_boilerplate_in_full_mode(tmp_path, monkeypatch):
    """full_04：重建后 bm25 与样板判据（counts 快照随索引变化）必须重新载入。

    rerank/CLIP 显式关闭：断言只需要 hybrid（样板判据）与 bm25，开着重模型会让
    这条单测在默认套件里加载 1.1GB reranker（终审修正B：测试自身也要轻）。
    """
    from rag04.pipeline import IngestStats

    made: list[_FakeStore] = []

    def factory(settings):
        st = _FakeStore()
        made.append(st)
        return st

    monkeypatch.setattr("rag04.ingest.store.VectorStore", factory)
    monkeypatch.setattr("rag04.retrieve.embed.embed_one", lambda text, s: [0.0] * 1024)
    monkeypatch.setattr("rag04.pipeline.build_all",
                        lambda s, reset=False, names=None:
                        [IngestStats(doc_id="d", n_chunk=1)])

    p = _pipeline(tmp_path, mode="full_04", use_rerank=False,
                  use_clip_retrieval=False)
    p._ensure_loaded()
    old_bm25, old_bp = p.bm25, p.boilerplate
    assert old_bm25 is not None and old_bp is not None

    p.build()
    assert p.store is made[-1]
    assert p.bm25 is not None and p.bm25 is not old_bm25, "BM25 必须重新载入"
    assert p.boilerplate is not None and p.boilerplate is not old_bp, \
        "样板判据必须在新索引上重新载入"
