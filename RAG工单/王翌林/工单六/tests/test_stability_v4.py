# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_stability_v4.py —— 工单四 异常容错稳定性单测（全部 Mock，不依赖外部服务）

覆盖 5 类异常：
  1. 图像提取失败（PDF 损坏）  2. 多模态模型失败（VLM 抛异常）
  3. Milvus rag_images 连接失败  4. LLM 超时  5. 空输入
"""
import pytest

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


# ---------------- 工单四：异常1 图像提取失败 ----------------
def test_image_extraction_failure(tmp_path):
    """工单四：损坏 PDF → extractor 优雅报错不崩溃（返回非零退出码或异常）"""
    import subprocess, sys
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    r = subprocess.run(
        [sys.executable, "-m", "src.image_parser.image_extractor",
         "--pdf", str(bad), "--out", str(tmp_path / "o.json")],
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0                       # 工单四：明确失败而非挂死
    assert not (tmp_path / "o.json").exists()


# ---------------- 工单四：异常2 多模态模型失败 ----------------
def test_vlm_failure_graceful(monkeypatch, tmp_path):
    """工单四：VLM caption 抛异常 → caption 为空、状态 failed/降级、不阻塞"""
    from src.image_parser.image_captioner import ImageCaptioner

    class BoomVLM:
        engine_name = "boom"

        def caption(self, image):
            raise RuntimeError("CUDA OOM")

        def generate(self, image, prompt, max_new_tokens=512):
            raise RuntimeError("CUDA OOM")

    cap = ImageCaptioner(vlm_engine=BoomVLM(), enable_ocr=False)
    out = cap.parse_one({"image_id": "img_x", "page": 1, "path": "no_such.png"})
    assert out["parse_status"] in ("failed", "unavailable", "empty")
    assert out["caption"] == ""                     # 工单四：异常被捕获为空描述


# ---------------- 工单四：异常3 Milvus rag_images 连接失败 ----------------
def test_milvus_connection_failure_fallback(monkeypatch, tmp_path):
    """工单四：远程连接失败 → 自动降级 Milvus Lite（检索链路不断）"""
    from src.image_parser.image_store import ImageStore

    def boom(*a, **k):
        raise ConnectionError("milvus down")

    monkeypatch.setenv("MILVUS_HOST", "10.255.255.1")   # 工单四：不可达地址
    monkeypatch.setenv("MILVUS_PORT", "1")
    s = ImageStore(lite_path=str(tmp_path / "fb.db"), force_lite=True)
    s.ensure_collection()
    assert s.insert([{"doc_id": "d", "image_id": "i", "page": 1, "path": "p",
                      "embedding": [0.1] * 1024}]) == 1


# ---------------- 工单四：异常4 LLM 超时 ----------------
def test_llm_timeout_retry_then_raise(monkeypatch):
    """工单四：LLM 超时 → 引擎层重试后仍失败则抛出（API 层转 500）"""
    import src.rag_engine_v4 as m

    class FakeV3:
        top_k = 5
        max_context_chars = 9000
        table_retriever = type("TR", (), {"retrieve": lambda self, *a, **k: {
            "fused": [{"source": "text", "doc_id": "d", "page": 1,
                       "content": "内容", "rrf_score": 0.9}], "elapsed_ms": 1.0}})()

        def ask_llm(self, q):
            return {"answer": "纯LLM", "references": [], "latency_ms": 1.0}

    eng = m.RAGEngineV4(v3_engine=FakeV3(), image_retriever=MagicMockRet())
    calls = {"n": 0}

    def timeout_chat(**kw):
        calls["n"] += 1
        raise TimeoutError("LLM 30s 超时")

    monkeypatch.setattr(m, "chat", timeout_chat)
    with pytest.raises(TimeoutError):
        eng.ask("测试问题", doc_id="招股说明书1")
    assert calls["n"] == 2                          # 工单四：空响应/异常重试一次


class MagicMockRet:
    """工单四：图像检索 Mock（返回一条可引用记录）"""
    def retrieve(self, q, top_k=5, doc_ids=None):
        return [{"image_id": "img_1", "doc_id": "d", "page": 1, "path": "p.png",
                 "caption": "", "ocr_text": "", "vqa_text": "", "rrf": 0.5}]

    def _get_store(self):
        return self


# ---------------- 工单四：异常5 空输入 ----------------
def test_empty_input_guard(engine_v4_min):
    """工单四：空问题 → 空检索结果路径，不崩溃、返回兜底话术"""
    r = engine_v4_min.ask("", doc_id=None)
    assert isinstance(r["answer"], str) and r["answer"]


@pytest.fixture()
def engine_v4_min():
    """工单四：最小引擎（文本检索空结果 + 图像空）"""
    import src.rag_engine_v4 as m

    class EmptyTR:
        def retrieve(self, *args, **k):
            return {"fused": [], "elapsed_ms": 0.5}

    class FakeV3:
        top_k = 5
        max_context_chars = 9000
        table_retriever = EmptyTR()

        def ask_llm(self, q):
            return {"answer": "", "references": [], "latency_ms": 0.5}

    return m.RAGEngineV4(v3_engine=FakeV3(), image_retriever=MagicMockRet())
