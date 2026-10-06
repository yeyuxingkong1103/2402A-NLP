# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_api_optimized.py —— Step 9 单元测试 + 异常测试

覆盖范围（要求 1、2）：
  1. 单元测试：PDF 解析（优化版）、语义分块（父子块/滑动窗口）、混合检索（RRF 融合）、
     重排序（bge-reranker 降级/打分排序）、FastAPI 接口（health/ask/feedback/questions）
  2. 异常测试：PDF 文件不存在 / PDF 加密 / PyMuPDF 缺失、Milvus 连接失败、
     LLM 超时（openai.APITimeoutError）/ API Key 缺失、用户输入空 / 越界参数、
     RAG 引擎未就绪 / 引擎内部异常

策略：全部使用本地构造数据 + unittest.mock，不依赖真实 Milvus / MySQL / bge 模型 / LLM，
     秒级可重复执行；真实链路由 scripts/evaluate_optimized.py 与 8002 服务联调验证。
"""
import pytest
from unittest.mock import MagicMock, patch

# ======================================================================
# 1. PDF 解析（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
from src import pdf_parser_optimized as ppo


def _make_pdf(path, encrypted=False):
    """构造单页测试 PDF：标题 + 正文（china-s 字体支持中文）"""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(60, 120), "第一节 公司概况", fontsize=16, fontname="china-s")
    page.insert_text(fitz.Point(60, 160), "公司营业收入持续增长，净利润稳步提升。",
                     fontsize=10, fontname="china-s")
    if encrypted:
        # 工单二异常测试：用户密码加密的 PDF（PyMuPDF AES-256）
        doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
                 owner_pw="secret", user_pw="secret")
    else:
        doc.save(str(path))
    doc.close()
    return str(path)


def test_pdf_parse_success(tmp_path):
    """正常 PDF：必填字段齐全、页数正确、正文可提取、无错误"""
    data = ppo.parse_pdf_optimized(_make_pdf(tmp_path / "ok.pdf"), enable_ocr=False)
    assert data["errors"] == [], data["errors"]
    assert data["total_pages"] == 1
    assert data["doc_id"] and len(data["doc_id"]) == 16
    assert "营业收入持续增长" in data["text"]
    assert isinstance(data["layout_blocks"], list) and data["layout_blocks"]
    assert isinstance(data["headings"], list)


def test_pdf_file_not_found():
    """异常：PDF 文件不存在 —— 返回 errors 且不抛异常（优雅失败）"""
    data = ppo.parse_pdf_optimized("/nonexistent/path/工单二_missing.pdf")
    assert data["total_pages"] == 0
    assert any("不存在" in e for e in data["errors"])


def test_pdf_encrypted(tmp_path):
    """异常：PDF 已加密需要密码 —— errors 含"需要密码"，不崩溃"""
    path = _make_pdf(tmp_path / "locked.pdf", encrypted=True)
    data = ppo.parse_pdf_optimized(path, enable_ocr=False)
    assert any("密码" in e for e in data["errors"]), data["errors"]


def test_pdf_parser_without_pymupdf(tmp_path, monkeypatch):
    """异常：PyMuPDF 未安装 —— 明确报错而非 ImportError 崩溃"""
    path = _make_pdf(tmp_path / "real.pdf")  # 文件真实存在，排除文件检查分支
    monkeypatch.setattr(ppo, "fitz", None)
    data = ppo.parse_pdf_optimized(path)
    assert data["errors"] == ["PyMuPDF 未安装"]


# ======================================================================
# 2. 语义分块（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
from src.chunker_optimized import (build_children, build_parent_blocks,
                                   chunk_semantic, sliding_window)


def test_sliding_window_short_text_single_piece():
    assert sliding_window("短文本", chunk_size=600, overlap=100) == ["短文本"]


def test_sliding_window_size_and_overlap():
    """长文本：多片、每片不超过窗口、相邻片存在 overlap 重叠区"""
    text = "这是一个完整句子。" * 200  # 9*200 = 1800 字
    pieces = sliding_window(text, chunk_size=300, overlap=100)
    assert len(pieces) >= 2
    assert all(len(p) <= 300 for p in pieces)
    # 第二片开头 100 字来自第一片尾部（滑动重叠）
    assert pieces[1][:100] == pieces[0][-100:]


@pytest.fixture
def parsed_for_chunk():
    """构造版面分析产物：标题 + 正文 + 页眉噪声 + 结构化表格"""
    return {
        "doc_id": "dtest", "filename": "x.pdf",
        "layout_blocks": [
            {"page": 1, "type": "header", "text": "招股说明书页眉噪声", "y0": 0, "y1": 10},
            {"page": 1, "type": "heading", "text": "第一章 释义",
             "y0": 20, "y1": 40, "level": 1, "font_size": 16.0},
            {"page": 1, "type": "body",
             "text": "本公司营业收入持续增长。" * 30, "y0": 50, "y1": 200},
            {"page": 2, "type": "heading", "text": "第二章 风险",
             "y0": 20, "y1": 40, "level": 1, "font_size": 16.0},
            {"page": 2, "type": "body", "text": "公司面临市场竞争风险。" * 30,
             "y0": 50, "y1": 200},
        ],
        "tables_structured": [
            {"page": 1, "table_index": 1, "n_rows": 2, "n_cols": 2,
             "headers": ["项目", "金额"], "rows": [["营业收入", "100万"]],
             "markdown": "| 项目 | 金额 |\n| --- | --- |\n| 营业收入 | 100万 |"},
        ],
    }


def test_parent_blocks_structure(parsed_for_chunk):
    parents = build_parent_blocks(parsed_for_chunk)
    assert len(parents) >= 2  # 两个章节 + 表格
    # 页眉页脚噪声不得进入任何父块
    assert all("页眉噪声" not in p["text"] for p in parents)
    # 标题开启新父块且 heading 元数据保留
    headings = [p for p in parents if p["heading"]]
    assert any("第一章" in p["heading"] for p in headings)
    # 表格独立成父块
    assert any(p["chunk_type"] == "table" and "营业收入" in p["text"] for p in parents)
    # chunk_id 连续编号
    assert parents[0]["chunk_id"].endswith("_parent_0001")


def test_children_link_to_parents(parsed_for_chunk):
    parents = build_parent_blocks(parsed_for_chunk)
    children = build_children(parents, chunk_size=600, overlap=100)
    assert children, "至少应产生一个子块"
    parent_ids = {p["chunk_id"] for p in parents}
    for c in children:
        assert c["parent_id"] in parent_ids          # 子块指回父块
        assert c["role"] == "child" and c["page"]    # 元数据继承
    # 每个父块至少有一个子块（全覆盖）
    assert {c["parent_id"] for c in children} == parent_ids


def test_chunk_semantic_summary(parsed_for_chunk):
    result = chunk_semantic(parsed_for_chunk, chunk_size=600, overlap=100)
    assert result["strategy"] == "semantic_parent_child_v2"
    assert result["total_parent_chunks"] >= 2
    assert result["total_child_chunks"] >= result["total_parent_chunks"]
    assert result["chunks"] and result["parent_chunks"]


# ======================================================================
# 3. 混合检索 RRF 融合（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
from src.retriever import Retriever


def test_rrf_fuse_dedup_and_order():
    """两路召回同一 chunk_id 去重，双路命中的 c1 融合分最高"""
    vec = [
        {"chunk_id": "c1", "content": "共同命中", "page": 1},
        {"chunk_id": "c2", "content": "仅向量", "page": 2},
    ]
    bm25 = [
        {"chunk_id": "c1", "content": "共同命中", "page": 1},
        {"chunk_id": "c3", "content": "仅BM25", "page": 3},
    ]
    # 未绑定方法调用：self 传 None（_rrf_fuse 不使用实例状态）
    fused = Retriever._rrf_fuse(None, vec, bm25, k=60)
    ids = [d["chunk_id"] for d in fused]
    assert ids[0] == "c1"                 # 双路命中排第一
    assert len(ids) == len(set(ids)) == 3  # 去重
    assert all("rrf_score" in d for d in fused)


def test_rrf_fuse_empty():
    assert Retriever._rrf_fuse(None, [], []) == []


def test_format_for_prompt_citation():
    prompt = Retriever.format_for_prompt([
        {"page": 12, "chunk_id": "abc", "content": "营业收入100万"}])
    assert "[资料1]" in prompt and "第12页" in prompt and "abc" in prompt


# ======================================================================
# 4. 重排序（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
from src.reranker import Reranker


def test_reranker_empty_candidates():
    assert Reranker().rerank("q", [], top_k=5) == []


def test_reranker_degradation_when_model_unavailable():
    """异常：reranker 模型加载失败 —— 降级保序透传，不中断检索链路"""
    rr = Reranker()
    rr._failed = True
    cands = [{"content": "a", "score": 1.0, "chunk_id": "x"},
             {"content": "b", "score": 2.0, "chunk_id": "y"}]
    out = rr.rerank("q", cands, top_k=1)
    assert len(out) == 1 and out[0]["chunk_id"] == "x"
    assert out[0]["rerank_score"] == 1.0


def test_reranker_scores_reorder():
    """模型正常时按 compute_score 降序取 top_k"""
    rr = Reranker()
    rr._model = MagicMock()
    rr._model.compute_score.return_value = [0.1, 0.9, 0.5]
    cands = [{"content": "低", "chunk_id": "a"},
             {"content": "高", "chunk_id": "b"},
             {"content": "中", "chunk_id": "c"}]
    out = rr.rerank("查询", cands, top_k=2)
    assert [d["chunk_id"] for d in out] == ["b", "c"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)


def test_reranker_compute_error_fallback():
    """异常：rerank 推理抛错 —— 降级保序，不向上抛"""
    rr = Reranker()
    rr._model = MagicMock()
    rr._model.compute_score.side_effect = RuntimeError("GPU OOM")
    out = rr.rerank("q", [{"content": "a", "score": 0.3, "chunk_id": "x"}], top_k=1)
    assert out[0]["chunk_id"] == "x"


# ======================================================================
# 5. LLM 客户端异常（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
def test_llm_timeout_propagates(monkeypatch):
    """异常：LLM 请求超时（openai.APITimeoutError）必须向上抛出由 API 层转 500"""
    import openai
    from src import llm_client

    fake = MagicMock()
    err = openai.APITimeoutError(request=MagicMock())
    fake.chat.completions.create.side_effect = err
    monkeypatch.setattr(llm_client, "_client", fake)
    with pytest.raises(openai.APITimeoutError):
        llm_client.chat([{"role": "user", "content": "超时问题？"}])


def test_llm_missing_api_key(monkeypatch):
    """异常：未配置 DEEPSEEK_API_KEY —— RuntimeError 明确提示"""
    from src import llm_client
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm_client.get_client()


# ======================================================================
# 6. Milvus 连接失败（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
def test_vector_store_all_endpoints_unavailable(tmp_path, monkeypatch):
    """异常：远程 Milvus 与 Milvus Lite 均不可用 —— 构造时抛出连接异常"""
    from src import vector_store

    monkeypatch.setattr(vector_store, "MilvusClient",
                        MagicMock(side_effect=ConnectionError("milvus down")))
    with pytest.raises(ConnectionError):
        vector_store.VectorStore(lite_path=str(tmp_path / "fail.db"))


# ======================================================================
# 7. FastAPI 接口（人工智能NLP-RAG-基于PDF文档的问答系统优化）
# ======================================================================
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    # 注意：不使用 with TestClient(app)，避免触发 lifespan 连接真实
    # MySQL / Milvus / 加载 RAGEngine（工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    return TestClient(app)


@pytest.fixture
def fake_engine(client, monkeypatch):
    """注入伪 RAG 引擎并屏蔽问答日志落库（不依赖 MySQL）"""
    engine = MagicMock()
    engine.ask_optimized.return_value = {
        "mode": "rag_optimized", "answer": "优化链路答案",
        "references": [{"page": 12, "chunk_id": "c_0001", "score": 0.98,
                        "preview": "营业收入 7,955.46 万元"}],
        "latency_ms": 88.5, "token_usage": {"total_tokens": 120},
        "query_understanding": None,
        "breakdown": {"retrieve_ms": 40.0, "llm_ms": 48.0},
        "cache_hit": False,
    }
    engine.ask_rag.return_value = {
        "mode": "rag", "answer": "基线答案", "references": [],
        "latency_ms": 13520.0, "token_usage": {}, "query_understanding": {},
        "breakdown": {"retrieve_ms": 12000.0, "llm_ms": 1520.0}}
    engine.ask_llm.return_value = {
        "mode": "pure_llm", "answer": "纯LLM答案", "references": [],
        "latency_ms": 300.0, "token_usage": {}}
    app.state.rag_engine = engine
    monkeypatch.setattr("src.api._log_qa", lambda *a, **k: None)
    yield engine
    if hasattr(app.state, "rag_engine"):
        delattr(app.state, "rag_engine")


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "uptime_seconds" in r.json()


def test_api_questions(client):
    r = client.get("/api/questions")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_api_ask_optimized_ok(client, fake_engine):
    r = client.post("/api/ask", json={"question": "公司营业收入是多少？", "top_k": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "rag_optimized"
    assert body["answer"] == "优化链路答案"
    assert body["references"][0]["page"] == 12
    assert body["breakdown"]["llm_ms"] == 48.0
    assert body["cache_hit"] is False
    fake_engine.ask_optimized.assert_called_once()


def test_api_ask_baseline_chain(client, fake_engine):
    """chain=baseline 必须路由到 ask_rag（前端优化前后对比用）"""
    r = client.post("/api/ask", json={"question": "q", "chain": "baseline"})
    assert r.status_code == 200
    assert r.json()["mode"] == "rag"
    fake_engine.ask_rag.assert_called_once()
    fake_engine.ask_optimized.assert_not_called()


def test_api_ask_pure_llm(client, fake_engine):
    r = client.post("/api/ask", json={"question": "你好", "use_rag": False})
    assert r.status_code == 200
    assert r.json()["mode"] == "pure_llm"
    assert r.json()["references"] == []


def test_api_ask_empty_question_422(client, fake_engine):
    """异常：用户输入空字符串 —— Pydantic 422 参数校验"""
    r = client.post("/api/ask", json={"question": ""})
    assert r.status_code == 422


def test_api_ask_question_too_long_422(client, fake_engine):
    """异常：问题超过 2000 字上限 —— 422"""
    r = client.post("/api/ask", json={"question": "问" * 2001})
    assert r.status_code == 422


def test_api_ask_topk_out_of_range_422(client, fake_engine):
    """异常：top_k 越界（0 与 21）—— 422"""
    assert client.post("/api/ask", json={"question": "q", "top_k": 0}).status_code == 422
    assert client.post("/api/ask", json={"question": "q", "top_k": 21}).status_code == 422


def test_api_ask_engine_not_ready_503(client):
    """异常：RAG 引擎未就绪（启动失败）—— 503 友好提示"""
    if hasattr(app.state, "rag_engine"):
        delattr(app.state, "rag_engine")
    r = client.post("/api/ask", json={"question": "q"})
    assert r.status_code == 503
    assert "未就绪" in r.json()["detail"]


def test_api_ask_engine_exception_500(client, fake_engine):
    """异常：引擎内部失败（如 LLM 超时透传）—— API 层兜底 500，不泄露堆栈"""
    fake_engine.ask_optimized.side_effect = TimeoutError("LLM request timeout")
    r = client.post("/api/ask", json={"question": "q"})
    assert r.status_code == 500
    assert "timeout" in r.json()["detail"]


def test_api_feedback_rating_validation(client):
    """异常：feedback rating 越界（允许 -1/0/1）—— 422"""
    payload = {"qa_log_id": 1, "rating": 5, "comment": ""}
    assert client.post("/api/feedback", json=payload).status_code == 422
    payload["rating"] = -2
    assert client.post("/api/feedback", json=payload).status_code == 422


def test_api_feedback_ok(client, monkeypatch):
    """点赞/点踩/评论落库：db 会话 mock，返回自增 id（工单二 rating 语义 +1/-1/0）"""

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def add(self, obj):
            self.obj = obj

        def commit(self):
            self.obj.id = 7

        def refresh(self, obj):
            pass

    fake_session = _FakeSession()
    monkeypatch.setattr("src.db.get_session", lambda: fake_session)
    monkeypatch.setattr("src.models.Feedback", MagicMock())
    r = client.post("/api/feedback",
                    json={"qa_log_id": 3, "rating": -1, "comment": "答案有误"})
    assert r.status_code == 200, r.text
    assert r.json() == {"id": 7, "ok": True}


def test_api_feedback_db_failure_500(client, monkeypatch):
    """异常：反馈落库失败 —— 500 且不崩进程"""
    def _boom():
        raise RuntimeError("mysql gone away")

    monkeypatch.setattr("src.db.get_session", _boom)
    r = client.post("/api/feedback", json={"qa_log_id": 1, "rating": 1})
    assert r.status_code == 500
