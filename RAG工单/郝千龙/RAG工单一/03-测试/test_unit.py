# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【单元测试 · test_unit.py】覆盖 PDF解析/切分/缓存/Embedding/LLM/RRF/BM25/反馈
# 运行：pytest test_unit.py -v
# 编写日期：2026-09-28   修订日期：2026-10-04
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 将研发目录加入 path
CODE_DIR = Path(__file__).resolve().parent.parent / "02-研发"
sys.path.insert(0, str(CODE_DIR))

import config          # noqa: E402
import pdf_parser      # noqa: E402
import text_splitter   # noqa: E402


# ---------- PDF 解析 ----------
def test_parse_pdf_smoke():
    """冒烟测试：从缓存读取招股书，记录非空且页码有序"""
    data = pdf_parser.parse_pdf(config.SOURCE_PDF, cache=True)
    assert len(data) > 0
    assert all("page" in d and "text" in d for d in data)
    pages = [d["page"] for d in data]
    assert pages == sorted(pages)


def test_parse_pdf_cache_file():
    """解析缓存文件格式正确"""
    assert config.JSONL_CACHE.exists()
    lines = config.JSONL_CACHE.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert "page" in rec and "text" in rec


# ---------- 文本切分 ----------
def test_split_pages_chunk_size():
    """切块大小受控"""
    pages = [{"page": 1, "text": "测试。" * 2000, "has_table": False}]
    chunks = text_splitter.split_pages(pages)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP + 20


def test_split_pages_metadata_kept():
    """切分后页码与 chunk_id 保留"""
    pages = [{"page": 5, "text": "武汉兴图新科电子股份有限公司。", "has_table": False}]
    chunks = text_splitter.split_pages(pages)
    assert all(c["page"] == 5 for c in chunks)
    assert all("chunk_id" in c for c in chunks)


# ---------- 配置 ----------
def test_config_basic():
    """核心配置符合当前架构"""
    assert config.EMBED_DIM == 768
    assert config.RETRIEVE_TOP_K == 5
    assert len(config.WORKORDER_QUESTIONS) == 10
    assert all("id" in q and "question" in q for q in config.WORKORDER_QUESTIONS)


def test_workorder_questions_ids():
    """工单 10 题 ID 完整"""
    ids = {q["id"] for q in config.WORKORDER_QUESTIONS}
    assert ids == {260, 95, 33, 34, 957, 793, 795, 543, 531, 207}


def test_prompt_bilingual():
    """双语 Prompt 可获取且含占位符"""
    zh = config.get_rag_prompt("zh")
    en = config.get_rag_prompt("en")
    assert "{context}" in zh and "{question}" in zh
    assert "{context}" in en and "{question}" in en


# ---------- Cache mock ----------
def test_cache_get_miss(monkeypatch):
    """缓存未命中返回 None"""
    from cache import LocalCache
    fake = MagicMock(); fake.get.return_value = None
    monkeypatch.setattr(LocalCache, "_store", {}, raising=False)
    monkeypatch.setattr(config, "ENABLE_CACHE", True)
    from cache import Cache
    assert Cache.get_answer("foo", 5) is None


def test_cache_get_hit(monkeypatch):
    """缓存命中返回负载"""
    import time as _t
    from cache import LocalCache, _key
    payload = {"answer": "cached", "refs": []}
    # 必须用与门面相同的哈希键，否则查不到
    real_key = _key("foo", 5, "zh")
    monkeypatch.setattr(
        LocalCache, "_store",
        {real_key: {"ts": _t.time(), "payload": payload}},
        raising=False,
    )
    monkeypatch.setattr(config, "CACHE_BACKEND", "local")
    from cache import Cache
    assert Cache.get_answer("foo", 5)["answer"] == "cached"


# ---------- Embedder mock ----------
def test_embedder_singleton():
    """Embedder 应为单例"""
    with patch("embedder.SentenceTransformer") as MockST:
        import numpy as np
        MockST.return_value.get_sentence_embedding_dimension.return_value = 768
        MockST.return_value.encode.return_value = np.zeros((1, 768), dtype="float32")
        from embedder import Embedder
        Embedder._instance = None
        a, b = Embedder(), Embedder()
        assert a is b


# ---------- LLM mock ----------
def test_llm_answer_with_context():
    """RAG 生成应包含上下文事实"""
    with patch("llm_client.openai") as MockOpenAI:
        MockOpenAI.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="法定代表人为程家明"))]
        )
        from llm_client import LLMClient
        LLMClient._sync = None
        cli = LLMClient()
        ans = cli.answer_with_context(
            "法定代表人是谁？", [("公司法定代表人：程家明", 22)]
        )
        assert "程家明" in ans


# ---------- RRF 融合 ----------
def test_rrf_fuse():
    """RRF 融合：两路共有片段排名应最高"""
    import rag_engine
    a = [{"chunk_id": "x1", "text": "a", "page": 1, "doc_id": "d", "score": 0.9}]
    b = [{"chunk_id": "x1", "text": "a", "page": 1, "doc_id": "d", "score": 5.0}]
    out = rag_engine._rrf_fuse(a, b)
    assert len(out) == 1
    # 两路各贡献 1/(k+1)
    assert abs(out[0]["fusion_score"] - 2.0 / (config.RRF_K + 1)) < 1e-9


def test_rrf_fuse_ordering():
    """RRF 结果按融合分降序"""
    import rag_engine
    a = [
        {"chunk_id": "a", "text": "", "page": 1, "doc_id": "d", "score": 1},
        {"chunk_id": "b", "text": "", "page": 2, "doc_id": "d", "score": 1},
    ]
    b = [
        {"chunk_id": "a", "text": "", "page": 1, "doc_id": "d", "score": 1},
        {"chunk_id": "c", "text": "", "page": 3, "doc_id": "d", "score": 1},
    ]
    out = rag_engine._rrf_fuse(a, b)
    assert out[0]["chunk_id"] == "a"


# ---------- BM25 分词 ----------
def test_bm25_tokenize():
    """BM25 分词非空且保留中文词"""
    from bm25_retriever import _tokenize
    toks = _tokenize("公司注册资本为5520万元")
    assert len(toks) > 0
    assert "公司" in toks


# ---------- 反馈保存 ----------
def test_save_feedback(tmp_path, monkeypatch):
    """反馈追加写入文件"""
    fb_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "FEEDBACK_FILE", fb_file)
    import rag_engine
    rec = rag_engine.save_feedback("q", "a", "up", comment="好")
    assert rec["rating"] == "up"
    lines = fb_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["comment"] == "好"


def test_save_feedback_bad_rating(tmp_path, monkeypatch):
    """非法 rating 应断言失败"""
    monkeypatch.setattr(config, "FEEDBACK_FILE", tmp_path / "f.jsonl")
    import rag_engine
    with pytest.raises(AssertionError):
        rag_engine.save_feedback("q", "a", "bad")


# ---------- RAG engine mock 端到端 ----------
def test_rag_ask_pipeline():
    """端到端 mock：无需真实向量库/LLM"""
    import numpy as np
    with patch("rag_engine.Embedder") as MockEmb, \
         patch("rag_engine.VectorStore") as MockVS, \
         patch("rag_engine.LLMClient") as MockLLM, \
         patch("rag_engine.Cache") as MockCache, \
         patch("rag_engine.BM25Retriever"):
        MockEmb.return_value.encode_one.return_value = np.zeros(768, dtype="float32")
        MockVS.search.return_value = [
            {"text": "公司注册资本 5,520 万元", "page": 22,
             "doc_id": "p1", "chunk_id": "x", "score": 0.9}
        ]
        MockCache.get_answer.return_value = None
        MockLLM.return_value.answer_with_context.return_value = "注册资本 5,520 万元"
        import rag_engine
        out = rag_engine.ask("注册资本？", use_cache=True)
        assert "5,520" in out["answer"]
        assert out["refs"][0]["page"] == 22
