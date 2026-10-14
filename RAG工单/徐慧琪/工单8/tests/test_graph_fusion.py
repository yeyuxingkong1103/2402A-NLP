# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.graph_fusion import build_context_text, fuse

V = [{"chunk_id": "v1", "text": "平安银行不良贷款率1.65%", "source": "a.pdf",
      "page_idx": 23, "score": 0.9},
     {"chunk_id": "v2", "text": "招商银行不良贷款率1.16%", "source": "b.pdf",
      "page_idx": 19, "score": 0.8}]
F = [{"chunk_id": "f1", "text": "平安银行不良贷款率1.65%", "source": "a.pdf",
      "page_idx": 23, "score": 5.0},
     {"chunk_id": "f2", "text": "拨备覆盖率183.12%", "source": "a.pdf",
      "page_idx": 24, "score": 4.0}]
G = ["平安银行2019年末的不良贷款率为 1.65%（来源：a.pdf 第 23 页）"]


def test_fuse_includes_all_three_channels():
    out = fuse(G, V, F)
    assert out.graph_facts == G
    assert out.text_hits, "必须有文本块"


def test_fuse_dedupes_same_page_across_channels():
    """同一文档同一页在向量与全文两路都命中时，只保留一条。"""
    out = fuse(G, V, F)
    keys = [(h["source"], h["page_idx"]) for h in out.text_hits]
    assert len(keys) == len(set(keys))


def test_fuse_respects_top_k():
    out = fuse(G, V, F, top_k=2)
    assert len(out.text_hits) <= 2


def test_fuse_weights_sum_to_one():
    out = fuse(G, V, F)
    assert abs(sum(out.weights.values()) - 1.0) < 1e-6
    assert set(out.weights) == {"graph", "vector", "fulltext"}


def test_context_text_has_both_sections_with_pages():
    out = fuse(G, V, F)
    text = build_context_text(out)
    assert "图谱事实" in text
    assert "文本证据" in text
    assert "1.65%" in text
    assert "第 23 页" in text


def test_context_text_empty_when_nothing_found():
    out = fuse([], [], [])
    assert out.text_hits == []
    assert "图谱事实" not in build_context_text(out) or out.graph_facts == []


def test_fuse_handles_empty_graph_facts():
    out = fuse([], V, F)
    assert out.text_hits
    assert out.graph_facts == []
