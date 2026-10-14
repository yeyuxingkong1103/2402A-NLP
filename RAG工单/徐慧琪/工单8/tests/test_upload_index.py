# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""上传文档（会话级索引）的规格测试。

工单产出物要求用户手册覆盖「如何上传 PDF 文档」，系统需支持用户上传自己的
PDF 并就该文档提问。规格：
  · 上传的 PDF 解析 → 分块，保留 source/page_idx（答案要能回到页码）；
  · 检索走会话级内存索引，**不触碰**全局语料库与知识图谱；
  · 非法文件给出可读错误，而不是让界面崩。

测试用注入的词袋嵌入器，不加载真实模型（保持测试秒级、可在 CI 跑）。
"""
import math
import re

import fitz
import pytest

from src.upload_index import UploadError, build_index_from_pdf


class _BagEmbedder:
    """极简词袋嵌入器：仅用于测试，无模型依赖、结果确定。"""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, texts):
        for t in texts:
            for w in re.findall(r"[a-z0-9%.]+", t.lower()):
                self.vocab.setdefault(w, len(self.vocab))
        return self

    def __call__(self, texts):
        out = []
        for t in texts:
            v = [0.0] * max(len(self.vocab), 1)
            for w in re.findall(r"[a-z0-9%.]+", t.lower()):
                if w in self.vocab:
                    v[self.vocab[w]] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _pdf_bytes(pages) -> bytes:
    """每页 = 短编号标题 + 正文段落。

    分块器按**章节**（标题层级）切分，两个约束来自解析器既有规则：
      · 标题靠编号正则识别 → 测试数据必须带编号，否则整册并成一个块、丢页码；
      · 编号行**超过 45 字会被判为正文**（防止「（一）xxx」式引用误判），
        所以标题要短、正文另起一行；
      · 用 ASCII 的「1.」而非中文「一、」—— fitz 默认字体编码不了 CJK，
        中文会渲染成占位点，正则匹配不上。
    """
    doc = fitz.open()
    for head, body in pages:
        page = doc.new_page()
        page.insert_text((72, 90), head, fontsize=12)
        page.insert_text((72, 120), body, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


PAGES = [
    ("1. Ping An Bank 2019",
     "Ping An Bank annual report: NPL ratio was 1.65 percent at the end of "
     "2019, down 0.10 percentage points year on year."),
    ("2. China Merchants Bank 2019",
     "China Merchants Bank annual report: NPL ratio was 1.16 percent at the "
     "end of 2019, down 0.20 percentage points year on year."),
    ("3. CITIC Securities 2020",
     "CITIC Securities annual report: brokerage fee income was 8.6 billion "
     "yuan and investment banking fee income was 2.5 billion yuan in 2020."),
]


def _identity_rerank(query, candidates, top_k=None):
    """测试用精排替身：保持顺序。

    必须注入：默认精排会 `import src.rerankers` → 拉入 transformers，
    在 pytest 主线程的小栈上会触发本机已知的原生崩溃（进程级，不是断言失败）。
    """
    return list(candidates)[:top_k] if top_k else list(candidates)


def _index(pages=None, rerank_fn=_identity_rerank):
    pages = pages or PAGES
    bag = _BagEmbedder().fit([h + " " + b for h, b in PAGES])   # 词表覆盖全部页
    return build_index_from_pdf(_pdf_bytes(pages), "上传测试.pdf",
                                embed_fn=bag, rerank_fn=rerank_fn)


def test_build_index_keeps_source_and_page():
    idx = _index()
    assert idx.n_chunks > 0
    for c in idx.chunks:
        assert c["source"] == "上传测试.pdf"
        assert isinstance(c["page_idx"], int)
        assert c["text"].strip()
    assert {c["page_idx"] for c in idx.chunks} <= {0, 1, 2}


def test_search_ranks_relevant_page_first():
    idx = _index()
    hits = idx.search("NPL ratio China Merchants Bank", top_k=3)
    assert hits, "检索不应为空"
    assert "Merchants" in hits[0]["text"]
    assert hits[0]["page_idx"] == 1


def test_search_hit_carries_score_and_page():
    idx = _index()
    hit = idx.search("CITIC Securities brokerage fee", top_k=1)[0]
    assert hit["page_idx"] == 2
    assert hit["score"] > 0
    assert hit["source"] == "上传测试.pdf"


def test_search_is_hybrid_vector_and_keyword():
    """上传文档也走混合检索（工单07 的教训：单路向量检索答不准数值题）。"""
    idx = _index()
    hits = idx.search("China Merchants Bank NPL ratio", top_k=3,
                      rerank=False)
    assert hits
    assert "vector" in hits[0]["arms"]
    assert "keyword" in hits[0]["arms"]


def test_search_applies_rerank_by_default():
    calls: list[str] = []

    def spy(query, candidates, top_k=None):
        calls.append(query)
        return list(candidates)[:top_k] if top_k else list(candidates)

    bag = _BagEmbedder().fit([h + " " + b for h, b in PAGES])
    idx = build_index_from_pdf(_pdf_bytes(PAGES), "上传测试.pdf",
                               embed_fn=bag, rerank_fn=spy)
    idx.search("CITIC Securities brokerage fee", top_k=2)
    assert calls, "默认必须过精排（与主链路口径一致）"


def test_search_rerank_order_is_respected():
    """精排给出的顺序即最终顺序。"""
    bag = _BagEmbedder().fit([h + " " + b for h, b in PAGES])
    pdf = _pdf_bytes(PAGES)

    def pick_last(query, candidates, top_k=None):
        return [list(candidates)[-1]]

    idx = build_index_from_pdf(pdf, "上传测试.pdf", embed_fn=bag,
                               rerank_fn=pick_last)
    hits = idx.search("NPL ratio", top_k=1)
    assert len(hits) == 1

    idx2 = build_index_from_pdf(pdf, "上传测试.pdf", embed_fn=bag)
    plain = idx2.search("NPL ratio", top_k=3, rerank=False)
    assert hits[0]["chunk_id"] == plain[-1]["chunk_id"], \
        "精排返回的是融合后的最后一条，最终顺序应随之改变"


def test_search_blank_query_returns_empty():
    idx = _index()
    assert idx.search("   ") == []


def test_build_index_rejects_non_pdf():
    with pytest.raises(UploadError):
        build_index_from_pdf(b"this is not a pdf", "bad.txt")


def test_build_index_respects_page_limit():
    idx = _index()
    idx1 = build_index_from_pdf(_pdf_bytes(PAGES), "上传测试.pdf",
                                embed_fn=_BagEmbedder(), limit_pages=1)
    assert idx1.n_chunks < idx.n_chunks
    assert {c["page_idx"] for c in idx1.chunks} == {0}


# ---------------------------------------------------------------------------
# 问答：上传模式下只依据上传文档作答（不碰全局图谱/语料）
# ---------------------------------------------------------------------------
def test_ask_uploaded_answers_from_uploaded_doc_only():
    from src import graph_qa
    idx = _index()
    seen: dict = {}

    def fake_llm(messages, **kw):
        seen["messages"] = messages
        return "招商银行 2019 年末不良贷款率为 1.16%。"

    d = graph_qa.ask_uploaded("招商银行2019年末不良贷款率是多少？", idx,
                              call_llm=fake_llm).to_dict()

    assert d["mode"] == "upload"
    assert d["answer"].startswith("招商银行")
    assert d["graph_facts"] == []                    # 上传模式不使用全局图谱
    assert d["text_hits"], "必须给出上传文档的文本块"
    assert d["text_hits"][0]["source"] == "上传测试.pdf"
    assert d["degraded"] is False
    # 上下文里确实带上了上传文档的原文（LLM 才能据文作答）
    ctx = seen["messages"][-1]["content"]
    assert "Merchants" in ctx or "Ping An" in ctx


def test_ask_uploaded_degrades_when_llm_fails():
    from src import graph_qa
    idx = _index()

    def boom(messages, **kw):
        raise RuntimeError("Ollama 未启动")

    d = graph_qa.ask_uploaded("NPL ratio", idx, call_llm=boom).to_dict()

    assert d["degraded"] is True
    assert "LLM" in d["degraded_reason"]
    assert d["answer"], "降级时也要给出抽取式答案（不能空手而归）"
    assert d["text_hits"], "降级时仍要给出证据文本块"


def test_ask_uploaded_handles_empty_index():
    """空索引：不抛异常、不调用 LLM，明确告知没有可用内容。"""
    from src import graph_qa
    from src.upload_index import UploadIndex

    empty = UploadIndex(chunks=[], vectors=[])
    d = graph_qa.ask_uploaded("任意问题", empty,
                              call_llm=lambda m, **kw: "不该被调用").to_dict()
    assert d["degraded"] is True
    assert "未" in d["answer"] or "上传" in d["answer"]
