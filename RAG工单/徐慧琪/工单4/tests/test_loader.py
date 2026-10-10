# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from pathlib import Path
import pytest

from rag04.config import get_settings, PROJECT_ROOT
from rag04.ingest.loader import doc_id_of, open_pdf, iter_text_blocks, PdfOpenError

GY2 = PROJECT_ROOT / "招股说明书2.pdf"
needs_corpus = pytest.mark.skipif(not GY2.exists(), reason="语料缺失")


def test_doc_id_strips_extension_and_is_stable():
    assert doc_id_of(Path("招股说明书2.pdf")) == "招股说明书2"
    assert doc_id_of(Path("/a/b/招股说明书1.pdf")) == "招股说明书1"


def test_open_pdf_raises_on_missing_file(tmp_path):
    with pytest.raises(PdfOpenError):
        open_pdf(tmp_path / "nope.pdf")


def test_open_pdf_raises_on_non_pdf(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(PdfOpenError):
        open_pdf(bad)


@needs_corpus
def test_iter_text_blocks_yields_page_and_bbox():
    doc = open_pdf(GY2)
    blocks = []
    for b in iter_text_blocks(doc, "招股说明书2"):
        blocks.append(b)
        if len(blocks) >= 30:
            break
    assert blocks
    for b in blocks:
        assert b.doc_id == "招股说明书2"
        assert 1 <= b.page <= doc.page_count
        assert len(b.bbox) == 4
        assert b.bbox[2] > b.bbox[0] and b.bbox[3] > b.bbox[1]
        assert b.text.strip()
        assert b.block_type == "text"


@needs_corpus
def test_rotated_span_detected_on_org_chart_page():
    """p38 组织结构图的文字是 90° 竖排，必须被标记为旋转文本。"""
    doc = open_pdf(GY2)
    flagged = [b for b in iter_text_blocks(doc, "招股说明书2")
               if b.page == 38 and b.has_rotated_text]
    assert flagged, "p38 应检出旋转文本（组织结构图竖排标签）"


@needs_corpus
def test_loader_survives_bad_page(monkeypatch):
    """坏页不应中断整体解析（硬性要求 7 容错）。

    注意：必须打在模块级 `_page_blocks` 上，不能在 doc 实例上挂 `__getitem__`
    —— Python 的特殊方法在**类型**上查找，实例属性对 `doc[i]` 不生效，
    那样打桩会静默失效、测试变成空测试。
    """
    import rag04.ingest.loader as m

    real = m._page_blocks

    def boom(page, doc_id, pno):
        if pno == 6:
            raise RuntimeError("模拟损坏页")
        return real(page, doc_id, pno)

    monkeypatch.setattr(m, "_page_blocks", boom)

    doc = open_pdf(GY2)
    got = list(iter_text_blocks(doc, "招股说明书2"))

    assert got, "坏页之外仍应有产出"
    assert all(b.page != 6 for b in got), "损坏页应被跳过，不应产出内容"
