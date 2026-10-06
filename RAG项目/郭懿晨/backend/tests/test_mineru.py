from pathlib import Path

from reportlab.pdfgen import canvas

import backend.app.mineru as mineru_mod
from backend.app.mineru import MinerUParser


def make_pdf(path: Path, pages: list[str]) -> None:
    pdf = canvas.Canvas(str(path))
    for index, text in enumerate(pages):
        pdf.setFont("Helvetica", 12)
        pdf.drawString(72, 720, text)
        if index < len(pages) - 1:
            pdf.showPage()
    pdf.save()


def test_parse_pdf_falls_back_to_pypdf_when_mineru_raises(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["page one content", "page two content"])

    def boom(*args, **kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(mineru_mod, "doc_analyze_streaming", boom)

    blocks = MinerUParser().parse_pdf(pdf_path)

    assert [block.page for block in blocks] == [1, 2]
    assert [block.label for block in blocks] == ["正文", "正文"]
    assert "page one content" in blocks[0].text
    assert "page two content" in blocks[1].text
    assert blocks[0].source_span == "page=1:fallback=1"


def test_parse_pdf_falls_back_when_mineru_returns_empty(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["page one content", "page two content"])

    def emit_empty(*args, **kwargs):
        on_doc_ready = args[3]
        on_doc_ready(0, [], {"pdf_info": []}, False)

    monkeypatch.setattr(mineru_mod, "doc_analyze_streaming", emit_empty)

    blocks = MinerUParser().parse_pdf(pdf_path)

    assert [block.page for block in blocks] == [1, 2]
    assert [block.label for block in blocks] == ["正文", "正文"]
    assert "page one content" in blocks[0].text
    assert "page two content" in blocks[1].text
