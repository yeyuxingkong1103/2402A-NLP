# -*- coding: utf-8 -*-
"""文档解析分流的测试。

只测**不需要 GPU / 不需要下载**的部分：
    - PDF 类型判定（probe_pdf）
    - markdown 清洗（_strip_markdown）
    - 引擎可用性与降级开关的默认值

MinerU 的实际解析要跑 GPU（单次约 95 秒），不放进默认测试套件 ——
否则整套测试会因显卡占用时红时绿。它由 `docs/09-OCR分流说明.md` 记录的手工验证覆盖。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import ocr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MED_PDF = REPO / "datasets" / "medical" / "国家基层高血压防治管理指南2025版.pdf"


# ---------------------------------------------------------------- 类型判定
def test_probe_detects_text_layer_pdf():
    """带文本层的 PDF 必须判为 text，走轻量路径（不要白白上 GPU OCR）。"""
    if not MED_PDF.exists():
        pytest.skip("测试语料不存在")
    info = ocr.probe_pdf(MED_PDF)
    assert info["kind"] == "text"
    assert info["avg_chars_per_page"] > ocr.TEXT_THRESHOLD
    assert info["total_pages"] > 0


def test_probe_detects_scanned_pdf(tmp_path):
    """无文本层的 PDF 必须判为 scanned。"""
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    for _ in range(2):
        pdf.new_page(width=200, height=200)      # 空白页 = 0 字符
    p = tmp_path / "blank.pdf"
    pdf.save(str(p))
    pdf.close()

    info = ocr.probe_pdf(p)
    assert info["kind"] == "scanned"
    assert info["avg_chars_per_page"] == 0.0


def test_probe_only_samples_not_whole_doc():
    """抽样页数不应超过设定值 —— 几百页的 PDF 全读一遍只为算均值不划算。"""
    if not MED_PDF.exists():
        pytest.skip("测试语料不存在")
    info = ocr.probe_pdf(MED_PDF, sample_pages=3)
    assert info["sampled"] == 3
    assert info["total_pages"] > 3         # 而总页数照常报告


# ---------------------------------------------------------------- markdown 清洗
def test_strip_markdown_removes_images():
    """图片链接对纯文本 RAG 无价值，留着只是噪声 token。"""
    out = ocr._strip_markdown("正文一\n![](images/a.jpg)\n正文二")
    assert "images" not in out
    assert "正文一" in out and "正文二" in out


def test_strip_markdown_keeps_link_text():
    out = ocr._strip_markdown("见[国家指南](http://x.com)说明")
    assert "国家指南" in out
    assert "http" not in out


def test_strip_markdown_removes_heading_marks():
    out = ocr._strip_markdown("# 一级标题\n## 二级标题\n- 列表项\n1. 有序项")
    assert "一级标题" in out and "#" not in out
    assert "列表项" in out and "-" not in out.split("列表项")[0][-3:]


def test_strip_markdown_keeps_table_content_but_drops_separator():
    md = "| 药名 | 剂量 |\n|---|---|\n| 氨氯地平 | 5mg |"
    out = ocr._strip_markdown(md)
    assert "氨氯地平" in out and "5mg" in out
    assert "---" not in out


# ---------------------------------------------------------------- 引擎与降级
def test_paddleocr_disabled_by_default():
    """PaddleOCR 模型本机不存在，默认必须关闭以免触发下载。"""
    if ocr.ALLOW_DOWNLOAD:
        pytest.skip("本次运行显式开启了下载权限")
    assert ocr.available()["paddleocr"]["ok"] is False


def test_parse_pdf_off_forces_text_layer(tmp_path):
    """ocr='off' 时即使判定为扫描件也不走 OCR（用于确认分流开关真的生效）。"""
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    pdf.new_page(width=200, height=200)
    p = tmp_path / "blank.pdf"
    pdf.save(str(p))
    pdf.close()

    pages, meta = ocr.parse_pdf(p, ocr="off")
    assert meta["used_ocr"] is False
    assert meta["engine"] is None


def test_parse_pdf_routes_scanned_to_ocr(tmp_path, monkeypatch):
    """扫描件应被路由到 OCR，并记录判定依据（页均字数）。

    ⚠️ 用 monkeypatch 替掉真实 OCR 引擎：本测试要验证的是**路由决策**，
    不是 MinerU 的解析能力。真跑 MinerU 每个用例要 95 秒（且空白 PDF 会让它失败），
    放进默认套件会让测试又慢又脆。MinerU 的实际解析由
    `docs/09-OCR分流说明.md` §4.2 记录的手工验证覆盖。
    """
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    pdf.new_page(width=200, height=200)
    p = tmp_path / "blank.pdf"
    pdf.save(str(p))
    pdf.close()

    called = {}

    def fake_mineru(path, timeout=900):
        called["path"] = str(path)
        return "模拟 OCR 结果正文"

    monkeypatch.setattr(ocr, "extract_with_mineru", fake_mineru)

    pages, meta = ocr.parse_pdf(p, ocr="auto")
    assert meta["pdf_kind"] == "scanned"
    assert meta["used_ocr"] is True
    assert meta["avg_chars_per_page"] == 0.0
    assert meta["engine"] == "mineru"
    assert "path" in called, "判定为扫描件却没有调用 OCR 引擎"
    assert pages and pages[0][1] == "模拟 OCR 结果正文"
