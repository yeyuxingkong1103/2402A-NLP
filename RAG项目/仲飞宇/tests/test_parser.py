"""OCR 接线测试：图片 / 扫描件 PDF 走 OCR，开关关闭时退回。

真实 OCR 引擎不参与这些断言（模型加载慢、依赖字体），统一 mock parser 里的
``ocr_image``。另附一个真实 OCR 冒烟，找得到 CJK 字体才跑，否则 skip。
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

import app.core.document.parser as parser_mod
from app.core.document.parser import DocumentParser, _decode_text


@pytest.fixture(autouse=True)
def _ocr_on(monkeypatch):
    """本模块默认把 OCR 打开。

    这些用例考的是「图片/扫描件有没有被正确接到 OCR 上」，不该受环境里 `OCR_ENABLED`
    的影响——而 `.env` 是 true、`.env.test` 是 false，于是同一个测试文件
    「不带 APP_ENV 跑全过、按文档 `APP_ENV=test ... pytest` 跑就 4 个失败」。
    关掉 OCR 的那条路径由 test_image_ocr_disabled_returns_empty 自己显式 monkeypatch 覆盖，
    它在本 fixture 之后执行，所以能正常生效。
    """
    monkeypatch.setattr(parser_mod.settings, "ocr_enabled", True)


def _png_bytes() -> bytes:
    """用 PIL 画一张白底小图（内容随便，OCR 已被 mock 掉）。"""
    from PIL import Image

    img = Image.new("RGB", (200, 60), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_parse_file_image_routes_to_ocr(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(parser_mod, "ocr_image", lambda img: calls.append(img) or "识别出的文字")
    p = tmp_path / "shot.png"
    p.write_bytes(_png_bytes())

    doc = DocumentParser().parse_file(p)

    assert len(calls) == 1
    assert doc["ext"] == ".png"
    assert doc["text"] == "识别出的文字"


def test_parse_bytes_image_routes_to_ocr(monkeypatch):
    monkeypatch.setattr(parser_mod, "ocr_image", lambda img: "识别出的文字")
    doc = DocumentParser().parse_bytes("photo.jpg", _png_bytes())
    assert doc["text"] == "识别出的文字"
    assert doc["source"] == "upload://photo.jpg"


def test_image_ocr_disabled_returns_empty(tmp_path, monkeypatch):
    # 关掉 OCR 要连引擎都不碰：只「不采用结果」的话，照样付一遍模型加载的代价。
    monkeypatch.setattr(parser_mod.settings, "ocr_enabled", False)
    called = []
    monkeypatch.setattr(parser_mod, "ocr_image", lambda img: called.append(img) or "不该调")
    p = tmp_path / "shot.png"
    p.write_bytes(_png_bytes())

    doc = DocumentParser().parse_file(p)

    assert doc["text"] == ""
    assert called == []


def test_scanned_pdf_pages_are_ocred(tmp_path, monkeypatch):
    """无文本层的 PDF，每页都渲染成图走 OCR。"""
    import fitz

    pdf_path = tmp_path / "scan.pdf"
    d = fitz.open()
    d.new_page()  # 空白页，无文本层
    d.save(pdf_path)
    d.close()

    seen = []
    monkeypatch.setattr(parser_mod, "ocr_image", lambda img: seen.append(img) or "扫描页文本" * 20)

    result = DocumentParser().parse_file(pdf_path)

    assert len(seen) == 1  # 一页一图
    assert "扫描页文本" in result["text"]


def test_text_pdf_not_ocred(tmp_path, monkeypatch):
    """有文本层的 PDF 不触发 OCR。"""
    import fitz

    pdf_path = tmp_path / "text.pdf"
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "This page has a real text layer with enough content")
    d.save(pdf_path)
    d.close()

    called = []
    monkeypatch.setattr(parser_mod, "ocr_image", lambda img: called.append(img) or "不该调")

    result = DocumentParser().parse_file(pdf_path)

    assert called == []
    assert "text layer" in result["text"]


def _find_cjk_font() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["fc-list", ":lang=zh", "file"], capture_output=True, text=True, timeout=5
        ).stdout
        first = out.splitlines()[0].split(":")[0].strip() if out.strip() else ""
        if first:
            return first
    except Exception:  # noqa: BLE001
        pass
    for cand in ("/mnt/c/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttc"):
        if Path(cand).exists():
            return cand
    return ""


def test_real_ocr_smoke(tmp_path):
    """真实 OCR 端到端冒烟：渲染中文图 → parse_file → 识别出关键汉字。"""
    pytest.importorskip("rapidocr_onnxruntime")
    font = _find_cjk_font()
    if not font:
        pytest.skip("找不到 CJK 字体，跳过真实 OCR 冒烟")

    from PIL import Image, ImageDraw, ImageFont

    p = tmp_path / "zh.png"
    img = Image.new("RGB", (700, 80), "white")
    ImageDraw.Draw(img).text((10, 10), "高血压患者每日食盐 5 克", font=ImageFont.truetype(font, 40), fill="black")
    img.save(p)

    doc = DocumentParser().parse_file(p)
    assert "高血压" in doc["text"]


# ---- 文本编码 ----
def test_decode_text_handles_gbk_and_utf16():
    """非 UTF-8 的中文文本不能被静默读成乱码。

    以前是 `decode("utf-8", errors="ignore")`：GBK/GB18030（中文 Windows 记事本默认）
    或 UTF-16 的 txt 会静默变成乱码——实测 108 字正文只剩 44 字垃圾，照样入库进 prompt，
    全程无报错，用户只会觉得"检索答非所问"。
    """
    zh = "高血压患者每日食盐应控制在 5 克以内。"

    assert _decode_text(zh.encode("utf-8"), "a.txt") == zh
    assert _decode_text(zh.encode("gbk"), "b.txt") == zh
    assert _decode_text(zh.encode("utf-16"), "c.txt") == zh  # 带 BOM


def test_parse_file_reads_gbk_txt(tmp_path):
    """走完整解析入口也要能读对 GBK（不只单测解码函数）。"""
    p = tmp_path / "gbk.txt"
    p.write_bytes("# 饮食建议\n\n高血压患者每日食盐应控制在 5 克以内。".encode("gbk"))

    doc = DocumentParser().parse_file(p)

    assert "高血压患者" in doc["text"]
    assert "食盐" in doc["text"]


# ---- 单页 OCR 失败不能连累整篇 ----
def test_ocr_failure_on_one_page_keeps_other_pages(monkeypatch):
    """某页 OCR 抛异常时，其它页已抽出的正文必须保住。

    异常穿透页循环后会被上层的 `except: text = ""` 吞掉，整篇正文一起丢，
    上传只报「无有效文本（扫描件 OCR 未识别成功）」，把真因（OCR 引擎本身报错）掩盖。
    """
    class _Page:
        def __init__(self, number, text):
            self.number = number
            self._text = text

        def get_text(self):
            return self._text

    pages = [_Page(1, "第一页有文本层，内容足够长以通过页级阈值。"), _Page(2, "短"), _Page(3, "短")]

    class _Doc:
        def __iter__(self):
            return iter(pages)

    def fake_ocr(self, page):
        if page.number == 2:
            raise RuntimeError("ONNX 引擎加载失败")
        return "第三页 OCR 出来的正文"

    monkeypatch.setattr(DocumentParser, "_ocr_pdf_page", fake_ocr)

    out = DocumentParser()._parse_pdf_doc(_Doc())

    assert "第一页有文本层" in out
    assert "第三页 OCR 出来的正文" in out, "第二页失败不该影响第三页"
