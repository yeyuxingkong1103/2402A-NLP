# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_image_extractor.py —— 工单四图像提取模块单测

策略：
  - 用 PyMuPDF + PIL 构造合成 PDF（大图/小图标/重复图），结果可确定性断言；
  - 真实 PDF（../附件/招股说明书2.pdf）存在时追加集成冒烟测试，否则 skip。
"""
import json
import os
from pathlib import Path

import pymupdf
import pytest
from PIL import Image
import io

from src.image_parser.image_extractor import ImageExtractor  # 工单四：被测模块

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
ROOT = Path(__file__).resolve().parents[1]          # 工单四：项目根 工单四/
REAL_PDF = ROOT.parent / "附件" / "招股说明书2.pdf"  # 工单四：真实附件（相对项目根../附件）


# ----------------------------------------------------------------------
def _make_png(width: int, height: int, color: str) -> bytes:
    """工单四：生成纯色 PNG 字节流（合成 PDF 用）"""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_pdf(path: Path, specs: list) -> None:
    """工单四：按 specs=[(page_idx, png_bytes, rect), ...] 生成合成 PDF"""
    doc = pymupdf.open()
    pages = max(p + 1 for p, _, _ in specs)
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    for page_idx, png, rect in specs:
        doc.load_page(page_idx).insert_image(pymupdf.Rect(*rect), stream=png)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def workdir(tmp_path):
    """工单四：临时输出目录 fixture"""
    return tmp_path


# ----------------------------------------------------------------------
def test_extract_big_image_and_filter_small(workdir):
    """工单四：大图保留、<50px 小图标被过滤"""
    pdf = workdir / "合成.pdf"
    _make_pdf(pdf, [
        (0, _make_png(300, 200, "red"), (50, 50, 350, 250)),   # 工单四：大图 300x200
        (0, _make_png(30, 30, "blue"), (400, 400, 430, 430)),  # 工单四：30x30 小图标
    ])
    ex = ImageExtractor(out_dir=str(workdir / "images"))
    result = ex.extract_pdf(str(pdf), doc_id="合成")

    assert result["stats"]["extracted"] == 1
    assert result["stats"]["filtered_small"] == 1
    img = result["images"][0]
    # 工单四：元数据字段完整
    for key in ("doc_id", "page", "image_index", "bbox", "width", "height", "format"):
        assert key in img
    assert img["doc_id"] == "合成"
    assert img["page"] == 1
    assert img["width"] == 300 and img["height"] == 200
    assert img["format"] == "png"
    # 工单四：文件落盘且为 page_{page}_img_{index}.png 命名
    assert Path(img["path"]).exists()
    assert Path(img["path"]).name == "page_001_img_1.png"


def test_dedup_same_image(workdir):
    """工单四：跨页重复图像按 md5 去重（模拟水印/页眉 Logo）"""
    pdf = workdir / "重复.pdf"
    dup_png = _make_png(200, 150, "green")
    _make_pdf(pdf, [
        (0, dup_png, (50, 50, 250, 200)),
        (1, dup_png, (50, 50, 250, 200)),   # 工单四：与第 1 页完全相同
        (2, _make_png(120, 120, "white"), (60, 60, 180, 180)),
    ])
    ex = ImageExtractor(out_dir=str(workdir / "images"))
    result = ex.extract_pdf(str(pdf), doc_id="重复")

    assert result["stats"]["extracted"] == 2
    assert result["stats"]["duplicates"] == 1


def test_json_output_structure(workdir):
    """工单四：清单 JSON 结构与 stats 可追踪"""
    pdf = workdir / "结构.pdf"
    _make_pdf(pdf, [(0, _make_png(100, 100, "black"), (10, 10, 110, 110))])
    out_json = workdir / "out" / "结构_images.json"
    ex = ImageExtractor(out_dir=str(workdir / "images"))
    ex.extract_pdf(str(pdf), doc_id="结构", out_json=str(out_json))

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["doc_id"] == "结构"
    assert data["work_order"] == WORK_ORDER          # 工单四：工单编号写入清单
    assert isinstance(data["images"], list) and len(data["images"]) == 1
    assert {"pages", "extracted", "filtered_small", "duplicates", "failed"} <= set(data["stats"])


def test_missing_pdf_raises(workdir):
    """工单四：PDF 不存在时抛 FileNotFoundError（容错规范）"""
    ex = ImageExtractor(out_dir=str(workdir / "images"))
    with pytest.raises(FileNotFoundError):
        ex.extract_pdf(str(workdir / "不存在.pdf"))


def test_l2_vector_region_render(workdir):
    """工单四：L2——矢量绘制的组织结构图（无位图）应被聚类渲染"""
    import pymupdf as pm
    pdf = workdir / "矢量图.pdf"
    doc = pm.open()
    page = doc.new_page(width=595, height=842)
    # 工单四：页面文字命中触发关键词（模拟"组织结构"章节页；内置中文字体 china-s）
    page.insert_textbox(pm.Rect(60, 60, 540, 90), "发行人组织结构如下图所示：",
                        fontsize=12, fontname="china-s")
    # 工单四：纯矢量绘制图元（矩形网格 + 连线，模拟组织结构图；间距15pt<合并阈值20pt）
    for r in range(4):
        for c in range(3):
            page.draw_rect(pm.Rect(80 + c * 105, 120 + r * 65,
                                   170 + c * 105, 170 + r * 65), width=1)
    for c in range(3):  # 工单四：竖向连线
        page.draw_line(pm.Rect(80 + c * 105, 120, 170 + c * 105, 170).tl,
                       pm.Rect(80 + c * 105, 300, 170 + c * 105, 390).bl)
    doc.save(str(pdf))
    doc.close()

    ex = ImageExtractor(out_dir=str(workdir / "images"))
    result = ex.extract_pdf(str(pdf), doc_id="矢量图")
    assert result["stats"]["l2_renders"] >= 1
    l2 = [m for m in result["images"] if m["extract_source"] == "L2_vector"]
    assert l2, "应存在 L2_vector 元数据"
    assert Path(l2[0]["path"]).exists()          # 工单四：渲染 PNG 落盘
    assert l2[0]["width"] >= 350                 # 工单四：dpi=200 放大后尺寸合理


def test_l2_disabled(workdir):
    """工单四：enable_l2=False 时不产生矢量渲染（回归保护）"""
    import pymupdf as pm
    pdf = workdir / "矢量图2.pdf"
    doc = pm.open()
    page = doc.new_page()
    page.insert_textbox(pm.Rect(60, 60, 540, 90), "组织结构如下图所示",
                        fontsize=12, fontname="china-s")
    for r in range(4):
        page.draw_rect(pm.Rect(80, 120 + r * 90, 480, 170 + r * 90), width=1)
    doc.save(str(pdf))
    doc.close()

    ex = ImageExtractor(out_dir=str(workdir / "images"), enable_l2=False)
    result = ex.extract_pdf(str(pdf), doc_id="矢量图2")
    assert result["stats"]["l2_renders"] == 0


@pytest.mark.skipif(not REAL_PDF.exists(), reason="真实附件不存在，跳过集成冒烟")
def test_real_pdf_smoke(workdir):
    """工单四：真实 PDF（招股说明书2）提取冒烟——数量>0、清单落盘"""
    ex = ImageExtractor(out_dir=str(workdir / "images"))
    result = ex.extract_pdf(str(REAL_PDF), doc_id=REAL_PDF.stem,
                            out_json=str(workdir / "招股说明书2_images.json"))
    assert result["stats"]["extracted"] > 0
    assert Path(workdir / "招股说明书2_images.json").exists()
    # 工单四：每张保留图均有 path 且文件存在
    for img in result["images"][:5]:
        assert Path(img["path"]).exists()
