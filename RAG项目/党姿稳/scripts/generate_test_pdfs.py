"""
generate_test_pdfs.py — 生成三个领域的测试 PDF

每个领域生成一份 35 页的 PDF，内容含正文、表格和图形（柱状图 / 流程图），
用于验证 PDF 解析（文本 / 表格 / 图片三条链路）与分块效果。

    python scripts/generate_test_pdfs.py
    → data/generated_pdfs/劳动法常见问题.pdf
      data/generated_pdfs/常见疾病诊疗指南.pdf
      data/generated_pdfs/英语语法精讲.pdf

正文素材见 scripts/pdf_content.py；中文字体使用 reportlab 内置的
STSong-Light，无需额外字体文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import config
from pdf_content import ENGLISH, LEGAL, MEDICAL
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

TITLE_STYLE = ParagraphStyle("title", fontName=FONT, fontSize=20, leading=28, spaceAfter=6,
                             textColor=colors.HexColor("#1a3d6d"))
HEAD_STYLE = ParagraphStyle("head", fontName=FONT, fontSize=14, leading=22, spaceBefore=10, spaceAfter=8,
                            textColor=colors.HexColor("#1a3d6d"))
BODY_STYLE = ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=17, spaceAfter=8, firstLineIndent=21)
CAPTION_STYLE = ParagraphStyle("cap", fontName=FONT, fontSize=9, leading=14,
                               textColor=colors.HexColor("#666666"))


def _bar_chart(width: float = 150 * mm, height: float = 55 * mm) -> Drawing:
    """画一个示意柱状图，用于测试 PDF 中的图形解析。"""
    drawing = Drawing(width, height)
    bars = [(48, "#4a90d9"), (72, "#5aa469"), (96, "#e0a458"), (60, "#c0504d")]
    baseline = 12 * mm

    drawing.add(Line(10 * mm, baseline, width - 10 * mm, baseline, strokeColor=colors.HexColor("#999999")))
    for index, (value, color) in enumerate(bars):
        x = 20 * mm + index * 28 * mm
        bar_height = value * 0.35 * mm
        drawing.add(Rect(x, baseline, 16 * mm, bar_height, fillColor=colors.HexColor(color), strokeColor=None))
        drawing.add(String(x + 4 * mm, baseline + bar_height + 2 * mm, str(value), fontName=FONT, fontSize=8))
    drawing.add(String(10 * mm, height - 8 * mm, "示意图（数值仅用于排版测试）", fontName=FONT, fontSize=8,
                       fillColor=colors.HexColor("#666666")))
    return drawing


def _flow_chart(width: float = 150 * mm, height: float = 45 * mm) -> Drawing:
    """画一个示意流程图，用于测试 PDF 中的图形解析。"""
    drawing = Drawing(width, height)
    steps = ["发生事故", "收集材料", "提交申请", "审核认定"]
    box_width, box_height = 28 * mm, 14 * mm
    y = 14 * mm

    for index, step in enumerate(steps):
        x = 6 * mm + index * 35 * mm
        drawing.add(Rect(x, y, box_width, box_height, fillColor=colors.HexColor("#eaf1fa"),
                         strokeColor=colors.HexColor("#1a3d6d"), strokeWidth=0.8))
        drawing.add(String(x + 4 * mm, y + 5 * mm, step, fontName=FONT, fontSize=9))
        if index < len(steps) - 1:
            arrow_x = x + box_width + 1.5 * mm
            drawing.add(Line(arrow_x, y + box_height / 2, arrow_x + 4 * mm, y + box_height / 2,
                             strokeColor=colors.HexColor("#1a3d6d")))
            drawing.add(Polygon([arrow_x + 4 * mm, y + box_height / 2,
                                 arrow_x + 2 * mm, y + box_height / 2 + 1.5,
                                 arrow_x + 2 * mm, y + box_height / 2 - 1.5],
                                fillColor=colors.HexColor("#1a3d6d"), strokeColor=None))
    return drawing


def _make_table(rows: list[list[str]], width: float = 150 * mm) -> Table:
    """构造带表头样式的表格。"""
    body = [[Paragraph(f"<b>{cell}</b>", CAPTION_STYLE) for cell in rows[0]]]
    body += [[Paragraph(str(cell), CAPTION_STYLE) for cell in row] for row in rows[1:]]

    table = Table(body, colWidths=[width / len(rows[0])] * len(rows[0]), hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1fa")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def _decorate(canvas, doc, title: str) -> None:
    """页眉与页脚。"""
    canvas.saveState()
    canvas.setFont(FONT, 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 285 * mm, title)
    canvas.drawRightString(190 * mm, 285 * mm, f"第 {doc.page} 页")
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.line(20 * mm, 282 * mm, 190 * mm, 282 * mm)
    canvas.drawString(20 * mm, 12 * mm, "本文档由脚本自动生成，仅用于 RAG 系统测试")
    canvas.restoreState()


def build_pdf(spec: dict, output: Path) -> Path:
    """按内容描述渲染一份 PDF。"""
    from functools import partial

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=30 * mm, bottomMargin=20 * mm,
        title=spec["title"], author="RAG 测试数据生成器",
    )

    story = [Paragraph(spec["title"], TITLE_STYLE), Paragraph(spec["subtitle"], CAPTION_STYLE), Spacer(1, 12)]

    for index, (heading, body, table_rows, chart) in enumerate(spec["sections"]):
        story.append(Paragraph(heading, HEAD_STYLE))
        for paragraph in body.split("|"):
            story.append(Paragraph(paragraph, BODY_STYLE))

        if table_rows:
            story.append(Spacer(1, 4))
            story.append(_make_table(table_rows))
        if chart == "bar":
            story.append(Spacer(1, 8))
            story.append(_bar_chart())
        elif chart == "flow":
            story.append(Spacer(1, 8))
            story.append(_flow_chart())

        if index < len(spec["sections"]) - 1:
            story.append(PageBreak())

    doc.build(story, onFirstPage=partial(_decorate, title=spec["title"]),
              onLaterPages=partial(_decorate, title=spec["title"]))
    return output


def main() -> None:
    config.ensure_dirs()
    targets = [
        (LEGAL, config.GENERATED_PDF_DIR / "劳动法常见问题.pdf"),
        (MEDICAL, config.GENERATED_PDF_DIR / "常见疾病诊疗指南.pdf"),
        (ENGLISH, config.GENERATED_PDF_DIR / "英语语法精讲.pdf"),
    ]

    for spec, output in targets:
        path = build_pdf(spec, output)
        size_kb = path.stat().st_size / 1024
        print(f"已生成 {path.name:24s} {len(spec['sections'])} 页  {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
