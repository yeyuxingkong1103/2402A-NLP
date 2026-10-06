import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT = Path("output/pdf/multimodal_medical_test.pdf")
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def create_ocr_image(path: Path) -> None:
    image = Image.new("RGB", (1400, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), 48)
    bold = ImageFont.truetype(str(FONT_PATH), 62)
    draw.rectangle((20, 20, 1380, 340), outline="#0F766E", width=6)
    draw.text((65, 65), "图片 OCR 健康提示", font=bold, fill="#0F766E")
    draw.text((65, 165), "每日记录血压读数，异常时联系医生。", font=font, fill="#1F2937")
    draw.text((65, 245), "测试关键词：OCR图片血压记录", font=font, fill="#1F2937")
    image.save(path)


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName="STSong-Light", fontSize=20)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="STSong-Light", fontSize=11, leading=18)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm)
    with tempfile.TemporaryDirectory(prefix="rag_pdf_") as folder:
        image_path = Path(folder) / "ocr_health_tip.png"
        create_ocr_image(image_path)
        table = Table(
            [["项目", "建议"], ["测量前", "安静休息 5 分钟"], ["测量后", "记录收缩压和舒张压"]],
            colWidths=[45 * mm, 115 * mm],
        )
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        document.build([
            Paragraph("图文表格医疗知识检索测试", title),
            Spacer(1, 8),
            Paragraph("这是原生文本测试：血压由收缩压和舒张压组成，规范测量有助于健康管理。", body),
            Spacer(1, 10), table, Spacer(1, 14),
            Paragraph("下面图片中的文字只可通过 OCR 读取：", body), Spacer(1, 8),
            PdfImage(str(image_path), width=160 * mm, height=41 * mm),
        ])


if __name__ == "__main__":
    build()
