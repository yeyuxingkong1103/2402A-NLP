from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT = Path("output/pdf/medical_knowledge_sample.pdf")
SOURCE = "https://www.nhc.gov.cn/cms-search/downFiles/63f752a17cfd4b4781f744477561866f.pdf"


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ChineseTitle", parent=styles["Title"], fontName="STSong-Light", fontSize=22,
        leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#155E75"), spaceAfter=16,
    )
    heading = ParagraphStyle(
        "ChineseHeading", parent=styles["Heading2"], fontName="STSong-Light", fontSize=14,
        leading=21, textColor=colors.HexColor("#0F766E"), spaceBefore=10, spaceAfter=6,
    )
    body = ParagraphStyle(
        "ChineseBody", parent=styles["BodyText"], fontName="STSong-Light", fontSize=10.5,
        leading=18, textColor=colors.HexColor("#24313A"), spaceAfter=7,
    )
    note = ParagraphStyle(
        "ChineseNote", parent=body, backColor=colors.HexColor("#ECFEFF"),
        borderColor=colors.HexColor("#67E8F9"), borderWidth=0.7, borderPadding=9,
    )
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=22 * mm, leftMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
    )
    story = [
        Paragraph("高血压健康知识示例", title),
        Paragraph("RAG 项目测试文档 | 依据公开指南整理", body),
        Paragraph(
            "重要声明：本文只用于软件检索与健康教育测试，不能替代医生诊断、治疗或个体化用药建议。"
            "若出现胸痛、呼吸困难、意识不清等急症信号，请立即拨打 120。", note,
        ),
        Spacer(1, 8),
        Paragraph("1. 什么是血压", heading),
        Paragraph(
            "血压通常由收缩压和舒张压表示。一次测量偏高不能自行确诊高血压，应在规范条件下重复测量，"
            "并由医疗专业人员结合病史和其他检查进行判断。家庭血压监测可以帮助了解日常变化。", body,
        ),
        Paragraph("2. 日常饮食原则", heading),
        Paragraph(
            "高血压人群的饮食管理强调食物多样、控制能量、少盐并减少高钠加工食品。可优先选择新鲜蔬菜、"
            "水果、全谷物、豆类、低脂奶类以及适量鱼禽。调味时可用天然香辛料减少对盐和高钠酱料的依赖。", body,
        ),
        Paragraph("3. 生活方式", heading),
        Paragraph(
            "规律作息、戒烟限酒、在身体条件允许时进行持续且适量的运动，并维持适宜体重，都有助于心血管健康。"
            "运动种类和强度应结合年龄、基础疾病和医生建议，不宜在明显不适时勉强运动。", body,
        ),
        Paragraph("4. 家庭测量提示", heading),
    ]
    rows = [
        ["步骤", "建议"],
        ["测量前", "安静休息，避免刚运动、吸烟或饮用刺激性饮料后立即测量。"],
        ["测量时", "坐姿稳定，手臂获得支撑，袖带尺寸合适并与心脏大致同高。"],
        ["记录", "记录日期、时间和读数，不凭单次结果自行增减药物。"],
    ]
    table = Table(rows, colWidths=[32 * mm, 116 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEADING", (0, 0), (-1, -1), 15),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        table, Paragraph("5. 就医与用药安全", heading),
        Paragraph(
            "降压药应按医生处方使用。即使家庭读数改善，也不应自行停药、换药或改变剂量。"
            "若读数持续异常，或出现头晕、乏力等不适，应记录情况并及时联系医生。", body,
        ),
        Paragraph("6. 信息来源", heading),
        Paragraph(
            "本示例依据国家卫生健康委员会《成人高血压食养指南（2023年版）》的公开健康教育方向进行简要整理。"
            f"原始来源：{SOURCE}", body,
        ),
        Paragraph(
            "知识更新日期：2026-09-16。上传到本项目时，请把上述来源 URL 同时填写到知识库来源字段。", note,
        ),
    ])
    document.build(story)


if __name__ == "__main__":
    build()
