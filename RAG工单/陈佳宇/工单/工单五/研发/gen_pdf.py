# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
font_name = 'STSong-Light'

c = canvas.Canvas("招股说明书.pdf", pagesize=A4)
width, height = A4

text_lines = [
    "武汉兴图新科电子股份有限公司 招股说明书",
    "",
    "一、主营业务",
    "公司主要从事军用音视频指挥系统的研发、生产与销售。",
    "报告期内，军用领域收入分别为1.23亿元、1.56亿元、1.89亿元。",
    "公司参与某大型国防工程，荣获国家科技进步一等奖。",
    "法定代表人：程家荣。",
    "",
    "二、武汉力源信息技术股份有限公司",
    "主营半导体分销与芯片设计。",
    "组织结构：总部下设销售部，销售部包含：华北销售处、华东销售处、华南销售处、西南销售处。销售部一共4个销售处。",
    "法定代表人：赵依农。",
]

y_pos = height - 50
for line in text_lines:
    c.setFont(font_name,14)
    c.drawString(50, y_pos, line)
    y_pos -= 22
    if y_pos < 50:
        c.showPage()
        y_pos = height - 50

c.save()
print("✅ 招股说明书.pdf 生成成功")
