# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 注册内置中文字体，不用依赖系统字体
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT_NAME = 'STSong-Light'

# 文档内容
pdf_content = """长江新能源股份有限公司
首次公开发行股票招股说明书（申报稿）

一、发行人基本情况
公司名称：长江新能源股份有限公司
成立时间：2018年03月12日
注册地址：湖北省武汉市东湖新技术开发区
法定代表人：张明
注册资本：30000万元人民币

二、主营业务
公司主营业务为光伏电池片、光伏组件的研发、生产与销售，同时提供光伏电站工程设计与运维服务。
公司产品主要分为两类：
1、N型TOPCon高效光伏电池片，转换效率最高可达26.2%；
2、大功率双面光伏组件，单块组件功率580W~720W。
下游客户主要为国内大型新能源电站运营商、海外光伏系统集成商。
公司收入来源：光伏组件销售占营业收入78%，电池片外销占17%，电站运维服务占5%。

三、经营模式
采购模式：向上游采购高纯硅料、光伏玻璃、铝边框、银浆等原材料。
生产模式：自主建设智能制造工厂，位于湖北宜昌，产能20GW/年。
销售模式：境内直销+海外经销，海外市场主要覆盖东南亚、中东、拉美。

四、财务数据（最近三年）
2022年：营业收入 42.6亿元，归母净利润 3.12亿元
2023年：营业收入 57.8亿元，归母净利润 4.75亿元
2024年：营业收入 71.3亿元，归母净利润 6.20亿元

五、核心技术
1、N型TOPCon电池钝化工艺；
2、组件无损切割技术；
3、光伏智能运维云平台算法。
公司拥有发明专利42项，实用新型专利116项。

六、股权结构
控股股东：长江控股集团有限公司，持股45.20%
实际控制人：李建国，通过长江控股集团控制本公司。
员工持股平台：新能汇智合伙企业，持股6.50%。

七、风险因素
1、行业周期性波动风险，光伏产品价格存在下行压力；
2、原材料价格波动风险，硅料价格变化影响毛利率；
3、海外贸易政策变化，存在关税、贸易壁垒风险。

八、募集资金用途
本次IPO拟募集资金总额28亿元：
1、高效N型光伏组件扩产项目：18亿元
2、研发中心升级项目：5亿元
3、补充流动资金：5亿元
"""

# 生成PDF
doc = SimpleDocTemplate("招股说明书.pdf", pagesize=A4)
styles = getSampleStyleSheet()
# 自定义中文样式
title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontName=FONT_NAME, fontSize=18, alignment=1)
h1_style = ParagraphStyle('CustomH1', parent=styles['Heading2'], fontName=FONT_NAME, fontSize=14)
normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=FONT_NAME, fontSize=11, leading=16)

story = []
lines = pdf_content.split("\n")
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("长江新能源"):
        story.append(Paragraph(line, title_style))
        story.append(Spacer(1, 20))
    elif line.startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、")):
        story.append(Paragraph(line, h1_style))
        story.append(Spacer(1,8))
    else:
        story.append(Paragraph(line, normal_style))
        story.append(Spacer(1,4))

doc.build(story)
print("✅ 招股说明书.pdf 生成成功！放在当前目录")
