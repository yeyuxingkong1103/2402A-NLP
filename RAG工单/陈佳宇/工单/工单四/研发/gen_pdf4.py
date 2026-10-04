# -*- coding:utf-8 -*-
"""
工单编号：人工智能NLP‑RAG‑图像内容解析及检索优化
gen_pdf4.py 生成工单4测试PDF，PDF内部包含2张绘图图像(组织结构图、IC市场增长图)
注意：这两张是PDF内置图形对象，不是可复制文本，用于多模态CLIP图像检索测试
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 注册内置中文字体
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT_CN = 'STSong-Light'

pdf_name = "招股说明书2.pdf"
c = canvas.Canvas(pdf_name, pagesize=A4)
width, height = A4

# ========== 第1页 文字 + 绘制【组织结构图】(PDF图像对象) ==========
c.setFont(FONT_CN,14)
c.drawString(80, height-60, "武汉力源信息技术股份有限公司招股说明书")
c.setFont(FONT_CN,11)
c.drawString(80, height-90, "章节：组织架构")

# 绘制组织结构框图（矢量图形，属于PDF图片对象，会被pymupdf识别为图片）
# 顶层：销售部
c.setFillColor(colors.lightblue)
c.rect(220, height-180, 160,40, fill=1)
c.setFillColor(colors.black)
c.setFont(FONT_CN,12)
c.drawString(260, height-155, "销售部")

# 4个下属部门
c.setStrokeColor(colors.black)
c.line(300, height-180,300, height-210)
c.line(120, height-210,480, height-210)

# 渠道销售部
c.rect(100, height-230,120,35)
c.drawString(120, height-208, "渠道销售部")
c.line(160, height-230,160, height-245)

# 电话及网络销售部
c.rect(240, height-230,120,35)
c.drawString(250, height-208, "电话及网络销售部")
c.line(300, height-230,300, height-245)

# 大客户销售部
c.rect(380, height-230,140,35)
c.drawString(390, height-208, "大客户销售部")
c.line(450, height-230,450, height-245)

# 国际贸易部
c.rect(540, height-230,120,35)
c.drawString(550, height-208, "国际贸易部")

# 大客户销售部下6个销售处
c.line(450, height-245,450, height-280)
c.line(320, height-280,580, height-280)
pos_y = height-300
for name in ["北京销售处","深圳销售处","广州销售处","成都销售处","珠海销售处","武汉销售处"]:
    c.rect(pos_y-80, pos_y-20,130,30)
    c.drawString(pos_y-70, pos_y, name)
    pos_y -=45

c.showPage()

# ==========第2页，绘制【2008 IC市场应用结构与增长柱状图】PDF图像对象 ==========
c.setFont(FONT_CN,14)
c.drawString(80, height-60, "2008年中国IC市场应用结构与增长")
base_x = 100
base_y = 120
bar_width = 60
data = [
    ("消费电子", 28),
    ("汽车电子",42),   # 增长率最快
    ("工业控制",15),
    ("通信",8),
    ("计算机",-12)     # 负增长
]
#坐标轴
c.line(base_x,base_y, base_x+400, base_y)
c.line(base_x,base_y, base_x, base_y+220)

x_offset =0
for name,grow in data:
    bar_h = grow*4
    if bar_h>0:
        c.setFillColor(colors.green)
        c.rect(base_x+x_offset, base_y, bar_width, bar_h, fill=1)
    else:
        c.setFillColor(colors.red)
        c.rect(base_x+x_offset, base_y+bar_h, bar_width, abs(bar_h), fill=1)
    c.setFillColor(colors.black)
    c.setFont(FONT_CN,10)
    c.drawCentredString(base_x+x_offset+bar_width/2, base_y-18, name)
    c.drawCentredString(base_x+x_offset+bar_width/2, base_y+bar_h+5, f"{grow}%")
    x_offset += 85

c.showPage()

#第3页，部分文本题目对应的文字内容
c.setFont(FONT_CN,11)
c.drawString(80, height-60,"本次发行股数：3600万股，占发行后总股本20%")
c.drawString(80, height-90,"募集资金投向：智能产线升级、研发中心、补充流动资金")
c.save()
print(f"✅ {pdf_name} 生成完成！PDF包含组织结构图、IC市场增长率图表（PDF内置绘图图像，用于工单4多模态RAG测试）")
