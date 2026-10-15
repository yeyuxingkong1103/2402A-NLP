# -*- coding: utf-8 -*-
# pdf_ocr_fix.py 工单14 低质量图片PDF本地OCR预处理
# 输入：扫描图片型PDF；输出：叠加OCR可编辑文本层的PDF，给到RAGFlow
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
import os

INPUT_PDF = "./dataset/CN100342976C.pdf"
OUTPUT_OCR_PDF = "./dataset/CN100342976C_ocr_fixed.pdf"

def ocr_enhance_pdf(input_path, output_path):
    """
    对图片PDF每页做OCR识别，生成带文本层新PDF，用于RAGFlow导入
    """
    print("开始PDF转图片...")
    pages = convert_from_path(input_path)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    c = canvas.Canvas(output_path)

    for idx, page_img in enumerate(pages):
        print(f"处理第 {idx+1} 页")
        w, h = page_img.size
        # 绘制原图背景
        c.drawImage(page_img, 0,0, width=w, height=h)
        # OCR识别
        res = ocr.ocr(page_img, cls=True)
        for page_res in res:
            if not page_res:
                continue
            for line in page_res:
                pts = line[0]
                txt = line[1][0]
                x1,y1 = pts[0]
                # OCR识别文本叠加，生成可复制文本层
                c.setFont("SimSun", 8)
                c.drawString(x1, h‑y1, txt)
        c.showPage()
    c.save()
    print(f"OCR增强PDF输出完成：{output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_PDF):
        raise FileNotFoundError(f"文件不存在 {INPUT_PDF}")
    ocr_enhance_pdf(INPUT_PDF, OUTPUT_OCR_PDF)
