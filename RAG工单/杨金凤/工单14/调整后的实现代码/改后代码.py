# deepdoc/parser/pdf_parser.py
import os
import re
import copy
import logging
import numpy as np
import pdfplumber
from PIL import Image
from io import BytesIO
from deepdoc.vision import OCR, Recognizer
from deepdoc.parser.utils import extract_pdf_outlines

class RAGFlowPdfParser:
    """增强版 PDF 解析器，针对低质量工业 PDF 优化"""

    def __init__(self):
        self.ocr = OCR()
        self.page_images = []
        self.page_layout = []
        self.page_cum_height = [0]
        self.boxes = []

    def __call__(self, filename, binary=None, from_page=0, to_page=100000,
                 callback=None, **kwargs):
        """主解析入口"""
        if callback:
            callback(0.1, "Start PDF parsing...")

        pdf = pdfplumber.open(filename if not binary else BytesIO(binary))
        self.page_images = []
        self.page_layout = []
        self.page_cum_height = [0]

        for pn, page in enumerate(pdf.pages):
            if pn < from_page or pn > to_page:
                continue

            # 高 DPI 渲染提升低质量 PDF 的 OCR 精度
            img = page.to_image(resolution=300).original
            self.page_images.append(img)

            # 版面分析
            layout = self._analyze_layout(img, pn)
            self.page_layout.append(layout)

            # 累计高度用于坐标转换
            self.page_cum_height.append(
                self.page_cum_height[-1] + page.height
            )

            if callback:
                callback(0.1 + 0.4 * (pn + 1) / len(pdf.pages),
                         f"Processed page {pn + 1}")

        # 提取文本和表格
        sections, tables = self._extract_content(pdf)

        if callback:
            callback(0.6, f"Extracted {len(sections)} sections, "
                          f"{len(tables)} tables")

        return sections, tables

    def _analyze_layout(self, image, page_num):
        """版面分析：识别文本、标题、表格、图片区域"""
        img_array = np.array(image)
        boxes = self.ocr(img_array, page_num)

        layout = []
        for box in boxes:
            text = box[1][0]
            if not text or len(text.strip()) < 2:
                continue

            # 根据字体大小和位置推断区域类型
            bbox = box[0]  # [x0, y0, x1, y1]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]

            if height > 30 and width > 200:
                layout.append({
                    "type": "title",
                    "text": text,
                    "bbox": bbox
                })
            elif height < 15:
                layout.append({
                    "type": "text",
                    "text": text,
                    "bbox": bbox
                })
            else:
                layout.append({
                    "type": "text",
                    "text": text,
                    "bbox": bbox
                })

        return layout

    def _extract_content(self, pdf):
        """提取文本和表格内容"""
        sections = []
        tables = []

        for pn, page in enumerate(pdf.pages):
            # 提取文本
            text = page.extract_text()
            if text:
                sections.append({
                    "page": pn,
                    "text": text
                })

            # 提取表格
            for table in page.extract_tables():
                if table:
                    tables.append({
                        "page": pn,
                        "data": table
                    })

        return sections, tables

    def _extract_figure(self, image, page_num):
        """提取图片区域的 OCR 文字（针对工业图纸优化）"""
        img_array = np.array(image)
        bxs = self.ocr(img_array, page_num)

        # 提取所有文字及其位置
        texts = []
        for box in bxs:
            text = box[1][0]
            if text and len(text.strip()) > 0:
                texts.append({
                    "text": text,
                    "bbox": box[0],
                    "confidence": box[1][1] if len(box[1]) > 1 else 0.0
                })

        # 按位置排序（从上到下，从左到右）
        texts.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # 合并相邻文字
        combined_text = " ".join([t["text"] for t in texts])

        return {
            "image": image,
            "text": combined_text,
            "page": page_num,
            "boxes": texts
        }

    def crop(self, position_tag):
        """根据位置标签裁剪图片"""
        positions = self.extract_positions(position_tag)
        if not positions:
            return None

        pn, left, right, top, bottom = positions[0]
        if pn >= len(self.page_images):
            return None

        img = self.page_images[pn]
        return img.crop((left, top, right, bottom))

    @staticmethod
    def extract_positions(tag):
        """解析位置标签"""
        positions = []
        parts = tag.strip().split("##")
        for part in parts:
            if not part:
                continue
            nums = re.findall(r'\d+', part)
            if len(nums) >= 5:
                positions.append([int(n) for n in nums[:5]])
        return positions