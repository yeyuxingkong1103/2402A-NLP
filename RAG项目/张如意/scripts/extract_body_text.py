# -*- coding: utf-8 -*-
"""从 PDF 文本层抽出正文条款 -> data/processed/body_text/{作物}_body.md

覆盖范围：三份标准的**正文 p3–p7**。附录 A（p8–p19）与附录 B（p20–p21）不走这里：
  - 附录 B 表格页是字形码/旋转排版，走视觉通道 → vision_transcribe_appendixB.md
  - 附录 A 记录表同理 → vision_transcribe_appendixA.md
  - 附录 A 里文本层可抽的页，字段已进 appendixA_cards.jsonl 的 record_form，不另存正文

预处理（configs/prompts/parse_pdf_to_knowledge.md 第 2 节）：
  P2 页眉页脚剔除：形如 `GB/Z26581—2011` 的页眉、单独成行的页码，不进正文
  P1 上标还原：这里的 `m²`/`hm²` 上标在文本层被拆到下一行，由下游 build_cards_body.py 还原
     （保持本文件"接近原始文本层"的形态，便于和 MinerU 做同层比对）

输出带 `===== PDF pN =====` 分页标记，下游按它切页。
"""
import os
import re

import fitz  # PyMuPDF：PDF 文本层提取库

RAW = os.path.join("data", "raw")                            # 原始国标 PDF 目录
OUT = os.path.join("data", "processed", "body_text")         # 正文 markdown 输出目录

# (作物, PDF 文件名, 正文起始页, 正文结束页) —— 页码为 PDF 物理页，1-based
BOOKS = [
    ("黄瓜", "GB_Z 26581-2011 黄瓜生产技术规范.pdf", 3, 7),
    ("辣椒", "GB_Z 26583-2011 辣椒生产技术规范.pdf", 3, 7),
    ("大蒜", "GB_Z 26578-2011 大蒜生产技术规范.pdf", 3, 7),
]

HEADER = re.compile(r'^\s*GB/Z\s*265\d\d—2011\s*$')   # 页眉
PAGENO = re.compile(r'^\s*\d{1,2}\s*$')                # 单独成行的页码


def extract(pdf_path, a, b):
    """抽取一册 PDF 的正文页文本。

    参数：
        pdf_path: PDF 文件路径
        a, b:     正文起止页（PDF 物理页码，1-based，含两端）
    返回：
        带 `===== PDF pN =====` 分页标记的正文文本字符串（供下游按标记切页）
    """
    doc = fitz.open(pdf_path)
    buf = []
    for pno in range(a, b + 1):
        # 逐页取文本层，按行拆开；doc 页索引从 0 起，故 pno-1
        lines = doc[pno - 1].get_text("text").splitlines()
        # 剔除页眉（GB/Z 265xx—2011）与单独成行的页码，其余行拼回正文
        body = "\n".join(l for l in lines if not HEADER.match(l) and not PAGENO.match(l))
        buf.append(f"\n\n===== PDF p{pno} =====\n{body}")
    doc.close()
    return "".join(buf)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for crop, fn, a, b in BOOKS:
        src = os.path.join(RAW, fn)
        if not os.path.exists(src):          # 缺 PDF 不中断，跳过并提示
            print(f"[{crop}] 缺 PDF：{src}")
            continue
        text = extract(src, a, b)
        dst = os.path.join(OUT, f"{crop}_body.md")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(text)
        n = len(re.findall(r'[一-鿿]', text))  # 统计中文字符数，作为抽取量 sanity check
        print(f"[{crop}] p{a}-p{b} -> {dst}  ({n} 中文字)")
