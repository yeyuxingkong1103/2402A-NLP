# 工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
"""探针3：验证自研字体模板 OCR 能否恢复第7页附图标记。"""
import os, sys, json
import numpy as np, pymupdf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.font_ocr import FontTemplateOCR, extract_numerals, merge_into_tokens, binarize

SRC = r"E:\gao6gongdan\_共享输入\original_problems\documents\CN100342976C.pdf"
OUT = r"E:\gao6gongdan\工单14\研发\data\_probe"
os.makedirs(OUT, exist_ok=True)

DPI = 200
doc = pymupdf.open(SRC)
page = doc[6]
pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
doc.close()
print("page7 render", gray.shape)

ocr = FontTemplateOCR()
print("templates:", len(ocr.tpl_chars), "glyph templates")

ink = binarize(gray)
print("ink ratio", round(float(ink.mean()), 5))

hits = extract_numerals(gray, ocr, min_score=0.50, min_h=16, max_h=80)
print("hits:", len(hits))
tokens = merge_into_tokens(hits, gap_x=22, gap_y=12)
print("tokens:", len(tokens))
for t in tokens[:60]:
    print(f"   {t['text']:>6s}  @({t['x0']:4d},{t['y0']:4d}) score={t['score']:.3f} n={t['n']}")

with open(os.path.join(OUT, "page7_tokens.json"), "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=1)
print("saved tokens ->", os.path.join(OUT, "page7_tokens.json"))
