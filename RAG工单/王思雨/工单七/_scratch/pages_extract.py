# -*- coding: utf-8 -*-
import os, re, fitz
SRC = "D:/Users/雨子/Desktop/\u4e13\u9ad8\u516d/liu-\u4e13\u9ad8\u516d/RAG \u5de5\u5355/\u9644\u4ef6/ccf_competition/pdf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
os.makedirs(OUT, exist_ok=True)
for n in sorted(os.listdir(SRC)):
    m = re.search(r"__(\d{6})__", n)
    code = m.group(1) if m else n[:10]
    year = re.search(r"__(\d{4})\u5e74__", n)
    stem = f"{code}_{year.group(1) if year else 'xxxx'}"
    doc = fitz.open(os.path.join(SRC, n))
    parts = []
    for i, pg in enumerate(doc):
        parts.append(f"\n===== PAGE {i+1} =====\n" + pg.get_text())
    doc.close()
    with open(os.path.join(OUT, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(stem, len(parts))
