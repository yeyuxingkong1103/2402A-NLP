"""抽全文到 _scratch/raw/*.txt，文件名用股票代码（盘上中文名已被 U+FFFD 破坏，不可逆）"""
import os, re, fitz
SRC = "D:/Users/雨子/Desktop/专高六/liu-专高六/RAG 工单/附件/ccf_competition/pdf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(OUT, exist_ok=True)
for n in sorted(os.listdir(SRC)):
    m = re.search(r"__(\d{6})__", n)                       # 从文件名提取 6 位股票代码作为稳定 ID
    code = m.group(1) if m else n[:10]
    year = re.search(r"__(\d{4})年__", n)
    stem = f"{code}_{year.group(1) if year else 'xxxx'}"
    doc = fitz.open(os.path.join(SRC, n))
    txt = "\n".join(pg.get_text() for pg in doc); doc.close()
    with open(os.path.join(OUT, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(stem, len(txt))
