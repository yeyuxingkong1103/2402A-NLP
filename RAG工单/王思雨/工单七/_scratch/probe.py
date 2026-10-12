import os, fitz
D = r"D:/Users/雨子/Desktop/专高六/liu-专高六/RAG 工单/附件/ccf_competition"
# 用 bytes 列目录，绕开文件名编码问题
raw = os.listdir(os.fsencode(os.path.join(D, "pdf")))
print("== 解码后的文件名 ==")
names = []
for b in sorted(raw):
    try: n = b.decode("gbk")
    except Exception: n = b.decode("utf-8", "replace")
    names.append((b, n)); print("  ", n)
print("\n== 逐 PDF 页数与文本层 ==")
for b, n in names:
    p = os.path.join(os.fsencode(os.path.join(D,"pdf")), b)
    doc = fitz.open(p); tot = sum(len(pg.get_text().strip()) for pg in doc)
    print(f"  {n[:38]:40s} pages={len(doc):5d} chars={tot:9d} avg/pg={tot//max(len(doc),1):5d}")
    doc.close()
