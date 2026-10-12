import os, fitz
D = "D:/Users/雨子/Desktop/专高六/liu-专高六/RAG 工单/附件/ccf_competition/pdf"
names = sorted(os.listdir(D))
print(f"files={len(names)}")
for n in names:
    p = os.path.join(D, n)
    try:
        doc = fitz.open(p); pages = len(doc)
        tot = sum(len(pg.get_text().strip()) for pg in doc); doc.close()
        ok = "OK"
    except Exception as e:
        pages = tot = -1; ok = f"ERR {type(e).__name__}"
    print(f"{ok:8s} pages={pages:5d} chars={tot:9d} name_utf8_bytes={n.encode('utf-8','replace')[:26]!r}")
