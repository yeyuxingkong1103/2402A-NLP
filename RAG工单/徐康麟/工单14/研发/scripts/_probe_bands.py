# 工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
# 探针2：抽取候选字符连通域并渲染成 PNG，供人眼确认页面性质
import os, numpy as np, pymupdf
from PIL import Image, ImageDraw

SRC = r"E:\gao6gongdan\_共享输入\original_problems\documents\CN100342976C.pdf"
OUT = r"E:\gao6gongdan\工单14\研发\data\_probe"
os.makedirs(OUT, exist_ok=True)

doc = pymupdf.open(SRC)
for pno in [6]:
    page = doc[pno]
    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    h, w = img.shape
    # 保存整页 PNG
    Image.fromarray(img).save(os.path.join(OUT, f"来源第{pno+1}页_整页.png"))

    ink = (img < 140).astype(np.uint8)
    # 行投影分段
    rowink = ink.sum(axis=1)
    bands = []
    inb = False
    for y, v in enumerate(rowink):
        if v > 2 and not inb:
            start = y; inb = True
        elif v <= 2 and inb:
            if y - start >= 6: bands.append((start, y))
            inb = False
    print(f"page {pno+1} size {w}x{h} bands={len(bands)}")
    for s, e in bands:
        seg = ink[s:e]
        cols = seg.sum(axis=0)
        print(f"  band y={s}-{e} h={e-s} ink={int(seg.sum())} maxcol={int(cols.max())} "
              f"xspan=({int(np.nonzero(cols)[0].min()) if cols.any() else -1},{int(np.nonzero(cols)[0].max()) if cols.any() else -1})")
doc.close()
print("saved to", OUT)
