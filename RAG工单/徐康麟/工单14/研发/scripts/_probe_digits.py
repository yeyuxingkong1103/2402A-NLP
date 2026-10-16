# 工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
# 探针脚本：判断第7页图页能否用「纯像素 + 字体模板匹配」定位附图标记数字
import os, sys, numpy as np
import pymupdf

SRC = r"E:\gao6gongdan\_共享输入\original_problems\documents\CN100342976C.pdf"
OUT = r"E:\gao6gongdan\工单14\研发\data\_probe"
os.makedirs(OUT, exist_ok=True)

doc = pymupdf.open(SRC)
page = doc[6]  # 第7页 (0-based 6)

# 高分辨率渲染
pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
print("render", img.shape, "dtype", img.dtype, "min", img.min(), "max", img.max())

# 二值化 (Otsu)
hist = np.bincount(img.ravel(), minlength=256).astype(float)
total = hist.sum()
sum_all = np.dot(np.arange(256), hist)
sumB = 0.0; wB = 0.0; best = 0.0; thr = 127
for t in range(256):
    wB += hist[t]
    if wB == 0: continue
    wF = total - wB
    if wF == 0: break
    sumB += t * hist[t]
    mB = sumB / wB; mF = (sum_all - sumB) / wF
    var = wB * wF * (mB - mF) ** 2
    if var > best:
        best = var; thr = t
print("otsu threshold", thr, "ink ratio", float((img < thr).mean()))

ink = (img < thr).astype(np.uint8)
print("ink pixels", int(ink.sum()))

# --- 连通域标记（纯 numpy 实现，两遍扫描 + 并查集）---
def connected_components(binary):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    def find(x):
        root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    nxt = 1
    for y in range(h):
        row = binary[y]; lab = labels[y]; above = labels[y-1] if y else None
        for x in range(w):
            if not row[x]: continue
            left = lab[x-1] if x else 0
            up = above[x] if above is not None else 0
            ul = above[x-1] if (above is not None and x) else 0
            neigh = [l for l in (left, up, ul) if l]
            if not neigh:
                lab[x] = nxt; parent.append(nxt); nxt += 1
            else:
                m = min(neigh); lab[x] = m
                for l in neigh: union(l, m)
    # 重映射
    remap = {}
    out = np.zeros_like(labels)
    for y in range(h):
        for x in range(w):
            l = labels[y, x]
            if l:
                r = find(l)
                if r not in remap: remap[r] = len(remap) + 1
                out[y, x] = remap[r]
    return out, len(remap)

# 全页连通域太慢，先只在「下部区域」和几个候选带宽区域做测试
# 先输出整页缩略图与行投影，帮助判断版面
rowink = ink.sum(axis=1)
colink = ink.sum(axis=0)
print("row ink bands (top 15):")
order = np.argsort(rowink)[::-1][:15]
for y in sorted(order.tolist()):
    print(f"   y={y:5d} ink={rowink[y]}")

# 保存二值图供人工/后续检查
import struct, zlib
def save_pgm(path, arr):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(arr.astype(np.uint8).tobytes())
save_pgm(os.path.join(OUT, "p7_gray.pgm"), img)
save_pgm(os.path.join(OUT, "p7_ink.pgm"), (ink ^ 1) * 255)
print("saved", OUT)

# 在下半页做连通域（图区通常在下半部）
sub = ink
t0 = __import__("time").time()
labels, n = connected_components(sub)
print("components", n, "in", round(__import__("time").time() - t0, 1), "s")

# 统计连通域 bbox，找“数字大小”的团块
ys, xs = np.nonzero(labels)
boxes = {}
for y, x in zip(ys.tolist(), xs.tolist()):
    l = labels[y, x]
    b = boxes.get(l)
    if b is None: boxes[l] = [x, y, x, y, 1]
    else:
        if x < b[0]: b[0] = x
        if y < b[1]: b[1] = y
        if x > b[2]: b[2] = x
        if y > b[3]: b[3] = y
        b[4] += 1
print("total boxes", len(boxes))
sizes = sorted(((b[4], k, b) for k, b in boxes.items()), reverse=True)[:20]
for cnt, k, b in sizes:
    print(f"   comp{k} px={cnt} bbox=({b[0]},{b[1]})-({b[2]},{b[3]}) wh=({b[2]-b[0]+1}x{b[3]-b[1]+1})")
