import pymupdf, os

pdf_path = r"D:\专高六 工单\工单14\pdf\CN100342976C.pdf"
out_dir = r"D:\专高六 工单\工单14\images"
os.makedirs(out_dir, exist_ok=True)

doc = pymupdf.open(pdf_path)
print("总页数:", len(doc))

# 第 7 页（索引 6）
page = doc[6]
pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
out7 = os.path.join(out_dir, "page7.png")
pix.save(out7)
print(f"已保存 {out7}，尺寸: {pix.width} x {pix.height}")

# 第 1 页
page1 = doc[0]
pix1 = page1.get_pixmap(matrix=pymupdf.Matrix(3, 3))
out1 = os.path.join(out_dir, "page1.png")
pix1.save(out1)
print(f"已保存 {out1}，尺寸: {pix1.width} x {pix1.height}")
