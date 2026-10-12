# -*- coding: utf-8 -*-
"""构建模拟数据集（模拟 IMDR documents/ 约 1700 文件的特征子集），用于演示质检全流程。"""
import os, shutil, random

ROOT = "/home/dabaie/code/工单/工单十八/demo_dataset"
SRC_PDF_DIR = "/mnt/c/Users/Lenovo/Desktop/zg6工单"

random.seed(42)


def build():
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)

    # 1) 真实 PDF（文字型，含可提取文本）— 复制 4 份
    real_pdfs = [f for f in os.listdir(SRC_PDF_DIR) if f.endswith(".pdf")]
    for i, f in enumerate(real_pdfs[:4]):
        shutil.copy(os.path.join(SRC_PDF_DIR, f), os.path.join(ROOT, f"manual_{i}.pdf"))

    # 2) 扫描型 PDF：用 pypdf 生成空白页（无可提取文本 → 判定为 Scan_PDF）
    try:
        from pypdf import PdfWriter
        for i in range(3):
            w = PdfWriter()
            for _ in range(5):
                w.add_blank_page(width=595, height=842)
            with open(os.path.join(ROOT, f"scan_doc_{i}.pdf"), "wb") as fp:
                w.write(fp)
    except Exception as e:
        print("blank pdf gen skipped:", e)

    # 3) Markdown / 文本：含精确重复、相似版本、敏感信息、不同长度
    base = ("工业设备维护规程：淬火工艺要求将钢件加热至临界温度以上，"
            "保温后快速冷却以获得高硬度。操作人员须佩戴防护装备。") * 8
    with open(os.path.join(ROOT, "op_a.md"), "w", encoding="utf-8") as f:
        f.write("# 规程A\n" + base)
    shutil.copy(os.path.join(ROOT, "op_a.md"), os.path.join(ROOT, "op_a_copy.md"))  # MD5 重复
    with open(os.path.join(ROOT, "op_b.md"), "w", encoding="utf-8") as f:
        f.write("# 规程B\n" + base.replace("快速冷却", "迅速冷却") + "\n（修订版）")  # 高相似
    with open(os.path.join(ROOT, "contacts.txt"), "w", encoding="utf-8") as f:
        f.write("紧急联系：张工 13912345678，邮箱 zhang@factory.com，身份证 320101198805051234。\n" * 3)
    with open(os.path.join(ROOT, "short_note.txt"), "w", encoding="utf-8") as f:
        f.write("待办：检查设备。")

    # 4) Office / 图片占位（按扩展名统计）
    for nm in ["spec.docx", "report.pptx", "photo.png"]:
        open(os.path.join(ROOT, nm), "wb").write(b"placeholder")

    print("demo dataset built at", ROOT, "files:", len(os.listdir(ROOT)))


if __name__ == "__main__":
    build()
