# -*- coding: utf-8 -*-
"""把 data/knowledge/*.txt 转成中文 PDF（用 Windows 系统字体嵌入，提取正常）。

注意：insert_textbox 对 .ttc 字体可能静默失败，这里用逐行 insert_text。
用法：python scripts/prepare_eval_data.py
"""
from pathlib import Path
# 解析：路径

import fitz
# 解析：PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
# 解析：项目根
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
# 解析：知识文档目录
FONT_CANDIDATES = [
    # 解析：候选中文字体
    "C:/Windows/Fonts/msyh.ttc",
    # 解析：微软雅黑
    "C:/Windows/Fonts/simsun.ttc",
    # 解析：宋体
    "C:/Windows/Fonts/simhei.ttf",
    # 解析：黑体
]
LINE_CHARS = 45  # 10.5pt 下每行约容纳的中文字符数
# 解析：每行字符数
LINE_HEIGHT = 16
# 解析：行高（磅）
MARGIN_TOP, MARGIN_BOTTOM = 60, 40
# 解析：上下边距
PAGE_W, PAGE_H = 595, 842
# 解析：A4 页面尺寸（磅）


# 按字符宽度折行（中文 PDF 排版用）
def wrap_line(line: str, width: int) -> list[str]:
    # 解析：按字符宽度折行
    return [line[i : i + width] for i in range(0, len(line), width)] or [""]
    # 解析：等宽切分（空行返回空串列表）


# 知识文档 txt 转中文 PDF（系统字体嵌入，逐行写入）
def txt_to_pdf(txt_path: Path) -> Path:
    # 解析：txt 转 PDF
    text = txt_path.read_text(encoding="utf-8")
    # 解析：读源文本
    pdf_path = txt_path.with_suffix(".pdf")
    # 解析：同名 PDF 路径
    fontfile = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    # 解析：取第一个存在的中文字体
    if fontfile is None:
        # 解析：无中文字体
        raise RuntimeError("未找到中文字体，请安装微软雅黑/宋体/黑体")
        # 解析：报错（否则中文变点无法提取）

    doc = fitz.open()
    # 解析：新建 PDF
    page = doc.new_page()
    # 解析：新建页
    page.insert_font(fontname="cjk", fontfile=fontfile)
    # 解析：嵌入中文字体
    y = MARGIN_TOP
    # 解析：起始 y 坐标
    for line in text.split("\n"):
        # 解析：逐源行
        for chunk in wrap_line(line, LINE_CHARS):
            # 解析：超长行折行
            if y > PAGE_H - MARGIN_BOTTOM:  # 翻页
                # 解析：超出页底
                page = doc.new_page()
                # 解析：新建页
                page.insert_font(fontname="cjk", fontfile=fontfile)
                # 解析：嵌入字体
                y = MARGIN_TOP
                # 解析：重置 y
            page.insert_text((50, y), chunk, fontsize=10.5, fontname="cjk")
            # 解析：写入一行（insert_text 逐行可靠，insert_textbox 对 ttc 有坑）
            y += LINE_HEIGHT
            # 解析：下移一行
    doc.save(str(pdf_path))
    # 解析：保存
    doc.close()
    # 解析：关闭
    print(f"[生成] {pdf_path}")
    # 解析：提示
    return pdf_path
    # 解析：返回路径


# 把 data/knowledge/*.txt 全部转为 PDF
def main() -> None:
    # 解析：批量转换
    for txt in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        # 解析：逐 txt 文件
        txt_to_pdf(txt)
        # 解析：转换


if __name__ == "__main__":
    # 解析：入口
    main()
    # 解析：执行
