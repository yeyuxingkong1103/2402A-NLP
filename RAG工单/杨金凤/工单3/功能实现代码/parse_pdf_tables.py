# ============================================================
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 项目名称：PDF文档的表格解析及检索优化
# 文件：parse_pdf_tables.py
# 说明：用 pdfplumber 提取 PDF 文本和表格，表格转 Markdown 追加，
#       为后续向量化检索提供结构化表格数据
# ============================================================
import pdfplumber
from pathlib import Path


def table_to_markdown(table):
    """把 pdfplumber 提取的表格（二维列表）转成 Markdown 表格"""
    if not table or not table[0]:
        return ""

    # 清理单元格：去换行、去首尾空格
    cleaned = []
    for row in table:
        new_row = []
        for cell in row:
            if cell is None:
                new_row.append("")
            else:
                new_row.append(str(cell).replace("\n", " ").strip())
        cleaned.append(new_row)

    # 补全列数
    max_cols = max(len(r) for r in cleaned)
    for r in cleaned:
        while len(r) < max_cols:
            r.append("")

    # 表头 + 分隔行 + 数据行
    lines = []
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * max_cols) + "|")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_pdf(pdf_path, output_path):
    """解析单个 PDF，输出含 Markdown 表格的文本文件"""
    full_text = []
    table_total = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"正在解析：{pdf_path}，共 {total} 页")

        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            tables = page.extract_tables()
            table_md_list = []
            for t in tables:
                md = table_to_markdown(t)
                if md:
                    table_md_list.append(md)
            table_total += len(table_md_list)

            # 拼接：页文本 + 表格（表格用特殊标记包裹，便于后续分块识别）
            block = f"\n\n===== 第 {i+1} 页 =====\n{page_text}"
            if table_md_list:
                block += "\n\n【本页表格】\n"
                for idx, md in enumerate(table_md_list, 1):
                    block += f"\n### 表格 {idx} ###\n{md}\n### 表格 {idx} 结束 ###\n"
            full_text.append(block)

    final_text = "\n".join(full_text)
    Path(output_path).write_text(final_text, encoding="utf-8")

    print(f"解析完成，输出到：{output_path}")
    print(f"文本总长度：{len(final_text)} 字符，共提取 {table_total} 个表格")
    return final_text


if __name__ == "__main__":
    base = "/root/autodl-tmp/projects/RAG"
    tasks = [
        (f"{base}/0_raw_data/招股说明书1.pdf",
         f"{base}/1_trans_data/招股说明书1.txt"),
        (f"{base}/0_raw_data/招股说明书2.pdf",
         f"{base}/1_trans_data/招股说明书2.txt"),
    ]
    for pdf, out in tasks:
        parse_pdf(pdf, out)
        print("=" * 60)