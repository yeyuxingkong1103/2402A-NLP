import fitz  # PyMuPDF
import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    """用PyMuPDF提取PDF纯文本"""
    doc = fitz.open(pdf_path)
    text_list = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_list.append(f"【第{page_num+1}页】\n{text.strip()}")
    doc.close()
    return text_list

def extract_tables_from_pdf(pdf_path):
    """用pdfplumber提取PDF表格"""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 0:
                    md_table = []
                    header = table[0]
                    md_table.append("| " + " | ".join([str(c) if c else "" for c in header]) + " |")
                    md_table.append("| " + " | ".join(["---"] * len(header)) + " |")
                    for row in table[1:]:
                        md_table.append("| " + " | ".join([str(c) if c else "" for c in row]) + " |")
                    tables.append(f"【第{page_num+1}页表格】\n" + "\n".join(md_table))
    return tables

def parse_pdf(pdf_path):
    """完整解析PDF：文本 + 表格"""
    print(f"📄 正在解析PDF：{pdf_path}")
    text_blocks = extract_text_from_pdf(pdf_path)
    print(f"  提取到 {len(text_blocks)} 页文本")
    table_blocks = extract_tables_from_pdf(pdf_path)
    print(f"  提取到 {len(table_blocks)} 个表格")
    return text_blocks + table_blocks

def split_by_sentence(text, max_chunk_size=300):
    """按句子分块，每块不超过max_chunk_size字符"""
    # 中文句子结束符：。！？；
    sentences = re.split(r'(?<=[。！？；])', text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current_chunk) + len(sent) <= max_chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def split_by_paragraph(text, max_chunk_size=500):
    """按段落分块，段落太长再按句子拆"""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chunk_size:
            # 段落太长，按句子拆
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            sub_chunks = split_by_sentence(para, max_chunk_size)
            chunks.extend(sub_chunks)
        elif len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def split_by_heading(text, max_chunk_size=500):
    """按标题分块（识别 # 或 数字标题）"""
    lines = text.split("\n")
    chunks = []
    current_heading = ""
    current_content = []
    for line in lines:
        line_stripped = line.strip()
        # 识别标题：# 开头 或 一、二、 或 1. 2.
        if re.match(r'^#{1,6}\s', line_stripped) or re.match(r'^[一二三四五六七八九十]+、', line_stripped) or re.match(r'^\d+\.\s', line_stripped):
            if current_content:
                chunk = current_heading + "\n" + "\n".join(current_content)
                if len(chunk) > max_chunk_size:
                    chunks.extend(split_by_paragraph(chunk, max_chunk_size))
                else:
                    chunks.append(chunk.strip())
            current_heading = line_stripped
            current_content = []
        else:
            if line_stripped:
                current_content.append(line_stripped)
    if current_content:
        chunk = current_heading + "\n" + "\n".join(current_content)
        if len(chunk) > max_chunk_size:
            chunks.extend(split_by_paragraph(chunk, max_chunk_size))
        else:
            chunks.append(chunk.strip())
    return chunks if chunks else [text]

def split_text(text, method="paragraph", chunk_size=300, overlap=50):
    """
    分块入口
    method: sentence(按句子) / paragraph(按段落) / heading(按标题) / fixed(固定长度)
    """
    if method == "sentence":
        return split_by_sentence(text, chunk_size)
    elif method == "paragraph":
        return split_by_paragraph(text, chunk_size)
    elif method == "heading":
        return split_by_heading(text, chunk_size)
    else:
        # 固定长度分块，带重叠
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
