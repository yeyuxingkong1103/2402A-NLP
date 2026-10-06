# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
PDF解析模块：使用 PyMuPDF (fitz) 提取PDF文本，并按固定大小分块
"""
import fitz  # PyMuPDF
import re
import json
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_FILE


def extract_text_from_pdf(pdf_path):
    """提取PDF中所有页面的文本，返回整段文本"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        full_text += text + "\n"
    doc.close()
    return full_text


def extract_text_from_bytes(pdf_bytes):
    """从内存字节（上传文件）提取PDF文本，返回整段文本"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text


def clean_text(text):
    """清洗文本：去除多余空白、乱码字符等"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除一些特殊的不可见字符
    text = text.replace('\x00', '').replace('\ufeff', '')
    return text.strip()


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    将长文本切分为固定大小的块，块之间有重叠以保留上下文
    策略：先按句子切分，再合并到 chunk_size 左右
    """
    # 先按句号、问号、感叹号、分号等切分成句子
    sentences = re.split(r'(?<=[。！？；])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""
    for sent in sentences:
        # 如果当前块加上这个句子还没超过 chunk_size，就加进去
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            # 当前块已满，保存
            if current_chunk:
                chunks.append(current_chunk)
            # 新块：取上一块末尾 overlap 长度的内容作为开头，保证上下文连续
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + sent
            else:
                current_chunk = sent
    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_chunks(pdf_path=PDF_PATH, save_path=CHUNKS_FILE):
    """
    完整流程：提取PDF文本 -> 清洗 -> 分块 -> 保存
    返回文本块列表
    """
    print(f"[PDF解析] 正在解析: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"[PDF解析] 提取到原始文本 {len(raw_text)} 字符")

    clean = clean_text(raw_text)
    print(f"[PDF解析] 清洗后文本 {len(clean)} 字符")

    chunks = split_text_into_chunks(clean)
    print(f"[PDF解析] 共切分为 {len(chunks)} 个文本块")

    # 保存到文件，避免重复解析
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[PDF解析] 文本块已保存到: {save_path}")

    return chunks


def build_chunks_from_bytes(pdf_bytes, save_path=None):
    """
    从上传的PDF字节构建文本块（用于Streamlit上传场景）
    返回: (chunks列表, 原始字符数, 清洗后字符数)
    """
    raw_text = extract_text_from_bytes(pdf_bytes)
    clean = clean_text(raw_text)
    chunks = split_text_into_chunks(clean)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    return chunks, len(raw_text), len(clean)


def load_chunks(save_path=CHUNKS_FILE):
    """从文件加载已保存的文本块"""
    with open(save_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\n前3个文本块预览:")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- 块 {i+1} (长度{len(c)}) ---")
        print(c[:200] + "..." if len(c) > 200 else c)
