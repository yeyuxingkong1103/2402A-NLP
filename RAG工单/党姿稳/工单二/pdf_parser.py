# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
PDF解析模块（优化版）：
  相比工单一的固定字符切分，这里按"页面段落块"切分：
  1. 用 get_text("blocks") 取段落级文本，保留原始换行与表格行结构；
  2. 归并段落时不在段落内部切断，保证语义完整；
  3. 每个块附带页码元数据，便于溯源与评估。
"""
import fitz  # PyMuPDF
import re
import json
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, CHUNKS_FILE


def _clean_block(text):
    """清洗单个段落块：去掉多余空格与不可见字符，但保留换行结构"""
    text = text.replace('\x00', '').replace('﻿', '').replace('\xa0', ' ')
    # 同一行内多余空格压缩，保留换行
    lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.split('\n')]
    lines = [ln for ln in lines if ln]
    return '\n'.join(lines)


def extract_blocks(pdf_path=PDF_PATH):
    """
    提取页面段落块，返回 [ {"text":..., "page":页码}, ... ]
    仅保留有实际文字内容的块（过滤页眉页脚式的超短块）
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc, 1):
        for b in page.get_text("blocks"):
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) >= 7 and b[6] != 0:
                continue  # 跳过图片块（图像内容由工单四处理）
            txt = _clean_block(b[4])
            if txt:
                blocks.append({"text": txt, "page": page_num})
    doc.close()
    return blocks


def split_blocks_into_chunks(blocks, chunk_size=CHUNK_SIZE,
                             overlap=CHUNK_OVERLAP, min_size=MIN_CHUNK_SIZE):
    """
    将段落块归并为文本块：
    - 累加段落直到接近 chunk_size，不在段落内部切断；
    - 新块开头带上上一块末尾 overlap 字符，保持上下文连续；
    - 返回 [ {"text":..., "page":起始页, "pages":[...]}, ... ]
    """
    chunks = []
    cur, cur_pages = "", []

    def flush():
        nonlocal cur, cur_pages
        text = cur.strip()
        if len(text) >= min_size or not chunks:
            chunks.append({
                "text": text,
                "page": cur_pages[0] if cur_pages else 1,
                "pages": sorted(set(cur_pages)),
            })
        else:
            # 太短的尾块并入上一块
            chunks[-1]["text"] += "\n" + text
            chunks[-1]["pages"] = sorted(set(chunks[-1]["pages"] + cur_pages))
        cur, cur_pages = "", []

    for blk in blocks:
        txt = blk["text"]
        if len(cur) + len(txt) + 1 <= chunk_size:
            cur += ("\n" if cur else "") + txt
            cur_pages.append(blk["page"])
        else:
            tail = cur[-overlap:] if len(cur) > overlap else cur
            flush()
            cur = tail + "\n" + txt
            cur_pages = [blk["page"]]
    if cur.strip():
        flush()
    return chunks


def build_chunks(pdf_path=PDF_PATH, save_path=CHUNKS_FILE):
    """完整流程：提取段落块 -> 归并为文本块 -> 保存，返回文本块列表"""
    print(f"[PDF解析] 正在解析: {pdf_path}")
    blocks = extract_blocks(pdf_path)
    print(f"[PDF解析] 共提取段落块 {len(blocks)} 个")

    chunks = split_blocks_into_chunks(blocks)
    print(f"[PDF解析] 归并为文本块 {len(chunks)} 个")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[PDF解析] 已保存: {save_path}")
    return chunks


def load_chunks(save_path=CHUNKS_FILE):
    """加载已保存的文本块"""
    with open(save_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_texts(chunks):
    """从文本块列表中取出纯文本列表（便于向量化）"""
    return [c["text"] for c in chunks]


if __name__ == "__main__":
    cs = build_chunks()
    for i, c in enumerate(cs[:3]):
        print(f"\n--- 块{i+1} 页码{c['page']} 长度{len(c['text'])} ---")
        print(c["text"][:200])
