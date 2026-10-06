# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
PDF解析模块（表格解析版）：
  1. 文本：按段落块提取，保留换行结构（同工单二的优化分块）；
  2. 表格：用 PyMuPDF find_tables() 识别，转成 Markdown 文本块，
     并把表格上方的标题文字作为"表标题"拼入块中，保证表格语义完整、可被检索到。
输出块结构：{"text":..., "page":页码, "doc":文档名, "type": "text"/"table"}
"""
import fitz  # PyMuPDF
import re
import json
from config import PDF_PATHS, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, TABLE_CAPTION_LEN, CHUNKS_FILE


def _clean_block(text):
    """清洗段落文本：压缩行内空白，保留换行"""
    text = text.replace('\x00', '').replace('﻿', '').replace('\xa0', ' ')
    lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.split('\n')]
    return '\n'.join(ln for ln in lines if ln)


def _clean_row(row):
    """清洗表格行：去掉合并单元格产生的空列与相邻重复列"""
    cells = []
    for c in row:
        c = (c or "").replace('\n', ' ').strip()
        if c and (not cells or c != cells[-1]):
            cells.append(c)
    return cells


def table_to_markdown(rows):
    """把二维表格转成 Markdown；首行作为表头"""
    cleaned = [_clean_row(r) for r in rows]
    cleaned = [r for r in cleaned if r]
    if not cleaned:
        return ""
    header = cleaned[0]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in cleaned[1:]:
        # 列数不足时补齐，多余时截断，保证 Markdown 表格合法
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        lines.append("| " + " | ".join(r[:len(header)]) + " |")
    return "\n".join(lines)


def _caption_above(page_blocks, table_bbox):
    """取表格上方最近的一段文字作为表标题（如"存在控制关系的关联方"）"""
    above = [b for b in page_blocks if len(b) >= 7 and b[6] == 0 and b[3] <= table_bbox[1] + 2]
    if not above:
        return ""
    above.sort(key=lambda b: b[3])
    return _clean_block(above[-1][4])[:TABLE_CAPTION_LEN].replace("\n", " ")


def extract_from_pdf(pdf_path, doc_name):
    """从单个PDF提取文本块与表格块，返回 chunk 列表"""
    doc = fitz.open(pdf_path)
    blocks, tables = [], []

    for page_num, page in enumerate(doc, 1):
        page_blocks = page.get_text("blocks")
        for b in page_blocks:
            if len(b) >= 7 and b[6] != 0:
                continue
            txt = _clean_block(b[4])
            if txt:
                blocks.append({"text": txt, "page": page_num, "doc": doc_name})

        # ---- 表格解析 ----
        try:
            found = page.find_tables()
        except Exception:
            found = None
        if found:
            for t in found.tables:
                md = table_to_markdown(t.extract())
                if not md:
                    continue
                caption = _caption_above(page_blocks, t.bbox)
                head = f"【表格·第{page_num}页】" + (f"{caption}\n" if caption else "")
                tables.append({
                    "text": head + md,
                    "page": page_num, "doc": doc_name, "type": "table",
                })
    doc.close()
    print(f"[PDF解析] {doc_name}: 段落块 {len(blocks)} 个, 表格 {len(tables)} 个")
    return blocks, tables


def _merge_blocks(blocks, chunk_size, overlap, min_size):
    """把段落块归并为文本块（不在段落内部切断）"""
    chunks, cur, cur_pages = [], "", []

    def flush():
        nonlocal cur, cur_pages
        text = cur.strip()
        if len(text) >= min_size or not chunks:
            chunks.append({"text": text, "page": cur_pages[0] if cur_pages else 1,
                           "pages": sorted(set(cur_pages))})
        else:
            chunks[-1]["text"] += "\n" + text
        cur, cur_pages = "", ""

    for blk in blocks:
        txt = blk["text"]
        if len(cur) + len(txt) + 1 <= chunk_size:
            cur += ("\n" if cur else "") + txt
            cur_pages.append(blk["page"])
        else:
            tail = cur[-overlap:] if len(cur) > overlap else cur
            flush()
            cur, cur_pages = tail + "\n" + txt, [blk["page"]]
    if cur.strip():
        flush()
    return chunks


def build_chunks(save_path=CHUNKS_FILE):
    """解析全部PDF（文本+表格），合并为一个知识库并保存"""
    all_chunks = []
    for doc_name, path in PDF_PATHS.items():
        blocks, tables = extract_from_pdf(path, doc_name)
        text_chunks = _merge_blocks(blocks, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
        for c in text_chunks:
            all_chunks.append({"text": c["text"], "page": c["page"],
                               "doc": doc_name, "type": "text"})
        all_chunks.extend(tables)     # 表格块单独入索引，语义完整不被切碎

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    n_tab = sum(1 for c in all_chunks if c["type"] == "table")
    print(f"[PDF解析] 知识库共 {len(all_chunks)} 块（其中表格 {n_tab} 块）已保存: {save_path}")
    return all_chunks


def load_chunks(save_path=CHUNKS_FILE):
    with open(save_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_texts(chunks):
    return [c["text"] for c in chunks]


if __name__ == "__main__":
    cs = build_chunks()
    for c in cs:
        if c["type"] == "table" and c["page"] in (22, 157):
            print(f"\n--- 表格 页{c['page']} {c['doc']} ---\n{c['text'][:300]}")
