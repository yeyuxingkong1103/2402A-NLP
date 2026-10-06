# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
ccf_competition 语料解析：解析附件 ccf_competition/pdf 下的 9 份金融年报 PDF，
按段落合并为 600 字左右的文本块（块内不切断段落），输出统一的知识库块。
公司名从文件名中的股票代码映射（规避文件名编码问题）。
"""
import os
import re
import json
import fitz

from config import CCF_PDF_DIR, CHUNKS_FILE, COMPANY_BY_CODE, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


def doc_name_of(filename):
    """由文件名解析出 '公司名+年份+年报' 形式的文档名（取文件名中最后一个 4 位数年份）"""
    code = re.search(r"__(\d{6})__", filename)
    years = re.findall(r"\d{4}", filename)
    year = years[-1] if years else ""
    comp = COMPANY_BY_CODE.get(code.group(1) if code else "", "未知公司")
    return f"{comp}{year}年报"


def _is_noise(txt):
    """过滤目录/索引类噪声块：点线占比过高、或几乎无中文/字母内容"""
    dots = txt.count(".") + txt.count("…")
    if dots >= 8 and dots / max(len(txt), 1) > 0.12:
        return True
    if len(re.findall(r"[一-鿿A-Za-z]", txt)) < 8:
        return True
    return False


def extract_pdf(path):
    """按页抽取文本块（get_text('blocks')，保留段落边界，过滤目录/页码噪声）"""
    doc = fitz.open(path)
    blocks = []
    for pno, page in enumerate(doc, 1):
        for b in page.get_text("blocks"):
            txt = (b[4] or "").strip()
            if len(txt) >= 10 and not _is_noise(txt):
                blocks.append({"text": txt, "page": pno})
    doc.close()
    return blocks


def merge_blocks(blocks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """把段落块合并为 chunk_size 左右的文本块，块内不切断段落，块间保留 overlap 字符重叠"""
    chunks, buf, pages, size = [], [], [], 0
    for b in blocks:
        t = b["text"]
        if size + len(t) > chunk_size and size >= MIN_CHUNK_SIZE:
            text = "".join(buf)
            chunks.append({"text": text, "page": pages[0]})
            tail = text[-overlap:]
            buf, pages, size = [tail], [b["page"]], len(tail)
        buf.append(t + "\n")
        pages.append(b["page"])
        size += len(t)
    if size >= MIN_CHUNK_SIZE:
        chunks.append({"text": "".join(buf), "page": pages[0]})
    return chunks


def build_chunks(save_path=CHUNKS_FILE):
    """解析全部 PDF，生成块并保存"""
    files = sorted(f for f in os.listdir(CCF_PDF_DIR) if f.lower().endswith(".pdf"))
    print(f"[语料解析] 发现 PDF {len(files)} 个")
    all_chunks = []
    for fn in files:
        name = doc_name_of(fn)
        blocks = extract_pdf(os.path.join(CCF_PDF_DIR, fn))
        cs = merge_blocks(blocks)
        for c in cs:
            c["doc"] = name
            c["type"] = "text"
        all_chunks.extend(cs)
        print(f"  - {name}: 段落 {len(blocks)} -> 块 {len(cs)}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[语料解析] 共生成块 {len(all_chunks)} 个 -> {save_path}")
    return all_chunks


def load_chunks(save_path=CHUNKS_FILE):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    cs = build_chunks()
    print(cs[0]["doc"], cs[0]["page"], cs[0]["text"][:120])
