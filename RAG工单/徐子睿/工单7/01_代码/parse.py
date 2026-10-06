# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：parse —— PDF 解析（MinerU 主通道 + PyMuPDF 兜底；支持表格结构化与多文档）
# 说明：调用 MinerU CLI 解析 PDF，读取其 content_list.json，转成统一的
#       {text, type, page, level, doc, table_rows} blocks；MinerU 不可用时回退 PyMuPDF 逐页抽取。
#       多文档：每本 PDF 单独落盘 blocks_<doc>.jsonl，并汇总为 blocks.jsonl。

import os
import sys
import json
import glob
import re
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MINERU_DIR, PARSED_DIR, BLOCKS_PATH,  # noqa: E402
                    doc_of, blocks_path, list_pdfs, DEFAULT_PDF)


def find_content_list(mineru_root: str):
    """在 MinerU 输出目录中定位 content_list.json（兼容不同版本目录结构）。"""
    pats = [
        os.path.join(mineru_root, "**", "*_content_list.json"),
        os.path.join(mineru_root, "**", "*content_list.json"),
    ]
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=lambda x: (-os.path.getmtime(x), len(x)))
    return hits[0]


def find_content_list_for(mineru_root: str, pdf_name: str):
    """按 PDF 文件名定位其 MinerU 输出（多文档时避免解析串本）。"""
    stem = os.path.splitext(os.path.basename(pdf_name))[0]
    pats = [os.path.join(mineru_root, "**", stem, "*content_list.json"),
            os.path.join(mineru_root, "**", stem + "*content_list.json")]
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=lambda x: (-os.path.getmtime(x), len(x)))
    return hits[0]


def run_mineru(pdf_path: str, out_dir: str, method: str = "txt", lang: str = "ch",
               start=None, end=None, log_path=None):
    """调用 MinerU CLI 解析 PDF。返回 (ok, log_tail)。"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "mineru", "-p", pdf_path, "-o", out_dir,
           "-b", "pipeline", "-m", method, "-l", lang]
    if start is not None:
        cmd += ["-s", str(start)]
    if end is not None:
        cmd += ["-e", str(end)]
    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    tail = (proc.stdout or "")[-4000:] + "\n" + (proc.stderr or "")[-2000:]
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(tail)
    ok = find_content_list(out_dir) is not None
    return ok, tail


def html_table_to_rows(body: str):
    """把 MinerU 的 HTML 表格解析成行列二维数组（供结构化检索/数值抽取）。"""
    if "<t" not in body:
        return []
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S | re.I):
        cells = []
        for td in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I):
            txt = re.sub(r"<[^>]+>", " ", td)
            txt = re.sub(r"\s+", " ", txt).strip()
            cells.append(txt)
        if any(cells):
            rows.append(cells)
    return rows


def table_block_text(caption, body, footnote):
    """表格块文本 = 标题 + 行结构化文本（“列1: 值 | 列2: 值”）+ 表内 HTML 压平。"""
    parts = []
    if caption:
        parts.append("表：" + caption)
    rows = html_table_to_rows(body)
    if rows:
        for r in rows:
            cells = [c for c in r]
            if cells and cells[0]:
                # 首列作行名，其余列 "列名: 值" 线性化，便于“X 是多少”的数值题命中
                head = cells[0]
                tail = " | ".join("%s: %s" % (rows[0][i] if i < len(rows[0]) else "", c)
                                  for i, c in enumerate(cells[1:], start=1) if c)
                parts.append(("%s -> %s" % (head, tail)) if tail else head)
            else:
                parts.append(" | ".join(cells))
    if body and not rows:
        parts.append(body)
    if footnote:
        parts.append("注：" + footnote)
    return "\n".join(p for p in parts if p)


def content_list_to_blocks(cl: list, doc: str = ""):
    """把 MinerU content_list 转成统一 block 结构（页码 1-based），带 doc 与表格结构化信息。"""
    blocks = []
    for item in cl:
        t = item.get("type")
        page = int(item.get("page_idx", 0)) + 1
        if t == "text":
            txt = (item.get("text") or "").strip()
            if not txt:
                continue
            blocks.append({"text": txt, "type": "text", "page": page, "doc": doc,
                           "level": item.get("text_level") or 0})
        elif t == "table":
            body = item.get("table_body") or ""
            cap = " ".join(item.get("table_caption") or [])
            fnt = " ".join(item.get("table_footnote") or [])
            rows = html_table_to_rows(body)
            text = table_block_text(cap, body, fnt)
            if text:
                blocks.append({"text": text, "type": "table", "page": page, "doc": doc,
                               "level": 0, "table_rows": rows, "caption": cap})
        elif t == "image":
            cap = " ".join(item.get("image_caption") or [])
            fnt = " ".join(item.get("image_footnote") or [])
            img_path = item.get("img_path") or item.get("image_path") or ""
            if cap or fnt or img_path:
                blocks.append({"text": (cap + " " + fnt).strip() or "（图片）",
                               "type": "image", "page": page, "doc": doc,
                               "level": 0, "img_path": img_path})
    return blocks


def blocks_from_mineru(mineru_root: str = MINERU_DIR, pdf_name: str = None, doc: str = ""):
    cl_path = find_content_list_for(mineru_root, pdf_name) if pdf_name else find_content_list(mineru_root)
    if not cl_path:
        return None, None
    with open(cl_path, encoding="utf-8") as f:
        cl = json.load(f)
    return content_list_to_blocks(cl, doc=doc), cl_path


def blocks_from_pymupdf(pdf_path: str, doc: str = ""):
    """兜底：PyMuPDF 逐页抽取。用 find_tables() 识别表格（无需 OCR），
    表格区单独成块并结构化，正文块剔除与表格重叠的部分，尽量保留数值结构。"""
    import pymupdf
    document = pymupdf.open(pdf_path)
    out = []
    for i, page in enumerate(document):
        pno = i + 1
        table_rects = []
        try:
            tabs = page.find_tables()
            for t in tabs.tables:
                table_rects.append(pymupdf.Rect(t.bbox))
                try:
                    rows = t.extract()
                except Exception:  # noqa: BLE001
                    rows = []
                lines, clean_rows = [], []
                for row in rows or []:
                    cells = ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
                    if any(cells):
                        clean_rows.append(cells)
                        lines.append(" | ".join(cells))
                if lines:
                    out.append({"text": "\n".join(lines), "type": "table", "page": pno,
                                "doc": doc, "level": 0, "table_rows": clean_rows})
        except Exception:  # noqa: BLE001
            pass

        def in_table(rect):
            return any(rect.intersects(tr) for tr in table_rects)

        for b in page.get_text("blocks"):
            x0, y0, x1, y1, txt, _no, btype = b[0], b[1], b[2], b[3], b[4], b[5], b[6]
            if btype != 0:
                continue
            txt = (txt or "").strip()
            if not txt:
                continue
            if table_rects and in_table(pymupdf.Rect(x0, y0, x1, y1)):
                continue
            out.append({"text": txt, "type": "text", "page": pno, "doc": doc, "level": 0})
    return out


def detect_title(blocks):
    """从首页文本猜公司名（用于清洗该文档的页眉；多文档各用各的）。"""
    head = " ".join(b["text"] for b in blocks[:40] if b.get("type") == "text")[:600]
    m = re.search(r"([\u4e00-\u9fff]{2,20}?(?:股份)?有限公司)", head)
    return m.group(1) if m else ""


def build_blocks(pdf_path=None, mineru_root=None, prefer="mineru"):
    """解析单本 PDF，产出统一 blocks 并落盘（含每文档文件）。返回 (blocks, source, title)。"""
    pdf_path = pdf_path or DEFAULT_PDF
    mineru_root = mineru_root or MINERU_DIR
    doc = doc_of(pdf_path)
    blocks, source = None, None
    if prefer == "mineru":
        blocks, cl_path = blocks_from_mineru(mineru_root, os.path.basename(pdf_path), doc=doc)
        if not blocks:  # 兼容旧目录结构
            blocks, cl_path = blocks_from_mineru(mineru_root, None, doc=doc)
        source = ("mineru:" + cl_path) if blocks else None
    if not blocks:
        blocks = blocks_from_pymupdf(pdf_path, doc=doc)
        source = "pymupdf"
    title = detect_title(blocks)
    for b in blocks:
        b["doc"] = doc
        b["title"] = title
    os.makedirs(PARSED_DIR, exist_ok=True)
    with open(blocks_path(doc), "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    return blocks, source, title


IMAGE_BLOCKS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "vision", "image_blocks.jsonl")


def load_image_blocks():
    """读取 CLIP 生成的图像语义块（工单 04，tools/vision_build.py 产出）。"""
    if not os.path.isfile(IMAGE_BLOCKS_PATH):
        return []
    out = []
    for line in open(IMAGE_BLOCKS_PATH, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_all_blocks(pdfs=None, mineru_root=None, prefer="mineru", with_images=True):
    """解析 data/pdfs 下全部 PDF，并（可选）并入 CLIP 图像块，汇总写入 blocks.jsonl。"""
    pdfs = pdfs or list_pdfs()
    all_blocks, sources = [], {}
    for p in pdfs:
        blocks, source, title = build_blocks(p, mineru_root=mineru_root, prefer=prefer)
        all_blocks += blocks
        sources[doc_of(p)] = (source, len(blocks), title)
    if with_images:
        imgs = load_image_blocks()
        for b in imgs:
            b.setdefault("title", "")
        all_blocks += imgs
        if imgs:
            sources["(image blocks)"] = ("clip", len(imgs), "")
    os.makedirs(PARSED_DIR, exist_ok=True)
    with open(BLOCKS_PATH, "w", encoding="utf-8") as f:
        for b in all_blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    return all_blocks, sources


def load_blocks(path=None):
    out = []
    with open(path or BLOCKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    blocks, sources = build_all_blocks()
    docs = {}
    for b in blocks:
        docs.setdefault(b.get("doc") or "?", set()).add(b["page"])
    print("sources:", sources)
    for d, ps in docs.items():
        print("doc=%-16s blocks=%d pages=%d" % (d, sum(1 for b in blocks if b.get("doc") == d), len(ps)))
