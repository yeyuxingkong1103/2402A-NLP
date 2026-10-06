# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：parse —— PDF 解析（MinerU 主通道 + PyMuPDF 兜底）
# 说明：调用 MinerU CLI 解析 PDF，读取其 content_list.json，转成统一的
#       {text, type, page, level} blocks；MinerU 不可用时回退 PyMuPDF 逐页抽取。

import os
import sys
import json
import glob
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MINERU_DIR, PARSED_DIR, BLOCKS_PATH  # noqa: E402


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
    # 取最外层/最新的一个
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
    # MinerU 偶发以非 0 退出码正常完成，故以“是否产出 content_list”为准
    ok = find_content_list(out_dir) is not None
    return ok, tail


def content_list_to_blocks(cl: list):
    """把 MinerU content_list 转成统一 block 结构（页码 1-based）。"""
    blocks = []
    for item in cl:
        t = item.get("type")
        page = int(item.get("page_idx", 0)) + 1
        if t == "text":
            txt = (item.get("text") or "").strip()
            if not txt:
                continue
            blocks.append({
                "text": txt, "type": "text", "page": page,
                "level": item.get("text_level") or 0,
            })
        elif t == "table":
            body = item.get("table_body") or ""
            cap = " ".join(item.get("table_caption") or [])
            fnt = " ".join(item.get("table_footnote") or [])
            parts = [x for x in (cap, body, fnt) if x]
            if parts:
                blocks.append({"text": "\n".join(parts), "type": "table", "page": page, "level": 0})
        elif t == "image":
            cap = " ".join(item.get("image_caption") or [])
            fnt = " ".join(item.get("image_footnote") or [])
            if cap or fnt:
                blocks.append({"text": (cap + " " + fnt).strip(), "type": "image",
                               "page": page, "level": 0})
    return blocks


def blocks_from_mineru(mineru_root: str = MINERU_DIR):
    cl_path = find_content_list(mineru_root)
    if not cl_path:
        return None, None
    with open(cl_path, encoding="utf-8") as f:
        cl = json.load(f)
    return content_list_to_blocks(cl), cl_path


def blocks_from_pymupdf(pdf_path: str):
    """兜底：PyMuPDF 逐页抽取。用 find_tables() 识别表格（无需 OCR），
    表格区单独成块，正文块剔除与表格重叠的部分，尽量保留数值结构。
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
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
                lines = []
                for row in rows or []:
                    cells = ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
                    if any(cells):
                        lines.append(" | ".join(cells))
                if lines:
                    out.append({"text": "\n".join(lines), "type": "table",
                                "page": pno, "level": 0})
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
            out.append({"text": txt, "type": "text", "page": pno, "level": 0})
    return out


def build_blocks(pdf_path=None, mineru_root=None, prefer="mineru"):
    """产出统一 blocks 并落盘。返回 (blocks, source)。"""
    from config import DEFAULT_PDF
    pdf_path = pdf_path or DEFAULT_PDF
    mineru_root = mineru_root or MINERU_DIR
    blocks, source = None, None
    if prefer == "mineru":
        blocks, cl_path = blocks_from_mineru(mineru_root)
        source = ("mineru:" + cl_path) if blocks else None
    if not blocks:
        blocks = blocks_from_pymupdf(pdf_path)
        source = "pymupdf"
    os.makedirs(PARSED_DIR, exist_ok=True)
    with open(BLOCKS_PATH, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    return blocks, source


def load_blocks():
    out = []
    with open(BLOCKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    blocks, source = build_blocks()
    pages = sorted({b["page"] for b in blocks})
    print("source:", source)
    print("blocks:", len(blocks), "| pages:", len(pages), "| range:", pages[0], "-", pages[-1])
