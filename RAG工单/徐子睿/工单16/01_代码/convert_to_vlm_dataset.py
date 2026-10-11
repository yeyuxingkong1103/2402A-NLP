# -*- coding: utf-8 -*-
"""
工单 16 · 数据格式转换脚本：IMDR QA + 图文块 → VLM 微调 JSONL
（人工智能NLP-RAG-调试与微调专用视觉语言模型）

输出格式（LLaMA-Factory VLM / LLaVA 风格）：
  {"messages":[{"role":"user","content":"<image>问题+选项"},
               {"role":"assistant","content":"标准答案（含术语解释）"}],
   "images":["images/xxx_p7.png"],
   "meta":{"document":"...","page":7,"group":2}}

用法：
  python convert_to_vlm_dataset.py \
    --documents <IMDR documents 目录> \
    --questions <questions.jsonl> \
    --out <输出 jsonl 路径> \
    [--images-dir <图片输出目录>] [--limit 200] [--zoom 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import pymupdf  # PyMuPDF

RE_PAGE = re.compile(r"第\s*(\d{1,4})\s*页")


def parse_page(question: str) -> Optional[int]:
    m = RE_PAGE.search(question)
    return int(m.group(1)) if m else None


def render_page(pdf_path: str, page_no: int, out_png: str, zoom: float = 2.0) -> bool:
    """把 PDF 的第 page_no 页（1-based）渲染为 PNG。"""
    try:
        doc = pymupdf.open(pdf_path)
        idx = page_no - 1
        if not (0 <= idx < len(doc)):
            idx = 0
        pix = doc[idx].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        pix.save(out_png)
        doc.close()
        return True
    except Exception as e:  # 损坏的 PDF 兜底
        print(f"[WARN] render failed {pdf_path} p{page_no}: {e}")
        return False


def build_answer(item: dict) -> str:
    """把选择题的答案字母还原成完整选项文本，并保留字母。"""
    opts = item.get("options") or []
    ans = str(item.get("answer", "")).strip()
    for o in opts:
        if o.startswith(ans + ".") or o.startswith(ans + "、"):
            return f"{o}"
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--zoom", type=float, default=2.0)
    args = ap.parse_args()

    images_dir = args.images_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for line in open(args.questions, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            item = json.loads(line)
            doc_name = item.get("document")
            pdf_path = os.path.join(args.documents, doc_name)
            if not os.path.exists(pdf_path):
                continue
            page_no = parse_page(item["question"]) or 1
            stem = os.path.splitext(doc_name)[0]
            png_name = f"{stem}_p{page_no}.png"
            png_path = os.path.join(images_dir, png_name)
            if not os.path.exists(png_path):
                if not render_page(pdf_path, page_no, png_path, args.zoom):
                    continue

            user = item["question"]
            if item.get("options"):
                user += "\n" + "\n".join(item["options"])
            rec = {
                "messages": [
                    {"role": "user", "content": "<image>" + user},
                    {"role": "assistant", "content": build_answer(item)},
                ],
                "images": [os.path.relpath(png_path, os.path.dirname(os.path.abspath(args.out))).replace("\\", "/")],
                "meta": {"document": doc_name, "page": page_no, "group": item.get("group")},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"questions={n_in}  written={n_out}  ->  {args.out}")


if __name__ == "__main__":
    main()
