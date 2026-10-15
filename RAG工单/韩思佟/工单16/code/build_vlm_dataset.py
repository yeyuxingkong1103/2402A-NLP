# -*- coding: utf-8 -*-
"""
工单16 —— VLM 微调数据转换脚本

把 IMDR 的 QA 对 + RAGFlow 解析出的图文块，转换成 LLaMA-Factory（Qwen2-VL）所需
的微调格式：JSONL，每行含 image / question(conversations) / answer。

输入：
  --qa   questions.jsonl（IMDR 原始问答，字段：question/document/answer[/options]）
  --docs documents/ 目录（图片型 PDF）或 RAGFlow 导出的 chunk 目录
  --out  sharegpt.jsonl

输出两种格式（可切换）：
  1) sharegpt（LLaMA-Factory 推荐）：
     {"messages":[{"role":"user","content":"<image>问题"},{"role":"assistant","content":"答案"}],
      "images":["abs/path/xxx.png"]}
  2) alpaca：
     {"instruction":"问题","input":"","output":"答案","image":"abs/path/xxx.png"}

同时导出工业术语词典（用于评估的"专业术语准确性"）与统计报告。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

TERM_HINTS = ["淬火", "公差配合", "退火", "回火", "渗碳", "轴承", "齿轮", "螺栓",
              "阀", "泵", "轴", "壳体", "密封", "润滑", "力矩", "浇注", "模具",
              "除尘器", "筛板", "带孔盘", "圆锥", "法兰", "焊接"]


def pdf_to_images(pdf: Path, out_dir: Path, dpi: int = 150, max_pages: int = 4):
    """把 PDF 每页渲染成 PNG（图片型 PDF 的"图像"就是整页）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    paths = []
    for i in range(min(doc.page_count, max_pages)):
        out = out_dir / f"{pdf.stem}_p{i+1}.png"
        if not out.exists():
            doc[i].get_pixmap(dpi=dpi).save(str(out))
        paths.append(str(out.resolve()))
    doc.close()
    return paths


def pick_page_for_question(q: dict) -> int:
    """从问题里猜页码（"第7页"/"图3"），用于选最相关的页面图"""
    m = re.search(r"第\s*(\d+)\s*页", q.get("question", ""))
    return int(m.group(1)) if m else 1


def build(qa_path: Path, docs_dir: Path, out_path: Path, img_dir: Path,
          fmt: str = "sharegpt", max_items: int | None = None,
          page_mode: str = "all") -> dict:
    qas = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    qas.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if max_items:
        qas = qas[:max_items]

    records, term_counter, missing_doc = [], Counter(), 0
    for q in qas:
        doc_name = q.get("document") or q.get("doc") or ""
        pdf = docs_dir / doc_name
        if not pdf.exists():
            missing_doc += 1
            continue
        imgs = pdf_to_images(pdf, img_dir)
        if page_mode == "hit":
            pg = pick_page_for_question(q)
            imgs = [p for p in imgs if f"_p{pg}.png" in p] or imgs[:1]
        question = q.get("question", "").strip()
        answer = (q.get("answer") or "").strip()
        if not question or not answer:
            continue
        for t in TERM_HINTS:
            if t in question + answer:
                term_counter[t] += 1

        if fmt == "sharegpt":
            records.append({
                "messages": [
                    {"role": "user", "content": "<image>" + question},
                    {"role": "assistant", "content": answer},
                ],
                "images": imgs if page_mode == "all" else imgs[:1],
                "meta": {"document": doc_name, "source": "IMDR"},
            })
        else:
            records.append({
                "instruction": question, "input": "", "output": answer,
                "image": imgs[0], "meta": {"document": doc_name},
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "QA原始条数": len(qas),
        "成功转换": len(records),
        "缺失文档": missing_doc,
        "格式": fmt,
        "图片目录": str(img_dir),
        "输出": str(out_path),
        "出现的工业术语Top20": term_counter.most_common(20),
    }
    (out_path.parent / "数据集统计.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent.parent
    ap.add_argument("--qa", default=str(base / "data" / "imdr" / "questions.jsonl"))
    ap.add_argument("--docs", default=str(base / "data" / "imdr"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent /
                                         "产出物" / "vlm_sft_sharegpt.jsonl"))
    ap.add_argument("--img-dir", default=str(Path(__file__).resolve().parent.parent /
                                             "产出物" / "images"))
    ap.add_argument("--format", default="sharegpt", choices=["sharegpt", "alpaca"])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--page-mode", default="all", choices=["all", "hit"])
    args = ap.parse_args()
    rep = build(Path(args.qa), Path(args.docs), Path(args.out), Path(args.img_dir),
                args.format, args.max_items, args.page_mode)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
