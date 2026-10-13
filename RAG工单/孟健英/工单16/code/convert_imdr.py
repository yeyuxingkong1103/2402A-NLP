# 工单编号：人工智能 NLP-RAG-微调专用视觉语言模型工单
"""把 IMDR questions.jsonl 转成 LLaMA-Factory JSONL 格式"""

import json
import os
import re
import random
from pathlib import Path

import pymupdf

# 配置
IMDR_DIR = "/mnt/d/专高六 工单/工单14/original_problems"
QUESTIONS = os.path.join(IMDR_DIR, "questions.jsonl")
DOCS_DIR = os.path.join(IMDR_DIR, "documents")

OUT_DIR = os.path.expanduser("~/LLaMA-Factory/data")
IMG_DIR = os.path.join(OUT_DIR, "imdr_images")
OUT_FILE = os.path.join(OUT_DIR, "imdr_500.json")

SAMPLE_SIZE = 500
os.makedirs(IMG_DIR, exist_ok=True)
random.seed(42)


def extract_page_num(question):
    """从问题里提取页码，如 '第7页' → 7"""
    m = re.search(r"第\s*(\d+)\s*页", question)
    return int(m.group(1)) if m else 1


def build_answer(item):
    """把答案字母拼成完整选项"""
    ans_letter = item["answer"]
    options = item["options"]
    for opt in options:
        if opt.startswith(ans_letter + "."):
            return opt
    # 兜底
    idx = ord(ans_letter) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return ans_letter


def convert_one(item, idx):
    """转换一条"""
    doc_name = item["document"]
    pdf_path = os.path.join(DOCS_DIR, doc_name)
    if not os.path.exists(pdf_path):
        return None

    page_num = extract_page_num(item["question"])

    try:
        doc = pymupdf.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            page_num = 1
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        img_name = f"imdr_{idx:05d}_{doc_name.replace('.pdf','')}_p{page_num}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        pix.save(img_path)
        doc.close()
    except Exception as e:
        print(f"  [WARN] PDF 处理失败 {doc_name}: {e}")
        return None

    # 拼选项
    opts_text = "\n".join(item["options"])
    user_content = f"<image>{item['question']}\n{opts_text}"
    assistant_content = build_answer(item)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [img_path],
    }


def main():
    print("读取 questions.jsonl...")
    with open(QUESTIONS, encoding="utf-8") as f:
        all_items = [json.loads(l) for l in f if l.strip()]
    print(f"总条数: {len(all_items)}")

    random.shuffle(all_items)
    sample = all_items[:SAMPLE_SIZE]
    print(f"抽样: {len(sample)} 条")

    results = []
    for i, item in enumerate(sample):
        r = convert_one(item, i)
        if r:
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(sample)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(results)} 条 → {OUT_FILE}")
    print(f"图像: {IMG_DIR}")


if __name__ == "__main__":
    main()
