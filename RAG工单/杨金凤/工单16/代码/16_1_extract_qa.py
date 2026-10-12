"""
从 IMDR 抽取视觉类 QA 对并抽图。
输入：original_problems/questions.jsonl + documents/*.pdf
输出：finetune_data/intermediate_qa.jsonl + finetune_data/images/*.png
"""
import os
import re
import json
import fitz
from tqdm import tqdm

IMDR_QUESTIONS = "/root/autodl-tmp/original_problems/questions.jsonl"
DOCS_DIR = "/root/autodl-tmp/original_problems/documents"
OUT_DIR = "/root/autodl-tmp/ragflow/finetune_data"
IMG_DIR = os.path.join(OUT_DIR, "images")
MAX_SAMPLES = 200  # 先用 200 条做快速实验

os.makedirs(IMG_DIR, exist_ok=True)


def parse_page(question: str):
    m = re.search(r'第\s*(\d+)\s*页', question)
    return int(m.group(1)) if m else None


def extract_page(pdf_path: str, page_num: int, dpi: int = 200):
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception as e:
        print(f"[warn] extract failed {pdf_path} p{page_num}: {e}")
        return None


def main():
    samples = []
    with open(IMDR_QUESTIONS, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 只取视觉类问题（含"第X页"），前 MAX_SAMPLES 条
    for line in tqdm(lines, desc="Processing"):
        if len(samples) >= MAX_SAMPLES:
            break
        try:
            item = json.loads(line)
        except Exception:
            continue

        question = item.get("question", "").strip()
        doc_name = item.get("document", "").strip()
        options = item.get("options", [])
        answer = item.get("answer", "").strip()

        if not question or not doc_name or not options or not answer:
            continue

        # 只处理含"第X页"的视觉问题
        page = parse_page(question)
        if page is None:
            continue

        pdf_path = os.path.join(DOCS_DIR, doc_name)
        if not os.path.exists(pdf_path):
            continue

        # 抽图
        img_bytes = extract_page(pdf_path, page)
        if not img_bytes:
            continue

        img_name = f"{doc_name.replace('.pdf','')}_p{page}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as f_img:
                f_img.write(img_bytes)

        # 拼接标准答案文本（选项字母 + 选项内容）
        answer_text = ""
        for opt in options:
            if opt.strip().startswith(answer + "."):
                answer_text = opt.strip()
                break
        if not answer_text:
            answer_text = answer

        # 拼接完整问题（含选项）
        full_question = question + "\n" + "\n".join(options)

        samples.append({
            "image": os.path.relpath(img_path, OUT_DIR),
            "question": full_question,
            "answer": answer_text,
            "doc": doc_name,
            "page": page,
        })

    out_path = os.path.join(OUT_DIR, "intermediate_qa.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 共生成 {len(samples)} 条视觉 QA 样本")
    print(f"   输出：{out_path}")
    print(f"   图片：{IMG_DIR}")


if __name__ == "__main__":
    main()
