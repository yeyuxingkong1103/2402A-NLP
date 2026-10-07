# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单 - 数据转换脚本（全文档版）"""

import json
import os
import re
import fitz
from tqdm import tqdm


def pdf_page_to_image(pdf_path, page_num, out_dir="./finetune_data/images", dpi=150):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    img_name = f"{os.path.basename(pdf_path)}_p{page_num}.png"
    img_path = os.path.join(out_dir, img_name)
    pix.save(img_path)
    doc.close()
    return img_path


def main():
    qa_path = "./data/original_problems/original_problems/questions.jsonl"
    doc_dir = "./data/original_problems/original_problems/documents"

    # 1. 读取所有 QA 对
    with open(qa_path, "r", encoding="utf-8") as f:
        all_qa = [json.loads(line) for line in f]
    print(f"总 QA 对：{len(all_qa)}")

    # 2. 按文档分组，优先取视觉问题
    docs = {}
    for qa in all_qa:
        doc = qa.get("document", "")
        if doc not in docs:
            docs[doc] = []
        docs[doc].append(qa)

    print(f"总文档数：{len(docs)}")

    # 3. 按文档轮流取 QA，凑够 100 条
    target_size = 100
    qa_pairs = []
    doc_list = list(docs.keys())
    max_per_doc = 3
    for i in range(max_per_doc):
        for doc in doc_list:
            if len(qa_pairs) >= target_size:
                break
            if i < len(docs[doc]):
                qa_pairs.append(docs[doc][i])
        if len(qa_pairs) >= target_size:
            break

    print(f"选中 QA 对：{len(qa_pairs)}")

    # 4. 转换成微调格式
    train_data = []
    visual_count = 0
    text_count = 0

    for qa in tqdm(qa_pairs):
        doc_name = qa["document"]
        question = qa["question"]
        answer = qa["answer"]
        options = qa.get("options", [])

        answer_text = answer
        for opt in options:
            if opt.startswith(answer + ".") or opt.startswith(answer + " "):
                answer_text = opt
                break

        pdf_path = os.path.join(doc_dir, doc_name)
        if not os.path.exists(pdf_path):
            continue

        has_visual = any(kw in question for kw in ["图", "页", "编号", "部件"])

        image_path = None
        if has_visual:
            m = re.search(r"第\s*(\d+)\s*页", question)
            page_num = int(m.group(1)) if m else 1
            try:
                image_path = pdf_page_to_image(pdf_path, page_num)
                visual_count += 1
            except Exception as e:
                print(f"  提取图失败：{e}")
                image_path = None
        else:
            text_count += 1

        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})

        train_data.append({
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": answer_text}
            ]
        })

    # 5. 保存
    os.makedirs("./finetune_data", exist_ok=True)
    output_path = "./finetune_data/train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 已保存：{output_path}（{len(train_data)} 条）")
    print(f"  视觉问题：{visual_count}")
    print(f"  文本问题：{text_count}")


if __name__ == "__main__":
    main()
