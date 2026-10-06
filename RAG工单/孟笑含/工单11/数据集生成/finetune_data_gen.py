# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
模块：微调数据集生成
功能：从 PDF 抽取 chunk，用 Qwen2.5-VL-3B 生成 (query, positive_doc) 对
"""

import os
import json
import re
import random
import torch
import fitz
from typing import List, Dict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

os.environ['HF_HUB_DISABLE_XET'] = '1'


class FinetuneDataGenerator:
    """微调数据集生成器"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"正在加载 LLM：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✅ LLM 加载完成")

    def extract_chunks(self, pdf_paths: List[str], chunk_size: int = 500) -> List[Dict]:
        """从 PDF 抽取 chunks"""
        all_chunks = []
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), 30)):  # 每个 PDF 取前 30 页
                text = doc[page_num].get_text()
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < chunk_size:
                    continue
                # 切块
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) >= 200:
                        all_chunks.append({
                            "doc": doc_name,
                            "page": page_num + 1,
                            "content": chunk,
                        })
            doc.close()
        print(f"✅ 抽取 {len(all_chunks)} 个 chunks")
        return all_chunks

    def generate_question(self, chunk: str) -> str:
        """用 LLM 从 chunk 生成一个问题"""
        prompt = f"""请根据以下文档内容，生成一个能用该内容回答的中文问题。

要求：
1. 问题要具体，涉及文档中的关键信息
2. 只输出问题，不要回答
3. 不要包含"根据文档"等词

文档内容：
{chunk[:800]}

问题："""
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        q = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return q.strip().split("\n")[0]

    def generate_dataset(self, pdf_paths: List[str], target_size: int = 500,
                         output_path: str = "train_data.json"):
        """生成微调数据集"""
        chunks = self.extract_chunks(pdf_paths)
        random.shuffle(chunks)
        chunks = chunks[:target_size]

        data = []
        for i, chunk in enumerate(chunks):
            try:
                q = self.generate_question(chunk["content"])
                if len(q) >= 5:
                    data.append({
                        "query": q,
                        "positive": chunk["content"],
                        "doc": chunk["doc"],
                        "page": chunk["page"],
                    })
                    if (i + 1) % 50 == 0:
                        print(f"  已生成 {i+1}/{len(chunks)}")
            except Exception as e:
                print(f"  生成失败：{e}")
                continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 数据集已保存：{output_path}（{len(data)} 对）")
        return data


if __name__ == "__main__":
    PDF_PATHS = [
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ]
    # 加 9 个年报
    for f in sorted(os.listdir("./data/ccf_competition/pdf")):
        if f.endswith(".pdf"):
            PDF_PATHS.append(f"./data/ccf_competition/pdf/{f}")

    print(f"共 {len(PDF_PATHS)} 个 PDF")
    gen = FinetuneDataGenerator()
    gen.generate_dataset(PDF_PATHS, target_size=500, output_path="train_data.json")
