# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
模块：图像解析模块（Qwen2.5-VL 多模态）
功能：PDF页面渲染为图 + 多模态模型解析图像语义
"""

import os
import fitz
import torch
from typing import List, Dict, Any
from PIL import Image

os.environ['HF_HUB_DISABLE_XET'] = '1'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ImageParser:
    """图像解析器（Qwen2.5-VL-3B）"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"正在加载多模态模型：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✅ 多模态模型加载完成")

    def render_page_to_image(self, pdf_path: str, page_num: int,
                              dpi: int = 300, out_dir: str = "/tmp/pdf_images") -> str:
        """渲染 PDF 某页为 PNG"""
        os.makedirs(out_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}_p{page_num}.png")
        pix.save(img_path)
        return img_path

    def ask_image(self, img_path: str, question: str) -> str:
        """向多模态模型提问"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    def describe_page(self, pdf_path: str, page_num: int) -> str:
        """通用描述：这一页的图里有什么"""
        img_path = self.render_page_to_image(pdf_path, page_num)
        q = "请详细描述这张图片中的所有内容，包括文字、图表、结构、层级关系。"
        return self.ask_image(img_path, q)

    def answer_from_image(self, pdf_path: str, page_num: int, question: str) -> str:
        """针对问题从图中找答案"""
        img_path = self.render_page_to_image(pdf_path, page_num)
        prompt = f"请仔细观察这张图片，并回答问题：{question}\n\n直接给出答案，不要解释。"
        return self.ask_image(img_path, prompt)


if __name__ == "__main__":
    parser = ImageParser()

    print("\n" + "=" * 70)
    print("【ID:5】组织结构图（第39页）")
    print("=" * 70)
    q5 = "请仔细阅读图中组织结构图的全部文字。销售部下设哪几个部门？请逐个列出。大客户销售部下设哪几个销售处？请逐个列出每个销售处的名称。"
    ans5 = parser.answer_from_image("./data/招股说明书2.pdf", 39, q5)
    print(ans5)

    print("\n" + "=" * 70)
    print("【ID:6】IC市场图（第72页）")
    print("=" * 70)
    q6 = "请仔细阅读图中柱状图和文字标签。1) 增长率最快的是哪个行业？2) 负增长的是哪个行业？请根据图中数字给出答案。"
    ans6 = parser.answer_from_image("./data/招股说明书2.pdf", 72, q6)
    print(ans6)
