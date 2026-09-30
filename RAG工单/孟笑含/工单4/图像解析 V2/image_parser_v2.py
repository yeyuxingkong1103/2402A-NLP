# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
模块：图像解析模块 V2（裁剪区域增强）
功能：只渲染图表关键区域，提高 Qwen-VL 识别率
"""

import os
import fitz
import torch

os.environ['HF_HUB_DISABLE_XET'] = '1'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ImageParserV2:
    """图像解析器 V2（裁剪区域版）"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"正在加载多模态模型：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✅ 多模态模型加载完成")

    def render_region(self, pdf_path, page_num, clip_rect, dpi=400):
        out_dir = "/tmp/pdf_images"
        os.makedirs(out_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
        img_path = os.path.join(out_dir, f"region_p{page_num}.png")
        pix.save(img_path)
        return img_path

    def ask_image(self, img_path, question):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    parser = ImageParserV2()

    # ========== ID5：裁切销售部区域 ==========
    print("\n" + "=" * 70)
    print("【ID:5】销售部区域（页面 y 320~560）")
    print("=" * 70)
    clip = fitz.Rect(0, 300, 595, 640)
    img = parser.render_region("./data/招股说明书2.pdf", 39, clip, dpi=400)
    print(f"裁切图：{img}")
    q5 = "请仔细阅读图中所有文字。1) 销售部下设哪几个部门？逐个列出。2) 大客户销售部下设哪几个销售处？逐个列出每个销售处名称。"
    print(parser.ask_image(img, q5))


    # ========== ID6：裁切 IC 市场图区域 ==========
    print("\n" + "=" * 70)
    print("【ID:6】IC 市场图区域（第72页，y 280~500）")
    print("=" * 70)
    clip2 = fitz.Rect(50, 280, 545, 500)
    img2 = parser.render_region("./data/招股说明书2.pdf", 72, clip2, dpi=400)
    print(f"裁切图：{img2}")
    q6 = "请仔细阅读图中柱状图和文字标签。1) 增长率最快的是哪个行业？2) 负增长的是哪个行业？请根据图中数字给出答案。"
    print(parser.ask_image(img2, q6))
