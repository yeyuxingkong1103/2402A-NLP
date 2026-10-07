# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
模块：多模态 PDF 解析器
功能：用 Qwen2.5-VL 解析图像型 PDF（无文字层）
"""

import os
import fitz
import torch
from typing import List, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info

os.environ['HF_HUB_DISABLE_XET'] = '1'


class MultimodalPDFParser:
    """多模态 PDF 解析器（图像型 PDF → 文本）"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"正在加载多模态模型：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✅ 多模态模型加载完成")

    def render_page_to_image(self, pdf_path: str, page_num: int,
                              dpi: int = 200, out_dir: str = "/tmp/pdf_images") -> str:
        """渲染 PDF 某页为 PNG（高 DPI 保证清晰度）"""
        os.makedirs(out_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}_p{page_num}.png")
        pix.save(img_path)
        doc.close()
        return img_path

    def ask_image(self, img_path: str, question: str) -> str:
        """向多模态模型提问"""
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """解析整个 PDF → 每页一个文本块"""
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()

        print(f"共 {n_pages} 页，逐页用 Qwen-VL 解析...")
        chunks = []
        for page_num in range(1, n_pages + 1):
            try:
                img_path = self.render_page_to_image(pdf_path, page_num, dpi=200)
                question = "请完整提取这张专利说明书页面中的所有文字内容，包括标题、编号、正文、图表中的标注。保持原文格式。"
                text = self.ask_image(img_path, question)
                chunks.append({
                    "chunk_id": f"cn_pdf_p{page_num}",
                    "content": f"[专利页 {page_num}]\n{text}",
                    "page": page_num,
                    "source": f"{pdf_path}#page={page_num}",
                    "doc": os.path.basename(pdf_path),
                    "type": "multimodal_pdf",
                })
                print(f"  第 {page_num}/{n_pages} 页：{len(text)} 字符")
            except Exception as e:
                print(f"  第 {page_num} 页失败：{e}")
        return chunks


if __name__ == "__main__":
    parser = MultimodalPDFParser()
    chunks = parser.parse_pdf("./data/CN100342976C.pdf")

    # 保存
    import json
    with open("cn_pdf_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 共解析 {len(chunks)} 页，已保存 cn_pdf_chunks.json")

    # 打印第 1 页
    if chunks:
        print(f"\n===== 第 1 页内容预览 =====")
        print(chunks[0]["content"][:500])
