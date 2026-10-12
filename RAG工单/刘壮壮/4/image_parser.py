# -*- coding: utf-8 -*-
"""
image_parser.py — PDF 图像内容解析模块
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

把 PDF 中的图像（组织结构图、统计图表等）解析为可检索的文本描述，
使用多模态模型（多模态大模型 / CLIP）实现图像语义解析。

两种实现：
  1. 多模态大模型（OpenAI 兼容视觉接口，如 gpt-4o / qwen-vl / glm-4v）：
     直接把图像送入 VLM 生成自然语言描述，描述文本参与检索。
  2. CLIP：用 CLIP 将图像与文本映射到同一向量空间，实现图文互检索。

无多模态模型时的兜底：以图像所在页的正文文本作为近似描述（文字型图表仍可命中）。
"""
from __future__ import annotations

import base64
import os
from typing import List, Dict, Any, Optional

import rag_core


def extract_images(pdf_path: str) -> List[Dict[str, Any]]:
    """提取 PDF 中的图像，附带页码与所在页文本（作为兜底上下文）。

    返回：[{page_no, width, height, bytes, page_text}]
    """
    pages = rag_core.load_pdf_pages(pdf_path)
    images: List[Dict[str, Any]] = []
    for p in pages:
        for img in p["images"]:
            images.append({
                "page_no": p["page_no"],
                "width": img["width"],
                "height": img["height"],
                "bytes": img["bytes"],
                "page_text": p["text"][:2000],
            })
    return images


class MultimodalLLM:
    """OpenAI 兼容多模态大模型（视觉理解）。"""

    def __init__(self, api_base: str, api_key: str, model: str):
        from openai import OpenAI
        kwargs = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model

    def describe(self, image_bytes: bytes, prompt: str = "请详细描述这张图片的内容，包括其中的人物、结构、数据和图表信息。") -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class CLIPImageParser:
    """基于 CLIP 的图像解析（需要 transformers + torch）。"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image_bytes_list: List[bytes]) -> "list":
        import io
        from PIL import Image
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().numpy()

    def embed_text(self, texts: List[str]) -> "list":
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().numpy()


def describe_images(images: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
    """为图像生成文本描述（优先多模态大模型，回退到页文本）。"""
    results = []
    vlm = None
    if config.llm_api_base and config.llm_api_key:
        try:
            vlm = MultimodalLLM(config.llm_api_base, config.llm_api_key,
                                config.llm_model)
        except Exception:
            vlm = None

    for img in images:
        desc = None
        if vlm is not None:
            try:
                desc = vlm.describe(img["bytes"])
            except Exception:
                desc = None
        if not desc:
            # 兜底：用图像所在页文本作为近似描述（图表标题、图注等）
            desc = f"[图片/第{img['page_no']}页，{img['width']}x{img['height']}px] " + img["page_text"][:500]
        results.append({
            "page_no": img["page_no"],
            "description": desc,
            "source": "vlm" if vlm else "page_text",
        })
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../附件/招股说明书2.pdf"
    imgs = extract_images(path)
    print(f"共提取 {len(imgs)} 张图像")
    for i in imgs[:5]:
        print(f"第 {i['page_no']} 页：{i['width']}x{i['height']}px")
