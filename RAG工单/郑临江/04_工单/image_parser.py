# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
图像语义解析模块：使用多模态模型理解 PDF 中的图片。

支持两种后端：
  1. CLIP（本地）：对图片与候选标签做图文匹配，识别图片语义类型；
  2. 多模态大模型（OpenAI 兼容 vision API）：直接生成图片描述。

未安装/未配置时，降级为“上下文描述”（图片所在页 + 附近文字）。
"""
import base64
import os


class ClipImageParser:
    """使用 CLIP 进行图像语义解析（需要 transformers + torch）。"""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            pass

    def available(self):
        return self.model is not None

    def classify(self, image_path: str, labels):
        """返回 (最匹配标签, 相似度得分)。"""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=labels, images=img, return_tensors="pt", padding=True)
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])


class MultimodalLLMParser:
    """使用多模态大模型生成图片描述（OpenAI 兼容 vision API）。"""

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("VISION_MODEL", "gpt-4o-mini")

    def available(self):
        return bool(self.api_key)

    def describe(self, image_path: str, question: str = "") -> str:
        import urllib.request
        import json
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        prompt = question or "请详细描述这张图片的内容，包括其中的文字、数据与结构。"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def describe_with_context(info, page_text_map):
    """降级方案：用图片所在页面的文字作为上下文描述。"""
    pno = info["page"]
    ctx = page_text_map.get(pno, "")
    return f"[图片] 位于第{pno}页，附近文字：{ctx[:300]}"


def build_image_descriptions(image_infos, page_text_map, labels, question=None):
    """
    为每张图片生成语义描述。优先多模态大模型，其次 CLIP，最后上下文降级。
    返回 [(描述文本, 图片路径, 页码)]。
    """
    mm = MultimodalLLMParser()
    clip = ClipImageParser()
    descriptions = []
    for info in image_infos:
        desc = None
        if mm.available():
            try:
                desc = mm.describe(info["path"], question or "")
            except Exception:
                desc = None
        if desc is None and clip.available():
            try:
                label, score = clip.classify(info["path"], labels)
                desc = f"[图片] 语义类型：{label}（置信度 {score:.2f}）"
            except Exception:
                desc = None
        if desc is None:
            desc = describe_with_context(info, page_text_map)
        descriptions.append((desc, info["path"], info["page"]))
    return descriptions
