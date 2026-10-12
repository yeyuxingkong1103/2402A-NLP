# -*- coding: utf-8 -*-
"""
工单十四：Qwen2-VL 多模态引擎
职责：
  1. OCR：识别图片型 PDF 页面中的文字（解决无文本层 PDF 的信息丢失）
  2. VQA：对技术图纸进行视觉问答（部件位置、尺寸含义、气流方向等）
"""
import os
import time
from typing import List, Optional
from loguru import logger
from PIL import Image

from src.config import VLM_MODEL_DIR, VLM_DEVICE

_model = None
_processor = None
_device = None


def _load():
    global _model, _processor, _device
    if _model is not None:
        return
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    _device = VLM_DEVICE if VLM_DEVICE in ("cuda", "cpu") else (
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"加载 Qwen2-VL: {VLM_MODEL_DIR} device={_device}")
    _processor = AutoProcessor.from_pretrained(VLM_MODEL_DIR, trust_remote_code=True)
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_DIR, torch_dtype=torch.float16, device_map=_device)
    _model.eval()
    logger.info("Qwen2-VL 就绪")


def generate(image_path: str, prompt: str, max_new_tokens: int = 1024) -> str:
    """单图单轮问答"""
    _load()
    import torch
    image = Image.open(image_path).convert("RGB")
    # 限制图像最大边长，加速推理
    max_side = 1024
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[image], return_tensors="pt").to(_model.device)
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def ocr_image(image_path: str) -> str:
    """OCR：完整识别图片中的所有文字"""
    prompt = "请完整识别图片中的所有文字内容，按原始排版顺序逐行输出，不要遗漏任何文字、数字、符号。"
    return generate(image_path, prompt, max_new_tokens=2048)


def vqa_image(image_path: str, question: str, context: str = "") -> str:
    """VQA：针对技术图纸的视觉问答，可结合说明书文本上下文"""
    context_hint = ""
    if context:
        context_hint = f"\n以下是专利说明书中对该图纸的相关描述：\n{context[:1500]}\n"
    prompt = f"""这是一张专利技术图纸（图1：静电除尘器管状入口）。请仔细观察图纸并回答问题。
{context_hint}
图纸关键部件说明：
- 1=管状入口, 2=外壳, 3=中心轴线, 4=圆柱形部分, 5=台阶形截止面
- 6,6',6"=配气带孔盘, 7=气流方向箭头, 10=圆锥形部分
- D=外壳直径, h1=圆柱形部分4的高度, h2=圆锥形部分10的高度
- X1,X2,X3=配气带孔盘之间的间隔距离

请根据图纸中的标注和箭头方向回答：
- 部件的相对位置（左/右/上/下，以气流方向为参考）
- 气流依次经过的部件顺序
- 尺寸标注的含义

问题：{question}"""
    return generate(image_path, prompt, max_new_tokens=256)


def batch_ocr_pages(page_images: List[str]) -> List[dict]:
    """批量 OCR 多页"""
    results = []
    for i, img_path in enumerate(page_images, 1):
        t0 = time.time()
        text = ocr_image(img_path)
        results.append({"page": i, "image_path": img_path, "text": text,
                        "ocr_time_s": round(time.time() - t0, 1)})
        logger.info(f"OCR 第{i}页: {len(text)}字, {time.time()-t0:.1f}s")
    return results
