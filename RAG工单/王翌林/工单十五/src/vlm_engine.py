# -*- coding: utf-8 -*-
"""
工单十五：Qwen2-VL 多模态引擎
职责：OCR（图片型PDF文本恢复）+ 图纸结构化解析（图号/部件/位置关系）
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
    _load()
    import torch
    image = Image.open(image_path).convert("RGB")
    max_side = 1280
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
    prompt = "请完整识别图片中的所有文字内容，按原始排版顺序逐行输出，不要遗漏任何文字、数字、符号。"
    return generate(image_path, prompt, max_new_tokens=2048)


def parse_figure(image_path: str, fig_no: int, page: int) -> dict:
    """对技术图纸做结构化解析：部件编号、位置关系、物料流向"""
    prompt = f"""这是一张专利技术图纸（图{fig_no}，位于说明书第{page}页），主题是块状散料分配装置。
请仔细观察图纸并输出结构化分析，包含：
1. 图中所有带编号的部件及其名称/作用
2. 各部件之间的位置关系（谁在谁内部、谁在顶部、物料依次经过谁）
3. 物料的流动路径（从哪个部件进入，依次经过哪些部件）
4. 若有调节装置，指出操作哪个部件可调节链条位置
请用清晰的条目输出。"""
    desc = generate(image_path, prompt, max_new_tokens=1024)
    return {"fig_no": fig_no, "page": page, "image_path": image_path, "analysis": desc}


def batch_ocr_pages(page_images: List[str]) -> List[dict]:
    results = []
    for i, img_path in enumerate(page_images, 1):
        t0 = time.time()
        text = ocr_image(img_path)
        results.append({"page": i, "image_path": img_path, "text": text,
                        "ocr_time_s": round(time.time() - t0, 1)})
        logger.info(f"OCR 第{i}页: {len(text)}字, {time.time()-t0:.1f}s")
    return results
