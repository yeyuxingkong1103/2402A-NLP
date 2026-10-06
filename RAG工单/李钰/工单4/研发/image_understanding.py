# -*- coding: utf-8 -*-
"""
图像语义理解模块
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

功能:
    1. 从 PDF 提取内嵌图像 / 截取页面渲染为图像
    2. CLIP 跨模态图像-文本向量 (可选,需安装 sentence-transformers)
    3. 多模态 LLM 图像理解 (可选,需 GPT-4V/Claude Vision)
    4. 降级: 预设图像描述 JSON (开箱即用)
"""
import os
import json
import logging
from typing import List, Dict

import config_v4 as config

logger = logging.getLogger(__name__)


# ============================================================
# CLIP 跨模态 (可选)
# ============================================================

def _load_clip():
    """懒加载 CLIP 模型"""
    if not hasattr(_load_clip, "_model"):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[CLIP] 加载模型: {config.CLIP_MODEL_NAME}")
            _load_clip._model = SentenceTransformer(config.CLIP_MODEL_NAME)
            config.CLIP_AVAILABLE = True
        except Exception as e:
            logger.warning(f"[CLIP] 不可用: {e}")
            _load_clip._model = None
            config.CLIP_AVAILABLE = False
    return _load_clip._model


def encode_image(image_path: str):
    """CLIP 编码图像"""
    model = _load_clip()
    if model is None:
        return None
    try:
        from PIL import Image
        img = Image.open(image_path)
        return model.encode(img, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"图像编码失败: {e}")
        return None


def encode_text(text: str):
    """CLIP 编码文本"""
    model = _load_clip()
    if model is None:
        return None
    return model.encode([text], normalize_embeddings=True)


def clip_similarity(query_text: str, image_path: str) -> float:
    """计算文本与图像的 CLIP 相似度"""
    import numpy as np
    q_vec = encode_text(query_text)
    i_vec = encode_image(image_path)
    if q_vec is None or i_vec is None:
        return 0.0
    return float(np.dot(q_vec[0], i_vec.T))


# ============================================================
# PDF 图像提取 (可选)
# ============================================================

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[Dict]:
    """
    从 PDF 提取内嵌图像 (使用 PyMuPDF)

    Returns:
        List[{"image_id", "path", "page", "doc_id"}]
    """
    if not os.path.exists(pdf_path):
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("未安装 PyMuPDF, 无法提取图像: pip install PyMuPDF")
        return []

    os.makedirs(output_dir, exist_ok=True)
    images = []
    doc = fitz.open(pdf_path)

    for page_idx, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_path = os.path.join(output_dir, f"p{page_idx}_img{img_idx+1}.{ext}")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images.append({
                "image_id": f"img_{page_idx}_{img_idx}",
                "path": img_path,
                "page": page_idx,
                "doc_id": os.path.basename(pdf_path),
            })

    doc.close()
    logger.info(f"[图像提取] {pdf_path}: {len(images)} 张图像")
    return images


# ============================================================
# 预设图像描述 (开箱即用)
# ============================================================

def load_preset_images(path: str = None) -> List[Dict]:
    """加载预设图像描述 JSON"""
    path = path or config.PRESET_IMAGES_PATH
    if not os.path.exists(path):
        logger.warning(f"预设图像描述不存在: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", [])
        logger.info(f"[预设图像] 加载 {len(images)} 条图像描述")
        return images
    except Exception as e:
        logger.error(f"加载预设图像失败: {e}")
        return []


# ============================================================
# 多模态 LLM (可选)
# ============================================================

def analyze_image_with_llm(image_path: str, question: str) -> str:
    """
    用多模态 LLM (GPT-4V/Claude Vision) 分析图像

    降级: 返回预设描述
    """
    if not config.VISION_API_KEY and not config.LLM_API_KEY:
        logger.warning("未配置 VISION_API_KEY, 使用降级描述")
        return ""

    try:
        import base64
        import requests

        # 读取图像并编码
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # 构造多模态请求 (OpenAI 兼容)
        url = f"{config.VISION_BASE_URL or config.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.VISION_API_KEY or config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        ext = image_path.rsplit(".", 1)[-1] if "." in image_path else "png"
        payload = {
            "model": config.VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"请分析这张招股说明书中的图像, 回答问题: {question}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/{ext};base64,{img_b64}"
                    }},
                ]
            }],
            "max_tokens": 800,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"多模态 LLM 调用失败: {e}")
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    imgs = load_preset_images()
    for img in imgs:
        print(f"\n=== {img['image_id']} ({img['image_type']}) ===")
        print(f"  描述: {img['description'][:80]}")
        print(f"  关键词: {img['keywords']}")
        sd = img.get("structured_data", {})
        if sd:
            print(f"  结构化数据键: {list(sd.keys())}")
