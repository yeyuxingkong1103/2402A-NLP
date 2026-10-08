# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_embedding.py —— 工单四图像向量化模块（新增文件）

双通道嵌入（见 docs/02_图像检索优化方案.md §2）：
  1. 文本通道：caption + ocr_text + vqa_text 拼接 → bge-m3（1024 维）
     —— 复用工单三 get_embedder() 单例（增量复用，不重复加载模型）
  2. 图像通道：Chinese-CLIP 图像/文本嵌入（512 维，双编码器支持中文 query）
"""
from typing import List, Optional

from loguru import logger

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


class ImageTextEmbedder:
    """工单四：图像描述文本嵌入（bge-m3 1024 维，复用工单三单例）"""

    def __init__(self):
        self._embedder = None
        self.dim = 1024

    def _get(self):
        if self._embedder is None:
            from src.embedding import get_embedder  # 工单四：复用工单三 bge-m3 单例
            self._embedder = get_embedder()
        return self._embedder

    def embed_text(self, text: str) -> List[float]:
        """工单四：单条文本嵌入（L2 归一化，配 COSINE metric；空文本护栏）"""
        if not text or not text.strip():            # 工单四：空文本 bge-m3 会崩，用占位词
            text = "无文字图像"
        return self._get().encode([text], show_progress_bar=False)[0].tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """工单四：批量文本嵌入（batch=8 防 OOM，见项目 Lessons）"""
        texts = [t if t and t.strip() else "无文字图像" for t in texts]  # 工单四：空文本护栏
        vecs = self._get().encode(texts, batch_size=batch_size, show_progress_bar=False)
        return [v.tolist() for v in vecs]


def build_image_text(caption: str = "", ocr_text: str = "",
                     vqa_text: str = "") -> str:
    """工单四：拼接图像描述文本（caption 优先，OCR/VQA 结构化补充，控制长度）"""
    parts = [p.strip() for p in (caption, ocr_text, vqa_text) if p and p.strip()]
    text = "；".join(parts)
    return text[:4000]                              # 工单四：bge-m3 max_seq=512 截断保护


class ImageClipEmbedder:
    """工单四：Chinese-CLIP 图像嵌入（复用 image_parser.CLIPEngine 惰性引擎）"""

    def __init__(self, model_dir: Optional[str] = None):
        from src.image_parser.image_parser import CLIPEngine
        self._clip = CLIPEngine(model_dir=model_dir)
        self.dim = 512

    def embed_image_file(self, path: str) -> Optional[List[float]]:
        """工单四：图像文件嵌入；CLIP 模型缺失返回 None（容错规范）"""
        from PIL import Image
        if not self._clip._ensure_loaded():
            return None
        with Image.open(path) as img:
            return self._clip.embed_image(img)

    def embed_query_text(self, text: str) -> Optional[List[float]]:
        """工单四：中文 query 走 CLIP 文本编码器（图文同空间，图像感知检索用）"""
        if not self._clip._ensure_loaded():
            return None
        import torch
        inputs = self._clip._proc(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._clip._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._clip._model.get_text_features(**inputs)
        # 工单四：兼容 transformers>=5.x（投影特征在 pooler_output）与旧版（tensor）
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().tolist()
