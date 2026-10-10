# -*- coding: utf-8 -*-
"""多模态图像编码模块（Chinese-CLIP）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

对应工单【备注】要求：PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现。

本模块封装中文 CLIP（OFA-Sys/chinese-clip-vit-base-patch16），提供：
    - encode_images(): 图形区域图像 → 归一化图像向量（用于「以文搜图」）
    - encode_texts():  文本 → 归一化文本向量（与图像向量同一语义空间）
    - classify():      零样本图形类型分类（组织结构图 / 柱状图 / 饼图 …）
    - similarity():    单图-单文本跨模态相似度

工程要点：
1. 模型首次使用时自动下载（走 HF 镜像），下载后本地缓存，后续离线可用；
2. CPU 推理，批量编码；图像向量在「构建知识库」阶段离线预计算，
   查询阶段仅需编码一次文本（毫秒级），满足「响应 ≤ 3s」；
3. 模型不可用时优雅降级（available=False），主链路仍可运行（容错要求）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# 本地模型目录（优先使用；不存在则回退到 HF 仓库名在线加载）
_LOCAL_MODEL_DIR = os.getenv("CLIP_MODEL_PATH", "")


class ClipEncoder:
    """中文 CLIP 多模态编码器（懒加载 + 线程内复用）。"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.CLIP_MODEL
        self.device = device or "cpu"
        self._model = None
        self._processor = None
        self._load_failed = False

    # ---- 懒加载 -----------------------------------------------------------
    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch  # noqa: F401
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            src = _LOCAL_MODEL_DIR if (_LOCAL_MODEL_DIR and Path(_LOCAL_MODEL_DIR).exists()) \
                else self.model_name
            logger.info("加载多模态模型: %s", src)
            self._model = ChineseCLIPModel.from_pretrained(src)
            self._processor = ChineseCLIPProcessor.from_pretrained(src)
            self._model.to(self.device).eval()
            return True
        except Exception as exc:                      # 容错：模型缺失不阻断主链路
            logger.warning("多模态模型加载失败（将降级为无 CLIP 模式）: %s", exc)
            self._load_failed = True
            return False

    @property
    def available(self) -> bool:
        return self._load()

    @staticmethod
    def _projected(out):
        """取跨模态投影后的特征张量。

        transformers 版本差异：旧版 `get_image_features` / `get_text_features` 直接返回
        张量；5.x 起返回 `BaseModelOutputWithPooling`，投影后的特征放在 `pooler_output`
        （图像经 visual_projection、文本经 text_projection）。此处统一兼容，避免
        「'BaseModelOutputWithPooling' object has no attribute 'norm'」导致整条图像链路失效。
        """
        feat = getattr(out, "pooler_output", None)
        return feat if feat is not None else out

    # ---- 图像编码 ---------------------------------------------------------
    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """把图形区域图像编码为归一化向量矩阵 (N, D)；失败返回空矩阵。"""
        paths = [str(p) for p in image_paths]
        if not paths or not self._load():
            return np.zeros((0, 0), dtype="float32")
        import torch
        from PIL import Image

        feats: List[np.ndarray] = []
        bs = max(1, config.CLIP_BATCH_SIZE)
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            images = []
            for p in batch:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as exc:
                    logger.warning("图像读取失败 %s: %s", p, exc)
                    images.append(Image.new("RGB", (224, 224), "white"))
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.get_image_features(**inputs)
            out = self._projected(out)
            out = out / out.norm(p=2, dim=-1, keepdim=True)
            feats.append(out.cpu().numpy().astype("float32"))
        return np.vstack(feats) if feats else np.zeros((0, 0), dtype="float32")

    # ---- 文本编码 ---------------------------------------------------------
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """把文本编码到与图像同一语义空间；失败返回空矩阵。"""
        items = [t or "" for t in texts]
        if not items or not self._load():
            return np.zeros((0, 0), dtype="float32")
        import torch

        feats: List[np.ndarray] = []
        bs = max(1, config.CLIP_BATCH_SIZE)
        for i in range(0, len(items), bs):
            inputs = self._processor(
                text=items[i:i + bs], return_tensors="pt",
                padding=True, truncation=True, max_length=52,
            ).to(self.device)
            with torch.no_grad():
                out = self._model.get_text_features(**inputs)
            out = self._projected(out)
            out = out / out.norm(p=2, dim=-1, keepdim=True)
            feats.append(out.cpu().numpy().astype("float32"))
        return np.vstack(feats) if feats else np.zeros((0, 0), dtype="float32")

    # ---- 跨模态相似度 -----------------------------------------------------
    def similarity(self, image_path: str | Path, text: str) -> float:
        iv = self.encode_images([image_path])
        tv = self.encode_texts([text])
        if iv.size == 0 or tv.size == 0:
            return 0.0
        return float((iv @ tv.T).ravel()[0])

    def classify(self, image_path: str | Path,
                 labels: Sequence[str] | None = None) -> List[tuple[str, float]]:
        """零样本图形类型分类：返回按相似度降序的 (标签, 分数)。"""
        labels = list(labels or config.FIGURE_TYPE_LABELS)
        iv = self.encode_images([image_path])
        tv = self.encode_texts(labels)
        if iv.size == 0 or tv.size == 0:
            return [(lb, 0.0) for lb in labels]
        sims = (iv @ tv.T).ravel()
        return sorted(zip(labels, [float(s) for s in sims]), key=lambda x: -x[1])


# ---- 进程级单例（避免重复加载模型） ------------------------------------------
_SINGLETON: ClipEncoder | None = None


def get_encoder() -> ClipEncoder:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ClipEncoder()
    return _SINGLETON


def clip_available() -> bool:
    """CLIP 是否可用（用于界面状态提示与容错分支）。"""
    if not config.CLIP_ENABLE:
        return False
    return get_encoder().available