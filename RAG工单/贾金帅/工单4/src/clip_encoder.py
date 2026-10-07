"""
CLIP 跨模态图像向量模块（工单4 核心 · CLIP 路线）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单备注写的是「使用多模态模型（**CLIP 或**多模态大模型）实现」，两者选一即可。
本工程**两条都落地**，因为它们解决的不是同一件事：

| 路线 | 产物 | 作用 |
|---|---|---|
| 多模态大模型 qwen-vl-plus（主力，见 image_semantics.py） | 图的**文本描述** | 让文本检索能命中图里的内容（id=5/id=6 靠它） |
| CLIP（本模块） | 图的**跨模态向量** | 让「以文搜图」直接成立：问题向量与图向量在同一空间里比相似度 |

CLIP 通道的独立价值在于：当问题的措辞与图的文本描述**用词对不上**时
（例如问「哪个行业的增长是负数」，而描述里写的是"IC 卡：−2.0%"），
纯文本检索会失手，而跨模态向量仍可能把那张柱状图排在前面。

模型：Chinese-CLIP（OFA-Sys/chinese-clip-vit-base-patch16，中文原生），
本地 CPU 推理。不引入 GPU 依赖 —— 本机是 RTX 3050 Laptop 4GB，
且 torch 装的是 CPU 版，没有必要为一个离线批处理去动显卡。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config
from .config import WORK_ORDER_NOS  # noqa: F401  (工单编号，模块标识)


class ClipEncoder:
    """Chinese-CLIP 图文编码器（单例，懒加载）。"""

    _instance: "ClipEncoder | None" = None

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.CLIP_MODEL_PATH
        self._model = None
        self._processor = None
        self._dim = config.CLIP_DIM
        self.load_error = ""
        # 记住「已经失败过」：检索端会对每个问题调一次 encode_texts，
        # 若每次失败都重试一遍 from_pretrained（几十 MB 的文件 IO + 异常构造），
        # 单个问题的耗时会从毫秒级劣化成秒级。失败一次就永久降级。
        self._failed = False

    @classmethod
    def instance(cls) -> "ClipEncoder":
        if cls._instance is None:
            cls._instance = ClipEncoder()
        return cls._instance

    # -------------------------------------------------------------- 加载
    @property
    def available(self) -> bool:
        return Path(self.model_path).exists()

    def load(self) -> bool:
        """加载模型。**失败不抛异常**，由调用方决定降级 —— 检索链路不能因为
        一个可选通道挂了就整体不可用。"""
        if self._model is not None:
            return True
        if self._failed:
            return False
        if not self.available:
            self.load_error = f"模型目录不存在：{self.model_path}"
            self._failed = True
            return False
        try:
            # 内存吃紧的机器上必须限制线程数：OpenBLAS 会按 CPU 核数（本机 12）预分配
            # 线程缓冲，分配失败时是**直接 abort、没有 Python 异常栈**的。
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            self._model = ChineseCLIPModel.from_pretrained(self.model_path)
            self._processor = ChineseCLIPProcessor.from_pretrained(self.model_path)
            self._model.eval()
            self._dim = int(self._model.config.projection_dim)
            self._torch = torch
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._model = None
            self._failed = True
            return False

    # -------------------------------------------------------------- 编码
    def encode_images(self, paths: list[Path], batch_size: int = 8) -> np.ndarray:
        """图 → 归一化向量矩阵 (N, D)。"""
        if not self.load():
            raise RuntimeError(f"CLIP 不可用：{self.load_error}")
        from PIL import Image

        feats: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self._processor(images=imgs, return_tensors="pt")
            with self._torch.no_grad():
                out = self._model.get_image_features(**inputs)
            feats.append(out.cpu().numpy().astype(np.float32))
        return _l2norm(np.vstack(feats)) if feats else np.zeros((0, self._dim), np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """文本 → 归一化向量矩阵 (N, D)，与图向量同一空间。"""
        if not self.load():
            raise RuntimeError(f"CLIP 不可用：{self.load_error}")
        feats: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._processor(text=batch, return_tensors="pt",
                                     padding=True, truncation=True, max_length=64)
            with self._torch.no_grad():
                out = self._model.get_text_features(**inputs)
            feats.append(out.cpu().numpy().astype(np.float32))
        return _l2norm(np.vstack(feats)) if feats else np.zeros((0, self._dim), np.float32)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


# ------------------------------------------------------------------ 索引产物
def build_and_save(figures: list[dict], out_path: Path | None = None) -> dict:
    """对 figures.json 里每张图算 CLIP 向量并落盘，同时回填 fig['clip_index']。

    返回统计信息（供报告使用）。向量顺序与 figures 里**有图且编码成功**的子集一一对应，
    clip_index 记录该图在矩阵里的行号。
    """
    out = out_path or config.CLIP_EMB_NPY
    enc = ClipEncoder.instance()
    paths: list[Path] = []
    targets: list[dict] = []
    for f in figures:
        if not f.get("image"):
            continue
        p = config.ROOT_DIR / f["image"]
        if not p.exists():
            continue
        paths.append(p)
        targets.append(f)
    if not paths:
        return {"count": 0, "error": "没有可编码的图"}

    vecs = enc.encode_images(paths)
    for i, f in enumerate(targets):
        f["clip_index"] = i
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vecs)
    return {"count": int(vecs.shape[0]), "dim": int(vecs.shape[1]),
            "path": str(out.relative_to(config.ROOT_DIR)),
            "model": Path(enc.model_path).name}


def load_clip_index() -> tuple[np.ndarray | None, list[dict]]:
    """读图向量矩阵 + 与之对齐的图清单。

    返回 (矩阵, 图列表)。矩阵缺失时返回 (None, [])，调用方据此降级。
    """
    if not config.CLIP_EMB_NPY.exists() or not config.FIGURES_JSON.exists():
        return None, []
    import json

    vecs = np.load(config.CLIP_EMB_NPY)
    figures = json.loads(config.FIGURES_JSON.read_text(encoding="utf-8"))
    aligned = [f for f in figures if isinstance(f.get("clip_index"), int)]
    aligned.sort(key=lambda f: f["clip_index"])
    if len(aligned) != vecs.shape[0]:
        # 清单与矩阵对不上（重新裁过图但没重算向量）→ 宁可不给，也不能错位
        return None, []
    return vecs, aligned
