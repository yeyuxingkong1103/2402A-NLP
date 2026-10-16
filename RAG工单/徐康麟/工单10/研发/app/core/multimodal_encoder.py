# -*- coding: utf-8 -*-
"""工单4 多模态语义编码（multimodal_encoder）。

任务：**人工智能NLP-RAG 项目-金融问答系统部署**（RAG 内核沿用 工单7 实现）。

用**中文 CLIP**（``OFA-Sys/chinese-clip-vit-base-patch16``）给图像块生成语义嵌入，
并提供「文本 query ↔ 图像块」的相似度检索通道，供 RAG 的**排名融合（RRF）**使用。

设计依据：``设计/图像解析方案.md`` §5。

------------------------------------------------------------------------
实测事实与坑（2026-10，本机，务必保留）
------------------------------------------------------------------------
1. ``huggingface_hub.snapshot_download`` **在本沙箱不可用**：它要把缓存 blob 软链到
   ``snapshots/<rev>/``，而 sandbox 在 ``<repo>/tmpxxxx/dummy_file_src`` 上直接
   ``PermissionError``。→ 用纯 HTTP 下载（见 ``研发/scripts/download_clip.py``）。
2. 仓库里的 ``clip_cn_vit-b-16.pt``（753,196,934 B）是 **cn_clip 训练检查点**，
   顶层键是 ``{epoch,name,step,state_dict}``，而 ``state_dict`` 里又是 timm 风格的
   ``module.visual.transformer.resblocks.N.attn.in_proj_weight``。
   把它当 ``pytorch_model.bin`` 加载 → **全部权重随机初始化**（实测已复现，366 个张量全 new）。
   → 本模块加载前会做**权重格式体检**，命中包装层直接报错，绝不静默用随机权重。
3. 两个解释器：主解释器 torch 是 **CPU 版**（2.11.0+cpu）；备用解释器
   ``工单1\\.gao6gongdan-src\\python.exe``（3.10.21, torch 2.6.0+cu124, RTX 2060 6GB）有 CUDA。
   两者**只能通过文件交换数据**（``.npy`` / ``.jsonl``）——本模块的落盘产物正是为此设计。
4. **中文 CLIP 的文本塔与 bge-m3 不是同一语义空间**：两种相似度**禁止直接相加**，
   必须走排名融合（RRF）或各自阈值后取并集。本模块只产出「相似度」，融合交给检索层。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_MODEL_DIR",
    "EncoderStatus",
    "ClipImageEncoder",
    "resolve_model_dir",
    "inspect_weight_file",
    "load_image_npy",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]          # 研发/
DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "models" / "chinese-clip-vit-base-patch16"
DEFAULT_INDEX_DIR = _REPO_ROOT / "data" / "index"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
REQUIRED_FILES = ["config.json", "preprocessor_config.json", "vocab.txt", "pytorch_model.bin"]


@dataclass
class EncoderStatus:
    """编码器可用性（**任何不可用都必须如实说明原因，不得伪造嵌入**）。"""

    available: bool
    reason: str = ""
    model_dir: str = ""
    model_name: str = MODEL_NAME
    device: str = ""
    load_seconds: float = 0.0
    projection_dim: int = 0
    weight_sha256: str = ""
    missing_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_dir(model_dir: Path | str | None = None) -> Path:
    """定位权重目录：显式参数 > 环境变量 ``RAG_CLIP_MODEL_DIR`` > 默认。"""

    if model_dir:
        return Path(model_dir)
    env = os.environ.get("RAG_CLIP_MODEL_DIR")
    if env:
        return Path(env)
    return DEFAULT_MODEL_DIR


def inspect_weight_file(model_dir: Path) -> tuple[bool, str]:
    """加载前的**权重格式体检**：拦截 cn_clip 训练检查点（会静默变成随机权重）。"""

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.exists():
        return False, "缺少 pytorch_model.bin"
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return True, "无法体检（torch 不可用：%s）" % exc
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, "torch.load 失败：%s: %s" % (type(exc).__name__, exc)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return False, ("pytorch_model.bin 是 cn_clip 训练检查点（顶层键=%s），"
                       "加载后会全部随机初始化；请改用真正的 HF state dict"
                       % sorted(obj.keys())[:8])
    return True, "格式正常"


class ClipImageEncoder:
    """中文 CLIP 图像/文本编码器（CPU / CUDA 自适应）。"""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 8,
        verbose: bool = True,
    ) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.batch_size = int(batch_size)
        self.status = EncoderStatus(available=False, model_dir=str(self.model_dir))
        self._model = None
        self._processor = None
        self._torch = None
        self._verbose = verbose
        self._load(device)

    # -- 加载 ---------------------------------------------------------------
    def _load(self, device: str) -> None:
        missing = [f for f in REQUIRED_FILES if not (self.model_dir / f).exists()]
        self.status.missing_files = missing
        if missing:
            self.status.reason = "权重目录缺少文件：%s（参见 研发/scripts/download_clip.py）" % missing
            return
        ok, why = inspect_weight_file(self.model_dir)
        if not ok:
            self.status.reason = why
            return
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        except Exception as exc:
            self.status.reason = "依赖不可用（torch/transformers）：%s: %s" % (type(exc).__name__, exc)
            return

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        t0 = time.time()
        try:
            model = ChineseCLIPModel.from_pretrained(str(self.model_dir))
            processor = ChineseCLIPProcessor.from_pretrained(str(self.model_dir))
            model.eval()
            model.to(device)
        except Exception as exc:
            self.status.reason = "from_pretrained 失败：%s: %s" % (type(exc).__name__, exc)
            return
        self._torch = torch
        self._model = model
        self._processor = processor
        self.status.available = True
        self.status.device = device
        self.status.load_seconds = round(time.time() - t0, 2)
        self.status.projection_dim = int(getattr(model.config, "projection_dim", 0))
        self.status.weight_sha256 = _sha256(self.model_dir / "pytorch_model.bin")
        self._text_workaround = False
        if self._verbose:
            print("[multimodal_encoder] 已加载 %s device=%s dim=%d %.2fs"
                  % (MODEL_NAME, device, self.status.projection_dim, self.status.load_seconds))

    @property
    def text_encoder_workaround(self) -> bool:
        """是否走了「CLS 代替 pooler」的上游 bug 规避（见 encode_texts 注释）。"""

        return bool(getattr(self, "_text_workaround", False))

    # -- 编码 ---------------------------------------------------------------
    def encode_images(self, images: Sequence[Any], *, batch_size: int | None = None) -> np.ndarray:
        """图像 → L2 归一化嵌入 (N, 512) float32。"""

        if not self.status.available:
            raise RuntimeError("中文 CLIP 不可用：%s" % self.status.reason)
        torch = self._torch
        bs = int(batch_size or self.batch_size)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), bs):
                chunk = list(images[i:i + bs])
                inputs = self._processor(images=chunk, return_tensors="pt")
                inputs = {k: v.to(self.status.device) for k, v in inputs.items()}
                feats = self._model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.status.projection_dim), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """文本 → L2 归一化嵌入 (N, 512) float32。

        ⚠️ **上游 bug 规避（实测 transformers 4.57.6）**：
        ``ChineseCLIPModel.get_text_features`` 内部取 ``text_outputs.pooler_output``，
        但 ``ChineseCLIPModel.__init__`` 是用 ``ChineseCLIPTextModel(cfg, add_pooling_layer=False)``
        构造文本塔的，且官方 checkpoint 里**没有任何 pooler 权重**
        （实测 ``[k for k in ckpt if 'pooler' in k] == []``）。
        于是 ``pooler_output is None`` → ``text_projection(None)`` 直接
        ``TypeError: linear(): argument 'input' must be Tensor, not NoneType``。
        ``get_text_features`` 的 docstring 明确写的是「final **[CLS] hidden state**」，
        所以这里显式取 ``last_hidden_state[:, 0]`` 再过 ``text_projection`` —— 与官方语义一致。
        该规避写进 manifest 的 ``text_encoder_workaround`` 字段，便于复核。
        """

        if not self.status.available:
            raise RuntimeError("中文 CLIP 不可用：%s" % self.status.reason)
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(text=list(texts), return_tensors="pt", padding=True,
                                     truncation=True, max_length=64)
            inputs = {k: v.to(self.status.device) for k, v in inputs.items()}
            out = self._model.text_model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]      # CLS，与 docstring 一致
                self._text_workaround = True
            feats = self._model.text_projection(pooled)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy().astype(np.float32)

    def similarity(self, text: str, image_embs: np.ndarray) -> np.ndarray:
        """一行 query 与全部图像嵌入的余弦相似度（两边都已 L2 归一化 → 内积）。"""

        t = self.encode_texts([text])[0]
        if image_embs.size == 0:
            return np.zeros((0,), np.float32)
        return (image_embs @ t).astype(np.float32)

    def classify_crop(self, crop: Any, candidates: Sequence[str]) -> list[tuple[str, float]]:
        """零样本分类：把一张小图与若干候选文本比对，返回 [(候选, 相似度)] 降序。

        用途：图表内位图文字的 **CLIP 通道**（与 chart_extractor 的字形模板通道互相校验）。
        """

        if not self.status.available or not candidates:
            return []
        im = self.encode_images([crop])
        tx = self.encode_texts(list(candidates))
        sims = (tx @ im[0]).tolist()
        return sorted(zip(candidates, sims), key=lambda t: -t[1])

    def rank(self, query: str, image_embs: np.ndarray, ids: Sequence[dict[str, Any]],
             top_k: int = 10) -> list[dict[str, Any]]:
        """按 CLIP 相似度给图像块排序（供 RRF 融合用；**不要与文本向量分数相加**）。"""

        sims = self.similarity(query, image_embs)
        order = np.argsort(-sims)[:top_k]
        out = []
        for r, i in enumerate(order):
            rec = dict(ids[int(i)]) if int(i) < len(ids) else {}
            rec["clip_similarity"] = float(sims[int(i)])
            rec["clip_rank"] = r + 1
            out.append(rec)
        return out


# ---------------------------------------------------------------------------
# 落盘 / 读取
# ---------------------------------------------------------------------------
def _sha256(path: Path, *, limit_mb: int | None = None) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_mb is not None and read >= limit_mb * (1 << 20):
                break
    return h.hexdigest()


def save_embeddings(
    index_dir: Path | str,
    embs: np.ndarray,
    ids: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    stem: str = "images_clip",
) -> dict[str, str]:
    """落盘 ``images_clip.npy`` + ``images_clip_ids.json`` + ``images_clip_manifest.json``。"""

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    npy = index_dir / ("%s.npy" % stem)
    idf = index_dir / ("%s_ids.json" % stem)
    mf = index_dir / ("%s_manifest.json" % stem)
    np.save(npy, np.asarray(embs, dtype=np.float32))
    idf.write_text(json.dumps(list(ids), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = dict(manifest)
    manifest.update({
        "npy": str(npy), "ids": str(idf), "n": int(len(ids)),
        "dim": int(embs.shape[1]) if embs.ndim == 2 and embs.size else 0,
    })
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"npy": str(npy), "ids": str(idf), "manifest": str(mf)}


def load_image_npy(index_dir: Path | str, *, stem: str = "images_clip") -> tuple[np.ndarray, list[dict[str, Any]]]:
    """读回嵌入与 id 表；不存在则返回 (空数组, [])，便于下游优雅降级。"""

    index_dir = Path(index_dir)
    npy = index_dir / ("%s.npy" % stem)
    idf = index_dir / ("%s_ids.json" % stem)
    if not npy.exists() or not idf.exists():
        return np.zeros((0, 0), np.float32), []
    return np.load(npy), json.loads(idf.read_text(encoding="utf-8"))
