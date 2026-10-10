# -*- coding: utf-8 -*-
"""图像索引模块（PDF 图像内容解析第三步）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

本模块把 `figure_semantics` 的**结构化语义**接入 RAG 检索链路，包含两条索引：

1. **图像语义文本块**（ctype="figure"）：
   把每个图形的语义文本（组织结构层级、图内数值、派生结论）作为一个可检索块，
   与正文/表格块一起进入同一 FAISS 文本向量库与 BM25 语料。这样图形内容即可被
   常规文本检索命中，并由抽取式答案合成直接给出答案（对应 id 5 / id 6）。

2. **CLIP 跨模态图像向量库**（vector_db/image/）：
   用 Chinese-CLIP 把图形区域图像编码为向量并离线持久化，支持「以文搜图」——
   查询文本编码后与图像向量做余弦相似度，定位答案所在图形。查询阶段仅编码一次
   文本（毫秒级），满足「响应 ≤ 3s」。

工程要点：
- 图像向量在「构建知识库」阶段离线预计算并落盘（figures.npy + figures.jsonl）；
- CLIP 不可用时优雅降级（不阻断文本链路，仅失去「以文搜图」能力）；
- 全部参数与路径取自 config，注释遵循工单编号要求。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import config
from src.chunking import Chunk

logger = logging.getLogger(__name__)

_VEC_FILE = "figures.npy"
_META_FILE = "figures.jsonl"


# ================= 一、图像语义文本块 ==========================================
def build_figure_chunks(pdf_paths: Optional[Sequence] = None,
                        out_dir=None,
                        use_clip: bool = True) -> List[Chunk]:
    """抽取图形 → 语义解析 → 生成 ctype="figure" 的检索块，并落盘 CLIP 图像向量。

    Returns:
        Chunk 列表（可并入知识库文本向量库）。
    """
    from src.figure_extractor import extract_all_figures
    from src.figure_semantics import analyze_figures

    paths = list(pdf_paths or config.PDF_PATHS)
    figs = extract_all_figures(paths, out_dir)
    if not figs:
        logger.warning("未抽取到任何图形区域")
        return []
    sems = analyze_figures(figs)

    chunks: List[Chunk] = []
    idx = 0
    for sem in sems:
        text = (sem.semantic_text or "").strip()
        if not text:
            continue
        section = f"{sem.figure_type or '图形'}|{sem.caption or ''}".strip("|")
        chunks.append(Chunk(content=text, page=sem.page, section=section,
                            ctype="figure", parent=text, source=sem.source, index=idx))
        idx += 1

    if use_clip and config.CLIP_ENABLE:
        _persist_image_vectors(sems)
    logger.info("生成图像语义块 %d 个（含 CLIP 图像向量持久化）", len(chunks))
    return chunks


def _persist_image_vectors(sems) -> None:
    """把图形区域图像编码为 CLIP 向量并落盘（离线预计算）。"""
    try:
        from src.image_encoder import get_encoder

        enc = get_encoder()
        if not enc.available:
            logger.warning("CLIP 不可用，跳过图像向量持久化（文本链路不受影响）")
            return
    except Exception as exc:
        logger.warning("CLIP 初始化失败，跳过图像向量持久化: %s", exc)
        return

    items = [s for s in sems if s.image_path and Path(s.image_path).exists()]
    if not items:
        return
    vecs = enc.encode_images([s.image_path for s in items])
    if vecs.size == 0:
        return

    out = Path(config.IMAGE_DB_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / _VEC_FILE, vecs.astype("float32"))
    with (out / _META_FILE).open("w", encoding="utf-8") as f:
        for s in items:
            f.write(json.dumps({
                "source": s.source, "page": s.page, "caption": s.caption,
                "figure_type": s.figure_type, "type_score": s.type_score,
                "image_path": s.image_path, "semantic_text": s.semantic_text[:1500],
            }, ensure_ascii=False) + "\n")
    logger.info("CLIP 图像向量已落盘: %d 个 → %s", len(items), out)


# ================= 二、CLIP 跨模态图像检索 ======================================
@dataclass
class ImageHit:
    """一次「以文搜图」命中。"""

    score: float
    meta: dict = field(default_factory=dict)


class ImageIndex:
    """CLIP 图像向量索引（以文搜图）。"""

    def __init__(self, db_dir=None):
        self.db_dir = Path(db_dir or config.IMAGE_DB_DIR)
        self.vectors: np.ndarray = np.zeros((0, 0), dtype="float32")
        self.meta: List[dict] = []
        self._load()

    def _load(self) -> None:
        vf, mf = self.db_dir / _VEC_FILE, self.db_dir / _META_FILE
        if not vf.exists() or not mf.exists():
            return
        try:
            self.vectors = np.load(vf).astype("float32")
            self.meta = [json.loads(ln) for ln in mf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as exc:
            logger.warning("图像向量加载失败: %s", exc)
            self.vectors, self.meta = np.zeros((0, 0), dtype="float32"), []

    @property
    def ready(self) -> bool:
        return self.vectors.size > 0 and len(self.meta) == self.vectors.shape[0]

    def search(self, query: str, top_k: int = config.IMAGE_TOP_K,
               source: Optional[str] = None) -> List[ImageHit]:
        """以文搜图：返回按余弦相似度降序的命中（可按源文档过滤）。"""
        if not self.ready or not query.strip():
            return []
        try:
            from src.image_encoder import get_encoder

            enc = get_encoder()
            if not enc.available:
                return []
            qv = enc.encode_texts([query])
        except Exception as exc:
            logger.debug("查询文本编码失败: %s", exc)
            return []
        if qv.size == 0:
            return []
        sims = (self.vectors @ qv.ravel()).ravel()
        order = np.argsort(-sims)
        out: List[ImageHit] = []
        for i in order:
            m = self.meta[int(i)]
            if source and m.get("source") != source:
                continue
            out.append(ImageHit(score=float(sims[int(i)]), meta=m))
            if len(out) >= top_k:
                break
        return out

    def stats(self) -> dict:
        return {"images": int(self.vectors.shape[0]) if self.vectors.size else 0,
                "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
                "ready": self.ready}


_SINGLETON: Optional[ImageIndex] = None


def get_image_index() -> ImageIndex:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ImageIndex()
    return _SINGLETON


def clear_image_index() -> None:
    global _SINGLETON
    _SINGLETON = None


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    cs = build_figure_chunks()
    print("figure chunks:", len(cs))
    for c in cs[:5]:
        print("-" * 50, c.source, c.page, c.section[:40])
        print(c.content[:300])