# -*- coding: utf-8 -*-
"""离线 RAG 流水线（OfflineRAG）。

把 RAG_2 的各模块串成一条**全离线**链路，无需 Milvus / MySQL 服务器：

    文件 → data_type 识别类型 → mineru/ocr/pdf_table 解析 → 分块 → 向量化 → OfflineStore 入库 → 检索

用法示例（详见 README）：

    from rag2 import OfflineRAG

    rag = OfflineRAG(db_path="kb.sqlite", embed_model="D:/modelscope/bge-m3", device="cuda")
    rag.add_file("农业知识.pdf")
    rag.add_file("扫描件.png")
    for h in rag.search("农业知识相关问题", top_k=3):
        print(h.score, h.text[:60])
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

from .data_type import detect, FileInfo
from .store import OfflineStore

logger = logging.getLogger("rag2.pipeline")

# 可直接按文本读取的文件类别
_TEXT_KINDS = {"text", "markdown", "code", "json", "jsonl", "csv", "xml", "html"}


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 0) -> list[str]:
    """把长文本按段落合并切块（约 chunk_size 字符），超长段落按滑窗切分。

    参数：
        text:          输入文本。
        chunk_size:    目标块大小（字符数）。
        chunk_overlap: 超长段落滑窗重叠字符数。

    返回：
        分块列表。
    """
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > chunk_size:
                step = max(chunk_size - chunk_overlap, 1)
                chunks.extend(p[i:i + chunk_size] for i in range(0, len(p), step))
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


class OfflineRAG:
    """全离线 RAG：识别 → 解析 → 分块 → 向量化 → SQLite 入库 → 检索。"""

    def __init__(
        self,
        db_path: str | Path,
        embed_fn: Any = None,
        embed_model: str | None = None,
        device: str = "cpu",
        chunk_size: int = 500,
        chunk_overlap: int = 0,
    ) -> None:
        """初始化。

        参数：
            db_path:       离线库文件路径（.sqlite）。
            embed_fn:      向量化函数 (texts) -> vectors；优先使用。
            embed_model:   sentence-transformers 模型路径；未提供 embed_fn 时懒加载。
            device:        embed_model 加载设备（cuda / cpu）。
            chunk_size:    分块字符数。
            chunk_overlap: 超长段落滑窗重叠字符数。
        """
        self.store = OfflineStore(db_path)
        self._embed_fn = embed_fn
        self.embed_model = embed_model
        self.device = device
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    # ------------------------------------------------------------------ 向量化
    def _ensure_embedder(self) -> None:
        if self._embed_fn is not None:
            return
        if not self.embed_model:
            raise RuntimeError("未配置 embed_fn 或 embed_model，无法向量化")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embed_model, device=self.device)
        self._embed_fn = lambda texts: [
            v.tolist() for v in model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        ]
        logger.info("加载向量化模型：%s（device=%s）", self.embed_model, self.device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化（懒加载模型）。"""
        self._ensure_embedder()
        return self._embed_fn(list(texts))

    # ------------------------------------------------------------------ 入库
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        doc_id: str | None = None,
    ) -> int:
        """把文本分块向量化后写入离线库，返回写入条数。"""
        texts = list(texts)
        if not texts:
            return 0
        vectors = self.embed(texts)
        metas: list[dict] = []
        for i, _ in enumerate(texts):
            m = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            m.setdefault("doc_id", doc_id or "")
            metas.append(m)
        return self.store.add(texts, vectors=vectors, metadatas=metas)

    def add_file(self, path: str | Path) -> dict:
        """识别并解析单个文件，分块入库，返回处理摘要。

        路由：
            pdf / office → MinerU（失败回退 pdfplumber 抽文本）；
            image → PaddleOCR-VL；
            文本类 → 直接读取；
            其余 → 跳过并告警。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        info = detect(path)
        text = self._extract_text(path, info)
        if not text.strip():
            logger.warning("未提取到文本，跳过：%s（%s）", info.name, info.kind)
            return {"file": str(path), "kind": info.kind, "chunks": 0}
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        self.add_texts(chunks, doc_id=info.name)
        logger.info("入库完成：%s（%s）→ %d 块", info.name, info.kind, len(chunks))
        return {"file": str(path), "kind": info.kind, "chunks": len(chunks)}

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 10, mode: str = "hybrid"):
        """离线检索（向量 / 关键词 / 混合）。"""
        return self.store.search(query, top_k=top_k, embed_fn=self.embed, mode=mode)

    def evaluate(self, question: str, answer: str | None = None, reference: str | None = None,
                 top_k: int = 3, mode: str = "hybrid", metrics=None, evaluator=None) -> dict:
        """用 RAGAS 对「检索上下文 + 生成答案」打分（懒加载 ragas_eval）。

        先检索 top_k 条上下文，再调用 :func:`rag2.ragas_eval.evaluate_rag`，
        返回 ``{指标名: 得分}``。详见 README 第九节。
        """
        from .ragas_eval import evaluate_rag

        return evaluate_rag(self, question, answer=answer, reference=reference,
                            top_k=top_k, mode=mode, metrics=metrics, evaluator=evaluator)

    def count(self) -> int:
        return self.store.count()

    # ------------------------------------------------------------------ 解析
    def _extract_text(self, path: Path, info: FileInfo) -> str:
        kind = info.kind
        if kind == "pdf":
            return self._pdf_to_text(path)
        if kind == "image":
            return self._image_to_text(path)
        if kind in _TEXT_KINDS:
            return self._read_text(path, info.encoding)
        if kind == "office":
            try:
                return self._pdf_to_text(path)  # MinerU 可能支持部分 office 文件
            except Exception as exc:  # noqa: BLE001
                logger.warning("office 文件解析失败：%s", exc)
                return ""
        logger.warning("不支持的文件类型，跳过：%s（%s）", info.name, kind)
        return ""

    def _pdf_to_text(self, path: Path) -> str:
        try:
            from .mineru_parser import parse_pdf
            return parse_pdf(path).markdown
        except Exception as exc:  # noqa: BLE001 - 回退 pdfplumber
            logger.warning("MinerU 解析失败，回退 pdfplumber 抽文本：%s", exc)
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join((p.extract_text() or "") for p in pdf.pages)

    def _image_to_text(self, path: Path) -> str:
        from .ocr import ocr_image
        return ocr_image(path, save_txt=False).text

    @staticmethod
    def _read_text(path: Path, encoding: str | None = None) -> str:
        data = path.read_bytes()
        encodings = ([encoding] if encoding else []) + ["utf-8", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode("utf-8", errors="replace")
