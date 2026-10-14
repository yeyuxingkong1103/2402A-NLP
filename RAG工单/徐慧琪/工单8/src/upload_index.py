# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：上传 PDF 的**会话级**索引

服务工单产出物「用户手册：提供用户操作指南，包括如何上传 PDF 文档」——
用户可在问答页上传自己的 PDF，并就这份文档提问。

设计取舍
--------
· **会话级、不改全局**：上传文档只进内存索引，**不写入** Qdrant / 倒排索引 /
  知识图谱 —— 本工程图谱是离线构建产物，一份临时文档若混进去，
  后续所有问答的数字来源都会被污染；
· **同口径复用**：解析走 `pdf_parser` 的 **PyMuPDF 回退分支**（与主链路
  建库时同一份代码），分块走 `chunker.build_chunks`（父子块 / 表格结构化），
  嵌入走 `embedder`（bge-m3）——不新造第二套解析口径；
· **可回溯**：每条命中保留 `source / page / page_idx`，答案能回到页码；
· **容错**：非 PDF、加密、扫描件（无文本层）都抛 `UploadError`，
  由界面捕获后给用户可读提示，而不是让进程崩。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

PDF_MAGIC = b"%PDF"
MAX_CHUNKS = 2000          # 单文档分块上限（防超大文件把内存/显存吃满）


class UploadError(ValueError):
    """上传文档无法处理（非 PDF / 解析失败 / 无可提取文本）。"""


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    """默认嵌入器：复用全局 bge-m3（首次调用会加载模型）。"""
    from src.embedder import embed_texts
    return embed_texts(list(texts), show_progress=False)


@dataclass
class UploadIndex:
    """上传文档的内存索引：分块 + 向量 + BM25，**混合检索**后精排。

    为什么不是单路向量：工单07 的结论是数值型问题靠单路向量检索答不准
    （召回得到同主题的表，却拿不到那一个数）。上传文档走与主链路同口径的
    「向量 + 关键词 → 加权融合 → Reranker 精排」。
    """

    chunks: list[dict] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    embed_fn: Callable[[Sequence[str]], list[list[float]]] = _default_embed
    filename: str = ""
    truncated: bool = False
    bm25: object | None = None
    rerank_fn: Callable | None = None

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def _hit(self, chunk: dict, score: float) -> dict:
        page_idx = int(chunk.get("page_idx", -1))
        return {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "source": chunk.get("source") or self.filename,
            "page_idx": page_idx,
            "page": page_idx + 1,          # 与 07 命中口径一致：1 基
            "heading_path": chunk.get("heading_path") or [],
            "score": float(score),
        }

    def _vector_hits(self, query: str, n: int) -> list[dict]:
        qv = np.asarray(self.embed_fn([query])[0], dtype="float32")
        mat = np.asarray(self.vectors, dtype="float32")
        if mat.ndim != 2 or qv.shape[0] != mat.shape[1]:
            return []
        denom = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(qv) or 1.0)
        denom[denom == 0] = 1.0
        sims = (mat @ qv) / denom
        order = np.argsort(-sims)[:n]
        out = []
        for i in order:
            if sims[int(i)] <= 0:
                continue
            h = self._hit(self.chunks[int(i)], sims[int(i)])
            h["vector_score"] = h["score"]           # 留档：展示/调试用
            out.append(h)
        return out

    def _keyword_hits(self, query: str, n: int) -> list[dict]:
        if self.bm25 is None:
            return []
        try:
            from src.tokenizer import tokenize_query
            scores = self.bm25.get_scores(tokenize_query(query))
        except Exception:                              # noqa: BLE001
            return []
        order = np.argsort(-np.asarray(scores))[:n]
        out = []
        for i in order:
            if scores[int(i)] <= 0:
                continue
            h = self._hit(self.chunks[int(i)], float(scores[int(i)]))
            h["bm25_score"] = h["score"]             # BM25 原始分（量纲与余弦不同）
            out.append(h)
        return out

    def search(self, query: str, top_k: int = 8,
               rerank: bool = True) -> list[dict]:
        """混合检索上传文档，返回 [{chunk_id,text,source,page,page_idx,score,...}]。"""
        q = (query or "").strip()
        if not q or not self.chunks:
            return []
        top_k = max(1, int(top_k))
        arm_n = max(top_k * 3, 12)
        vec = self._vector_hits(q, arm_n)
        kw = self._keyword_hits(q, arm_n)

        if vec and kw:
            from src.fusion import weighted_fuse
            fused = weighted_fuse([("vector", vec), ("keyword", kw)],
                                  weights={"vector": 0.6, "keyword": 0.4})
            # 融合后统一用归一化分数：payload 里残留的 `score` 是**单路原始分**
            # （向量是余弦、BM25 是无界分），直接用会在界面上显示成「相似度 9.7」。
            for h in fused:
                h["score"] = float(h.get("fused_score", h.get("score", 0.0)))
        else:
            fused = vec or kw

        if not rerank or not fused:
            return list(fused)[:top_k]
        fn = self.rerank_fn
        if fn is None:
            try:
                from src.rerankers import rerank as fn   # 与主链路同一精排模型
            except Exception:                            # noqa: BLE001
                return list(fused)[:top_k]
        try:
            out = fn(q, list(fused), top_k=top_k)
            return list(out) if out else list(fused)[:top_k]
        except Exception:                                # noqa: BLE001
            return list(fused)[:top_k]


def _build_bm25(chunks: list[dict]):
    """对上传文档建 BM25 倒排（关键词通道）。装不了就返回 None（退化为单路）。"""
    try:
        from rank_bm25 import BM25Okapi

        from src.tokenizer import tokenize
        corpus = [tokenize(c.get("text") or "") for c in chunks]
        if not corpus:
            return None
        return BM25Okapi(corpus)
    except Exception:                                  # noqa: BLE001
        return None


def build_index_from_pdf(pdf_bytes: bytes, filename: str,
                         embed_fn: Callable | None = None,
                         limit_pages: int | None = None,
                         max_chunks: int = MAX_CHUNKS,
                         rerank_fn: Callable | None = None) -> UploadIndex:
    """把上传的 PDF 字节流解析→分块→向量化，产出会话级索引。

    limit_pages：只解析前 N 页（界面提供「快速模式」时用）。
    """
    data = pdf_bytes or b""
    if not data.startswith(PDF_MAGIC):
        raise UploadError("不是有效的 PDF 文件（缺少 %PDF 文件头），"
                          "请上传 PDF 格式的文档。")

    from src import chunker, pdf_parser

    if limit_pages:
        # pdf_parser.parse_pdf 没有页数参数，这里先按页裁剪字节流；
        # 顺带完成一次 PDF 有效性校验（损坏/加密文件在此抛 UploadError）。
        try:
            import fitz
            src = fitz.open(stream=data, filetype="pdf")
            if src.page_count > int(limit_pages):
                out = fitz.open()
                out.insert_pdf(src, from_page=0, to_page=int(limit_pages) - 1)
                data = out.tobytes()
                out.close()
            src.close()
        except UploadError:
            raise
        except Exception as exc:                       # noqa: BLE001
            raise UploadError(f"PDF 无法读取：{type(exc).__name__}: {exc}") from exc

    tmp_dir = tempfile.mkdtemp(prefix="wo08_upload_")
    tmp_pdf = os.path.join(tmp_dir, "upload.pdf")
    try:
        with open(tmp_pdf, "wb") as f:
            f.write(data)
        try:
            # 走 PyMuPDF 分支：确定性、无外部进程，与建库时的回退路径同源
            parsed = pdf_parser.parse_pdf(tmp_pdf, tmp_dir, reuse=False,
                                          force_fallback=True)
        except Exception as exc:                       # noqa: BLE001
            raise UploadError(f"PDF 解析失败：{type(exc).__name__}: {exc}") from exc

        items = parsed.get("items") or []
        chunks = chunker.build_chunks(items, filename or "上传文档.pdf")
        chunks = [c for c in chunks if (c.get("text") or "").strip()]
        if not chunks:
            raise UploadError("未能从该 PDF 提取到文本：可能是扫描件/图片型 PDF"
                              "（无文本层），本工程未启用 OCR 兜底。")
        truncated = len(chunks) > max_chunks
        if truncated:
            chunks = chunks[:max_chunks]
        texts = [c["text"] for c in chunks]
        fn = embed_fn or _default_embed
        try:
            vectors = fn(texts)
        except Exception as exc:                       # noqa: BLE001
            raise UploadError(f"向量化失败：{type(exc).__name__}: {exc}") from exc
        return UploadIndex(chunks=chunks, vectors=list(vectors), embed_fn=fn,
                           filename=filename, truncated=truncated,
                           bm25=_build_bm25(chunks), rerank_fn=rerank_fn)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
