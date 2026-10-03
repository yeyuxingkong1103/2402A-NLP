# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单04 - 图像内容解析及检索优化
"""
入库流水线：解析 → 分块 → 去重 → 嵌入 → 写入 Milvus。

API（/api/ingest）与命令行（scripts/ingest.py）共用这一份逻辑，
避免两条路径行为不一致 —— "CLI 能跑但接口跑不通"是最常见的验收翻车方式。

【进度可观测】548 页 × 1100 chunk 全程约 1–3 分钟。工单01 要求
"交互友好"，因此每一步都更新 IngestState，前端轮询 /api/ingest/status 显示进度。
"""

from __future__ import annotations

from collections import Counter

import asyncio
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.chunker import chunk_pages
from app.core.dedup import dedup_chunks
from app.core.embedder import Embedder
from app.core.pdf_parser import parse_pdf_full
from app.core.vectorstore import VectorStore


@dataclass
class IngestState:
    running: bool = False
    stage: str = "idle"
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def doc_id_of(path: Path) -> str:
    """文档标识：文件名 + 大小 + mtime，换文件后能识别出是不同的库。"""
    raw = f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class IngestPipeline:
    """单例式流水线，进程内只允许一个入库任务同时运行。"""

    def __init__(self) -> None:
        self.state = IngestState()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    async def run(self, pdf_path: str | Path | None = None,
                  *, rebuild: bool = False,
                  limit_pages: int | None = None) -> IngestState:
        async with self._lock:
            if self.state.running:
                return self.state
            self.state = IngestState(running=True, stage="prepare", message="准备中")
            try:
                await self._run_inner(Path(pdf_path) if pdf_path else self._default_pdf(),
                                      rebuild=rebuild, limit_pages=limit_pages)
            except Exception as e:  # noqa: BLE001
                self.state.error = f"{type(e).__name__}: {e}"
                self.state.stage = "error"
                self.state.message = "入库失败"
            finally:
                self.state.running = False
            return self.state

    # ------------------------------------------------------------------
    def _default_pdf(self) -> Path:
        p = settings.data_path / "raw" / "招股说明书1.pdf"
        if not p.exists():
            raise FileNotFoundError(f"默认 PDF 不存在：{p}")
        return p

    async def _run_inner(self, pdf: Path, *, rebuild: bool, limit_pages: int | None) -> None:
        st = self.state
        t_all = time.perf_counter()

        # ---------------- 1. 解析 ----------------
        st.stage = "parse"
        st.message = f"解析 {pdf.name}"
        pages, pstats = await asyncio.to_thread(
            parse_pdf_full, pdf, settings.page_label_offset, limit_pages
        )
        st.stats["parse"] = {
            "pages": pstats.n_pages, "tables": pstats.n_tables,
            "text_blocks": pstats.n_text_blocks,
            "header_removed": pstats.n_header_removed,
            "footer_removed": pstats.n_footer_removed,
            "table_text_dropped": pstats.n_table_text_dropped,
        }

        # ---------------- 1b. 图像语义转写（工单04）----------------
        # 必须夹在 parse 与 chunk 之间：转写结果要回填到 Figure.text 上，
        # 而分块（chunk_pages）会把 figures 变成 chunk。
        # 必须是 async 阶段：parse 跑在 to_thread 里（同步、几十秒），
        # 而 VLM 调用是网络 I/O。
        if settings.image_enable and getattr(pstats, "n_figures", 0) > 0:
            st.stage = "figure"
            st.message = "图像语义解析（多模态模型）"
            # 显存排程：主动赶走聊天模型再转写（8GB 卡装不下两者）。
            # 不主动赶，Ollama 会自己淘汰一个 —— 但那是被动的：
            # 转写完的下一问要付一次数秒冷启动。
            from app.core.vision import (FigureTranscriber, unload_chat_model,
                                         unload_vlm)
            await unload_chat_model()
            tr = FigureTranscriber(pdf)
            fstats = await tr.run(pages)
            await unload_vlm()
            st.stats["figure"] = fstats.as_dict() | {
                "bitmap": pstats.n_figures_bitmap,
                "vector": pstats.n_figures_vector,
                "images_dropped": pstats.n_images_dropped,
            }
        elif not settings.image_enable:
            st.stats["figure"] = {"disabled": True}

        # ---------------- 2. 分块 ----------------
        st.stage = "chunk"
        st.message = "句子边界分块"
        chunks = await asyncio.to_thread(lambda: list(chunk_pages(pages)))
        counts: Counter[str] = Counter(c.chunk_type for c in chunks)
        st.stats["chunk"] = {
            "total": len(chunks),
            "text": counts.get("text", 0),
            "table": counts.get("table", 0),
            "image": counts.get("image", 0),
        }

        # ---------------- 3. 去重 ----------------
        st.stage = "dedup"
        st.message = "去重"
        dres = await asyncio.to_thread(dedup_chunks, chunks)
        st.stats["dedup"] = {
            "kept": len(dres.kept), "dropped": len(dres.dropped),
            "exact": dres.n_exact, "near": dres.n_near,
            "rate": round(dres.drop_rate, 4),
        }
        chunks = dres.kept

        # ---------------- 4. 向量库准备 ----------------
        st.stage = "milvus"
        store = VectorStore()
        ok, msg = await asyncio.to_thread(store.health)
        if not ok:
            raise RuntimeError(f"Milvus 不可用：{msg}")
        action = await asyncio.to_thread(store.create_collection, drop_existing=rebuild)
        st.stats["milvus"] = {"collection": store.collection, "action": action}

        if rebuild:
            await asyncio.to_thread(store.drop_collection)
            await asyncio.to_thread(store.create_collection)

        # ---------------- 5. 嵌入 ----------------
        st.stage = "embed"
        st.total = len(chunks)
        st.done = 0
        texts = [c.content for c in chunks]

        def on_progress(done: int, total: int) -> None:
            st.done = done
            st.message = f"向量化 {done}/{total}"

        embedder = Embedder()
        vectors, estats = await embedder.embed_many(texts, progress=on_progress)
        st.stats["embed"] = {"n": estats.n_texts, "batches": estats.n_batches,
                             "seconds": round(estats.seconds, 1),
                             "dim": estats.dim}

        # ---------------- 6. 写入 ----------------
        st.stage = "insert"
        st.message = "写入 Milvus"
        # Milvus 单次 insert 行数过大易超时，按 500 一批
        batch = 500
        inserted = 0
        for i in range(0, len(chunks), batch):
            n = await asyncio.to_thread(
                store.insert_chunks,
                chunks[i:i + batch], vectors[i:i + batch],
                doc_id_of(pdf), pdf.name,
            )
            inserted += n
            st.done = inserted
            st.message = f"写入 {inserted}/{len(chunks)}"
        await asyncio.to_thread(store.flush)

        st.stats["insert"] = {"rows": inserted}
        st.stats["total_seconds"] = round(time.perf_counter() - t_all, 1)
        st.stage = "done"
        st.message = f"完成：{inserted} 个片段入 {store.collection}"

    # ------------------------------------------------------------------
    def status(self) -> IngestState:
        return self.state


_pipeline: IngestPipeline | None = None


def get_pipeline() -> IngestPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestPipeline()
    return _pipeline
