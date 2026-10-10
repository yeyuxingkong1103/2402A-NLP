# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""端到端流水线编排：ingest（建库）与 ask（问答）两条主链路。"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag04.config import Settings, ensure_dirs, get_settings
from rag04.obs.logging import setup_logging, timed
from rag04.schema import Answer, Chunk, FigureBlock, Hit, TableBlock, TextBlock

# 必须先于 _preload_fragile_deps() 的导入期调用定义：否则预热失败时
# except 分支会抛 NameError，反而阻断导入（vlparser 同款顺序）。
logger = logging.getLogger("rag04.pipeline")


def _preload_fragile_deps() -> None:
    """Windows 本机环境修复（必修）：把 sklearn→pandas→pyarrow 链提前到导入期。

    实测（本机 Win11 + pytest）：进程内先加载 numpy/qdrant_client（例如
    ``RAGPipeline.health()`` 打开 Qdrant 嵌入式库）后，再惰性 ``import sklearn``
    会触发 pandas→pyarrow 的 C 初始化栈溢出 —— 无 faulthandler 输出、进程被
    直接杀掉（git-bash 报 Segmentation fault）。Task 13 的 rerank 惰性预加载
    在该场景下本身就是崩溃点。把该链提前到本模块导入期（此时 torch/PIL 尚未
    加载）即可稳定规避，与 vlparser 的既有修法同源。仅在 win32 预热，失败不
    阻断导入。
    """
    try:
        import sklearn  # noqa: F401
    except Exception:  # pragma: no cover - 预热尽力而为
        logger.debug("sklearn 预热失败", exc_info=True)


if sys.platform == "win32":
    _preload_fragile_deps()

_page_limit: int | None = None       # 测试用：限制解析页数


@dataclass
class IngestStats:
    doc_id: str
    n_text: int = 0
    n_table: int = 0
    n_figure: int = 0
    n_chunk: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    n_text_boilerplate: int = 0    # RC1-in：入库侧剔除的页眉/页脚文本块数


def _extract_figures(doc, doc_id: str, s: Settings, limit: int | None) -> list[FigureBlock]:
    """逐页检测并渲染图区。单页失败跳过。"""
    from rag04.ingest.figures import detect_figures, render_figure

    out: list[FigureBlock] = []
    total = doc.page_count if not limit else min(doc.page_count, limit)
    for i in range(total):
        pno = i + 1
        try:
            figs = detect_figures(doc[i], doc_id, pno, s)
            for f in figs:
                try:
                    render_figure(doc[i], f, s)
                    out.append(f)
                except Exception as e:
                    logger.warning("p%d 图渲染失败：%s", pno, e)
        except Exception as e:
            logger.warning("p%d 图区检测失败：%s", pno, e)
    return out


def _image_vector(c: Chunk, s: Settings) -> list[float] | None:
    """图像块向量：优先 CLIP 图像向量，缺失时退回 CLIP 文本塔（同一 512 维空间）。

    文本塔也失败则返回 None，由调用方丢弃该块 —— image_chunks 是 512 维，任何
    1024 维（bge-m3）向量都会让 Qdrant upsert 抛错，进而中断整个建库。
    """
    cv = (c.extra or {}).get("clip_vector") or []
    if len(cv) == 512:
        return list(cv)
    try:
        from rag04.ingest.vlparser import clip_encode_text
        tv = clip_encode_text(c.text, s)
        if len(tv) == 512:
            return list(tv)
        logger.warning("图块 %s（%s）CLIP 文本塔向量维度异常 %d，已丢弃",
                       c.chunk_id, c.doc_id, len(tv))
    except Exception as e:
        logger.warning("图块 %s（%s）无 CLIP 向量且文本塔兜底失败（%s: %s），已丢弃",
                       c.chunk_id, c.doc_id, type(e).__name__, e)
    return None


def ingest_document(pdf_path: Path, s: Settings, store=None,
                    page_limit: int | None = None) -> IngestStats:
    """解析单个 PDF 并入库。返回统计。"""
    from rag04.ingest.chunker import build_chunks
    from rag04.ingest.loader import doc_id_of, iter_text_blocks, open_pdf
    from rag04.ingest.tables import iter_all_tables
    from rag04.ingest.vlparser import parse_figures
    from rag04.retrieve.embed import embed_texts

    pdf_path = Path(pdf_path)
    doc_id = doc_id_of(pdf_path)
    st = IngestStats(doc_id=doc_id)
    t0 = time.perf_counter()
    limit = page_limit if page_limit is not None else _page_limit

    doc = open_pdf(pdf_path)

    # 1) 文本
    text_blocks: list[TextBlock] = []
    for b in iter_text_blocks(doc, doc_id):
        if limit and b.page > limit:
            continue
        text_blocks.append(b)

    # RC1-in：入库侧就剔掉跨页重复的页眉/页脚（与检索侧同一判据，K=20+长度护栏）。
    # 只在 full_04 生效；baseline_03 是对照组，索引内容保持原样。
    if s.pipeline_mode == "full_04":
        from rag04.retrieve.boilerplate import filter_boilerplate_blocks
        text_blocks, st.n_text_boilerplate = filter_boilerplate_blocks(text_blocks)
        if st.n_text_boilerplate:
            logger.info("[%s] 入库侧剔除样板文本块 %d 个", doc_id, st.n_text_boilerplate)

    st.n_text = len(text_blocks)
    logger.info("[%s] 文本块 %d 个", doc_id, st.n_text)

    # 2) 表格（baseline_03 不抽表）
    tables: list[TableBlock] = []
    if s.use_tables:
        total = limit or doc.page_count
        for tb in iter_all_tables(pdf_path, doc_id):
            if tb.page > total:
                continue
            tables.append(tb)
        st.n_table = len(tables)
        logger.info("[%s] 表格 %d 个", doc_id, st.n_table)

    # 3) 图像（baseline_03 不做）
    figures: list[FigureBlock] = []
    if s.use_figures:
        figures = _extract_figures(doc, doc_id, s, limit)
        figures = parse_figures(figures, s)
        for f in figures:
            if f.parse_warning:
                st.warnings.append(f"{f.figure_id}: {f.parse_warning}")
        st.n_figure = len(figures)
        logger.info("[%s] 图区 %d 个（含告警 %d）", doc_id, st.n_figure, len(st.warnings))

    # 4) 分块
    chunks = build_chunks(text_blocks, tables, figures, s)
    st.n_chunk = len(chunks)
    logger.info("[%s] 分块 %d 个", doc_id, st.n_chunk)

    # 5) 向量化 + 入库（图像块用 CLIP 向量，其余用 bge-m3）
    if store is not None and chunks:
        text_like = [c for c in chunks if c.block_type != "image"]
        image_like = [c for c in chunks if c.block_type == "image"]

        ordered: list[Chunk] = list(text_like)
        vecs: list[list[float]] = []
        if text_like:
            vecs.extend(embed_texts([c.text for c in text_like], s))

        for c in image_like:
            cv = _image_vector(c, s)
            if cv is None:
                continue          # 丢弃：图像库是 512 维，宁可少一块也不写错维度
            ordered.append(c)
            vecs.append(cv)

        n = store.upsert_chunks(ordered, vecs)
        n_dropped = len(image_like) - (len(ordered) - len(text_like))
        logger.info("[%s] 入库 %d 条（图块丢弃 %d）", doc_id, n, n_dropped)

    doc.close()
    st.seconds = time.perf_counter() - t0
    return st


def build_all(s: Settings, reset: bool = False,
              names: list[str] | None = None) -> list[IngestStats]:
    """构建语料的索引。``reset=True`` 先清空三库（RC-2 重建安全，见 store.py）。

    ``names`` 限定要重建的文件名（默认 ``s.corpus`` 全量），供 /api/ingest 指定
    单个语料重建；文件名不存在于 project_root 时跳过并告警。
    """
    from rag04.ingest.store import VectorStore
    from rag04.retrieve.bm25 import BM25Index

    ensure_dirs(s)
    store = VectorStore(s)
    store.ensure_collections()
    if reset:
        logger.info("重建模式：清空三库（清空前点数 %s）", store.clear_collections())

    stats: list[IngestStats] = []
    all_chunks = []
    for name in (names if names is not None else s.corpus):
        pdf = Path(s.project_root) / name
        if not pdf.exists():
            logger.warning("语料缺失，跳过：%s", pdf)
            continue
        st = ingest_document(pdf, s, store=store)
        stats.append(st)

    # BM25 索引（从库里全量重建）
    for coll in ("text_chunks", "table_chunks"):
        try:
            recs, _ = store.client.scroll(coll, limit=100_000, with_payload=True)
            from rag04.schema import Chunk
            for r in recs:
                pl = r.payload or {}
                all_chunks.append(Chunk(
                    chunk_id=pl.get("chunk_id", str(r.id)),
                    doc_id=pl.get("doc_id", ""), page=int(pl.get("page", 0)),
                    block_type=pl.get("block_type", "text"),
                    source_id=pl.get("source_id", ""), text=pl.get("text", ""),
                    section_path=pl.get("section_path", ""),
                    lang=pl.get("lang", "zh"), extra=pl.get("extra", {}) or {},
                ))
        except Exception as e:
            logger.warning("读取 %s 建 BM25 失败：%s", coll, e)

    idx = BM25Index()
    idx.build(all_chunks)
    idx.save(Path(s.data_dir) / "bm25.pkl")
    logger.info("BM25 索引已保存，共 %d 块", len(all_chunks))

    store.close()
    return stats


def _retrieve(question: str, s: Settings, store, bm25, embed_fn, clip_fn,
              boilerplate_ids=None):
    """检索入口，便于测试打桩。"""
    from rag04.retrieve.hybrid import hybrid_retrieve
    return hybrid_retrieve(question, s, store, bm25,
                           embed_fn=embed_fn, clip_fn=clip_fn,
                           boilerplate_ids=boilerplate_ids)


def ask(question: str, s: Settings, store=None, bm25=None, llm=None,
        boilerplate_ids=None) -> Answer:
    """问答主链路：检索 → 重排 → 生成。"""
    from rag04.generate.llm import LLMClient
    from rag04.retrieve.embed import embed_one
    from rag04.retrieve.rerank import rerank

    t0 = time.perf_counter()
    with timed(logger, f"ask[{s.pipeline_mode}]"):
        hits = _retrieve(question, s, store, bm25, embed_one, _clip_fn(s),
                         boilerplate_ids=boilerplate_ids)
        # 三路 RRF 最多可产出 3*k_each 个候选，截到 rerank_top_n 再重排：
        # CrossEncoder 逐对前向，候选越多本机 CPU 时延越高（3 秒硬指标）。
        hits = hits[:s.rerank_top_n]
        hits = rerank(hits, question, s, top_k=s.final_top_k)
        client = llm or LLMClient(s)
        ans = client.generate(question, hits)

    ans.latency_ms = (time.perf_counter() - t0) * 1000
    return ans


def _clip_fn(s: Settings):
    if not s.use_clip_retrieval:
        return None
    from rag04.retrieve.image_index import clip_search
    return clip_search


class RAGPipeline:
    """便捷门面：一次构建，多次问答。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.store = None
        self.bm25 = None
        self.boilerplate = None      # RC1：样板判据（懒加载 + 落盘缓存）
        self.log = setup_logging(self.s)
        # 延迟加载（存储/BM25/样板判据）的互斥锁：Qdrant 本地客户端对 data/qdrant
        # 持有独占文件锁，并发首查若各自开一个客户端即抛「already accessed」。
        self._load_lock = threading.Lock()

    def build(self, names: list[str] | None = None,
              reset: bool = True) -> list[IngestStats]:
        """重建索引。``names`` 限定文件名（None = 配置里的全部语料）。

        ``reset=True``（默认）先清空三库：分块改动（RC5 行块合并 / RC2 图描述
        文本块 / RC1-in 入库侧过滤）会改变 chunk_id，旧点不会被覆盖，直接
        upsert 会让新旧块共存并污染检索（判据见 store.clear_collections）。
        只有「追加同一批 chunk_id」的场景才可显式传 ``reset=False``。
        /api/ingest 与 UI 重建按钮调用本方法，可传 names 做单语料重建。
        """
        # build_all 自己开一个嵌入客户端；本进程若已持有句柄（例如服务里先查过
        # 一次 /health），必须让路，否则同进程也会撞 Qdrant 的文件锁。
        self._release_store()
        stats = build_all(self.s, reset=reset, names=names)
        self.warmup()                # 重建后重新载入存储/BM25/样板判据
        return stats

    def _release_store(self) -> None:
        """释放本进程持有的存储句柄并作废相关缓存（重建前调用，幂等）。"""
        with self._load_lock:
            store, self.store = self.store, None
            self.bm25 = None         # 重建会重写 bm25.pkl，必须重新载入
            self.boilerplate = None  # 索引变了，样板判据的 counts 快照随之失效
        if store is not None:
            try:
                store.close()
                self.log.info("已释放存储句柄，供重建独占索引")
            except Exception as e:                       # 释放失败不阻断重建
                self.log.warning("释放存储句柄失败（继续重建）：%s", e)

    def warmup(self) -> None:
        """预热问答热路径上的重模型，避免进程内首次提问付加载时延。

        reranker 约 8s / 1.1GB、CLIP 约数百 MB，均为模块级缓存：只在进程内
        首次调用时加载，此后复用（Task 13 教训）。逐项独立容错，任一失败仅
        记录不抛出 —— 预热是优化，不是正确性前提。服务启动时调用本方法即可
        让第一条问答满足 3 秒硬指标（Task 19/20 用）。

        第一步先把存储/BM25/样板判据打开：Qdrant 本地客户端独占 data/qdrant，
        留到首次请求才开，启动瞬间的并发首查会互抢文件锁（Task 19 复核）。
        提前串行打开后，问答热路径只剩纯计算。
        """
        try:
            self.log.info("预热存储/BM25/样板判据 …")
            with timed(self.log, "warmup[store]"):
                self._ensure_loaded()
        except Exception as e:
            self.log.warning("存储预热失败（不影响启动，问答时会重试）：%s: %s",
                             type(e).__name__, e)

        if self.s.use_rerank:
            try:
                self.log.info("预热 reranker …")
                with timed(self.log, "warmup[rerank]"):
                    from rag04.retrieve.rerank import rerank
                    rerank([Hit(chunk_id="_warmup", doc_id="", page=0,
                                block_type="text", source_id="", text="预热",
                                score=0.0)], "预热", self.s, top_k=1)
            except Exception as e:
                self.log.warning("reranker 预热失败（不影响问答，走降级）：%s", e)

        if self.s.use_clip_retrieval:
            try:
                self.log.info("预热 CLIP …")
                from rag04.ingest.vlparser import clip_encode_text
                with timed(self.log, "warmup[clip]"):
                    clip_encode_text("预热", self.s)
            except Exception as e:
                self.log.warning("CLIP 预热失败（不影响问答，该路会被跳过）：%s", e)

        try:
            from rag04.retrieve.embed import embed_one
            with timed(self.log, "warmup[embed]"):
                embed_one("预热", self.s)
        except Exception as e:
            self.log.warning("嵌入模型预热失败（Ollama 不可用时问答会走降级）：%s", e)

    def _ensure_loaded(self) -> None:
        """惰性加载存储 / BM25 / 样板判据（幂等）。

        加锁 + 锁内二次判空：并发首查只能有一个线程开存储，其余复用同一句柄。
        否则第二次 ``QdrantClient(path=...)`` 会抛「already accessed by another
        instance」——压测爬坡期的瞬时 500 与重复样板扫描都源于此。
        """
        with self._load_lock:
            if self.store is None:
                from rag04.ingest.store import VectorStore
                self.store = VectorStore(self.s)
            if self.bm25 is None:
                from rag04.retrieve.bm25 import BM25Index
                p = Path(self.s.data_dir) / "bm25.pkl"
                self.bm25 = BM25Index.load(p) if p.exists() else BM25Index()
            if self.s.use_hybrid and self.boilerplate is None:
                self._load_boilerplate()

    def _load_boilerplate(self) -> None:
        """RC1：载入/建立跨页样板集合。只读现有索引，一次扫描后走缓存。

        baseline_03 是控制组（use_hybrid=False），不引入本阶段检索侧修复。
        任何失败都只记录不抛出：过滤是质量优化，不是问答正确性的前提。
        """
        try:
            from rag04.retrieve.boilerplate import collect_chunks, load_or_build

            path = Path(self.s.data_dir) / "boilerplate.json"
            try:
                counts = self.store.counts()
            except Exception:
                counts = None
            self.boilerplate = load_or_build(
                path, chunks=lambda: collect_chunks(self.store), counts=counts)
        except Exception as e:
            self.boilerplate = None
            self.log.warning("样板判据不可用（%s: %s），本次问答不做样板过滤",
                             type(e).__name__, e)

    def ask(self, question: str) -> Answer:
        self._ensure_loaded()
        return ask(question, self.s, store=self.store, bm25=self.bm25,
                   boilerplate_ids=(self.boilerplate.ids if self.boilerplate
                                    else None))

    def health(self) -> dict:
        """健康检查：各组件可用性。"""
        out: dict = {"mode": self.s.pipeline_mode}
        try:
            self._ensure_loaded()
            out["qdrant"] = "ok" if self.store else "down"
            out["counts"] = self.store.counts() if self.store else {}
        except Exception as e:
            out["qdrant"] = f"down: {type(e).__name__}"
        out["bm25"] = "ok" if (self.bm25 and self.bm25.chunks) else "empty"

        import os
        import requests
        try:
            r = requests.get(f"{self.s.ollama_url.rstrip('/')}/api/tags", timeout=3)
            out["ollama"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            out["ollama"] = f"down: {type(e).__name__}"
        out["llm"] = "ok" if (os.environ.get("API") or os.environ.get("QWEN_API")) else "no key"

        clip_dir = Path(self.s.models_dir) / "clip-vit-base-patch32"
        out["clip"] = "ok" if (clip_dir / "pytorch_model.bin").exists() else "missing"
        rr_dir = Path(self.s.models_dir) / "bge-reranker-base"
        out["rerank"] = "ok" if (rr_dir / "pytorch_model.bin").exists() else "missing(启发式降级)"
        out["figures"] = "on" if self.s.use_figures else "off(baseline)"
        out["tables"] = "on" if self.s.use_tables else "off(baseline)"
        return out
