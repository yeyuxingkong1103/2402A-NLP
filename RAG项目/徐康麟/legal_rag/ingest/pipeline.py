# -*- coding: utf-8 -*-
"""入库编排：解析 -> 分块 -> 向量化 -> 入库，带 MD5 指纹增量去重。

链路（本阶段默认走真实后端）
----------------------------
    PDF/文本 -> pypdf 解析 -> 父子分块 -> Ollama bge-m3 嵌入 -> Milvus 写入

增量策略
--------
以文件 MD5 为指纹记录在 manifest（``index/ingest_manifest.json``）中，指纹未变则整篇
跳过；因此重复执行不会产生任何重复 chunk。另外 chunk 的 id 由内容派生（``stable_id``），
天然幂等，即便重复 upsert 也是覆盖语义——与 Milvus 的「同主键覆盖」一致。

可观测性（全部复用 t1 的产物）
------------------------------
* 每个阶段一条 ``[IO]`` 日志（在/出、条数、字节、耗时、失败原因）；
* 指标：``parse_seconds`` / ``chunk_seconds`` / ``upsert_seconds`` / ``embed_seconds``、
  ``docs_total`` / ``chunks_total`` / ``ingest_failed_total``；
* 整批结束打一条 ``METRIC`` 汇总行（``emit_request_metrics``）；
* 单文件失败**不阻断整批**，但一定 ``logger.exception`` + 计入 ``ingest_failed_total``。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import metrics as M
from ..config import ChunkConfig, RagConfig
from ..embedding.base import Embedder
from ..observability import bind_request, log_io, reset_request, timed, traced, wrap_errors
from ..roles import ROLE_LIBRARY
from ..store.base import VectorStore
from ..utils import md5_of_file, now_ts
from .chunker import build_chunks
from .loaders import SUPPORTED_SUFFIXES, iter_document_paths, load_document

logger = logging.getLogger(__name__)

MANIFEST_NAME = "ingest_manifest.json"


def group_sources_by_role(sources, *, default_role: str | None = None
                          ) -> list[tuple[str, list[str]]]:
    """把知识来源按**角色**分组，返回 ``[(role_id, [路径, ...]), ...]``。

    目录约定（角色隔离的落地点）：``knowledge/<role_id>/`` 下的语料归属该角色，
    例如 ``knowledge/lawyer/`` 全部打上 ``role_id="lawyer"``。

    规则：

    * 传入的路径**本身就是角色目录**（目录名命中 ``ROLE_LIBRARY``）-> 整目录归该角色；
    * 传入的是父目录且**含**角色子目录 -> 每个角色子目录各归其角色；
      该父目录根部的散落文件**不入库**（放错位置了，左右为难：给谁都可能串味，
      所以只告警不擅自归属）；
    * 其余来源（单个文件、无角色子目录的目录）-> 归 ``default_role``；
      若 ``default_role`` 为 None 则归 ``role_id=""``（未标注）。

    **为什么默认要给 default_role**：检索侧按 ``role_id`` 严格等值过滤，
    未标注的语料**任何角色都召不回**。如果不给默认值，``--source uploads/x.pdf``
    这类用法就会变成「入库成功、但谁也搜不到」的静默失败 ——
    所以标准工具一律带上 ``DEFAULT_ROLE_ID`` 兜底并告警，
    把「未标注」留给显式需要它的运维场景。
    """
    groups: dict[str, list[str]] = {}
    loose: list[str] = []
    for item in sources or []:
        path = Path(item)
        if path.is_dir() and path.name in ROLE_LIBRARY:
            groups.setdefault(path.name, []).append(str(path))
            continue
        if path.is_dir():
            subdirs = [child for child in sorted(path.iterdir())
                       if child.is_dir() and child.name in ROLE_LIBRARY]
            if subdirs:
                for subdir in subdirs:
                    groups.setdefault(subdir.name, []).append(str(subdir))
                loose.extend(str(child) for child in sorted(path.iterdir())
                             if child.is_file()
                             and child.suffix.lower() in SUPPORTED_SUFFIXES)
                continue
        groups.setdefault(default_role or "", []).append(str(path))

    if loose:
        logger.warning(
            "知识目录根部有 %d 个文件**不在角色子目录里**，已跳过不入库：%s —— "
            "请放进 knowledge/<role_id>/（如 knowledge/lawyer/），"
            "否则无法判断该知识属于哪个角色",
            len(loose), [Path(p).name for p in loose][:5])
    for role_id, paths in groups.items():
        if role_id != default_role:
            continue
        if role_id:
            logger.warning("来源 %s 没有角色子目录，已按默认角色 role_id=%r 入库",
                           [Path(p).name for p in paths][:5], role_id)
        else:
            logger.warning("来源 %s 没有角色归属（role_id=\"\"）："
                           "这些知识任何角色都检索不到，仅显式 where 的运维检索可见",
                           [Path(p).name for p in paths][:5])
    return sorted(groups.items())



@dataclass
class IngestReport:
    documents: int = 0
    skipped: int = 0
    failed: int = 0
    parents: int = 0
    children: int = 0
    elapsed: float = 0.0
    files: list[dict] = field(default_factory=list)

    @property
    def chunks(self) -> int:
        return self.parents + self.children

    def summary(self) -> str:
        return (
            f"入库文件 {self.documents}，跳过 {self.skipped}，失败 {self.failed}，"
            f"新增 chunk {self.chunks}（父块 {self.parents} / 子块 {self.children}），"
            f"耗时 {self.elapsed:.2f}s"
        )


class KnowledgeBasePipeline:
    def __init__(self, embedder: Embedder, store: VectorStore,
                 config: RagConfig | None = None,
                 manifest_path: Path | None = None,
                 manifest_save_every: int = 20) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or RagConfig()
        self.manifest_path = manifest_path or (self.config.index_dir / MANIFEST_NAME)
        #: 长任务（上千篇法规）中途挂掉时，进度不能只在整批结束时才落盘，
        #: 否则重跑要把已经嵌入好的上千篇再嵌入一遍。0/负数 = 只在结束时落盘。
        self.manifest_save_every = int(manifest_save_every)

    # ---------- manifest ----------
    def load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("manifest 损坏，按空处理: %s", self.manifest_path)
            return {}

    def save_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---------- 单文件 ----------
    # P0.5：单文件入库（解析 + 分块 + 嵌入 + 写入）整段耗时，阈值 15000ms ——
    # 扫描件走 OCR、或大 PDF 嵌几百块时才会触发，正是运维想知道的那些文件。
    @timed(operation="pipeline.ingest_one", slow_ms=15000.0)
    def ingest_one(self, path: Path, *, role_id: str = "",
                   detail: dict | None = None) -> list:
        """解析 -> 分块 -> 嵌入 -> 写入；返回本次写入的 chunk 列表。

        每个阶段都有独立指标与 ``[IO]`` 日志；异常一律带上下文抛出（不吞），
        由 :meth:`ingest` 决定「记失败并继续下一篇」。

        ``detail`` 是**可选出参**（默认 None，老调用方零影响）：传入 dict 时会把
        解析报告（``document.parse``，含 OCR 状态/表格数/页数）写进去，
        让调用方能向用户交代"这份文件是怎么被解析的"。
        """
        parse_started = time.perf_counter()
        with log_io(str(path), "document", operation="parse",
                    size=path.stat().st_size if path.is_file() else None) as io:
            try:
                document = load_document(path)
            except Exception as exc:  # noqa: BLE001 - 带上下文继续抛
                logger.error("解析文档失败：%s（文件=%s，后缀=%s）",
                             exc, path, path.suffix.lower())
                raise
            io["meta"]["suffix"] = document.suffix
            io["meta"]["chars"] = len(document.text)
            if isinstance(detail, dict):
                detail["parse"] = dict(document.parse or {})
                detail["chars"] = len(document.text)
        M.observe("parse_seconds", time.perf_counter() - parse_started,
                  suffix=path.suffix.lower().lstrip(".") or "unknown")

        chunk_started = time.perf_counter()
        with log_io(f"document:{document.doc_id}", "chunks", operation="chunk",
                    size=len(document.text)) as io:
            try:
                chunks = build_chunks(document, self.config.chunk)
            except Exception as exc:  # noqa: BLE001
                logger.error("分块失败：doc_id=%s 文件=%s 文本长度=%d",
                             document.doc_id, path, len(document.text))
                raise
            if role_id:
                for chunk in chunks:
                    chunk.role_id = role_id
            parents = sum(1 for c in chunks if c.is_parent)
            io["count"] = len(chunks)
            io["meta"].update({"parents": parents, "children": len(chunks) - parents,
                               "role_id": role_id or None})
        M.observe("chunk_seconds", time.perf_counter() - chunk_started,
                  strategy=self.config.chunk.strategy)
        if not chunks:
            logger.warning("文档未产出任何 chunk（文件=%s，文本长度=%d），跳过写入",
                           path, len(document.text))
            return []

        texts = [c.text for c in chunks]
        embed_started = time.perf_counter()
        with wrap_errors("pipeline.embed", file=path.name, chunks=len(chunks),
                         model=getattr(self.embedder, "model", self.embedder.name)):
            vectors = self.embedder.embed_texts(texts)
        embed_elapsed = time.perf_counter() - embed_started
        # 缓存命中/未命中是「走 Redis 还是走 Ollama」的分叉点，必须留痕：
        # 一行汇总回答「这次 ingest 到底有没有真的调用 Ollama」（t15）。
        # 说明：cache_hits 之类是本进程累计值，这里只做「本次」增量对比；
        # 支持该信息的后端会填 last_embed_stats，其它后端退化为「未命中=全部」。
        stats = getattr(self.embedder, "last_embed_stats", None)
        if isinstance(stats, dict) and stats.get("total") is not None \
                and int(stats.get("total") or 0) == len(texts):
            hits = int(stats.get("cache_hits") or 0)
            misses = int(stats.get("cache_misses") or 0)
            http_texts = int(stats.get("http_texts") or 0)
            unique = hits + misses
            dedup_note = "" if unique == len(texts) else f"，文本去重后 {unique} 条"
            logger.info(
                "[IO] ✓ embed 汇总：文件=%s 条数=%d%s（缓存命中 %d / 未命中 %d，"
                "真正走 Ollama %d 条；%s），耗时 %.1fms；"
                "embed_seconds 只统计真实 HTTP，命中项不产生样本（属设计）",
                path.name, len(texts), dedup_note, hits, misses, http_texts,
                "本次未调用 Ollama" if http_texts == 0 else "已调用 Ollama",
                embed_elapsed * 1000.0)
        else:
            logger.info("[IO] [OK] embed 汇总：文件=%s 条数=%d（该嵌入后端不上报缓存统计），"
                        "耗时 %.1fms", path.name, len(texts), embed_elapsed * 1000.0)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"嵌入结果数量与分块数量不一致：分块 {len(chunks)} 条，向量 {len(vectors)} 条"
                f"（文件={path}，embedder={self.embedder.name}）")
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector

        upsert_started = time.perf_counter()
        with log_io(f"chunks[{len(chunks)}]", f"{self.store.name}:{path.name}",
                    operation="upsert", count=len(chunks),
                    size=sum(len(t) for t in texts)) as io:
            try:
                written = self.store.upsert(chunks)
            except Exception as exc:  # noqa: BLE001
                logger.error("写入向量库失败：文件=%s 条数=%d 维度=%s store=%s",
                             path, len(chunks),
                             len(chunks[0].vector) if chunks and chunks[0].vector else "?",
                             self.store.name)
                raise
            io["meta"]["written"] = written
        M.observe("upsert_seconds", time.perf_counter() - upsert_started,
                  store=self.store.name)
        M.counter("chunks_total").inc(len(chunks), store=self.store.name)
        M.counter("docs_total").inc(store=self.store.name)
        return chunks

    # ---------- 主流程 ----------
    @traced(operation="pipeline.ingest", level=logging.INFO, log_result=False)
    def ingest(self, targets, rebuild: bool = False, role_id: str = "") -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        paths = iter_document_paths(targets)

        token = bind_request(role_id=role_id)
        manifest: dict = {}
        # 只有真正读到过 manifest 才允许在 finally 里回写：
        # 否则「加载之前就抛异常」会用空 dict 覆盖掉已有的入库清单（灾难性）。
        manifest_loaded = False
        try:
            logger.info("入口 ingest：目标 %s，待处理文件 %d 个，rebuild=%s，store=%s，embedder=%s",
                        targets if not isinstance(targets, (list, tuple)) else list(targets),
                        len(paths), rebuild, self.store.name, self.embedder.name)
            if rebuild:
                logger.info("rebuild=True：忽略指纹，全量重建")

            manifest = self.load_manifest()
            manifest_loaded = True
            chunk_config: ChunkConfig = self.config.chunk
            logger.debug("分块参数：strategy=%s chunk_size=%d overlap=%d parent_size=%d",
                         chunk_config.strategy, chunk_config.chunk_size,
                         chunk_config.overlap, chunk_config.parent_size)

            processed = 0
            for path in paths:
                processed += 1
                # 节流落盘：放在循环开头，任何分支（成功/跳过/失败）都已把上一篇
                # 记进 manifest，因此这里落盘一定不含「写了一半」的篇目。
                if (self.manifest_save_every > 0
                        and processed % self.manifest_save_every == 0):
                    self.save_manifest(manifest)
                    logger.info("manifest 已落盘（已处理 %d/%d 篇）", processed - 1, len(paths))
                logger.info("入口 ingest -> 处理 %s", path)

                try:
                    fingerprint = md5_of_file(path)
                except OSError as exc:
                    logger.exception("读取文件失败: %s", path)
                    M.counter("ingest_failed_total").inc(reason="read")
                    report.failed += 1
                    report.files.append({"file": path.name, "status": "failed",
                                         "reason": str(exc)})
                    continue

                record = manifest.get(path.name) or {}
                if not rebuild and record.get("md5") == fingerprint:
                    logger.info("跳过（MD5 指纹未变）: %s", path.name)
                    report.skipped += 1
                    report.files.append({"file": path.name, "status": "skipped",
                                         "reason": "md5 unchanged"})
                    continue

                try:
                    # `detail` 是**可选出参**：把解析报告（含 OCR 状态）带出来，
                    # 供上传接口向用户交代"这份文件是怎么被解析的"。
                    # 为什么用出参而不是改返回类型：`ingest_one` 的返回值被多处使用，
                    # 改类型会连带改一批调用方；出参对老调用方零影响。
                    detail: dict = {}
                    chunks = self.ingest_one(path, role_id=role_id, detail=detail)
                    parents = sum(1 for c in chunks if c.is_parent)
                    children = len(chunks) - parents

                    manifest[path.name] = {
                        "md5": fingerprint,
                        "doc_id": chunks[0].doc_id if chunks else Path(path).stem,
                        "chunks": len(chunks),
                        "parents": parents,
                        "children": children,
                        "role_id": role_id,
                        "embedder": self.embedder.name,
                        "store": self.store.name,
                        "ingested_at": now_ts(),
                    }

                    report.documents += 1
                    report.parents += parents
                    report.children += children
                    record = {
                        "file": path.name,
                        "status": "ingested",
                        "chunks": len(chunks),
                        "parents": parents,
                        "children": children,
                    }
                    if detail.get("parse"):
                        record["parse"] = dict(detail["parse"])
                    report.files.append(record)
                    logger.info("出口 ingest -> %s 写入 %d 个 chunk（父 %d / 子 %d）",
                                path.name, len(chunks), parents, children)
                except Exception as exc:  # noqa: BLE001 - 单文件失败不阻断整批
                    logger.exception("入库失败: %s（阶段见上方 [IO]/[ERR] 日志）", path)
                    M.counter("ingest_failed_total").inc(reason=type(exc).__name__)
                    report.failed += 1
                    report.files.append({"file": path.name, "status": "failed",
                                         "reason": f"{type(exc).__name__}: {exc}"})

            with log_io("pipeline", f"{self.store.name}:{self.store.name}",
                        operation="persist", count=report.chunks):
                self.store.persist()
            self.save_manifest(manifest)
            report.elapsed = time.perf_counter() - started
            logger.info("入口 ingest 完成: %s", report.summary())
        finally:
            # 异常/中断（含 Ctrl-C、单篇异常冒出循环）时也要保住已完成篇目：
            # 只记「已经完整入库」的文件，绝不记写了一半的那篇。
            if manifest_loaded:
                try:
                    self.save_manifest(manifest)
                except Exception:  # noqa: BLE001 - 落盘失败不能掩盖原始异常
                    logger.exception("异常退出时保存 manifest 失败：%s", self.manifest_path)
            # 指标汇总行（request_id 由上下文自动带上），CLI 一次性任务顺便清空
            try:
                M.emit_request_metrics(
                    extra={"phase": "ingest", "docs": report.documents,
                           "chunks": report.chunks, "failed": report.failed},
                    flush=False)
            except Exception:  # pragma: no cover - 指标失败不影响入库
                logger.debug("写指标汇总失败", exc_info=True)
            reset_request(token)
        return report

    # ---------- 查询辅助 ----------
    def stats(self) -> dict:
        manifest = self.load_manifest()
        return {
            "chunks_in_store": self.store.count(),
            "documents": len(manifest),
            "manifest_path": str(self.manifest_path),
            "embedder": self.embedder.name,
            "store": self.store.name,
        }

    def manifest_records(self) -> list[dict]:
        manifest = self.load_manifest()
        return [{"file": name, **record} for name, record in manifest.items()]
