# -*- coding: utf-8 -*-
"""文档服务：上传入库（job 状态）、列表、删除，以及「库内活条数」Gauge。

只用 t2 的公开接口
------------------
* ``KnowledgeBasePipeline.ingest()`` / ``load_manifest()`` / ``save_manifest()`` —— 不碰 ``ingest/``；
* ``VectorStore`` 的 ``all_chunks`` / ``delete`` / ``count``，以及 Milvus 实现额外提供的
  ``delete_by_doc``（没有时退回「按 doc_id 查 id 再删」的通用路径）。

「库里到底有多少条」只有**一个口径**
------------------------------------
一律用 ``store.count()``（= Milvus ``query(count(*))``），并且把**来源写进指标标签**。
**禁止**使用 ``num_entities`` / ``get_collection_stats()['row_count']`` —— 它们包含
``upsert`` 留下的软删除墓碑，compaction 之前会明显虚报（实测虚报过 5 倍：78 vs 真实 16）。
数字看着正常但它是错的，与「降级了却报 ok」是同一类缺陷。

同时注意区分两类数字（否则界面会把「累计写入」当成「当前存量」）：
* **Gauge** ``vector_store_rows{store=...}`` —— 当前**存量**（会随删除下降）；
* **Counter** ``docs_total`` / ``chunks_total`` —— **本进程生命周期内的累计写入量**，
  ``rebuild`` 会一路涨、进程重启归零，既不是存量也不是持久累计。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .. import metrics as M
from ..ingest.pipeline import KnowledgeBasePipeline
from .uploads import UploadStore

logger = logging.getLogger(__name__)

#: 活条数 Gauge 名（标签 store=memory|milvus）
VECTOR_STORE_ROWS = "vector_store_rows"

#: 活条数缓存秒数：/metrics 必须「轻」，不能让每次抓取都打一次库。
ROWS_CACHE_TTL = 5.0

#: job 状态
JOB_PENDING = "pending"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"


@dataclass
class UploadJob:
    """一次上传（可含多文件）的进度。"""

    job_id: str
    status: str = JOB_PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    files: list[dict] = field(default_factory=list)
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict] = field(default_factory=list)
    #: 顶层告警（**不是错误**：入库确实完成了，但有需要用户知道的事，例如"抽出 0 块"）
    warnings: list[dict] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": round(self.updated_at - self.created_at, 3),
            "documents": self.documents,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "failed": self.failed,
            "files": list(self.files),
        }
        if self.errors:
            payload["errors"] = list(self.errors)
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def _zero_chunks_message(parse: dict) -> str:
    """"抽出 0 块"时给用户看的话 —— 要**按 OCR 的实际状态**分别说清，
    不能一律说"不支持 OCR"（那是旧版文案，现在有 OCR 了）。"""
    status = str(parse.get("ocr_status") or "")
    engine = str(parse.get("ocr_engine") or "")
    if status == "ok":
        # OCR 成功却仍然 0 块：说明识别出来的文字也被切没了（极少见），如实说
        return ("抽出 0 个分块：OCR 已识别出文字，但分块后为空 —— "
                "请把该 PDF 发给我们定位（文件见 stored_name）")
    if status == "empty":
        return (f"抽出 0 个分块：文字层为空，已用 OCR（{engine or '未知引擎'}）识别但**没识别出文字**"
                "—— 可能是图片太糊/不是文字，建议换更清晰的扫描件")
    if status == "failed":
        return ("抽出 0 个分块：文字层为空，且 **OCR 执行失败**（详见服务端日志）"
                "—— 请重试或先把文件 OCR 成文本再上传")
    if status == "unavailable":
        return ("抽出 0 个分块：这份文件很可能是**扫描件/纯图片 PDF**，而**本机没有可用的 OCR 引擎**"
                "（未安装），因此它在检索里不会命中 —— "
                "请安装 OCR（`pip install rapidocr-onnxruntime`）或改用带文字层的 PDF")
    return ("抽出 0 个分块：这份文件很可能是**扫描件/纯图片 PDF**（或内容为空），"
            "因此它在检索里不会命中 —— 请改用带文字层的 PDF，或先把文件 OCR 成文本再上传")


def _warn_zero_chunks(job: "UploadJob", entry: dict) -> None:
    """把"入库成功但抽出 0 块"记进 job 顶层告警 + 日志 + 指标。

    为什么单拎出来：这类文件**状态是 completed、文件也在库里**，唯独检索不到 ——
    不显式告警的话，用户只会觉得"问了没结果"，而运维也看不到任何异常。
    """
    job.warnings.append({
        "file": entry.get("filename") or entry.get("stored_name"),
        "issue": "zero_chunks",
        "message": entry.get("warning", "抽出 0 个分块"),
    })
    logger.warning("上传文件抽出 0 个分块（疑似扫描件/空文件；当前不支持 OCR）："
                   "job=%s file=%s size=%s bytes",
                   job.job_id, entry.get("stored_name"), entry.get("size_bytes"))
    try:
        from .. import metrics as M
        M.counter("upload_zero_chunks_total").inc()
    except Exception:  # noqa: BLE001 - 指标失败不影响主链路
        logger.debug("zero_chunks 计数失败", exc_info=True)


class JobRegistry:
    """有界的内存 job 表（最新 N 条；job 只服务于「刚上传完的进度查询」）。"""

    def __init__(self, max_jobs: int = 200) -> None:
        self.max_jobs = max(int(max_jobs), 1)
        self._jobs: dict[str, UploadJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def new(self) -> UploadJob:
        job = UploadJob(job_id=f"job-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self.max_jobs:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> UploadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = list(reversed(self._order))[: max(int(limit), 1)]
            return [self._jobs[i].to_dict() for i in ids if i in self._jobs]

    def __len__(self) -> int:
        return len(self._jobs)


class DocumentService:
    """上传落盘 + 入库 + 列表 + 删除。线程安全（内部锁保护 job 与 manifest 读写）。"""

    def __init__(self, pipeline: KnowledgeBasePipeline, store: Any, config: Any,
                 *, jobs: JobRegistry | None = None,
                 upload_store: UploadStore | None = None) -> None:
        self.pipeline = pipeline
        self.store = store
        self.config = config
        self.jobs = jobs if jobs is not None else JobRegistry()
        self.uploads = upload_store if upload_store is not None else UploadStore(config)
        self._rows_lock = threading.Lock()
        self._rows_value: int | None = None
        self._rows_at: float = 0.0
        self._rows_error: str = ""
        #: manifest 是「读-改-写」，并发上传/删除时必须串行，否则条目会互相覆盖
        self._manifest_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 活条数（唯一口径：store.count()）
    # ------------------------------------------------------------------
    def store_rows(self, *, force: bool = False) -> int | None:
        """当前**活条数**（``store.count()``）；带 TTL 缓存，失败返回 None 并记原因。

        * 成功即发布 Gauge ``vector_store_rows{store=...}``；
        * **绝不**退回 ``num_entities`` / ``row_count``（见模块 docstring）。
        """
        now = time.time()
        with self._rows_lock:
            fresh = (self._rows_value is not None
                     and (now - self._rows_at) < ROWS_CACHE_TTL)
            if fresh and not force:
                return self._rows_value

        value: int | None = None
        error = ""
        try:
            value = int(self.store.count())
        except Exception as exc:  # noqa: BLE001 - 计数失败不影响服务
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("读取向量库活条数失败（store=%s）：%s",
                           getattr(self.store, "name", "?"), error)

        with self._rows_lock:
            if value is not None:
                self._rows_value = value
                self._rows_at = now
                self._rows_error = ""
                M.gauge(VECTOR_STORE_ROWS,
                        "向量库当前活条数（取自 store.count()，非 num_entities）",
                        "rows").set(float(value),
                                    store=str(getattr(self.store, "name", "unknown")))
            else:
                self._rows_error = error
        return value

    def rows_snapshot(self) -> dict:
        cached: int | None
        with self._rows_lock:
            cached = self._rows_value
            at = self._rows_at
            error = self._rows_error
        payload: dict = {
            "value": cached,
            "source": "store.count()",
            "observed_at": at or None,
            "age_seconds": round(time.time() - at, 3) if at else None,
            "store": str(getattr(self.store, "name", "unknown")),
        }
        if error:
            payload["error"] = error
            payload["hint"] = ("不要改用 num_entities / row_count：它们包含 upsert 墓碑，"
                               "会虚报（实测 78 vs 真实 16）")
        return payload

    # ------------------------------------------------------------------
    # 上传入库
    # ------------------------------------------------------------------
    def create_job(self) -> UploadJob:
        return self.jobs.new()

    def run_job(self, job: UploadJob, saved: list[dict],
                *, role_id: str = "") -> UploadJob:
        """把已落盘的文件入库（同步执行；调用方负责放到线程池里）。

        ``saved`` 每项形如 ``{"filename","stored_path","doc_id","md5","size","created"}``。
        """
        job.status = JOB_RUNNING
        job.touch()
        paths = [Path(item["stored_path"]) for item in saved]
        try:
            report = self.pipeline.ingest(paths, rebuild=False, role_id=role_id)
        except Exception as exc:  # noqa: BLE001 - 入库失败要落进 job，供轮询定位
            job.status = JOB_FAILED
            job.failed = len(paths)
            job.errors.append({"stage": "ingest", "error": f"{type(exc).__name__}: {exc}"})
            job.touch()
            logger.exception("上传入库失败：job=%s 文件=%d", job.job_id, len(paths))
            return job

        by_name = {item["stored_path"].name: item for item in saved}
        for record in report.files:
            item = by_name.get(record["file"], {})
            entry = {
                "filename": item.get("filename", record["file"]),
                "stored_name": record["file"],
                "doc_id": item.get("doc_id", Path(record["file"]).stem),
                "md5": item.get("md5", ""),
                "size_bytes": item.get("size", 0),
                "status": record["status"],
                "chunks": record.get("chunks", 0),
            }
            if record.get("reason"):
                entry["reason"] = record["reason"]
                job.errors.append({"file": record["file"], "reason": record["reason"]})
            # 解析报告（页数 / 表格 / **OCR 状态**）：让用户看得见"这份是怎么被解析的"。
            if record.get("parse"):
                entry["parse"] = dict(record["parse"])
            # ⚠️ **"入库成功但抽出 0 块"必须显式告警**（2026-09-26 补）：
            #    扫描件 PDF 走 pypdf 抽不出文字 ⇒ 分块返回空列表（`ingest/chunker.py`
            #    对空文本返回 `[]`）⇒ 文件状态是 **`ingested`**（不是 failed/skipped）、
            #    也进了 manifest，**但怎么问都检索不到** —— 用户以为入库成功，
            #    日志和指标里也没有任何异常。这类"看起来成功的失败"正是本项目
            #    最忌讳的静默失败。
            #    判定口径：状态是**成功态**（`ingested`）且分块数为 0。
            #    用"成功态"而不是"非失败"来判定，是为了不把 skipped/failed 误报成告警。
            if entry["status"] == "ingested" and int(entry["chunks"] or 0) == 0:
                entry["zero_chunks"] = True
                entry["warning"] = _zero_chunks_message(entry.get("parse") or {})
                _warn_zero_chunks(job, entry)
            job.files.append(entry)

        job.documents = report.documents
        job.chunks = report.chunks
        job.skipped = report.skipped
        job.failed = report.failed
        job.status = JOB_FAILED if report.failed else JOB_COMPLETED
        job.touch()
        self._remember_original_names(saved)
        self.store_rows(force=True)      # 入库后立刻刷新存量
        logger.info("上传入库完成：job=%s 文件=%d 新增 chunk=%d 跳过=%d 失败=%d",
                    job.job_id, len(paths), job.chunks, job.skipped, job.failed)
        return job

    def _remember_original_names(self, saved: list[dict]) -> None:
        """把「用户原始文件名」补进 manifest。

        为什么需要：ingest 用 ``path.stem`` 当 doc_id、``path.name`` 当 source，
        而落盘名是内容 MD5 —— 于是检索引用里只会出现 ``<md5>.pdf``，
        运维/用户根本认不出是哪份文件。这里把原始名作为**附加元数据**记进 manifest
        （不改 ``ingest/`` 的任何代码），``GET /documents`` 会把它展示出来。
        """
        if not saved:
            return
        with self._manifest_lock:
            manifest = self.pipeline.load_manifest()
            changed = False
            for item in saved:
                entry = manifest.get(item["stored_path"].name)
                if isinstance(entry, dict) and item.get("filename"):
                    if entry.get("original_name") != item["filename"]:
                        entry["original_name"] = item["filename"]
                        changed = True
            if changed:
                self.pipeline.save_manifest(manifest)

    # ------------------------------------------------------------------
    # 列表
    # ------------------------------------------------------------------
    def list_documents(self) -> list[dict]:
        """已入库文档列表（manifest 为准，并标注是否来自 uploads/）。"""
        with self._manifest_lock:
            manifest = self.pipeline.load_manifest()
        upload_dir = self.uploads.directory
        rows: list[dict] = []
        for source, record in sorted(manifest.items()):
            doc_id = str(record.get("doc_id") or Path(source).stem)
            path = upload_dir / source
            from_upload = path.is_file()
            if not from_upload:
                candidate = self.uploads.find_by_doc_id(doc_id)
                from_upload = candidate is not None
            rows.append({
                "doc_id": doc_id,
                "source": source,
                # 落盘名是内容 MD5（幂等所需）；original_name 才是用户当初传的文件名
                "original_name": record.get("original_name", ""),
                "md5": record.get("md5", ""),
                "chunks": int(record.get("chunks") or 0),
                "parents": int(record.get("parents") or 0),
                "children": int(record.get("children") or 0),
                "role_id": record.get("role_id", ""),
                "ingested_at": record.get("ingested_at"),
                "embedded_at": record.get("ingested_at"),
                "from_upload": from_upload,
                "file_exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            })
        return rows

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> dict:
        """删除该文档在向量库里的全部 chunk + 磁盘文件 + manifest 记录。"""
        doc_id = str(doc_id).strip()
        if not doc_id:
            raise ValueError("doc_id 不能为空")

        deleted_chunks = self._delete_chunks(doc_id)

        file_deleted = False
        path = self.uploads.find_by_doc_id(doc_id)
        if path is not None:
            file_deleted = self.uploads.delete(path)

        manifest_removed: list[str] = []
        with self._manifest_lock:
            manifest = self.pipeline.load_manifest()
            for source, record in list(manifest.items()):
                if str(record.get("doc_id") or Path(source).stem) == doc_id:
                    manifest.pop(source, None)
                    manifest_removed.append(source)
            if manifest_removed:
                self.pipeline.save_manifest(manifest)

        # 删完先 flush 再报数：Milvus 的删除要 flush 后才会体现在 ``count(*)`` 上，
        # 否则 ``rows_after`` 会显示「删了 2 条、存量还是 18」这种**似是而非的数字**
        # （与"降级了却报 ok"同属一类缺陷）。宁可多花一次 flush，也要报真数。
        try:
            self.store.persist()
        except Exception as exc:  # noqa: BLE001 - flush 失败不影响删除结果
            logger.warning("删除后 flush 失败（rows_after 可能滞后）：%s: %s",
                           type(exc).__name__, exc)

        rows_after = self.store_rows(force=True)
        result = {
            "doc_id": doc_id,
            "deleted_chunks": deleted_chunks,
            "file_deleted": file_deleted,
            "manifest_entries_removed": manifest_removed,
            "rows_after": rows_after,
        }
        logger.info("删除文档：doc_id=%s -> 删除 chunk=%d 文件=%s manifest=%s",
                    doc_id, deleted_chunks, file_deleted, manifest_removed)
        return result

    def _delete_chunks(self, doc_id: str) -> int:
        """删除该 doc_id 的分块，并做「应删 N / 实删 M」对账（t10 关闭 t9 §9 缺口 C）。

        Milvus 实现提供 ``delete_by_doc``（按表达式删，一次搞定）；它的对账在
        store 内部完成（``MilvusVectorStore._delete_by_filter`` 先用 ``count(*)``
        取「匹配 N 条」再删，见 ``milvus_delete_mismatch_total``）。
        其它后端退回通用路径「按 doc_id 查 id -> delete(ids)」——这条路的
        N/M 就是本函数实查到的「查到多少个 id」与「delete 返回删了多少」。

        两者不一致时 **WARNING（默认级别可见）+ ``documents_delete_mismatch_total``
        计数**：少删会留下残留分块，表现为「删了还能检索到」。
        """
        deleter = getattr(self.store, "delete_by_doc", None)
        if callable(deleter):
            return int(deleter(doc_id) or 0)
        chunks = self.store.all_chunks(where={"doc_id": doc_id})
        if not chunks:
            return 0
        expected = len(chunks)
        removed = int(self.store.delete([c.id for c in chunks]) or 0)
        if removed != expected:
            short = expected - removed
            M.counter("documents_delete_mismatch_total").inc(
                abs(short), store=getattr(self.store, "name", "unknown"))
            logger.warning(
                "删除文档 %s 的分块对不上：应删 %d 条、实删 %d 条（差 %+d）—— "
                "少删会留下残留分块（「删了还能检索到」），请查向量库删除路径",
                doc_id, expected, removed, short)
        return removed

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "upload_dir": str(self.uploads.directory),
            "uploaded_files": len(self.uploads.list_files()),
            "jobs_tracked": len(self.jobs),
            "manifest_entries": len(self.pipeline.load_manifest()),
            "rows": self.rows_snapshot(),
        }


__all__ = [
    "VECTOR_STORE_ROWS", "ROWS_CACHE_TTL",
    "JOB_PENDING", "JOB_RUNNING", "JOB_COMPLETED", "JOB_FAILED",
    "UploadJob", "JobRegistry", "DocumentService",
]
