# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

F9 修复：文档解析的**有界队列 + 工作线程池**（对应 RAGFlow 的 Task Executor）
—— ``研发/service/task_queue.py``。

## 与工单原文的对应

工单原文任务三-2-①：「对耗时较长的『文档解析』任务，确保其已通过 Redis 队列由
Task Executor 异步处理，**API Server 不阻塞**」。

本仓库基座（工单10）**没有上传/解析端点**，所以本工单：
* **优化前口径**（``PERF_ENABLED=0``）：解析**在请求线程里同步跑完**才返回
  —— 这正是原文描述的「API Server 被解析阻塞」反模式的可复现实现，
  也是场景B（5 问答 + 5 上传解析）里问答 P95 飙升的**直接原因**（实测数据见
  ``优化/评估结果/对比.md``）；
* **优化后口径**（``PERF_ENABLED=1``）：上传即 ``202 + task_id``，
  解析进 :class:`ParseQueue` 的有界队列，由 ``PERF_PARSE_WORKERS`` 个工作线程消费；
  队列满 → ``429``（不堆积）。

> 诚实性声明：这是**如实重建**原文描述的反模式，不是伪造性能差异；
> 两种口径都跑同一个解析函数（同一个 PyMuPDF 实现），唯一差别是「在哪个线程跑、有没有队列」。

## 解析实现

* ``fitz``(PyMuPDF) 打开 → 页数、前 ``PERF_PARSE_MAX_PAGES`` 页文本抽样、字符数；
* 只做**统计与分块估算**，不写索引入库（索引目录是只读语义，且全量入库耗时不可控）；
* 任何异常（加密 PDF、损坏文件）→ 任务 ``failed`` + ``error``，**服务不崩**；
* 任务表落盘 ``部署/日志/parse_tasks.jsonl``（边跑边 append，供报告与队列消费取证）。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .perf_config import PerfConfig, get_perf_config

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

#: PDF 魔数
_PDF_MAGIC = b"%PDF-"


class UploadRejected(Exception):
    """上传被拒（400 / 429）。"""

    def __init__(self, message: str, *, status: int = 400, **extra: Any) -> None:
        self.message = message
        self.status = int(status)
        self.extra = extra
        super().__init__(message)


@dataclass
class ParseTask:
    """一个解析任务。"""

    task_id: str
    filename: str
    size_bytes: int
    saved_path: str = ""
    status: str = "pending"          # pending | running | done | failed
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_ms: float | None = None
    pages: int = 0
    chars: int = 0
    chunks_est: int = 0
    error: str | None = None
    worker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ====================================================================== #
# 解析实现（两条口径共用同一个函数 —— 保证对比公平）
# ====================================================================== #
def parse_pdf(path: Path, *, max_pages: int = 40) -> dict[str, Any]:
    """真实解析一个 PDF：页数 + 文本抽样 + 分块估算。异常一律抛出由调用方转任务失败。"""
    import fitz  # PyMuPDF（惰性导入：不上传解析就不加载）

    doc = fitz.open(str(path))
    try:
        if doc.needs_pass:
            raise ValueError("PDF 已加密，需要密码")
        pages = int(doc.page_count)
        limit = max(1, min(int(max_pages), pages))
        chars = 0
        for i in range(limit):
            page = doc.load_page(i)
            chars += len(page.get_text("text") or "")
            # 及时释放页面对象（共享约定 §5.5 规则4：流式处理、立即释放）
            del page
        scale = (pages / limit) if limit else 1.0
        est_chars = int(chars * scale)
        return {"pages": pages, "pages_sampled": limit, "chars": est_chars,
                "chunks_est": max(1, est_chars // 500)}
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass


class ParseQueue:
    """有界解析队列 + 工作线程池（线程安全）。

    工单17 的**磁盘泄漏**治理（F12）：
        基座的上传路径每次都把整份文件复制到 ``DEPLOY_UPLOAD_DIR``
        （文件名前缀是 task_id），**没有任何内容去重**。
        实测一轮压测把同一份 4.6 MB 年报重复上传 → ``2304`` 份文件 / ``15430.2 MB``，
        其中 **99.61 %（15369.9 MB）是重复内容**（抽样 SHA256 证明内容完全一致，
        证据见 ``测试/测试数据/存储卷取证_清理前.json``）。
        现在：``PERF_UPLOAD_DEDUP=1`` 时按内容哈希查索引，命中即**复用已有文件**，不再重复落盘。
    """

    def __init__(self, cfg: PerfConfig, *, upload_dir: Path, log_dir: Path,
                 logger: Any = None) -> None:
        self.cfg = cfg
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger
        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=max(1, cfg.parse_queue_max))
        self._lock = threading.RLock()
        self._tasks: dict[str, ParseTask] = {}
        self._order: list[str] = []
        self._file = (self.log_dir / "parse_tasks.jsonl").open("a", encoding="utf-8")
        self._workers: list[threading.Thread] = []
        self.submitted = 0
        self.rejected = 0
        self.done = 0
        self.failed = 0
        self._stopping = False
        # —— 工单17 F12：上传内容去重（磁盘泄漏治理）——
        #   实测：2304 份上传文件只对应 10 组唯一内容，浪费 15 369.9 MB（99.61 %）。
        #   索引落盘 ``上传去重索引.json``（sha256 → 已存文件名），启动载入、写入时节流落盘。
        self.dedup_enabled = bool(cfg.enabled and cfg.upload_dedup)
        self.dedup_index_path = self.upload_dir / "上传去重索引.json"
        self._dedup: dict[str, str] = {}
        self.dedup_hits = 0
        self.dedup_writes = 0
        self.dedup_saved_bytes = 0
        self._dedup_dirty = 0
        if self.dedup_enabled:
            self._load_dedup_index()
        # ``parse_workers=0`` 表示**不起消费线程**（诊断/测试用：让队列可以被堆满）。
        # 默认 2；生产不要设 0，否则上传只会入队不会解析（会有一条显式告警）。
        workers = max(0, int(cfg.parse_workers))
        if workers == 0 and self.log is not None:
            try:
                self.log.log_event("parse.no_workers", level="WARNING",
                                   reason="PERF_PARSE_WORKERS=0：解析队列只入队不消费")
            except Exception:  # noqa: BLE001
                pass
        for i in range(workers):
            t = threading.Thread(target=self._worker, name=f"parse-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # -- 对外 ---------------------------------------------------------- #
    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    def validate(self, filename: str, content: bytes, *, max_mb: float = 32.0) -> None:
        """上传校验：扩展名 / 大小 / PDF 魔数 / 可解析性。失败抛 :class:`UploadRejected`。"""
        if not content:
            raise UploadRejected("空文件")
        if not str(filename).lower().endswith(".pdf"):
            raise UploadRejected(f"只接受 .pdf 文件（收到 {filename}）")
        size_mb = len(content) / (1024 * 1024)
        if size_mb > max_mb:
            raise UploadRejected(f"文件 {size_mb:.1f} MB 超过上限 {max_mb} MB")
        if not content[:5] == _PDF_MAGIC:
            raise UploadRejected(f"不是有效的 PDF（缺少 %PDF- 魔数，实际前缀 {content[:5]!r}）")

    # -- 工单17 F12：内容去重 -------------------------------------------- #
    def _hash_of(self, content: bytes) -> str:
        algo = (self.cfg.upload_dedup_algo or "sha256").lower()
        if algo == "md5":
            import hashlib

            return hashlib.md5(content).hexdigest()          # noqa: S324 —— 去重用途，非安全用途
        import hashlib

        return hashlib.sha256(content).hexdigest()

    def _load_dedup_index(self) -> None:
        try:
            if self.dedup_index_path.is_file():
                data = json.loads(self.dedup_index_path.read_text(encoding="utf-8"))
                self._dedup = {str(k): str(v) for k, v in (data.get("index") or {}).items()}
                # 只保留仍然存在的文件
                self._dedup = {k: v for k, v in self._dedup.items()
                               if (self.upload_dir / v).is_file()}
        except (OSError, ValueError):
            self._dedup = {}

    def _save_dedup_index(self, force: bool = False) -> None:
        self._dedup_dirty += 1
        if not force and self._dedup_dirty < 20:
            return
        self._dedup_dirty = 0
        try:
            payload = {"work_order": WORK_ORDER, "algo": self.cfg.upload_dedup_algo,
                       "saved_at": time.time(), "entries": len(self._dedup),
                       "index": self._dedup}
            tmp = self.dedup_index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.dedup_index_path)
        except OSError:
            pass

    def dedup_stats(self) -> dict[str, Any]:
        return {"enabled": self.dedup_enabled, "algo": self.cfg.upload_dedup_algo,
                "index_entries": len(self._dedup), "hits": self.dedup_hits,
                "writes": self.dedup_writes,
                "saved_bytes": self.dedup_saved_bytes,
                "saved_mb": round(self.dedup_saved_bytes / (1024 * 1024), 2)}

    def submit(self, filename: str, content: bytes, *, max_mb: float = 32.0,
               validate_parseable: bool = True) -> ParseTask:
        """校验 → 落盘（**内容去重**）→ 入队；队列满抛 ``UploadRejected(status=429)``。"""
        self.validate(filename, content, max_mb=max_mb)
        task_id = "t-" + uuid.uuid4().hex[:8]
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(filename))[:80]

        # ⚠️ 实测踩坑：**先落盘再校验可解析性**会留下无法删除的残file ——
        #    Windows 上 ``fitz.open(path)`` 失败时可能仍持有文件句柄，随后的
        #    ``unlink`` 抛 PermissionError 被静默吞掉，坏文件永久留在上传卷里
        #    （由 测试/test_06_容错与降级.py 抓出）。
        #    正解：**在内存里**先校验（``fitz.open(stream=...)``），不通过就根本不落盘。
        if validate_parseable:
            try:
                import fitz

                doc = fitz.open(stream=content, filetype="pdf")
                try:
                    if doc.needs_pass:
                        raise ValueError("PDF 已加密，需要密码")
                finally:
                    doc.close()
            except Exception as exc:  # noqa: BLE001
                raise UploadRejected(
                    f"PDF 已损坏或无法解析：{type(exc).__name__}: {exc}") from exc

        digest = self._hash_of(content) if self.dedup_enabled else ""
        reused = bool(digest) and digest in self._dedup
        if reused:
            target = self.upload_dir / self._dedup[digest]
            self.dedup_hits += 1
            self.dedup_saved_bytes += len(content)
        else:
            target = self.upload_dir / f"{task_id}__{safe}"
            target.write_bytes(content)
            self.dedup_writes += 1
            if digest:
                self._dedup[digest] = target.name
                self._save_dedup_index()

        task = ParseTask(task_id=task_id, filename=str(filename), size_bytes=len(content),
                         saved_path=str(target), status="pending")
        with self._lock:
            self._tasks[task_id] = task
            self._order.append(task_id)
        try:
            self.queue.put_nowait(task_id)
        except queue.Full:
            with self._lock:
                task.status = "failed"
                task.error = f"解析队列已满（{self.cfg.parse_queue_max}）"
                self.rejected += 1
            raise UploadRejected(f"解析队列已满（{self.queue_depth}/{self.cfg.parse_queue_max}）",
                                 status=429, queue_depth=self.queue_depth) from None
        self.submitted += 1
        self._write(task, event="submitted", queue_depth=self.queue_depth)
        return task

    def get(self, task_id: str) -> ParseTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order[-limit:]))
            return [self._tasks[i].to_dict() for i in ids if i in self._tasks]

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for t in self._tasks.values():
                out[t.status] = out.get(t.status, 0) + 1
        return out

    def stats(self) -> dict[str, Any]:
        return {"backend": "memory-queue", "workers": len(self._workers),
                "queue_depth": self.queue_depth, "queue_max": self.cfg.parse_queue_max,
                "submitted": self.submitted, "rejected": self.rejected,
                "done": self.done, "failed": self.failed, "counts": self.counts(),
                "tasks_total": len(self._tasks), "dedup": self.dedup_stats()}

    # -- 内部 ---------------------------------------------------------- #
    def _worker(self) -> None:
        while not self._stopping:
            try:
                task_id = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            task: ParseTask | None = None
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.status = "running"
                    task.started_at = time.time()
                    task.worker = threading.current_thread().name
            if task is None:
                self.queue.task_done()
                continue
            started = time.perf_counter()
            try:
                info = parse_pdf(Path(task.saved_path), max_pages=self.cfg.parse_max_pages)
                with self._lock:
                    task.status = "done"
                    task.pages = int(info["pages"])
                    task.chars = int(info["chars"])
                    task.chunks_est = int(info["chunks_est"])
                    self.done += 1
            except Exception as exc:  # noqa: BLE001 —— 任何解析失败都转任务失败，服务不崩
                with self._lock:
                    task.status = "failed"
                    task.error = f"{type(exc).__name__}: {exc}"
                    self.failed += 1
                if self.log is not None:
                    try:
                        self.log.log_event("parse.failed", level="WARNING", task_id=task_id,
                                           file=task.filename,
                                           trace=traceback.format_exc()[-500:])
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                with self._lock:
                    task.elapsed_ms = elapsed
                    task.finished_at = time.time()
                self._write(task, event="finished", queue_depth=self.queue_depth)
                self.queue.task_done()

    def _write(self, task: ParseTask, *, event: str, **extra: Any) -> None:
        row = {"ts": round(time.time(), 3), "event": event, "work_order": WORK_ORDER,
               **task.to_dict(), **extra}
        try:
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001 —— 留痕失败绝不影响解析
            pass

    def close(self) -> None:
        self._stopping = True
        for t in self._workers:
            t.join(timeout=2.0)
        if self.dedup_enabled:
            self._save_dedup_index(force=True)
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass


# ====================================================================== #
# 「优化前」口径：在请求线程里同步解析（如实重建原文描述的反模式）
# ====================================================================== #
_SYNC_LOCK = threading.Lock()


def parse_sync(filename: str, content: bytes, *, cfg: PerfConfig, upload_dir: Path,
               log_dir: Path) -> dict[str, Any]:
    """同步解析（阻塞请求线程）—— ``PERF_ENABLED=0`` 时的口径。

    与 :class:`ParseQueue` 用**同一个** :func:`parse_pdf`，唯一差别是不排队、不异步。
    """
    validator = ParseQueue.__new__(ParseQueue)          # 只借校验方法，不启线程池
    validator.validate(filename, content)
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    task_id = "t-" + uuid.uuid4().hex[:8]
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(filename))[:80]
    target = upload_dir / f"{task_id}__{safe}"
    target.write_bytes(content)
    started = time.perf_counter()
    row: dict[str, Any] = {"task_id": task_id, "filename": str(filename),
                           "size_bytes": len(content), "saved_path": str(target),
                           "submitted_at": time.time(), "started_at": time.time()}
    try:
        info = parse_pdf(target, max_pages=cfg.parse_max_pages)
        row.update({"status": "done", "pages": info["pages"], "chars": info["chars"],
                    "chunks_est": info["chunks_est"], "error": None})
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "failed", "pages": 0, "chars": 0, "chunks_est": 0,
                    "error": f"{type(exc).__name__}: {exc}"})
    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    row["finished_at"] = time.time()
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with _SYNC_LOCK:
            with (log_dir / "parse_tasks.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({"ts": time.time(), "event": "sync_finished",
                                     "work_order": WORK_ORDER, **row}, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return row


_QUEUE: ParseQueue | None = None
_QUEUE_LOCK = threading.Lock()


def get_parse_queue(cfg: PerfConfig | None = None, *, upload_dir: Path, log_dir: Path,
                    logger: Any = None, refresh: bool = False) -> ParseQueue:
    """进程级单例解析队列。"""
    global _QUEUE
    config = cfg or get_perf_config()
    with _QUEUE_LOCK:
        if _QUEUE is None or refresh:
            _QUEUE = ParseQueue(config, upload_dir=upload_dir, log_dir=log_dir, logger=logger)
        return _QUEUE


__all__ = ["WORK_ORDER", "ParseTask", "ParseQueue", "UploadRejected", "parse_pdf",
           "parse_sync", "get_parse_queue"]
