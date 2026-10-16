# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.skill
职责：DocumentQualityAssessmentSkill 主入口。
      * 目录/文件列表扫描与遍历
      * 逐文件解析（进程池并行）+ 进度输出
      * 断点续跑（state.json：按 路径+mtime+size 判定"已处理即跳过"）
      * 五大功能编排 + 分类标签 + 待确认/待审核列表
      * JSON 全量报告 + 自包含 HTML 简报

对外唯一入口：:class:`DocumentQualityAssessmentSkill`
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import detectors as det
from .config import Config, REPO_ROOT, load_config
from .extractors import extract, is_parseable_suffix, normalize_text

SKILL_NAME = "DocumentQualityAssessmentSkill"
SKILL_VERSION = "1.0.0"
# 状态文件结构版本：schema 变化时递增，旧 state 自动失效重建
STATE_SCHEMA = 3

LOG = logging.getLogger("dqa.skill")


# ===========================================================================
# 文件清单
# ===========================================================================
SUPPORTED_SUFFIX_HINT = {
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".md", ".markdown",
    ".txt", ".csv", ".json", ".jsonl", ".html", ".htm",
}


@dataclass
class FileEntry:
    path: str
    suffix: str
    size_bytes: int
    mtime: float

    @property
    def fingerprint(self) -> str:
        return f"{self.size_bytes}:{int(self.mtime)}"


def collect_files(
    folder: str | os.PathLike[str] | None,
    files: Sequence[str] | None,
    cfg: Config,
) -> list[FileEntry]:
    """汇总待检文件：``folder`` 递归遍历 + ``files`` 显式列表（去重、稳定排序）。

    只收集配置中声明过的扩展名（未知扩展名默认跳过，避免把 .exe/.zip 拖进来）。
    """
    recursive = cfg.boolean("performance.recursive", True)
    group_map = {k.lower() for k in (cfg.get("format_stats.extension_groups", {}) or {})}
    seen: set[str] = set()
    out: list[FileEntry] = []

    def _add(p: Path) -> None:
        try:
            if not p.is_file():
                return
            key = str(p.resolve()).lower()
            if key in seen:
                return
            seen.add(key)
            st = p.stat()
            out.append(FileEntry(path=str(p.resolve()), suffix=p.suffix.lower(), size_bytes=st.st_size, mtime=st.st_mtime))
        except OSError:
            return

    if folder:
        base = Path(folder)
        if base.is_file():
            _add(base)
        elif base.is_dir():
            it = base.rglob("*") if recursive else base.glob("*")
            for p in it:
                if p.is_file() and p.suffix.lower() in group_map:
                    _add(p)
    for f in files or []:
        fp = Path(f)
        if not fp.is_absolute() and folder:
            cand = Path(folder) / fp
            if cand.exists():
                fp = cand
        if fp.exists() and fp.is_file() and fp.suffix.lower() in group_map:
            _add(fp)
    out.sort(key=lambda e: e.path.lower())
    return out


# ===========================================================================
# 逐文件 worker（进程池：每个子进程各自加载一次配置）
# ===========================================================================
_WORKER_CFG: Config | None = None
_WORKER_OPTS: dict[str, Any] = {}


def _worker_init(config_path: str | None, opts: dict[str, Any]) -> None:  # pragma: no cover - 子进程
    global _WORKER_CFG, _WORKER_OPTS
    _WORKER_CFG = load_config(config_path, use_cache=False)
    _WORKER_OPTS = opts
    logging.getLogger("dqa.skill").setLevel(logging.WARNING)


def process_one(entry: FileEntry, cfg: Config, *, want_text: bool = False, timeout_sec: float = 0) -> dict[str, Any]:
    """处理单个文件 → 逐文件记录（绝不允许抛异常）。"""
    started = time.perf_counter()
    rec: dict[str, Any] = {
        "path": entry.path,
        "name": Path(entry.path).name,
        "suffix": entry.suffix,
        "size_bytes": entry.size_bytes,
        "mtime": entry.mtime,
        "corrupted": False,
        "encrypted": False,
        "error": "",
        "char_count": 0,
        "page_count": 0,
        "scanned_pages": 0,
        "low_text_pages": 0,
        "scan_ratio": 0.0,
        "scan_ratio_literal": 0.0,
        "has_text_layer": False,
        "pure_scan": False,
        "pdf_type": "",
        "md5": "",
        "simhash": None,
        "sensitive_hits": 0,
        "tags": [],
        "primary_tag": "Normal",
        "elapsed_sec": 0.0,
    }
    text_for_simhash = ""
    try:
        res = extract(entry.path, cfg)
        rec["corrupted"] = bool(res.corrupted)
        rec["encrypted"] = bool(res.encrypted)
        rec["error"] = res.error
        rec["char_count"] = int(res.char_count)
        rec["page_count"] = int(res.page_count)
        if res.is_pdf:
            verdict = det.classify_pdf(res.pages, res.page_count, cfg)
            sampled = int(verdict["sampled_pages"] or 0)
            has_text_layer = bool(res.char_count > 0)
            rec.update(
                {
                    "pdf_type": verdict["pdf_type"],
                    "scanned_pages": verdict["scan_pages"],
                    "low_text_pages": verdict.get("low_text_pages", 0),
                    "sampled_pages": sampled,
                    "scan_ratio": verdict["scan_ratio"],
                    "scan_ratio_literal": verdict.get("scan_ratio_literal", verdict["scan_ratio"]),
                    "avg_chars_per_page": verdict["avg_chars_per_page"],
                    "pdf_needs_confirm": verdict["needs_confirm"],
                    "pdf_confirm_reason": verdict["confirm_reason"],
                    "has_text_layer": has_text_layer,
                    # 纯扫描件：抽样页全部为扫描页且整册抽不到任何文本
                    "pure_scan": bool(
                        verdict["scan_pages"] >= sampled > 0 and not has_text_layer
                    ),
                }
            )
        # MD5（配置可只读前 N 字节）
        try:
            rec["md5"] = det._md5_of_file(Path(entry.path), cfg.integer("duplicate_detection.md5_read_bytes", 0))
        except OSError as exc:
            rec["error"] = (rec["error"] + "; " if rec["error"] else "") + f"md5_failed: {exc}"
        # SimHash：只用文本，扫描型 PDF 无文本层 → 跳过（不参与近似重复比对）
        if cfg.boolean("duplicate_detection.simhash.enabled", True) and res.char_count > 0:
            text_for_simhash = res.text
            rec["simhash"] = det.simhash64(text_for_simhash, cfg)
        # 敏感信息
        hits = det.detect_sensitive_in_text(res.text, cfg)
        rec["sensitive_hits"] = len(hits)
        rec["sensitive_detail"] = hits
    except Exception as exc:  # noqa: BLE001 - 最后一道防线
        rec["corrupted"] = True
        rec["error"] = f"unhandled: {type(exc).__name__}: {exc}"
    rec["elapsed_sec"] = round(time.perf_counter() - started, 4)
    if timeout_sec and rec["elapsed_sec"] > timeout_sec:
        rec["error"] = (rec["error"] + "; " if rec["error"] else "") + f"slow_file_over_{timeout_sec}s"
    if want_text:
        rec["_text"] = text_for_simhash
    return rec


def _worker_task(payload: tuple[dict[str, Any], bool]) -> dict[str, Any]:  # pragma: no cover - 子进程
    entry_dict, want_text = payload
    cfg = _WORKER_CFG or load_config(None, use_cache=False)
    entry = FileEntry(**entry_dict)
    timeout_sec = float(_WORKER_OPTS.get("per_file_timeout_sec", 0) or 0)
    return process_one(entry, cfg, want_text=want_text, timeout_sec=timeout_sec)


# ===========================================================================
# 断点续跑状态
# ===========================================================================
class StateStore:
    """``研发/data/state.json`` 读写：记录已处理文件指纹与逐文件结果。"""

    def __init__(self, path: Path, *, folder: str = "") -> None:
        self.path = path
        self.folder = folder
        self.processed: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}

    @classmethod
    def load(cls, path: Path, *, folder: str = "", reset: bool = False) -> "StateStore":
        store = cls(path, folder=folder)
        if reset or not path.exists():
            return store
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("状态文件损坏，忽略并重建：%s（%s）", path, exc)
            return store
        if not isinstance(raw, dict) or raw.get("schema") != STATE_SCHEMA:
            LOG.warning("状态文件 schema 不匹配（期望 %s），按空状态重建", STATE_SCHEMA)
            return store
        if folder and raw.get("folder") and Path(str(raw["folder"])) != Path(folder):
            LOG.warning("状态文件属于另一个目录（%s），忽略", raw.get("folder"))
            return store
        store.processed = {
            str(k): v for k, v in (raw.get("processed") or {}).items() if isinstance(v, dict)
        }
        store.meta = raw.get("meta") or {}
        return store

    def is_done(self, entry: FileEntry) -> bool:
        rec = self.processed.get(entry.path)
        return bool(rec) and rec.get("fingerprint") == entry.fingerprint

    def record(self, rec: dict[str, Any]) -> None:
        payload = dict(rec)
        payload.pop("_text", None)
        payload["fingerprint"] = f"{rec.get('size_bytes')}:{int(rec.get('mtime', 0))}"
        self.processed[str(rec.get("path"))] = payload

    def save(self, *, meta: dict[str, Any] | None = None) -> None:
        if meta:
            self.meta.update(meta)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": STATE_SCHEMA,
            "skill": SKILL_NAME,
            "skill_version": SKILL_VERSION,
            "folder": self.folder,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processed_count": len(self.processed),
            "meta": self.meta,
            "processed": self.processed,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def records(self) -> list[dict[str, Any]]:
        return list(self.processed.values())


# ===========================================================================
# Skill
# ===========================================================================
@dataclass
class SkillResult:
    report: dict[str, Any]
    json_path: Path | None = None
    html_path: Path | None = None
    per_file_jsonl: Path | None = None

    def to_summary(self) -> dict[str, Any]:
        r = self.report
        return {
            "skill": r.get("skill", {}).get("name"),
            "version": r.get("skill", {}).get("version"),
            "total_files": r.get("summary", {}).get("total_files"),
            "pdf_types": r.get("summary", {}).get("pdf_types"),
            "duplicate_groups": r.get("summary", {}).get("md5_duplicate_groups"),
            "version_conflicts": r.get("summary", {}).get("version_conflicts"),
            "sensitive_files": r.get("summary", {}).get("sensitive_files"),
            "tag_accuracy_note": "标签准确率需人工抽样核对（见 测试/测试数据/抽样核对表.csv）",
        }


class DocumentQualityAssessmentSkill:
    """知识库文档质量评估 Skill。

    用法::

        skill = DocumentQualityAssessmentSkill()
        result = skill.run(folder=r"E:\\...\\documents", resume=True)
        print(result.to_summary())
    """

    name = SKILL_NAME
    version = SKILL_VERSION

    def __init__(self, config: Config | str | os.PathLike[str] | None = None) -> None:
        self.cfg: Config = config if isinstance(config, Config) else load_config(config)
        self.log_lines: list[str] = []

    # -- 日志 ---------------------------------------------------------------
    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        LOG.info(msg)
        print(line, flush=True)

    # -- 主流程 -------------------------------------------------------------
    def run(
        self,
        *,
        folder: str | os.PathLike[str] | None = None,
        files: Sequence[str] | None = None,
        resume: bool = True,
        limit: int | None = None,
        workers: int | None = None,
        want_texts: bool = True,
        write_reports: bool = True,
        json_path: str | os.PathLike[str] | None = None,
        html_path: str | os.PathLike[str] | None = None,
        extra_meta: dict[str, Any] | None = None,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> SkillResult:
        """执行一次完整质检。

        ``resume=True`` 时读取 state.json，已处理（且 路径+size+mtime 未变）的文件直接跳过。
        """
        started = time.perf_counter()
        cfg = self.cfg
        workers = int(workers if workers is not None else cfg.integer("performance.max_workers", 4))
        workers = max(1, workers)
        if limit is None:
            limit = cfg.integer("performance.sample_limit", 0)
        per_file_timeout = cfg.number("performance.per_file_timeout_sec", 120)
        progress_every = max(1, cfg.integer("performance.progress_every", 50))
        checkpoint_every = max(1, cfg.integer("performance.checkpoint_every", 100))

        folder_resolved = str(Path(folder).resolve()) if folder else ""
        entries = collect_files(folder, files, cfg)
        discovered = len(entries)
        if limit and limit > 0:
            entries = entries[:limit]
        self._log(
            f"{self.name} v{self.version} 启动：发现 {discovered} 个受支持文件"
            + (f"，本次按 limit={limit} 处理 {len(entries)} 个" if limit and limit > 0 else "")
            + f"；workers={workers}；resume={resume}"
        )

        state_path = cfg.path_of("output.state_file")
        store = StateStore.load(state_path, folder=folder_resolved, reset=not resume)
        if resume and store.processed:
            self._log(f"断点续跑：state.json 已有 {len(store.processed)} 条历史结果（{state_path}）")

        todo = [e for e in entries if not (resume and store.is_done(e))]
        skipped = len(entries) - len(todo)
        if skipped:
            self._log(f"跳过已处理文件 {skipped} 个（指纹 路径+大小+mtime 未变）")

        records: list[dict[str, Any]] = []
        texts: dict[str, str] = {}
        errors: list[dict[str, str]] = []
        done = 0
        t0 = time.perf_counter()

        def _absorb(rec: dict[str, Any]) -> None:
            nonlocal done
            text = rec.pop("_text", "")
            if text and want_texts:
                texts[str(rec["path"])] = text
            store.record(rec)
            done += 1
            if rec.get("error"):
                errors.append({"path": str(rec["path"]), "error": str(rec["error"])})
            if done % progress_every == 0 or done == len(todo):
                elapsed = time.perf_counter() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                remain = (len(todo) - done) / speed if speed > 0 else 0.0
                eta = time.strftime("%H:%M:%S", time.gmtime(remain))
                self._log(
                    f"进度 {done}/{len(todo)}（{(done / max(1, len(todo))):.1%}）"
                    f" 已用 {elapsed:.1f}s  速度 {speed:.2f} 份/s  预计剩余 {eta}"
                )
                if progress_cb:
                    try:
                        progress_cb({"done": done, "total": len(todo), "elapsed": elapsed, "speed": speed})
                    except Exception:  # noqa: BLE001 - 回调异常不影响主流程
                        pass
            if done % checkpoint_every == 0:
                store.save(meta={"last_run_started": time.strftime("%Y-%m-%d %H:%M:%S"), "discovered": discovered})
                self._log(f"检查点已写入 {state_path}")

        if todo:
            if workers > 1 and len(todo) > 1:
                payloads = [({"path": e.path, "suffix": e.suffix, "size_bytes": e.size_bytes, "mtime": e.mtime}, want_texts) for e in todo]
                try:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_worker_init,
                        initargs=(str(cfg.path), {"per_file_timeout_sec": per_file_timeout}),
                    ) as pool:
                        futures = [pool.submit(_worker_task, p) for p in payloads]
                        for fut in as_completed(futures):
                            try:
                                _absorb(fut.result())
                            except Exception as exc:  # noqa: BLE001 - 子进程崩溃兜底
                                done += 1
                                errors.append({"path": "<worker>", "error": f"worker_failed: {exc}"})
                except Exception as exc:  # noqa: BLE001 - 进程池不可用则退回串行
                    self._log(f"进程池不可用（{type(exc).__name__}: {exc}），退回单进程串行执行")
                    gc_every = max(1, cfg.integer("performance.gc_every", 100))
                    for n_done, e in enumerate(todo, 1):
                        _absorb(process_one(e, cfg, want_text=want_texts, timeout_sec=per_file_timeout))
                        if n_done % gc_every == 0:
                            gc.collect()
            else:
                gc_every = max(1, cfg.integer("performance.gc_every", 100))
                for n_done, e in enumerate(todo, 1):
                    _absorb(process_one(e, cfg, want_text=want_texts, timeout_sec=per_file_timeout))
                    if n_done % gc_every == 0:
                        gc.collect()

        # ---- 瞬时可恢复故障自动重试（MemoryError 等资源类错误） ----
        retry_round = 0
        while True:
            retryable = self._retryable_paths(store, todo, cfg)
            if not retryable or retry_round >= max(0, cfg.integer("performance.max_attempts_per_file", 2) - 1):
                break
            retry_round += 1
            if cfg.boolean("performance.gc_before_retry", True):
                gc.collect()
            self._log(f"第 {retry_round} 轮重试：{len(retryable)} 个文件因资源类错误失败（MemoryError 等），逐个重跑")
            retry_map = {e.path: e for e in todo}
            fixed = 0
            for p in retryable:
                e = retry_map.get(p)
                if e is None:
                    continue
                rec = process_one(e, cfg, want_text=want_texts, timeout_sec=per_file_timeout)
                if not rec.get("error"):
                    fixed += 1
                text = rec.pop("_text", "")
                if text and want_texts:
                    texts[str(rec["path"])] = text
                store.record(rec)
            self._log(f"第 {retry_round} 轮重试完成：修复 {fixed} / {len(retryable)}")

        store.save(meta={"last_run_started": time.strftime("%Y-%m-%d %H:%M:%S"), "discovered": discovered})

        # 汇总最终记录：本次新算的 + 历史跳过的（保持清单顺序）
        final: list[dict[str, Any]] = []
        for e in entries:
            rec = store.processed.get(e.path)
            if rec:
                final.append(rec)
        for rec in store.processed.values():  # 兜底：state 里有但不在本次清单内的也算入（例如 limit 之外的续跑结果）
            if rec not in final:
                final.append(rec)
        records = final
        self._log(f"解析完成：{len(records)} 条记录，用时 {time.perf_counter() - t0:.1f}s（并行 {workers} 进程）")

        meta_extra = dict(extra_meta or {})
        meta_extra["retry_rounds"] = retry_round
        report = self.analyze(records, texts=texts, extra_meta=meta_extra, elapsed=time.perf_counter() - started,
                              discovered=discovered, skipped=skipped, workers=workers, errors=errors)

        json_out = html_out = jsonl_out = None
        if write_reports:
            from .report import write_html_report, write_json_report, write_per_file_jsonl

            json_out = Path(json_path) if json_path else cfg.path_of("output.json_report")
            html_out = Path(html_path) if html_path else cfg.path_of("output.html_report")
            jsonl_out = cfg.path_of("output.per_file_jsonl")
            write_json_report(report, json_out)
            write_per_file_jsonl(records, jsonl_out)
            write_html_report(report, html_out)
            self._log(f"报告已写出：{json_out}")
            self._log(f"HTML 简报已写出：{html_out}")
            self._log(f"逐文件明细：{jsonl_out}")
        return SkillResult(report=report, json_path=json_out, html_path=html_out, per_file_jsonl=jsonl_out)

    # -- 分析（可独立调用：给定 records 直接出报告） --------------------------
    def analyze(
        self,
        records: Sequence[dict[str, Any]],
        *,
        texts: dict[str, str] | None = None,
        extra_meta: dict[str, Any] | None = None,
        elapsed: float = 0.0,
        discovered: int = 0,
        skipped: int = 0,
        workers: int = 1,
        errors: Sequence[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        started = time.perf_counter()
        recs = [dict(r) for r in records]
        texts = texts or {}

        # --- 功能 4 重复检测（需要先算出 is_md5_duplicate 才能打标签） ---
        dup = det.detect_duplicates(recs, cfg, texts=texts)
        dup_paths = set()
        for g in dup["md5_exact"]["duplicate_groups"]:
            for p in g["files"][1:]:
                dup_paths.add(str(p))
        for r in recs:
            r["is_md5_duplicate"] = str(r.get("path")) in dup_paths

        # --- 分类标签 ---
        for r in recs:
            tags, primary = det.assign_tags(r, cfg)
            r["tags"] = tags
            r["primary_tag"] = primary

        # --- 功能 5 敏感信息（待审核列表） ---
        sensitive = det.detect_sensitive(recs, cfg, texts=texts)

        # --- 功能 1 / 2 / 3 ---
        fmt = det.analyze_formats(recs, cfg)
        lengths = det.analyze_lengths(recs, cfg)
        pdf_summary = self._pdf_summary(recs, cfg)
        tags = det.tag_distribution(recs)

        pending = self._pending_lists(recs, dup, sensitive, cfg)

        n_records = len(recs)
        corrupted = sum(1 for r in recs if r.get("corrupted"))
        elapsed_parse = sum(float(r.get("elapsed_sec", 0) or 0) for r in recs)
        report: dict[str, Any] = {
            "skill": {
                "name": self.name,
                "version": self.version,
                "config_path": str(cfg.path),
                "config_version": cfg.get("config_version"),
                "thresholds": cfg.describe_thresholds(),
                "env_overrides": cfg.env_overrides,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repo_root": str(REPO_ROOT),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "host": platform.node(),
            },
            "run": {
                "discovered_files": discovered or n_records,
                "analyzed_files": n_records,
                "skipped_from_state": skipped,
                "workers": workers,
                "wall_clock_sec": round(elapsed, 3),
                "sum_per_file_sec": round(elapsed_parse, 3),
                "avg_sec_per_file": round(elapsed_parse / n_records, 4) if n_records else 0.0,
                "resume_enabled": skipped >= 0,
                "error_count": len(errors or []),
            },
            "summary": {
                "total_files": n_records,
                "total_bytes": fmt["total_bytes"],
                "total_mb": fmt["total_mb"],
                "corrupted_files": corrupted,
                "pdf_types": pdf_summary["counts"],
                "pdf_type_ratios": pdf_summary["ratios"],
                "scan_pdf_ratio": pdf_summary["ratios"].get("Scan_PDF", 0.0),
                "md5_duplicate_groups": dup["md5_exact"]["total_groups"],
                "md5_duplicate_files": dup["md5_exact"]["duplicate_file_count"],
                "version_conflicts": dup["simhash_near"]["total_conflicts"],
                "sensitive_files": sensitive["files_with_hits"],
                "sensitive_hits": sensitive["total_hits"],
                "pdf_pending_confirm": len(pending["pdf_type_pending_confirm"]),
                "tag_primary_top": tags["by_primary"][0]["name"] if tags["by_primary"] else "",
            },
            "format_distribution": fmt,
            "pdf_type_analysis": pdf_summary,
            "length_distribution": lengths,
            "duplicate_detection": dup,
            "sensitive_detection": sensitive,
            "tag_system": {
                "priority": list(cfg.get("tagging.priority", []) or []),
                "definitions": cfg.get("tagging.definitions", {}) or {},
                "distribution": tags,
            },
            "pending_lists": pending,
            "routing_rules": cfg.get("routing", {}) or {},
            "errors": list(errors or [])[:200],
            "per_file": self._per_file(recs, cfg),
        }
        if extra_meta:
            report["run"].update(extra_meta)
        report["analysis_sec"] = round(time.perf_counter() - started, 3)
        return report

    # -- 内部 ---------------------------------------------------------------
    def _retryable_paths(self, store: "StateStore", todo: Sequence[FileEntry], cfg: Config) -> list[str]:
        """找出本次待办中"因资源类错误失败"的文件路径（可安全重跑）。"""
        keywords = [str(k).lower() for k in (cfg.get("performance.retryable_error_keywords", []) or [])]
        if not keywords:
            return []
        todo_paths = {e.path for e in todo}
        out: list[str] = []
        for path, rec in store.processed.items():
            if path not in todo_paths:
                continue
            err = str(rec.get("error", "")).lower()
            if err and any(k in err for k in keywords):
                out.append(path)
        return sorted(out)

    def _pdf_summary(self, recs: Sequence[dict[str, Any]], cfg: Config) -> dict[str, Any]:
        digits = cfg.integer("format_stats.ratio_digits", 4)
        pdf_recs = [r for r in recs if str(r.get("suffix", "")).lower() == ".pdf"]
        pages = [r for r in pdf_recs if not r.get("corrupted")]
        counts = {"Text_PDF": 0, "Scan_PDF": 0, "Mixed_PDF": 0, "Corrupted": 0}
        total_pages = 0
        scan_pages = 0
        low_text_pages = 0
        for r in pdf_recs:
            if r.get("corrupted"):
                counts["Corrupted"] += 1
                continue
            t = str(r.get("pdf_type") or "")
            if t in counts:
                counts[t] += 1
            total_pages += int(r.get("sampled_pages", 0) or 0)
            scan_pages += int(r.get("scanned_pages", 0) or 0)
            low_text_pages += int(r.get("low_text_pages", 0) or 0)
        denom = max(1, sum(counts.values()))
        ratios = {k: round(v / denom, digits) for k, v in counts.items()}
        ratios_of_valid = {k: round(v / max(1, len(pages)), digits) for k, v in counts.items() if k != "Corrupted"}
        return {
            "pdf_file_count": len(pdf_recs),
            "parsed_ok": len(pages),
            "counts": counts,
            "ratios": ratios,
            "ratios_of_valid_pdf": ratios_of_valid,
            "sampled_pages_total": total_pages,
            "scanned_pages_total": scan_pages,
            "low_text_pages_total": low_text_pages,
            "scanned_page_ratio": round(scan_pages / total_pages, digits) if total_pages else 0.0,
            "low_text_page_ratio": round(low_text_pages / total_pages, digits) if total_pages else 0.0,
            "pdfs_with_text_layer": sum(1 for r in pdf_recs if r.get("has_text_layer")),
            "pure_scan_pdfs": sum(1 for r in pdf_recs if r.get("pure_scan")),
            "thresholds": {
                "page_char_threshold": cfg.number("scan_detection.page_char_threshold"),
                "scan_ratio_threshold": cfg.number("scan_detection.scan_ratio_threshold"),
                "mixed_ratio_threshold": cfg.number("scan_detection.mixed_ratio_threshold"),
                "image_area_ratio_threshold": cfg.number("scan_detection.image_area_ratio_threshold"),
                "require_image_for_scan": cfg.boolean("scan_detection.require_image_for_scan", False),
            },
            "note": "启发式分流，不保证严格正确；所有偏离阈值的边角样本进入待确认列表。",
        }

    def _pending_lists(
        self,
        recs: Sequence[dict[str, Any]],
        dup: dict[str, Any],
        sensitive: dict[str, Any],
        cfg: Config,
    ) -> dict[str, Any]:
        """工单强调的三类"可操作列表"。"""
        pdf_confirm = []
        for r in recs:
            if str(r.get("suffix", "")).lower() != ".pdf" or r.get("corrupted"):
                continue
            if r.get("pdf_needs_confirm"):
                pdf_confirm.append(
                    {
                        "file": r.get("path"),
                        "pdf_type": r.get("pdf_type"),
                        "scan_ratio": r.get("scan_ratio"),
                        "scan_ratio_literal": r.get("scan_ratio_literal"),
                        "scanned_pages": r.get("scanned_pages"),
                        "low_text_pages": r.get("low_text_pages"),
                        "sampled_pages": r.get("sampled_pages"),
                        "page_count": r.get("page_count"),
                        "reason": r.get("pdf_confirm_reason"),
                        "need_manual_confirm": True,
                    }
                )
        corrupted = [
            {
                "file": r.get("path"),
                "size_bytes": r.get("size_bytes"),
                "error": r.get("error"),
                "action": "移入隔离区，人工复核或重新获取原件",
            }
            for r in recs
            if r.get("corrupted")
        ]
        anomalies = [
            {
                "file": r.get("path"),
                "tags": r.get("tags"),
                "primary_tag": r.get("primary_tag"),
                "char_count": r.get("char_count"),
                "action": "按标签路由分支处理",
            }
            for r in recs
            if any(t in (r.get("tags") or []) for t in ("Too_Short", "Too_Long", "Empty"))
        ]
        return {
            "pdf_type_pending_confirm": pdf_confirm,
            "pdf_type_pending_confirm_total": len(pdf_confirm),
            "version_conflicts": dup["pending_version_conflicts"],
            "version_conflicts_total": dup["pending_version_conflicts_total"],
            "sensitive_pending_review": sensitive["pending_review"],
            "sensitive_pending_review_total": sensitive["pending_review_total"],
            "corrupted_files": corrupted,
            "corrupted_files_total": len(corrupted),
            "anomaly_files": anomalies,
            "anomaly_files_total": len(anomalies),
        }

    def _per_file(self, recs: Sequence[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
        max_rows = cfg.integer("output.max_per_file_in_report", 0)
        rows = []
        for r in recs:
            rows.append(
                {
                    "path": r.get("path"),
                    "name": r.get("name"),
                    "suffix": r.get("suffix"),
                    "size_bytes": r.get("size_bytes"),
                    "char_count": r.get("char_count"),
                    "page_count": r.get("page_count"),
                    "pdf_type": r.get("pdf_type"),
                    "scan_ratio": r.get("scan_ratio"),
                    "scan_ratio_literal": r.get("scan_ratio_literal"),
                    "scanned_pages": r.get("scanned_pages"),
                    "low_text_pages": r.get("low_text_pages"),
                    "sampled_pages": r.get("sampled_pages"),
                    "md5": r.get("md5"),
                    "simhash": r.get("simhash"),
                    "sensitive_hits": r.get("sensitive_hits"),
                    "corrupted": r.get("corrupted"),
                    "encrypted": r.get("encrypted"),
                    "pure_scan": r.get("pure_scan"),
                    "error": r.get("error"),
                    "tags": r.get("tags"),
                    "primary_tag": r.get("primary_tag"),
                    "elapsed_sec": r.get("elapsed_sec"),
                }
            )
        return rows if max_rows <= 0 else rows[:max_rows]
