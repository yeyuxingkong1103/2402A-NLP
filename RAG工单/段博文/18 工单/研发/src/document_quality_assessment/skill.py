# -*- coding: utf-8 -*-
"""Skill 主编排：五大功能流水线 + checkpoint 中断恢复 + 结构化结果。"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from .doc_classifier import apply_global_tags, format_tag_for
from .duplicate_detector import md5_groups, md5_of, simhash, simhash_pairs
from .format_stats import collect_files, format_distribution
from .length_analysis import length_distribution
from .pdf_classifier import classify_pdf
from .sensitive_detector import detect_sensitive
from .text_reader import DocumentReadError, read_pdf_detailed, read_text

_DEFAULT_CONFIG = Path(__file__).parent / "assessment_config.yaml"

__skill_name__ = "document-quality-assessment"
__version__ = "1.0.0"


def load_config(config_path: Optional[str] = None) -> dict:
    with open(config_path or _DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AssessmentResult:
    """结构化评估报告容器。"""

    def __init__(self, report: dict):
        self.report = report

    def to_dict(self) -> dict:
        return self.report

    def to_json(self, path: str, indent: int = 2) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.report, ensure_ascii=False, indent=indent), encoding="utf-8")
        return path


class DocumentQualityAssessment:
    """文档质量评估 Skill 入口。"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)

    # ------------------------------------------------------------------
    def assess(self, target: Optional[str] = None, files: Optional[List[str]] = None,
               output_dir: str = "results", resume: bool = True) -> AssessmentResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # 1. 收集文件
        if files:
            all_files = [Path(f) for f in files]
            unsupported = []
        else:
            if not target:
                raise ValueError("必须提供 target 文件夹路径或 files 文件列表")
            all_files, unsupported = collect_files(
                Path(target), self.config["scan"]["extensions"], self.config["scan"]["recursive"])

        # 2. checkpoint 恢复
        ckpt_path = out / self.config["performance"]["checkpoint_file"]
        records = []
        done = {}
        if resume and self.config["performance"]["checkpoint"] and ckpt_path.exists():
            done = json.loads(ckpt_path.read_text(encoding="utf-8"))
            records = [done[str(p)] for p in all_files if str(p) in done]

        # 3. 逐文件评估
        total = len(all_files)
        for idx, path in enumerate(all_files):
            if str(path) in done:
                continue
            rec = self._process_one(path)
            records.append(rec)
            done[str(path)] = rec
            if (idx + 1) % self.config["performance"]["progress_every"] == 0:
                print(f"  进度 {idx + 1}/{total}，已用 {time.time() - t0:.0f}s")
                ckpt_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        ckpt_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        records.sort(key=lambda r: r["path"])

        # 4. 汇总分析
        report = self._aggregate(records, unsupported, target or "(文件列表模式)", out)
        report["elapsed_seconds"] = round(time.time() - t0, 1)

        (out / "quality_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return AssessmentResult(report)

    # ------------------------------------------------------------------
    def _process_one(self, path: Path) -> dict:
        """单文件五维评估。任何异常都不向上抛，转为记录字段。"""
        rec = {
            "path": str(path), "name": path.name,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "md5": "", "char_count": 0, "pdf": None,
            "tags": [], "sensitive_findings": [], "simhash": 0,
            "route": "", "read_error": None,
        }
        try:
            rec["md5"] = md5_of(path)
        except Exception as e:
            rec["read_error"] = f"MD5 计算失败: {e}"

        text, read_ok = "", True
        try:
            if path.suffix.lower() == ".pdf":
                page_count, page_chars, text = read_pdf_detailed(path)
                rec["pdf"] = classify_pdf(path, self.config["pdf"], pages_info=(page_count, page_chars))
            else:
                text = read_text(path)
        except DocumentReadError as e:
            read_ok = False
            rec["read_error"] = str(e)
            if path.suffix.lower() == ".pdf":
                rec["pdf"] = {"type": "Corrupt_PDF", "scanned_ratio": 0.0, "near_boundary": False}

        rec["char_count"] = len(text)
        rec["tags"] = [format_tag_for(path, rec["pdf"], read_ok)]

        # 敏感信息（仅在成功读取且有文本时）
        if read_ok and text:
            rec["sensitive_findings"] = detect_sensitive(text, self.config["sensitive"])
            # SimHash
            try:
                rec["simhash"] = simhash(text, self.config["duplicate"])
            except Exception as e:
                rec["simhash"] = 0
                rec["read_error"] = (rec["read_error"] or "") + f" | simhash失败: {e}"
        return rec

    # ------------------------------------------------------------------
    def _aggregate(self, records: list, unsupported: list, target: str, out: Path) -> dict:
        dcfg = self.config["duplicate"]
        # 格式统计
        fmt = format_distribution([Path(r["path"]) for r in records])
        # 长度分布
        len_stats = length_distribution([r["char_count"] for r in records], self.config["length"])

        # MD5 重复组
        md5_map = {r["path"]: r["md5"] for r in records if r["md5"]}
        md5_grps = md5_groups(md5_map) if dcfg["md5"] else []
        # 重复组内第一个文件保留，其余标记
        md5_dup_paths = set()
        for g in md5_grps:
            for f in g["files"][1:]:
                md5_dup_paths.add(f)

        # SimHash 两两比较
        sh_records = [{"path": r["path"], "char_count": r["char_count"], "simhash": r["simhash"]}
                      for r in records if r["simhash"]]
        sh_pairs = simhash_pairs(sh_records, dcfg) if dcfg["simhash"] else []
        # 对候选对重读文件，补充人工判断所需的相似片段
        if sh_pairs:
            from .duplicate_detector import _similar_snippet
            text_cache = {}
            for pair in sh_pairs:
                for key in ("file_a", "file_b"):
                    fp = pair[key]
                    if fp not in text_cache:
                        try:
                            text_cache[fp] = read_text(Path(fp))[:20000]
                        except Exception:
                            text_cache[fp] = ""
                pair["similar_snippet"] = _similar_snippet(
                    text_cache[pair["file_a"]], text_cache[pair["file_b"]])
        sh_paths = {p["file_a"] for p in sh_pairs} | {p["file_b"] for p in sh_pairs}

        # 全局标签 + 路由（就地）
        apply_global_tags(records, md5_dup_paths, sh_paths, len_stats,
                          self.config["length"], self.config.get("tags", {}))

        # PDF 类型汇总
        pdf_type_counter = Counter(
            r["pdf"]["type"] for r in records if r["pdf"])
        pending_pdf = [
            {"path": r["path"], "scanned_ratio": r["pdf"]["scanned_ratio"],
             "page_count": r["pdf"]["page_count"], "status": "待确认"}
            for r in records if r["pdf"] and r["pdf"].get("near_boundary")]

        # 敏感信息扁平待审列表
        flat_sensitive = []
        sensitive_counter = Counter()
        for r in records:
            for f in r["sensitive_findings"]:
                sensitive_counter[f["type"]] += 1
                flat_sensitive.append({"file": r["path"], **f})

        action_lists = {
            "scan_pdf_for_ocr": [r["path"] for r in records if r["pdf"] and r["pdf"]["type"] == "Scan_PDF"],
            "corrupt_files": [{"path": r["path"], "error": r["read_error"]}
                              for r in records if r["read_error"] or (r["pdf"] and r["pdf"]["type"] == "Corrupt_PDF")],
            "empty_files": [r["path"] for r in records if r["char_count"] <= self.config["length"]["empty_threshold"]],
            "pending_pdf_boundary": pending_pdf,
            "pending_version_conflicts": sh_pairs,
            "pending_sensitive_review": flat_sensitive,
        }

        return {
            "skill": __skill_name__,
            "version": __version__,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target": target,
            "config_used": self.config,
            "summary": {
                "total_files": len(records),
                "total_size_mb": round(sum(r["size_bytes"] for r in records) / 1024 / 1024, 1),
                "pdf_type_counts": dict(pdf_type_counter),
                "md5_duplicate_groups": len(md5_grps),
                "md5_duplicate_files": len(md5_dup_paths),
                "simhash_pending_pairs": len(sh_pairs),
                "sensitive_pending_items": len(flat_sensitive),
                "sensitive_by_type": dict(sensitive_counter),
                "routing_counts": dict(Counter(r["route"] for r in records)),
            },
            "format": fmt,
            "length": len_stats,
            "pdf_pending_confirm": pending_pdf,
            "duplicates": {"md5_groups": md5_grps, "simhash_pairs": sh_pairs},
            "sensitive_review": flat_sensitive,
            "unsupported_files": [str(p) for p in unsupported],
            "action_lists": action_lists,
            "records": records,
        }
