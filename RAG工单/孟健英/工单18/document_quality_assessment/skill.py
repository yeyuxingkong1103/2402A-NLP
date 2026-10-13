# 工单编号：人工智能 NLP-RAG-实现文档质量评估 Skill 并集成至智能体工作流工单
"""
DocumentQualityAssessmentSkill
文档质量评估 Skill - 5 大功能：
1. 格式分布统计
2. PDF 页面类型识别（文字型/扫描型/混合型）
3. 文档长度分布（P25/P50/P75/P90/P99）
4. 重复检测（MD5 + SimHash）
5. 敏感信息检测
"""

import os
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime

import yaml
import pymupdf

try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False


class DocumentQualityAssessmentSkill:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.scan_cfg = self.config["scan_detection"]
        self.len_cfg = self.config["length_distribution"]
        self.dup_cfg = self.config["duplicate_detection"]
        self.sens_cfg = self.config["sensitive_detection"]
        self.supported = tuple(self.config["supported_formats"])

    def assess_folder(self, folder_path: str, max_files: Optional[int] = None) -> Dict:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")

        files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in self.supported]
        if max_files:
            files = files[:max_files]

        total = len(files)
        print(f"[Skill] 共找到 {total} 个支持的文件")

        # 1. 格式分布
        format_stats = self._format_distribution(files)

        # 2. PDF 类型识别 + 文档长度
        pdf_types = []
        doc_lengths = []
        file_infos = []

        for i, f in enumerate(files):
            try:
                info = self._inspect_file(f)
                file_infos.append(info)
                if info["length"] > 0:
                    doc_lengths.append(info["length"])
                if info["pdf_type"]:
                    pdf_types.append({"path": str(f), "type": info["pdf_type"]})
            except Exception as e:
                print(f"[WARN] 处理失败 {f}: {e}")

            if (i + 1) % 100 == 0:
                print(f"[Skill] 已处理 {i+1}/{total}")

        # 3. 长度分布
        length_stats = self._length_distribution(doc_lengths)

        # 4. 重复检测
        duplicates = self._duplicate_detection(files)

        # 5. 敏感信息
        sensitive = self._sensitive_detection(files, file_infos)

        # PDF 分类统计
        pdf_type_counter = Counter(p["type"] for p in pdf_types)

        report = {
            "meta": {
                "folder": str(folder),
                "total_files": total,
                "scan_time": datetime.now().isoformat(),
            },
            "format_distribution": format_stats,
            "pdf_type_distribution": {
                "summary": dict(pdf_type_counter),
                "details": pdf_types,
            },
            "length_distribution": length_stats,
            "duplicate_detection": duplicates,
            "sensitive_detection": sensitive,
        }
        return report

    # ---------- 1. 格式分布 ----------
    def _format_distribution(self, files: List[Path]) -> Dict:
        counter = Counter(f.suffix.lower() for f in files)
        total = sum(counter.values()) or 1
        return {
            "total": total,
            "by_format": [
                {"ext": ext, "count": cnt, "ratio": round(cnt / total, 4)}
                for ext, cnt in counter.most_common()
            ],
        }

    # ---------- 2. 单文件检查 ----------
    def _inspect_file(self, f: Path) -> Dict:
        ext = f.suffix.lower()
        info = {
            "path": str(f),
            "ext": ext,
            "length": 0,
            "pdf_type": None,
        }
        if ext == ".pdf":
            info.update(self._inspect_pdf(f))
        elif ext == ".docx":
            info["length"] = self._read_docx(f)
        elif ext in (".md", ".txt"):
            info["length"] = self._read_text(f)
        return info

    def _inspect_pdf(self, f: Path) -> Dict:
        try:
            doc = pymupdf.open(str(f))
        except Exception as e:
            raise RuntimeError(f"PDF 打开失败: {e}")

        total_chars = 0
        scan_pages = 0
        page_count = len(doc)
        char_threshold = self.scan_cfg["char_threshold"]

        for page in doc:
            text = page.get_text().strip()
            total_chars += len(text)
            if len(text) < char_threshold:
                scan_pages += 1

        doc.close()
        ratio = scan_pages / page_count if page_count else 0
        if ratio >= self.scan_cfg["scan_ratio_threshold"]:
            pdf_type = "Scan_PDF"
        elif ratio >= self.scan_cfg["mixed_ratio_threshold"]:
            pdf_type = "Mixed_PDF"
        else:
            pdf_type = "Text_PDF"

        return {"length": total_chars, "pdf_type": pdf_type, "page_count": page_count,
                "scan_page_ratio": round(ratio, 3)}

    def _read_docx(self, f: Path) -> int:
        try:
            from docx import Document
            doc = Document(str(f))
            return sum(len(p.text) for p in doc.paragraphs)
        except Exception:
            return 0

    def _read_text(self, f: Path) -> int:
        try:
            return len(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return 0

    # ---------- 3. 长度分布 ----------
    def _length_distribution(self, lengths: List[int]) -> Dict:
        if not lengths:
            return {"count": 0, "percentiles": {}, "bins": []}
        sorted_len = sorted(lengths)
        n = len(sorted_len)

        percentiles = {}
        for p in self.len_cfg["percentiles"]:
            idx = min(int(n * p / 100), n - 1)
            percentiles[f"P{p}"] = sorted_len[idx]

        bins = self.len_cfg["bins"]
        labels = self.len_cfg["bin_labels"]
        bin_counts = [0] * (len(bins) - 1)
        for v in sorted_len:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    bin_counts[i] += 1
                    break

        return {
            "count": n,
            "percentiles": percentiles,
            "bins": [{"label": labels[i], "count": bin_counts[i]} for i in range(len(bin_counts))],
        }

    # ---------- 4. 重复检测 ----------
    def _duplicate_detection(self, files: List[Path]) -> Dict:
        md5_groups = defaultdict(list)
        simhash_list = []

        for f in files:
            try:
                data = f.read_bytes()
                md5 = hashlib.md5(data).hexdigest()
                md5_groups[md5].append(str(f))

                if self.dup_cfg.get("simhash_enabled") and SIMHASH_AVAILABLE and f.suffix.lower() in (".md", ".txt"):
                    text = f.read_text(encoding="utf-8", errors="ignore")[:5000]
                    if text.strip():
                        simhash_list.append((str(f), Simhash(text)))
            except Exception:
                pass

        # MD5 完全相同的组（>= 2 个文件）
        md5_dupes = [
            {"md5": md5, "files": paths}
            for md5, paths in md5_groups.items() if len(paths) > 1
        ]

        # SimHash 相似（待确认）
        simhash_pairs = []
        threshold = self.dup_cfg["simhash_hamming_threshold"]
        for i in range(len(simhash_list)):
            for j in range(i + 1, len(simhash_list)):
                p1, h1 = simhash_list[i]
                p2, h2 = simhash_list[j]
                if h1.distance(h2) <= threshold:
                    simhash_pairs.append({
                        "file1": p1, "file2": p2,
                        "hamming_distance": h1.distance(h2)
                    })

        return {
            "md5_duplicates": md5_dupes,
            "md5_duplicate_groups": len(md5_dupes),
            "simhash_pending_pairs": simhash_pairs[:100],  # 最多 100 条
            "simhash_pending_count": len(simhash_pairs),
        }

    # ---------- 5. 敏感信息 ----------
    def _sensitive_detection(self, files: List[Path], file_infos: List[Dict]) -> Dict:
        patterns = {}
        for key in ["phone", "email", "id_card", "bank_card"]:
            cfg = self.sens_cfg.get(key, {})
            if cfg.get("enabled"):
                patterns[key] = re.compile(cfg["pattern"])

        findings = []
        context_chars = self.sens_cfg.get("context_chars", 30)

        for f in files:
            if f.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for key, pat in patterns.items():
                for m in pat.finditer(text):
                    start = max(0, m.start() - context_chars)
                    end = min(len(text), m.end() + context_chars)
                    findings.append({
                        "file": str(f),
                        "type": key,
                        "match": m.group(),
                        "context": text[start:end].replace("\n", " "),
                    })

        # 汇总
        summary = Counter(f["type"] for f in findings)
        return {
            "summary": dict(summary),
            "total": len(findings),
            "pending_review_list": findings[:200],  # 最多 200 条
        }


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python skill.py <folder_path> [config_path]")
        sys.exit(1)

    folder = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "assessment_config.yaml")

    skill = DocumentQualityAssessmentSkill(cfg)
    report = skill.assess_folder(folder)

    out_dir = Path("/mnt/d/专高六 工单/工单18/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quality_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[Skill] 报告已保存: {out_path}")
    print(f"[Skill] 总文件数: {report['meta']['total_files']}")
    print(f"[Skill] PDF 类型: {report['pdf_type_distribution']['summary']}")
    print(f"[Skill] MD5 重复组: {report['duplicate_detection']['md5_duplicate_groups']}")
    print(f"[Skill] 敏感信息: {report['sensitive_detection']['summary']}")
