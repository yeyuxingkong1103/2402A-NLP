# -*- coding: utf-8 -*-
"""
DQA · DocumentQualityAssessmentSkill 主入口

用法：
  python scripts/assess.py --input D:/data/imdr/documents --out ../产出物
  python scripts/assess.py --input ... --config assessment_config.yaml --limit 200
  python scripts/assess.py --input ... --resume            # 断点续跑

产出：
  quality_report.json / quality_report.html / quality_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dqa import (analyze_format_distribution, classify_pdf_pages,          # noqa: E402
                 analyze_length_distribution, detect_duplicates,
                 detect_sensitive_info, classify_documents,
                 build_report, to_html, to_markdown)
from dqa.pdf_type import scan_pdf_types                                    # noqa: E402

DEFAULT_CONFIG = {
    "input": "",
    "output": "",
    "extensions": [".pdf", ".docx", ".doc", ".md", ".markdown", ".txt",
                   ".xlsx", ".xls", ".csv", ".pptx", ".ppt"],
    "pdf": {"min_chars_per_page": 50, "scan_ratio_threshold": 0.70,
            "mixed_ratio_threshold": 0.15, "max_pages_per_pdf": 40},
    "length": {"long_doc_threshold": 50000},
    "dedup": {"enable_simhash": True, "simhash_threshold": 0.90},
    "sensitive": {"enable": {"手机号": True, "邮箱": True, "身份证": True, "银行卡": False}},
    "limits": {"max_files": None, "checkpoint_every": 50},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).exists():
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml
            cfg = _deep_merge(cfg, yaml.safe_load(text) or {})
        except ImportError:
            print("[warn] 未安装 PyYAML，使用内置默认配置。pip install pyyaml")
        except Exception as e:                                    # noqa: BLE001
            print(f"[warn] 配置文件解析失败：{e}，使用默认配置")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="文档质量评估 Skill")
    ap.add_argument("--input", required=True, help="待评估的文件夹")
    ap.add_argument("--out", default=None, help="报告输出目录")
    ap.add_argument("--config", default=str(HERE.parent / "assessment_config.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（调试用）")
    ap.add_argument("--no-simhash", action="store_true")
    ap.add_argument("--resume", action="store_true", help="从断点续跑（跳过已完成的模块）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["input"] = args.input
    outdir = Path(args.out or cfg.get("output") or (HERE.parent / "产出物"))
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / "_checkpoint.json"
    done = {}
    if args.resume and ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        print(f"[resume] 已完成模块：{list(done.keys())}")

    t0 = time.perf_counter()
    root = Path(args.input)
    exts = set(cfg["extensions"])
    max_files = args.limit or cfg["limits"].get("max_files")

    def progress(i, n):
        print(f"  ... {i}/{n}", flush=True)

    # 1. 格式分布
    if "format" not in done:
        print(">>> [1/6] 格式分布统计")
        done["format"] = analyze_format_distribution(root)
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    # 2. PDF 类型
    if "pdf_type" not in done:
        print(">>> [2/6] PDF 文字型/扫描型分流")
        done["pdf_type"] = scan_pdf_types(
            root, min_chars_per_page=cfg["pdf"]["min_chars_per_page"],
            scan_ratio_threshold=cfg["pdf"]["scan_ratio_threshold"],
            max_files=max_files, progress=progress)
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    # 3. 长度分布
    if "length" not in done:
        print(">>> [3/6] 文档长度分布")
        done["length"] = analyze_length_distribution(root, exts, max_files, progress)
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    # 4. 重复检测
    if "dedup" not in done:
        print(">>> [4/6] 重复检测（MD5 + SimHash）")
        done["dedup"] = detect_duplicates(
            root, exts,
            simhash_threshold=cfg["dedup"]["simhash_threshold"],
            max_files=max_files, progress=progress)
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    # 5. 敏感信息
    if "sensitive" not in done:
        print(">>> [5/6] 敏感信息检测")
        done["sensitive"] = detect_sensitive_info(
            root, exts, enable=cfg["sensitive"]["enable"],
            max_files=max_files, progress=progress)
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    # 6. 分类标签
    if "labels" not in done:
        print(">>> [6/6] 文档分类标签")
        done["labels"] = classify_documents(
            done["format"].get("_files", []), done["pdf_type"], done["length"],
            done["dedup"], done["sensitive"],
            long_threshold=cfg["length"]["long_doc_threshold"])
        ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    rep = build_report(root, done["format"], done["pdf_type"], done["length"],
                       done["dedup"], done["sensitive"], done["labels"], elapsed,
                       config=cfg)
    (outdir / "quality_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "quality_report.html").write_text(to_html(rep), encoding="utf-8")
    (outdir / "quality_report.md").write_text(to_markdown(rep), encoding="utf-8")

    print("\n===== 评估完成 =====")
    print(json.dumps(rep["总览"], ensure_ascii=False, indent=2))
    print(f"\n报告已输出到：{outdir}")


if __name__ == "__main__":
    main()
