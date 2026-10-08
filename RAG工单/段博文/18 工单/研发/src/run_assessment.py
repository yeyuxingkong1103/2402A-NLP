# -*- coding: utf-8 -*-
"""CLI：对指定文件夹运行文档质量评估 Skill，输出 JSON + HTML。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_quality_assessment import DocumentQualityAssessment
from document_quality_assessment.report import render_html


def main():
    ap = argparse.ArgumentParser(description="文档质量评估 Skill CLI")
    ap.add_argument("--path", help="待评估文件夹")
    ap.add_argument("--files", nargs="*", help="显式文件列表")
    ap.add_argument("--config", help="配置文件路径")
    ap.add_argument("--output", default="results", help="输出目录")
    ap.add_argument("--no-resume", action="store_true", help="不使用 checkpoint 恢复")
    args = ap.parse_args()

    skill = DocumentQualityAssessment(args.config)
    result = skill.assess(target=args.path, files=args.files,
                          output_dir=args.output, resume=not args.no_resume)
    report = result.to_dict()

    json_path = Path(args.output) / "quality_report.json"
    html_path = Path(args.output) / "quality_report.html"
    html_path.write_text(render_html(report), encoding="utf-8")

    s = report["summary"]
    print("\n===== 质检完成 =====")
    print(f"JSON: {json_path}\nHTML: {html_path}")
    print(f"文件 {s['total_files']} | PDF类型 {s['pdf_type_counts']}")
    print(f"MD5重复 {s['md5_duplicate_files']} 文件/{s['md5_duplicate_groups']} 组 | "
          f"版本冲突待确认 {s['simhash_pending_pairs']} 对 | "
          f"敏感信息 {s['sensitive_pending_items']} 条")
    print(f"路由: {s['routing_counts']}")


if __name__ == "__main__":
    main()
