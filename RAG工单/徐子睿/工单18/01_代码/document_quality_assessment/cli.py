# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill · 命令行入口

用法：
  python -m document_quality_assessment.cli --root <目录> [--config assessment_config.yaml] \
      [--out report.json] [--html report.html] [--limit 0]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_quality_assessment import Config, assess, report as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--html", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)

    def cb(i, n, name):
        print(f"  [{i}/{n}] {name}", flush=True)

    rep = assess(args.root, cfg, limit=args.limit, progress_cb=cb, progress_file=args.progress)
    R.to_json(rep, args.out)
    if args.html:
        R.to_html(rep, args.html)
    fs = rep["format_stats"]
    pdf = rep["pdf_types"]
    print("\n==== 摘要 ====")
    print(f"文件总数: {fs['total_files']}")
    print(f"PDF: 扫描型 {pdf['scan']} / 文字型 {pdf['text']} / 混合型 {pdf['mixed']}")
    print(f"MD5 重复分组: {len(rep['duplicates']['md5'])}  |  SimHash 待确认: {len(rep['duplicates']['near'])}")
    print(f"敏感信息命中: {rep['sensitive']['counts']}")
    print(f"分类标签: {list(rep['classification']['label_counts'].items())[:6]}")
    if args.out:
        print("JSON:", args.out)
    if args.html:
        print("HTML:", args.html)


if __name__ == "__main__":
    main()
