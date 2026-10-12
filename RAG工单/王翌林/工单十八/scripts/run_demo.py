# -*- coding: utf-8 -*-
"""运行质检全流程：评估 -> 保存 JSON/HTML 报告 -> 生成图表与截图。"""
import os, sys, json
sys.path.insert(0, "/home/dabaie/code/工单/工单十八/document_quality_assessment")
from src.quality_assessor import assess, load_config
from src.report_renderer import render_html

BASE = "/home/dabaie/code/工单/工单十八"
DS = BASE + "/demo_dataset"
OUT = BASE + "/output"
SHOTS = BASE + "/docs/screenshots"
os.makedirs(OUT, exist_ok=True); os.makedirs(SHOTS, exist_ok=True)


def main():
    report = assess(DS, load_config())
    json.dump(report, open(OUT + "/quality_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(OUT + "/quality_report.html", "w", encoding="utf-8").write(render_html(report))
    print("saved report.json & report.html")
    print("文件数:", report["file_count"])
    print("格式:", report["format_distribution"]["distribution"])
    print("PDF:", [(os.path.basename(p["file"]), p["label"]) for p in report["pdf_classification"]["pdfs"]])
    print("精确重复:", len(report["duplicates"]["exact_duplicates"]))
    print("版本冲突:", len(report["action_items"]["version_conflicts"]))
    print("敏感信息:", report["sensitive_info"]["summary"])
    return report


if __name__ == "__main__":
    main()
