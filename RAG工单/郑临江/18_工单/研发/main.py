# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
主程序：在真实数据集上运行质检 Skill（IMDR documents/ 约 1700 文件）。
"""
import os
import sys

import config
from document_quality_assessment import DocumentQualityAssessor


def main():
    folder = config.DOC_DIR
    if not os.path.isdir(folder):
        print(f"[main] 目录不存在: {folder}")
        print("请先解压 original_problems.zip，或将目录作为参数传入")
        if len(sys.argv) > 1:
            folder = sys.argv[1]

    assessor = DocumentQualityAssessor()
    report = assessor.assess(folder)

    print("\n===== 文档质量评估报告 =====")
    print("总文件数:", report["total_files"])
    print("格式分布:", report["format_distribution"])
    print("长度分位数:", report["length_distribution"]["percentiles"])

    scan_count = sum(1 for v in report["pdf_types"].values()
                     if v["type"] == "Scan_PDF")
    print(f"扫描型 PDF: {scan_count} 个")

    conflicts = report["duplicates"]["version_conflicts"]
    sens = report["sensitive_info"]
    print(f"待确认版本冲突: {len(conflicts)} 组")
    print(f"待审核敏感信息: {len(sens)} 条")
    for s in sens[:5]:
        print("  ", s["file"], s["type"], s["value"])

    # 保存报告（支持中断恢复：结果落盘）
    import json
    with open("quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告已保存到 quality_report.json")


if __name__ == "__main__":
    main()
