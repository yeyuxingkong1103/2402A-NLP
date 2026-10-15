# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：命令行入口。调用 DocumentQualityAssessmentSkill 完成一次质检，
#           并把结构化报告与控制台摘要打印出来。

import os                                                          # 路径
import sys                                                         # 参数与编码
import argparse                                                    # 命令行参数

BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # 工单根目录
sys.path.insert(0, BASE_DIR)                                       # 保证可直接运行

import document_quality_assessment as dqa                          # 质检技能
from document_quality_assessment.report import summary_lines       # 控制台摘要


def _utf8_console():
    """把标准输出切到 UTF-8，避免中文在 Windows 控制台乱码。"""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv=None):
    """解析参数并执行质检。"""
    parser = argparse.ArgumentParser(description="文档质量评估（DocumentQualityAssessmentSkill）")
    parser.add_argument("--target", "-t", default=None,            # 评估目标
                        help="文件夹路径、文件路径或逗号分隔的文件列表；缺省用配置文件里的默认值")
    parser.add_argument("--out", "-o", default=None,               # 输出目录
                        help="报告输出目录（缺省 data/report）")
    parser.add_argument("--config", "-c", default=None,            # 配置文件
                        help="配置文件路径（缺省 assessment_config.yaml）")
    parser.add_argument("--no-resume", action="store_true",        # 关闭断点续跑
                        help="忽略已有进度，全量重新探测")
    parser.add_argument("--summary-only", action="store_true",     # 只打摘要
                        help="只打印摘要，不输出逐文档明细")
    args = parser.parse_args(argv)
    _utf8_console()                                                # 输出编码

    target = args.target                                           # 目标
    if target and "," in target:                                   # 逗号分隔的文件列表
        target = [p.strip() for p in target.split(",") if p.strip()]
    out = args.out or os.path.join(BASE_DIR, "data", "report")     # 输出目录

    print("[质检] 开始，目标：%s" % (target or "（配置文件默认值）"), flush=True)
    report = dqa.run_assessment(target=target, config_file=args.config,
                                output_dir=out, resume=not args.no_resume)
    print("\n================ 质检摘要 ================")
    for line in summary_lines(report):                             # 摘要
        print("  " + line)
    meta = report["meta"]                                          # 元信息
    print("\n  JSON 报告：%s" % meta["report_files"]["json"])
    print("  HTML 简报：%s" % meta["report_files"]["html"])
    print("  总耗时：%.1f 秒" % meta["elapsed_seconds"])

    tr = report["to_review"]                                       # 三类待确认列表
    print("\n================ 待确认/待审核列表 ================")
    print("  待确认 PDF 类型：%d 条" % len(tr["pdf_type_to_confirm"]))
    for it in tr["pdf_type_to_confirm"][:5]:
        print("    - %s（%s，扫描占比 %s）：%s"
              % (it["name"], it["kind"], it["scan_ratio"], it["confirm_reason"]))
    print("  待确认版本冲突：%d 条" % len(tr["version_conflicts"]))
    for it in tr["version_conflicts"][:5]:
        print("    - 相似度 %.3f（汉明距离 %d）：%s  ←→  %s"
              % (it["similarity"], it["hamming_distance"],
                 it["file_a"]["name"], it["file_b"]["name"]))
    print("  待审核敏感信息：%d 条" % len(tr["sensitive_items"]))
    for it in tr["sensitive_items"][:5]:
        print("    - [%s] %s　上下文：%s" % (it["type"], it["value"], it["context"]))
    if not args.summary_only:                                      # 逐文档分类明细
        print("\n================ 分类标签（前 10 条） ================")
        for d in report["classification"]["documents"][:10]:
            print("  %-24s %-12s %-28s → %s"
                  % (d["name"][:24], d["primary_label"],
                     "/".join(d["quality_tags"]) or "-", d["route"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
