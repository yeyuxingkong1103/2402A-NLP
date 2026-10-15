# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：DocumentQualityAssessmentSkill 主入口。编排“格式统计 → PDF 分诊 → 长度分布
#           → 重复检测 → 敏感信息 → 分类标签 → 报告落盘”的完整流水线，输出标准结构化报告。

import os                                                          # 路径
import time                                                        # 计时

from . import scanner                                              # 遍历与逐文件探测
from . import length_stats                                         # 长度分布
from . import dedup                                                # 重复检测
from . import classifier                                           # 分类标签
from . import report as report_mod                                 # 报告组装

SKILL_NAME = "DocumentQualityAssessmentSkill"                      # 技能名称
SKILL_VERSION = "1.0.0"                                            # 技能版本


def _pdf_triage(records):
    """汇总 PDF 页面类型分流结果，并挑出需要人工确认的文件。"""
    pdfs = [r for r in records if (r.get("ext") or "").lower() == ".pdf"]  # 只看 PDF
    counts = {"text": 0, "scan": 0, "mixed": 0, "failed": 0}       # 各类型计数
    files, to_confirm = [], []                                     # 明细与待确认
    for r in pdfs:                                                 # 逐 PDF
        kind = r.get("pdf_kind")                                   # 页面类型
        if r.get("error"):                                         # 打不开的 PDF
            counts["failed"] += 1                                  # 计入失败
            files.append({"name": r["name"], "path": r["path"], "kind": "failed",
                          "page_count": None, "scan_pages": None, "scan_ratio": None,
                          "image_page_ratio": None, "need_confirm": True,
                          "confirm_reason": "解析失败：%s" % r["error"]})
            continue
        counts[kind] = counts.get(kind, 0) + 1                      # 类型计数
        item = {"name": r["name"], "path": r["path"], "kind": kind,  # 明细项
                "page_count": r.get("page_count"),                  # 页数
                "scan_pages": r.get("scan_pages"),                  # 扫描页数
                "scan_ratio": r.get("scan_ratio"),                  # 扫描页占比
                "image_page_ratio": r.get("image_page_ratio"),      # 图像覆盖率（交叉验证）
                "need_confirm": bool(r.get("need_confirm")),        # 是否待确认
                "confirm_reason": r.get("confirm_reason") or ""}
        files.append(item)
        if item["need_confirm"]:                                    # 收集待确认
            to_confirm.append(item)
    to_confirm.sort(key=lambda i: (i["kind"], i["name"]))           # 稳定排序
    return {"total": len(pdfs), "text": counts["text"], "scan": counts["scan"],
            "mixed": counts["mixed"], "failed": counts["failed"],
            "scan_ratio": round(counts["scan"] / float(max(len(pdfs), 1)), 4),
            "thresholds": {"char_threshold": None, "scan_ratio_threshold": None},
            "files": files, "to_confirm": to_confirm}


def _errors(records, limit):
    """收集解析失败清单。"""
    items = [{"path": r["path"], "name": r.get("name"), "ext": r.get("ext"),
              "error": r.get("error")} for r in records if r.get("error")]
    return items[:limit] if limit else items                       # 超限截断


def run_assessment(target=None, config_file=None, output_dir=None,
                   resume=None, progress_cb=None):
    """执行一次完整质检，返回报告字典；同时把 JSON/HTML 报告写入 output_dir。

    target —— 文件夹路径、文件列表或单个文件；缺省用配置里的 default_target
    resume —— 是否启用断点续跑（缺省用配置值）
    """
    from .config_loader import load_config, WORK_ORDER_DIR         # 延迟导入，避免循环依赖
    t0 = time.time()                                               # 总计时
    cfg, note = load_config(config_file)                           # 配置
    target = target or cfg.get("paths.default_target")             # 目标
    if not target:
        raise ValueError("未指定评估目标，请传入 target 或在配置中设置 paths.default_target")
    out_dir = cfg.resolve_path(output_dir or cfg.get("paths.output_dir", "data"))
    os.makedirs(out_dir, exist_ok=True)                            # 输出目录

    paths = scanner.discover_files(target, cfg.get("scan.extensions", [".pdf"]),
                                   cfg.get("scan.recursive", True))  # 一、遍历与格式口径
    if not paths:
        raise ValueError("目标中没有受支持的文件：%s" % target)
    fmt = scanner.format_stats(paths)                              # 格式分布统计

    use_resume = cfg.get("runtime.resume", True) if resume is None else resume
    state_file = cfg.resolve_path(cfg.get("paths.state_file", "data/assessment_state.json"))
    state = scanner.StateStore(state_file, resume=use_resume)      # 断点续跑状态
    records = scanner.probe_files(paths, cfg, state,               # 逐文件探测
                                  progress_every=cfg.get("runtime.progress_every", 100),
                                  on_record=progress_cb)

    pdf_triage = _pdf_triage(records)                              # 二、PDF 分诊
    pdf_triage["thresholds"] = {"char_threshold": cfg.get("pdf_triage.char_threshold"),
                                "scan_ratio_threshold": cfg.get("pdf_triage.scan_ratio_threshold"),
                                "text_ratio_threshold": cfg.get("pdf_triage.text_ratio_threshold")}
    length = length_stats.compute(records, cfg)                    # 三、长度分布
    length["short_documents"] = length_stats.outliers(records, cfg)  # 过短清单
    dedup_res = dedup.compute(records, cfg)                        # 四、重复检测
    classify = classifier.compute(records, cfg, dedup_res)         # 五、分类标签
    errors = _errors(records, cfg.get("runtime.error_sample_limit", 50))  # 错误清单

    doc = report_mod.build(note, cfg, target if isinstance(target, str) else "(文件列表)",
                           fmt, pdf_triage, length, dedup_res, classify,
                           records, errors, time.time() - t0)      # 组装报告
    doc["meta"]["skill"] = {"name": SKILL_NAME, "version": SKILL_VERSION,
                            "entry": "document_quality_assessment.run_assessment"}
    json_path = os.path.join(out_dir, "quality_report.json")       # 报告文件路径
    html_path = os.path.join(out_dir, "quality_report.html")
    doc["meta"]["report_files"] = {"json": json_path,             # 先登记路径，保证写出的
                                   "html": html_path if cfg.get(   # JSON 里自带路径信息
                                       "runtime.html_report", True) else ""}
    report_mod.write_json(doc, json_path, cfg.get("runtime.json_indent", 2))
    if cfg.get("runtime.html_report", True):                       # HTML 简报
        report_mod.to_html(doc, html_path)
    return doc
