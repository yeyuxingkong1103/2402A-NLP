#!/usr/bin/env python3
"""比较两轮评测 JSON，输出发布验收所需的指标和口径差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("Recall@5", "recall_at_5"),
    ("MRR@10", "mrr_at_10"),
    ("引用正确率", "citation_accuracy"),
    ("拒答准确率", "refusal_accuracy"),
    ("非拒答误拒率", "false_refusal_rate"),
)


def load_report(report_path: Path) -> dict[str, Any]:
    """读取并校验评测 JSON，避免把任意 JSON 当成评测报告比较。"""
    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise ValueError(f"不是有效评测报告：{report_path}")
    return report


def get_prompt_md5(report: dict[str, Any]) -> str | None:
    """提取提示词文件哈希，并兼容单文件和多文件配置结构。"""
    prompt_files = report.get("run_config", {}).get("prompt_files", {})
    if not isinstance(prompt_files, dict) or not prompt_files:
        return None
    values = [str(value) for value in prompt_files.values()]
    return ", ".join(values)


def count_local_corpus_chunks(project_root: Path) -> int | None:
    """在报告未记录块数时，从当前交付语料计算可核验的块数。"""
    processed_root = project_root / "data" / "labor_law_processed"
    chunk_files = sorted(processed_root.glob("*/document_chunks.jsonl"))
    if not chunk_files:
        return None
    return sum(
        1
        for chunk_file in chunk_files
        for line in chunk_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def get_corpus_chunk_count(report: dict[str, Any], local_count: int | None) -> int | None:
    """提取报告记录的语料块数，旧报告缺失时使用当前语料统计。"""
    run_config = report.get("run_config", {})
    candidates = (
        run_config.get("corpus_chunk_count"),
        run_config.get("chunk_count"),
        run_config.get("corpus_chunks"),
        report.get("corpus_chunk_count"),
    )
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
    corpus_config = run_config.get("corpus")
    if isinstance(corpus_config, dict) and isinstance(corpus_config.get("chunk_count"), int):
        return corpus_config["chunk_count"]
    return local_count


def get_config_values(report: dict[str, Any], local_count: int | None) -> dict[str, Any]:
    """整理验收必须相同的模型、提示词、语料和阈值配置。"""
    run_config = report.get("run_config", {})
    return {
        "LLM 模型名": run_config.get("llm_model"),
        "提示词 md5": get_prompt_md5(report),
        "语料块数": get_corpus_chunk_count(report, local_count),
        "召回窗口": (
            run_config.get("recall_vector_limit"),
            run_config.get("recall_keyword_limit"),
            run_config.get("rerank_candidate_limit"),
        ),
        "拒答阈值": run_config.get("refusal_min_vector_score"),
    }


def format_value(value: Any) -> str:
    """将缺失值、元组和浮点数格式化成稳定的验收输出。"""
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, tuple):
        return "/".join(format_value(item) for item in value)
    return str(value)


def format_metric(value: Any) -> str:
    """格式化指标，保留空值语义而不是把空值显示为零。"""
    return "未计算" if value is None else f"{float(value):.4f}"


def print_report_comparison(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """打印完整对照表，并返回关键口径是否一致。"""
    before_summary = before["summary"]
    after_summary = after["summary"]
    print("评测基线对照")
    print(f"改动前：{before.get('run_at', '未记录')} | 改动后：{after.get('run_at', '未记录')}")
    print("\n五指标")
    print("指标 | 改动前 | 改动后 | 变化")
    print("---|---:|---:|---:")
    for label, key in METRICS:
        before_value = before_summary.get(key)
        after_value = after_summary.get(key)
        if before_value is None or after_value is None:
            change = "未计算"
        else:
            change = f"{float(after_value) - float(before_value):+.4f}"
        print(f"{label} | {format_metric(before_value)} | {format_metric(after_value)} | {change}")

    before_types = before_summary.get("by_type", {})
    after_types = after_summary.get("by_type", {})
    type_names = sorted(set(before_types) | set(after_types))
    print("\n逐题型")
    print("题型 | 题数（前/后） | Recall@5（前/后） | MRR@10（前/后） | 拒答准确率（前/后）")
    print("---|---:|---:|---:|---:")
    for type_name in type_names:
        before_type = before_types.get(type_name, {})
        after_type = after_types.get(type_name, {})
        before_count = before_type.get("count")
        after_count = after_type.get("count")
        print(
            f"{type_name} | {format_value(before_count)}/{format_value(after_count)} | "
            f"{format_metric(before_type.get('recall_at_5'))}/{format_metric(after_type.get('recall_at_5'))} | "
            f"{format_metric(before_type.get('mrr10'))}/{format_metric(after_type.get('mrr10'))} | "
            f"{format_metric(before_type.get('refusal_accuracy'))}/{format_metric(after_type.get('refusal_accuracy'))}"
        )

    local_count = count_local_corpus_chunks(Path.cwd())
    before_config = get_config_values(before, local_count)
    after_config = get_config_values(after, local_count)
    config_same = True
    print("\nrun_config 关键项")
    print("对比项 | 改动前 | 改动后 | 状态")
    print("---|---|---|---")
    for label in before_config:
        before_value = before_config[label]
        after_value = after_config[label]
        same = before_value == after_value and before_value is not None and after_value is not None
        config_same = config_same and same
        status = "一致" if same else "不同/缺失"
        print(f"{label} | {format_value(before_value)} | {format_value(after_value)} | {status}")
    if config_same:
        print("\n口径结论：关键 run_config 一致，可直接对比。")
    else:
        print("\n口径不同，对比仅供参考。")
    return config_same


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数，要求显式提供两份输入报告。"""
    parser = argparse.ArgumentParser(description="比较两份评测 JSON 的部署前后指标")
    parser.add_argument("before", type=Path, help="改动前评测 JSON")
    parser.add_argument("after", type=Path, help="改动后评测 JSON")
    return parser


def main() -> int:
    """加载两份报告并返回适合脚本验收的退出码。"""
    arguments = build_parser().parse_args()
    try:
        before = load_report(arguments.before)
        after = load_report(arguments.after)
        print_report_comparison(before, after)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
