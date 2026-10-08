# -*- coding: utf-8 -*-
"""产物定位、读取与校验。

不调用 MinerU——本模块只关心"磁盘上有什么"，可脱开外部依赖独立执行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import EXIT_VALIDATION, ScriptError

# docs/04 §5 记录的 3.x pipeline 后端类型集合。集合外的类型意味着可能有内容被无声
# 跳过，必须停止并人工决策，不得静默忽略。
KNOWN_BLOCK_TYPES = frozenset({
    "text", "image", "table", "chart", "equation", "code", "list",
    "header", "footer", "page_number", "aside_text", "page_footnote", "discarded",
})


def locate_content_list(doc_dir: Path) -> Path:
    """容错定位 content_list.json。

    两个都不能写死：目录层级随版本变（3.4.5 是 {pdf名}/{auto|txt}/），文件名也带
    PDF 名前缀（{pdf名}_content_list.json）。故用 * 兜住前缀、rglob 兜住层级。
    不能写成 *content_list*.json——那会连 _content_list_v2.json 一起捞进来。
    """
    hits = sorted(doc_dir.rglob("*content_list.json"))
    if not hits:
        raise ScriptError(
            EXIT_VALIDATION,
            f"未找到 content_list.json（目录：{doc_dir}）\n"
            f"       MinerU 输出契约可能已变化，请人工检查产物目录结构。")
    if len(hits) > 1:
        print(f"WARN: 发现 {len(hits)} 个 content_list.json，取 {hits[0]}，请人工确认")
    return hits[0]


def load_content_items(content_list_path: Path) -> list[dict[str, Any]]:
    """读取并做最基本的完整性校验。空产物一律视为失败，不得冒充成功。"""
    try:
        data = json.loads(content_list_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScriptError(EXIT_VALIDATION, f"content_list.json 不是合法 JSON：{exc}") from exc
    if not isinstance(data, list):
        raise ScriptError(EXIT_VALIDATION, f"content_list.json 顶层不是数组：{content_list_path}")
    if not data:
        raise ScriptError(EXIT_VALIDATION,
                          f"content_list.json 为空：{content_list_path}\n"
                          f"       拒绝写出空产物冒充成功。")
    return data


def validate_page_idx(items: list[dict[str, Any]]) -> None:
    """页码是引用功能的物理前提。缺失即退出码 2，禁止以 0/null 填充（docs/04 §11）。"""
    missing = [index for index, item in enumerate(items) if "page_idx" not in item]
    if missing:
        raise ScriptError(
            EXIT_VALIDATION,
            f"content_list 中有 {len(missing)} 个块缺少 page_idx（索引示例：{missing[:5]}）\n"
            f"       禁止以 0/null 填充后继续——那会让引用指向错误的页码。")


def is_artifact_complete(doc_dir: Path) -> bool:
    """判定产物是否可跳过。这是探针而非错误吞噬：产物损坏即视为未完成并重跑。"""
    if not doc_dir.is_dir():
        return False
    try:
        return len(load_content_items(locate_content_list(doc_dir))) > 0
    except (ScriptError, OSError):
        return False


def build_type_distribution(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    """统计类型分布，用于核对 MinerU 实际输出契约（docs/04 §15 待验证项 P1）。"""
    counts: dict[str, int] = {}
    for item in items:
        block_type = item.get("type", "<缺失>")
        counts[block_type] = counts.get(block_type, 0) + 1
    pages = [item["page_idx"] for item in items]
    return {
        "backend": backend,
        "total_blocks": len(items),
        "type_counts": counts,
        "page_idx_range": [min(pages), max(pages)],
        "unknown_types": sorted(t for t in counts if t not in KNOWN_BLOCK_TYPES),
    }


def print_report(distribution: dict[str, Any], mineru_version: str) -> None:
    """按固定格式打印类型分布报告（契约见 specs/001-mineru-pdf-parse/contracts/cli.md）。"""
    print(f"--- 类型分布 (backend={distribution['backend']}, mineru={mineru_version}) ---")
    for block_type, count in sorted(distribution["type_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {block_type:<20s} {count:>6d}")
    print(f"  {'合计':<19s} {distribution['total_blocks']:>6d}")
    print(f"  page_idx 范围        {distribution['page_idx_range']}")
    unknown = distribution["unknown_types"]
    print(f"  未知类型             {'、'.join(unknown) if unknown else '无'}")
