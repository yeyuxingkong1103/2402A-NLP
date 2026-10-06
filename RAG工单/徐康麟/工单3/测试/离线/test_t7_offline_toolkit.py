# -*- coding: utf-8 -*-
"""离线级：断言工具自检（合成数据，不依赖产品实现）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么先测工具：口径写错会把**正确实现判成失败**（captain 反复强调）。
本文件用合成数据证明工具本身的方向性正确 —— 纵向同值不误报、行内横向重复必报、
0-based 错页引用必被抓、首字逐题判定、互污染判定方向正确。
"""

from __future__ import annotations

import pytest

from common import assertions

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def test_toolkit_self_check_all_green() -> None:
    """工具自检必须全绿（8 组造形）。"""
    report = assertions.toolkit_self_check()
    report.print()
    assert report.ok, "断言工具自检失败：\n" + "\n".join(c.render() for c in report.failures())


def test_evidence_hit_prefers_verbatim_over_synthetic() -> None:
    """题 207 口径：命中判定必须优先 ``evidence_verbatim``，并注明判据是 ``evidence_contains``。"""
    chunk = {"content": "……拟使用本次发行募集\n金15,000 万元用于补充流动资金……"}
    outcome = assertions.evidence_hit(
        [chunk],
        evidence="（合成串：14 题中唯一非逐字原文，任何块都不可能包含它）",
        verbatim="拟使用本次发行募集\n金15,000 万元用于补充流动资金",
    )
    assert outcome.hit is True, outcome.detail
    assert outcome.used_field == "evidence_verbatim", outcome.detail
    assert assertions.hit_is_evidence_based(outcome), outcome.detail

    # 反向：只用合成串必须判不命中（这正是「严格单证据 13/14 属基准缺陷」的原因）
    only_synthetic = assertions.evidence_hit(
        [chunk], evidence="（合成串：14 题中唯一非逐字原文，任何块都不可能包含它）")
    assert only_synthetic.hit is False


def test_non_defect_whitelist_is_complete() -> None:
    """非缺陷白名单（验收标准.md §3）必须齐备，防止后续有人「顺手」加回错误判据。"""
    items = {row["item"] for row in assertions.non_defect_verdicts()}
    for required in ("跨行纵向同值", "退化表不进索引", "引用页 ≠ evidence_pages",
                     "闸门 counted=False", "表格占位符 [ ◆ ]", "首次请求冷启动"):
        assert required in items, f"非缺陷白名单缺少「{required}」"


def test_ragas_banner_is_exact_and_never_numeric() -> None:
    """RAGAS 标注必须是固定原文；且不得出现任何伪造的 RAGAS 数值字段。"""
    banner = assertions.ragas_banner()
    assert banner == "RAGAS 未运行（依赖不可用，本机断网）"
    report = assertions.Report("RAGAS 标注检查")
    payload = report.to_dict()
    assert payload["ragas"] == banner
    assert "ragas_score" not in payload and "faithfulness" not in payload
