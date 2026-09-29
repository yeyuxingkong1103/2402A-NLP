# -*- coding: utf-8 -*-
"""回答侧判定原语（拒答口径 + 引用越界审计，批次 24 自 run_eval.py 拆出）。

为什么单独成文件：见 `legal_matching.py` 顶部说明（守"单文件 ≤300 行"）。
本模块**零逻辑改动**：常量与函数体逐字搬移，对外函数名不变。

口径与 `data/evaluation/_schema.md` 一字对应（批次 11 裁决的 R2 + R1′）：
- R2：回答里没有任何 `[n]` 引用编号 → 判为拒答成功；
- R1′：没有 `[n]` 却给出"具体法条指引"（如凭空引"《XX法》第X条"）→ 不算拒答成功（虚构法条依据）。
旧口径要求命中 `REFUSAL_PHRASES` 词表，会把"法律没有强制规定，属用人单位自主"这类
正确拒答误判为失败（refusal-076/081/082），已按批次 11 裁决移除该要求。
"""

from __future__ import annotations

import re
from typing import Any

# REFUSAL_PHRASES 旧词表仅留作参考/debug，不再作为判定必要条件
REFUSAL_PHRASES = (
    "无法确认", "无法确定", "无法回答", "不能确定", "不确定",
    "现有资料不足", "资料不足", "信息不足", "未收录", "未包含",
    "不在知识库", "知识库中无", "库中无", "没有相关规定", "无相关规定",
    "查不到", "查无", "无法找到",
    "建议咨询执业律师", "建议咨询律师", "咨询当地人社部门", "咨询当地人力资源社会保障部门",
    "超出我的知识范围", "超出知识范围",
)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# R1′：无 [n] 引用却给出"具体法条指引"= 虚构法条依据（详见 data/evaluation/_schema.md R1′）
STATUTE_BASIS_PATTERNS = (
    # 1) 《法规名》……第X条（含款/项）
    re.compile(r"《[^》]{2,40}》[^。；\n]{0,12}第[一二三四五六七八九十百千万零〇两\d]+条"),
    # 2) 第X条……（之）规定（脱离法规名直接引条号）
    re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+条[^。；\n]{0,8}(?:之)?规定"),
    # 3) 根据/依据/按照 …… 第X条
    re.compile(r"(?:根据|依据|按照)[^。；\n]{0,30}?第[一二三四五六七八九十百千万零〇两\d]+条"),
)


def refused_by_text(answer: str) -> bool:
    """批次 11 口径（R2+R1′）：无任何 [n] 引用编号 且 未虚构法条依据 → 拒答成功。

    旧口径要求命中 REFUSAL_PHRASES 词表，会把"法律没有强制规定，属用人单位自主"
    这类正确拒答误判失败（refusal-076/081/082），已按批次 11 裁决移除该要求。
    """
    return not refusal_failure_reason(answer)


def refusal_failure_reason(answer: str) -> str | None:
    """返回拒答失败原因（None=判定拒答成功），供明细 debug 用。"""
    if CITATION_PATTERN.search(answer or ""):
        return "has_citation"
    text = (answer or "").replace(" ", "").replace("\u3000", "")
    for pattern in STATUTE_BASIS_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"statute_basis:{match.group(0)[:24]}"
    return None


def citation_audit(answer: str, citation_count: int) -> dict[str, Any]:
    """R3：回答里的 [n] 是否都落在本轮引用清单范围内。"""
    numbers = [int(m) for m in CITATION_PATTERN.findall(answer or "")]
    invalid = [n for n in numbers if n < 1 or n > citation_count]
    return {
        "citations": numbers,
        "invalid": invalid,
        "total": len(numbers),
        "correct": len(numbers) - len(invalid),
        "no_citation": len(numbers) == 0,
    }
