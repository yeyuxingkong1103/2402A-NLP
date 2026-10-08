# -*- coding: utf-8 -*-
"""功能5：敏感信息检测。输出"待审核列表"，每条命中附带上下文。"""
from __future__ import annotations

import re
from typing import Dict, List

# 手机号：11 位，1[3-9] 开头，前后不能再有数字
_RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 手机号（允许数字间夹空格/横线分隔，如 138 1234 5678 / 138-1234-5678）
_RE_PHONE_SEP = re.compile(r"(?<!\d)1[3-9]\d(?:[\s-]?\d){8}(?!\d)")
# 邮箱
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 身份证：18 位（末位 X）
_RE_IDCARD = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
# 银行卡：16~19 位（默认关闭）
_RE_BANKCARD = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")

_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_CHECK = "10X98765432"


def _idcard_valid(num: str) -> bool:
    """身份证校验位验证，大幅降低 18 位连续数字的误报。"""
    try:
        s = sum(int(num[i]) * _ID_WEIGHTS[i] for i in range(17))
        return _ID_CHECK[s % 11] == num[17].upper()
    except (ValueError, IndexError):
        return False


def _luhn_valid(num: str) -> bool:
    digits = [int(c) for c in num]
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _context(text: str, start: int, end: int, radius: int) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    snippet = text[lo:hi].replace("\n", " ").strip()
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def detect_sensitive(text: str, config: dict) -> List[Dict]:
    """返回待审核命中列表。每条含 type/match/context/offset。"""
    radius = config["context_chars"]
    max_per_type = config["max_findings_per_type"]
    findings: List[Dict] = []

    def add(kind: str, match: re.Match, display: str = None):
        if sum(1 for f in findings if f["type"] == kind) >= max_per_type:
            return
        val = display if display is not None else match.group(0)
        findings.append({
            "type": kind,
            "match": val,
            "offset": match.start(),
            "context": _context(text, match.start(), match.end(), radius),
            "status": "待审核",
        })

    if config.get("phone"):
        occupied = []  # 已被严格模式覆盖的区间，避免重复报
        for m in _RE_PHONE.finditer(text):
            add("手机号", m)
            occupied.append(m.span())
        for m in _RE_PHONE_SEP.finditer(text):
            # 与严格命中区间重叠则跳过；归一化（去分隔符）后必须恰为11位
            if any(not (m.end() <= lo or m.start() >= hi) for lo, hi in occupied):
                continue
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 11:
                add("手机号", m, digits)
    if config.get("email"):
        for m in _RE_EMAIL.finditer(text):
            add("邮箱", m)
    if config.get("idcard"):
        for m in _RE_IDCARD.finditer(text):
            if _idcard_valid(m.group(1)):
                add("身份证", m)
    if config.get("bankcard"):
        for m in _RE_BANKCARD.finditer(text):
            if _luhn_valid(m.group(1)):
                add("银行卡", m)

    return findings
