"""数值规范化：让“等价写法”在判分与比对时一致。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 通用工具（判分与英文金额换算共用；**判分口径不允许在此放宽**）

招股书里的金额有大量等价写法，例如::

    5,520 万元      5,520.00 万元      5520万元      5,520万
    15,000.00 万元  1.5 亿元           15000万元

如果按字符串比较，这些都会被判成“不一致”，导致准确率被严重低估
（实测：标准答案 5,520 万元 vs 抽取答案 5,520.00 万元 → 误判为错）。
本模块提供统一的数值解析，全部折算成**元**再做比较。

设计要点：
- 纯函数、无外部依赖，便于单元测试；
- 只处理中文财务文本里真实出现的单位，不猜测；
- 无法解析时返回 ``None``，调用方自行决定回退策略（绝不抛异常）。
"""

from __future__ import annotations

import re

# 单位 -> 折算成“元”的倍率
UNIT_SCALES: tuple[tuple[str, float], ...] = (
    ("亿元", 1e8),
    ("亿", 1e8),
    ("万元", 1e4),
    ("万", 1e4),
    ("元", 1.0),
)

# 数字核（千分位或小数）；单位单独一次匹配，允许中间有空白
_NUMBER_CORE = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_UNIT_CORE = r"(亿元|亿|万元|万|元)?"
_AMOUNT_RE = re.compile(_NUMBER_CORE + r"\s*" + _UNIT_CORE)
# 顺序扫描版本：用 finditer 逐个消费“数字+可选单位”，避免单位被漏掉
_SCAN_RE = re.compile(_NUMBER_CORE + r"\s*" + _UNIT_CORE)


def parse_amount(text: str) -> float | None:
    """把 ``"5,520.00 万元"`` 这样的片段解析成以**元**为单位的数值。

    Args:
        text: 含数字与可选单位的字符串。

    Returns:
        折算成元的浮点数；无法解析时返回 ``None``。
    """
    if not text:
        return None
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    return _scale(match.group(1), match.group(2))


def _scale(number_text: str, unit: str | None) -> float:
    """按单位把数字折算成元；无单位时返回原值。"""
    number = float(number_text.replace(",", ""))
    if not unit:
        return number
    for name, scale in UNIT_SCALES:
        if unit == name:
            return number * scale
    return number


def parse_all_amounts(text: str) -> list[float]:
    """抽取文本里所有金额并折算成元（用于“数字是否全部出现”的判断）。"""
    if not text:
        return []
    return [_scale(match.group(1), match.group(2)) for match in _SCAN_RE.finditer(text)]


def to_million_yuan(value_yuan: float) -> float:
    """把“元”换算成“百万元”（英文表述 ``million yuan`` 用）。"""
    return value_yuan / 1e6


def to_ten_thousand_yuan(value_yuan: float) -> float:
    """把“元”换算成“万元”。"""
    return value_yuan / 1e4


def format_cny_en(value_yuan: float) -> str:
    """把金额格式化成英文表述。

    例：``55200000`` -> ``"55.2 million yuan (RMB 5,520万元)"``

    为什么要用代码而不是让模型翻译单位：小模型（0.6B）会把
    “5,520 万元”错译成 “5,520 million yuan”（差 100 倍）。
    金额单位必须由确定性代码换算，模型只负责组织语言。
    """
    if value_yuan >= 1e8:
        return f"{value_yuan / 1e8:,.2f} hundred million yuan"
    if value_yuan >= 1e6:
        return f"{value_yuan / 1e6:,.2f} million yuan"
    if value_yuan >= 1e4:
        return f"{value_yuan / 1e4:,.2f} ten-thousand yuan"
    return f"{value_yuan:,.2f} yuan"


def amounts_equivalent(left: str, right: str, tolerance: float = 1e-6) -> bool:
    """判断两段文本里的金额是否等价（折算成元后比较）。"""
    left_value = parse_amount(left)
    right_value = parse_amount(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) <= tolerance * max(1.0, abs(right_value))


def normalize_number_strings(text: str) -> set[str]:
    """把文本中的数字抽取成“规范化字符串集合”。

    每个**带单位**的数字会折算成元后再规范化，因此
    ``5,520 万元`` / ``5,520.00 万元`` / ``5520万`` 得到同一个规范值，
    跨单位写法（``1.5 亿元`` 与 ``15,000 万元``）也能正确判等。

    只带单位的值才会被收录：若把不带单位的裸数字也收进来，
    “军用领域收入”这类纯数字串会与金额混在一起，反而制造误判。
    """
    normalized: set[str] = set()
    for match in _SCAN_RE.finditer(text or ""):
        number_text, unit = match.group(1), match.group(2)
        if unit:
            normalized.add(_trim_float(_scale(number_text, unit)))
    return normalized


def _trim_float(value: float | str) -> str:
    """把数值转成稳定字符串（去掉多余的 0 与小数点）。"""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def normalize_amount_text(text: str) -> str:
    """归一化金额字面量的写法，用于**生成答案文本**（不是判分）。

    招股书同一金额常有 ``5,520 万元`` / ``5,520.00 万元`` 两种写法，
    若不统一，同一问题会因命中的是正文页还是表格页而输出不同文本。
    这里保留千分位、去掉无意义的小数零：

        "5,520.00" -> "5,520"
        "15,000.00" -> "15,000"
        "1,234.50" -> "1,234.50"   （有意义的小数保留）
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return raw
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"
