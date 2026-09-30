"""中文数字转阿拉伯数字。

独立成文件而不是塞进 structure.py 的原因：它是纯函数、无依赖、边界值多，
单独放便于把 1~1260 的边界用例写全，也避免 structure.py 触到 300 行上限。
"""
from __future__ import annotations

DIGITS = "零一二三四五六七八九"


def cn2int(text: str) -> int:
    """把「一千零四十」这类法条编号转成 1040。

    只支持到「千」位——民法典条号最大 1260，不引入万/亿的复杂度。
    「两」按 2 处理：「第两百条」与「第二百条」等价，法条里偶见前一种写法。
    """
    # 「第十条」的「十」前面没有数字，单独返回；否则会被当成 0 + 10 也算对，
    # 但显式写明可以避免读者以为这里漏了分支
    if text == "十":
        return 10
    total = 0
    num = 0
    for ch in text:
        if ch in DIGITS:
            num = DIGITS.index(ch)
        elif ch == "两":
            num = 2
        elif ch == "十":
            # 「十五」= 15（num 为 0 时按 1 计），「二十」= 20
            total += 10 * (num or 1)
            num = 0
        elif ch == "百":
            total += 100 * (num or 1)
            num = 0
        elif ch == "千":
            total += 1000 * (num or 1)
            num = 0
    # 个位数字在循环结束后仍未结算，在这里补上
    return total + num
