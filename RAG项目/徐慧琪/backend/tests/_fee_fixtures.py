# 费用编排测试共用的夹具与常量。
#
# 为什么拆成独立模块（与 _fakes.py 同款理由）：test_fees.py 的非空行数顶到
# 「单文件 ≤ 300」的硬闸门，而本批还要往费用这一侧加用例（量级闸门、效力闸门），
# 只能按主题拆文件。拆开后「命中片段」这个夹具被两个测试文件共用 —— 它现在
# 多了一个 status 字段（estimate 的效力闸门要求现行，缺了判拒），留在多处各写
# 一份就会改一处漏一处：漏掉的那份会让对应用例静默变成 rejected，而断言里
# 只看状态时表现成「闸门误伤」。本模块名不带 test_ 前缀，pytest 不收集它。
from __future__ import annotations

# 语料片段取自政府指导价的典型表述。含量级词「万元」是**有意**的：真语料
# （data/raw/fee/北京智深律师事务所收费标准.txt）里「1 万元－50 万元」正是
# 「模型不报单位就静默错 10000 倍」那个真实标本
SNIPPET = "财产案件根据诉讼标的额分段累计交纳：不超过1万元的，每件交纳50元；超过1万元至10万元的部分，按照2.5%交纳。"

# 无量级词的片段：数字 1 与 10 各有字面出处、也都不紧邻量级词 —— 合法的
# 无单位口径（按件、按百分比）就长这样，缺省闸门不得把它们一起拒掉
FLAT_SNIPPET = "按阶段收费：每件 1 元至 10 元，另按 2.5% 收取风险代理费。"

# 命中片段的最小形状：text（生成要整段、回查要数字）+ 溯源两字段 + 效力。
# status 是 estimate 效力闸门的**必需项**（缺了按非现行判拒），不是可选装饰
HIT = {"text": SNIPPET, "source_doc": "诉讼费用交纳办法", "source_no": "第十三条",
       "status": "现行有效"}

# 同上，但片段无量级词：缺 unit 的 ok 方向用它（溯源两字段与 HIT 刻意同值，
# 便于两个测试文件的断言逐字对齐）
FLAT_HIT = {"text": FLAT_SNIPPET, "source_doc": "诉讼费用交纳办法",
            "source_no": "第十三条", "status": "现行有效"}


def search(hits):
    """假检索：固定返回这一批命中（estimate 只取 hits[0]）。"""
    return lambda cause: hits


def gen(low, high):
    """假生成器：不带 unit 上报（unit 缺省的两个方向都有专门用例）。"""
    return lambda snippet, cause: {"low": low, "high": high}


def gen_unit(low, high, unit):
    """带上 unit 上报的生成器：unit 是模型的可选字段（见编排的 unit 闸门）。"""
    return lambda snippet, cause: {"low": low, "high": high, "unit": unit}


def gen_basis(low, high, charge_basis):
    """带上计价基础上报的生成器：charge_basis 同为模型的可选字段（缺省不判拒）。"""
    return lambda snippet, cause: {"low": low, "high": high,
                                   "charge_basis": charge_basis}


def no_log(*args, **kwargs):
    """不留痕的 log_fn：只测分支、不关心 AC-21 的用例用它。"""
    return None
