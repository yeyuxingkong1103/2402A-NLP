# -*- coding: utf-8 -*-
"""知识卡自检：把 scripts/validate_cards.py 的检查挂进 pytest。

为什么要有这层：清洗 / 切分 / 向量化每一步都可能重新生成卡片，而 card_id 重复、
高危卡漏打核实提示、附录A 表号错配这类问题**在建索引之后才发现就晚了**
（重复 id 会互相覆盖）。这里每次都把 data/processed/ 的三份 JSONL 全量过一遍。

不联网，只读 data/processed/*.jsonl。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]      # 仓库根目录（tests/ 的上一级）
sys.path.insert(0, str(ROOT / "scripts"))       # 把 scripts/ 加入导入路径，才能 import validate_cards

import validate_cards as V  # noqa: E402   # 被测对象：scripts/validate_cards.py 的校验逻辑

PROCESSED = ROOT / "data/processed"
CARDS = [PROCESSED / f for f in V.FILES]        # 三份知识卡 JSONL 文件路径（由 V.FILES 定义）


@pytest.fixture(scope="module")
def report():
    """模块级只跑一次校验：卡片没生成就跳过，否则返回 (卡片列表, 错误列表, 统计)。"""
    if not all(p.exists() for p in CARDS):
        pytest.skip("知识卡未生成（先跑 scripts/build_cards_*.py）")
    return V.collect()


def test_全部检查通过(report):
    """任何一条校验错误都让用例失败，并把明细原样打出来。"""
    _, errs, _ = report
    assert not errs, "知识卡校验未通过:\n" + "\n".join(f"  ✗ {e}" for e in errs)


def test_card_id_全局唯一(report):
    """三份文件合起来也必须唯一 —— 重复 id 进索引会互相覆盖。"""
    cards, _, _ = report
    dup = [k for k, v in Counter(c.get("card_id") for _, _, c in cards).items() if v > 1]
    assert not dup, f"card_id 重复: {dup}"


def test_三份卡片文件齐备且非空(report):
    """每个来源文件（GB/Z 26581 / 26583 / 26578）都必须至少产出一张卡，不允许整份标准被漏采。"""
    cards, _, _ = report
    got = Counter(fn for fn, _, _ in cards)
    for fn in V.FILES:
        assert got[fn] > 0, f"{fn} 无卡片"


def test_高危卡与核实提示一一对应(report):
    """高危 ⇔ needs_verification_hint。两个方向都要查，防止漏打也防止误打。"""
    cards, _, _ = report
    bad = [c["card_id"] for _, _, c in cards
           if c.get("is_high_risk") != c.get("needs_verification_hint")]
    assert not bad, f"高危标记与核实提示不一致: {bad}"


def test_附录B_高危数值缺失必进_missing_fields(report):
    """安全间隔期拿不到就记 missing_fields，绝不用邻行数值补（核心约束 3）。"""
    cards, _, _ = report
    bad = []
    for fn, _, c in cards:
        if not fn.startswith("appendixB"):
            continue
        for i, ch in enumerate(c.get("chemicals", [])):
            if ch.get("pre_harvest_interval") is None and \
               f"chemicals[{i}].pre_harvest_interval" not in c.get("missing_fields", []):
                bad.append(f"{c['card_id']}[{i}]")
    assert not bad, f"间隔期缺失但未记 missing_fields: {bad}"


def test_跨作物安全间隔期未串用(report):
    """同一药剂在不同作物的安全间隔期不同（噻虫嗪 黄瓜30d/辣椒18d/大蒜18d），
    必须各自独立，不能出现三份标准共用一张数值表。"""
    cards, _, _ = report
    seen = {}
    for fn, _, c in cards:
        if not fn.startswith("appendixB"):
            continue
        for ch in c["chemicals"]:
            seen.setdefault(ch["product"], set()).add(str(ch["pre_harvest_interval"]))
    # 同名产品若出现在多个作物且天数不同 -> 说明确实按作物隔离了，是预期行为
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    assert multi, "未发现跨作物数值差异，疑似把三份标准的数值表合并了"


def test_附录A_一律不进本期范围(report):
    """附录A 属于资料性附录，不构成本期检索语料，任何卡都不允许 in_scope=true。"""
    cards, _, _ = report
    bad = [c["card_id"] for fn, _, c in cards
           if fn.startswith("appendixA") and c.get("in_scope")]
    assert not bad, f"附录A 卡应 in_scope=false: {bad}"


def test_无卡引用不存在的标准号(report):
    """卡片 std_no 必须来自 data/raw 里真实存在的三份标准。"""
    cards, _, _ = report
    ok = {"GB/Z 26581-2011", "GB/Z 26583-2011", "GB/Z 26578-2011"}
    bad = {c.get("std_no") for _, _, c in cards} - ok
    assert not bad, f"出现范围外标准号: {bad}"
