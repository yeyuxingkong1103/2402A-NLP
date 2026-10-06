# -*- coding: utf-8 -*-
"""症状覆盖层自检：校验器 + 切分合并的回归锁。

为什么单来一层：症状条目是**外部来源**（期刊/农科院/科普），挂在国标卡上。
挂空 = 症状静默丢失；挂错卡 = 把 A 病的症状安到 B 病头上，会给出错药。
不联网、不加载模型。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_symptoms as V  # noqa: E402  被测对象


def _ok_row():
    """一条合法样例。"""
    return {
        "symptom_id": "sym-test-01",
        "card_id": "CARD-A",
        "crop": "大蒜",
        "pest": "豌豆潜叶蝇",
        "symptom_text": "幼虫在叶片表皮下潜食叶肉，形成白色弯曲虫道",
        "part": ["叶片"],
        "source": {"kind": "journal", "title": "测试文献", "publisher": "《测试》2020(1)",
                   "url": None, "accessed": "2026-09-28", "quote": "原文"},
        "authority_level": 2,
        "needs_human_review": True,
    }


def test_合法样例通过():
    errs = V.validate([_ok_row()], {"CARD-A"}, {"CARD-A"})
    assert not errs, errs


def test_抓得住挂到不存在的卡():
    errs = V.validate([_ok_row()], set(), set())
    assert any("不存在" in e for e in errs), errs


def test_抓得住挂了不进索引的卡():
    """卡存在但没进索引（附录A / 光杆标题卡）——症状会静默丢失。"""
    errs = V.validate([_ok_row()], {"CARD-A"}, set())
    assert any("不进索引" in e for e in errs), errs


def test_抓得住空症状文本():
    r = _ok_row()
    r["symptom_text"] = "   "
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"})
    assert any("symptom_text" in e for e in errs), errs


def test_抓得住非法part():
    r = _ok_row()
    r["part"] = ["叶子"]
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"})
    assert any("part" in e for e in errs), errs


def test_抓得住挂错卡():
    """pest 必须是目标卡的独立词元；挂到别的病虫害卡上要拦下。"""
    cards = {"CARD-A": {"subtype": "蓟马", "source": {"quote": "蓟马｜单株2头~3头"}}}
    errs = V.validate([_ok_row()], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert any("独立词元" in e for e in errs), errs


def test_子串不算数_疫病挂不到早晚疫卡():
    """假阳性方向：「疫病」是「早疫病、晚疫病」的子串，必须判为对不上——
    挂上去就是把疫病症状安到早/晚疫方案卡上，等于给错药。"""
    r = _ok_row()
    r["pest"] = "疫病"
    cards = {"CARD-A": {"subtype": "早疫病、晚疫病",
                        "source": {"quote": "早疫病、晚疫病｜初见病叶｜方案一:…"}}}
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert any("独立词元" in e for e in errs), errs


def test_正文卡按quote认病虫害():
    """假阴性方向：正文卡 subtype 就是章节号，病虫害名只在 quote 里——必须放行。"""
    r = _ok_row()
    r["pest"] = "疫病"
    cards = {"CARD-A": {"subtype": "6.2",
                        "source": {"quote": "主要防治对象为:猝倒病、立枯病、疫病、病毒病等。"}}}
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert not errs, errs


def test_抓得住重复symptom_id():
    """两条同 id 会让复核清单与快照文件名互相覆盖。"""
    errs = V.validate([_ok_row(), _ok_row()], {"CARD-A"}, {"CARD-A"})
    assert any("symptom_id 重复" in e for e in errs), errs


@pytest.mark.skipif(not (ROOT / "data/processed/symptoms.jsonl").exists(),
                    reason="覆盖层未生成（Task 3 之前正常）")
def test_真实覆盖层全部通过():
    """真实文件必须零错误——有错就是挂错了卡或写漏了字段。"""
    rows, errs, _ = V.collect()
    assert not errs, "症状覆盖层校验未通过:\n" + "\n".join(f"  ✗ {e}" for e in errs)


sys.path.insert(0, str(ROOT / "src" / "index"))
import build_chunks as B  # noqa: E402  被测对象：切分合并逻辑


def _card():
    """一张最小可用的附录B 卡。"""
    return {
        "card_id": "CARD-A", "crop": "大蒜", "std_no": "GB/Z 00000-0000",
        "subtype": "豌豆潜叶蝇", "title": "大蒜豌豆潜叶蝇防治方案",
        "knowledge_type": "病虫害防治", "type_group": "病虫草害", "pest_kind": "虫害",
        "growth_stage": None, "trigger": "始见虫道", "precautions": None,
        "is_high_risk": True, "needs_verification_hint": True,
        "chemicals": [{"product": "10%落灭津SC", "dose": "30 g/667m²", "dilution": None,
                       "method": "对水喷雾", "interval_days": "7 d",
                       "pre_harvest_interval": None, "max_uses_per_season": None,
                       "companion_control": None, "raw": "方案一:10%落灭津SC 30 g/667m²"}],
        "measures": [], "source": {"section": "附录B 表B.1", "page": 20,
                                   "channel": "vision", "quote": "豌豆潜叶蝇｜始见虫道"},
    }


def _sym():
    return {"symptom_id": "sym-x", "card_id": "CARD-A", "crop": "大蒜", "pest": "豌豆潜叶蝇",
            "symptom_text": "叶片表皮下潜食叶肉，形成白色弯曲的蛇形虫道",
            "part": ["叶片"],
            "source": {"kind": "journal", "title": "测试文献", "publisher": "《测试》",
                       "url": None, "accessed": "2026-09-28", "quote": "原文"}}


def test_症状进sym_text不进embed和bm25():
    """症状文本只进 sym_text——embed_text/bm25_text 必须保持卡本体口径，
    否则会污染检索层从 bm25_text 现算的实体表（spec §4.1）。"""
    ch = B.build_chunk(_card(), [_sym()])
    assert "蛇形虫道" in ch["sym_text"]
    assert "蛇形虫道" not in ch["embed_text"]
    assert "蛇形虫道" not in ch["bm25_text"]
    assert ch["meta"]["has_symptoms"] is True
    assert ch["symptoms"][0]["symptom_id"] == "sym-x"


def test_无症状的卡字段为空():
    ch = B.build_chunk(_card(), [])
    assert ch["sym_text"] == ""
    assert ch["symptoms"] == []
    assert ch["meta"]["has_symptoms"] is False


def test_挂空卡直接报错():
    """症状挂到不存在的卡 → 必须拦下，不能静默丢。"""
    codes = B.check_orphans({"CARD-NOPE"}, {"CARD-A"})
    assert codes == ["CARD-NOPE"]
