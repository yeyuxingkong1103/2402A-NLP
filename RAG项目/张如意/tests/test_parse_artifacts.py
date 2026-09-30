# -*- coding: utf-8 -*-
"""解析提示词产物自检：不联网，校验 configs/prompts/*.md 与 docs/05 的一致性。

覆盖：schema 完整性（对齐《01-需求分析》7.4 元数据字段 + 核心约束 1/4）、
规则块齐全性、视觉转录提示词规则、以及 docs/05 里「需视觉通道页」与 PDF 实测一致。
"""
import json
import re
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/prompts/parse_pdf_to_knowledge.md"
VLM = ROOT / "configs/prompts/vlm_transcribe_table.md"
DOC = ROOT / "docs/05-解析提示词设计.md"

# 01-需求分析 7.4 元数据字段 -> schema 字段名
META_FIELDS = {"crop": "作物", "region": "地域", "growth_stage": "生育期",
               "knowledge_type": "知识类型", "source": "来源", "year": "年份",
               "authority_level": "权威等级", "is_high_risk": "是否高危"}
EXTRA_FIELDS = {"card_id", "std_no", "std_name", "title", "type_group", "subtype", "measures",
                "chemicals", "high_risk_reasons", "needs_human_review",
                "needs_verification_hint", "ocr_uncertain", "missing_fields", "in_scope",
                # 2026-09-16 新增：病/虫细分、兼治对象、附录A 表单结构
                "pest_kind", "record_form"}

# knowledge_type 只允许枚举内的值。注意**没有**「病害防治」「虫害防治」——
# 病/虫细分走 pest_kind。（CLAUDE.md 第六节该字段的「示例」列曾写作「虫害防治」，
# 与本枚举不一致，以本枚举为准。）
KNOWLEDGE_TYPES = {"病虫害防治", "杂草防治", "鼠害防治", "土肥水管理", "品种选择", "农事操作",
                   "灾害应对", "政策补贴", "基地与投入品管理", "劳动保护", "档案记录",
                   "加工贮运", "其他"}
PEST_KINDS = {"病害", "虫害", "杂草", None}

main = MAIN.read_text(encoding="utf-8")
vlm = VLM.read_text(encoding="utf-8")
doc = DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def card():
    """从主提示词中抽出 JSON schema 示例块，取第一张卡作为被检对象。"""
    block, = re.findall(r"```json\s*(\[.*?\])\s*```", main, re.S)
    return json.loads(block)[0]


def test_schema_覆盖_7_4_元数据字段(card):
    """8 个元数据字段（作物/地域/生育期等）必须一个不少地出现在示例 schema 里。"""
    assert not set(META_FIELDS) - set(card), f"缺 {set(META_FIELDS) - set(card)}"


def test_schema_覆盖约束附加字段(card):
    """核心约束要求的扩展字段（card_id/chemicals/pest_kind 等）必须齐全，source 子结构四要素齐备。"""
    assert not EXTRA_FIELDS - set(card), f"缺 {EXTRA_FIELDS - set(card)}"
    assert {"section", "page", "channel", "quote"} <= set(card["source"])


def test_示例高危卡带核实提示(card):
    """示例卡必须是高危卡且带核实提示——约束 1「必须提醒线下核实」的示范位。"""
    assert card["is_high_risk"] and card["needs_verification_hint"]


def test_示例_knowledge_type_在枚举内(card):
    """病/虫细分不得混进 knowledge_type —— 这是踩过的坑，见 pest_kind。"""
    assert card["knowledge_type"] in KNOWLEDGE_TYPES, card["knowledge_type"]
    assert card["pest_kind"] in PEST_KINDS
    assert "病害防治" not in KNOWLEDGE_TYPES and "虫害防治" not in KNOWLEDGE_TYPES


def test_示例兼治挂在方案级(card):
    """兼治对象是 chemicals[] 的字段，不是卡级字段。"""
    assert "companion_control" in card["chemicals"][0]
    assert "companion_control" not in card


def test_record_form_在_schema_中说明(card):
    """附录A 表单结构字段必须有定义与示例说明，否则只活在脚本里。"""
    assert "record_form" in card
    assert "3.1 `record_form`" in main
    for k in ("table_no", "table_name", "columns", "is_blank_form"):
        assert f'"{k}"' in main, f"record_form 缺 {k} 的示例"


def test_缺字段进_missing_fields_而非编值(card):
    """原文没有的安全间隔期必须显式置 None 并登记进 missing_fields——禁止模型编数值。"""
    assert card["chemicals"][0]["pre_harvest_interval"] is None
    assert any("pre_harvest_interval" in m for m in card["missing_fields"])


def test_规则块齐全():
    """主提示词的骨架检查：0 节前 6 条规则、P1–P5 预处理、H1–H6 输出格式、末尾 ≥7 条自检项。"""
    assert len(re.findall(r"^\d+\. \*\*", main.split("### 0.")[1].split("### 1.")[0], re.M)) == 6
    assert all(f"**{t} " in main for t in ("P1", "P2", "P3", "P4", "P5"))
    assert all(f"**{t}" in main for t in ("H1", "H2", "H3", "H4", "H5", "H6"))
    assert len(re.findall(r"^\d+\. ", main.split("### 6. 输出前自检")[1], re.M)) >= 7


def test_范围由占位符注入不写死():
    """范围必须在调用时注入，不得写死在提示词里；已定范围要与 CLAUDE.md / docs/01 一致。"""
    assert "{{IN_SCOPE_CROPS}}" in main
    sec3 = main.split("## 三、")[1]
    assert "作物范围已确定" in sec3 and "黄瓜、辣椒、大蒜" in sec3
    assert "新增作物" in sec3                     # 扩范围须先确认，不得只改提示词


def test_视觉转录提示词规则():
    """视觉通道提示词：10 条规则、禁止补全/换算、不确定标记〔?〕与上标单位，末尾 ≥5 条页内自查。"""
    assert len(re.findall(r"^\d+\. \*\*", vlm, re.M)) == 10
    assert "禁止补全" in vlm and "禁止换算" in vlm
    assert "〔?〕" in vlm and "m²" in vlm and "hm²" in vlm
    assert len(re.findall(r"^\d+\. ", vlm.split("### 页内自查")[1], re.M)) >= 5


def test_文档引用存在且无编号撞号():
    """提示词/文档里引用到的文件必须真实存在；且不允许残留旧编号 03 的语料清单。"""
    for ref in ("configs/prompts/parse_pdf_to_knowledge.md", "configs/prompts/vlm_transcribe_table.md",
                "docs/01-需求分析.md", "docs/04-语料清单-农业国标PDF.md"):
        assert (ROOT / ref).exists(), ref
    assert not (ROOT / "docs/03-语料清单-农业国标PDF.md").exists()


@pytest.mark.parametrize("pdf,stdno", [
    ("GB_Z 26581-2011 黄瓜生产技术规范.pdf", "26581-2011"),
    ("GB_Z 26583-2011 辣椒生产技术规范.pdf", "26583-2011"),
    ("GB_Z 26578-2011 大蒜生产技术规范.pdf", "26578-2011"),
])
def test_需视觉通道页与实测一致(pdf, stdno):
    """docs/05 表格里登记的异常页必须与 PDF 现算结果一致（CJK < 60 判异常）。"""
    path = ROOT / "data/raw" / pdf
    if not path.exists():
        pytest.skip("语料未就位")
    d = fitz.open(path)
    low = [str(p + 1) for p in range(d.page_count)
           if len(re.findall(r"[\u4e00-\u9fff]", d[p].get_text())) < 60]
    row = next(l for l in doc.splitlines() if l.startswith(f"| GB/Z {stdno}"))
    assert low == re.findall(r"p(\d+)", row.split("|")[3])
