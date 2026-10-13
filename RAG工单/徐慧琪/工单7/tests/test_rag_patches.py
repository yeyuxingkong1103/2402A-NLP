# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：生成侧的三个**确定性校正**（工单6 实测新增）
  · complete_row_list —— 表格列举题漏行补全（id4）
  · repair_quotes     —— 书名号名称取"最完整版本"（id95）
  · _take_with_arm_quota —— 融合截断时每路保底（id260）

均为纯函数测试，不加载模型、不联网。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rag, strategies  # noqa: E402

_TABLE = """2、不存在控制关系的关联方
| 企业名称 | 与本公司关系 |
|---|---|
| 融冰投资 | 持有公司股份5%以上的股东 |
| 武汉博润 | 持有公司股份5%以上的股东 |
| 上海博润 | 持有公司股份5%以上的股东 |
| 听音投资 | 持有公司股份5%以上的股东 |
| 联众聚源 | 持有公司股份5%以上的股东 |
| 力源贸易 | 同一实际控制人控制的企业 |
| 普芯达 | 实际控制人近亲属控制的公司 |
"""


def _blocks() -> list[dict]:
    return [{"is_table": True, "parent_text": _TABLE, "page_idx": 156}]


# ---------------------------------------------------------------------------
# complete_row_list
# ---------------------------------------------------------------------------
def test_row_patch_appends_missing_rows():
    q = "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"
    ans = "融冰投资、武汉博润、上海博润、听音投资、联众聚源。[1]"
    fixed, added = rag.complete_row_list(ans, _blocks(), q)
    assert set(added) == {"力源贸易", "普芯达"}
    for name in ("力源贸易", "普芯达"):
        assert name in fixed
    assert fixed.endswith("[1]")            # 引用标记仍留在末尾


def test_row_patch_skipped_for_non_list_question():
    q = "公司的注册资本是多少？"            # 非列举型
    ans = "5,520.00万元。[1]"
    fixed, added = rag.complete_row_list(ans, _blocks(), q)
    assert fixed == ans and added == []


def test_row_patch_skipped_when_answer_barely_covers_table():
    """答案只覆盖表里不到一半的行 → 更像"只有部分行相关"，不补（宁少勿错）。"""
    q = "存在控制关系的关联方有哪些？"
    ans = "武汉博润。[1]"
    fixed, added = rag.complete_row_list(ans, _blocks(), q)
    assert fixed == ans and added == []


def test_row_patch_noop_when_complete():
    q = "关联方企业有哪些？"
    ans = ("融冰投资、武汉博润、上海博润、听音投资、联众聚源、力源贸易、普芯达。[1]")
    fixed, added = rag.complete_row_list(ans, _blocks(), q)
    assert fixed == ans and added == []


def test_row_patch_ignores_non_table_blocks():
    q = "关联方企业有哪些？"
    ans = "融冰投资、武汉博润、上海博润、听音投资、联众聚源。[1]"
    blocks = [{"is_table": False, "parent_text": _TABLE}]
    fixed, added = rag.complete_row_list(ans, blocks, q)
    assert fixed == ans and added == []


# ---------------------------------------------------------------------------
# repair_quotes
# ---------------------------------------------------------------------------
def test_quote_repair_prefers_most_complete_name():
    ctx = ("公司参与制定了全军第一个视频指挥系统技术标准，即2019年制订的"
           "《某视频指挥系统技术规范（1.0版）》。另有噪声版本《某视频技术规范 1.0》出现于别处。")
    ans = "公司参与制定了全军第一个视频指挥系统技术标准（即《某视频技术规范1.0》）。[2]"
    fixed, fixes = rag.repair_quotes(ans, ctx)
    assert "《某视频指挥系统技术规范（1.0版）》" in fixed
    assert fixes and fixes[0][1] == "某视频指挥系统技术规范（1.0版）"


def test_quote_repair_noop_when_name_exact():
    ctx = "标准为《某视频指挥系统技术规范（1.0版）》。"
    ans = "答：《某视频指挥系统技术规范（1.0版）》。"
    fixed, fixes = rag.repair_quotes(ans, ctx)
    assert fixed == ans and fixes == []


def test_quote_repair_does_not_fabricate():
    """上下文里没有相近名称时不得造名字（宁可留着不一致，由重试/人工处理）。"""
    ctx = "标准为《某视频指挥系统技术规范（1.0版）》。"
    ans = "答：《完全不相干的名字》。"
    fixed, fixes = rag.repair_quotes(ans, ctx)
    assert fixed == ans and fixes == []


def test_quote_repair_handles_empty_inputs():
    assert rag.repair_quotes("", "《X》")[0] == ""
    assert rag.repair_quotes("《Y》", "")[0] == "《Y》"


# ---------------------------------------------------------------------------
# 每路保底（融合截断）
# ---------------------------------------------------------------------------
def _hit(cid: str, src: str = "a.pdf") -> dict:
    return {"source": src, "chunk_id": cid, "text": f"t-{cid}"}


def test_arm_quota_keeps_each_arm_top_hits():
    """保底：融合序之外，每路前 ARM_QUOTA 名必须进入精排候选（上限 RERANK_MAX_IN）。

    实测动机（id260）：正确块只在全文臂排第 12，若不保底就进不了精排。
    """
    quota = strategies.config.ARM_QUOTA
    dense = [_hit(f"d{i}") for i in range(quota + 10)]
    fulltext = [_hit(f"f{i}") for i in range(quota + 10)]
    # 融合结果全部来自两路共同命中的块（模拟真实噪声），单路头部不在里面
    fused = [_hit(f"x{i}") for i in range(20)]
    ranked = [("dense", dense), ("fulltext", fulltext)]
    cap = 20 + 2 * quota                      # 20 融合 + 两路各 quota
    out = strategies._take_with_arm_quota(fused, ranked, top_k=20, max_in=cap)
    keys = {strategies.fusion_mod.doc_key(h) for h in out}
    assert len(out) == cap
    for i in range(quota):
        assert f"a.pdf#d{i}" in keys          # dense 前 N 保底
        assert f"a.pdf#f{i}" in keys          # fulltext 前 N 保底
    # 融合序仍然打底（排在最前）
    assert strategies.fusion_mod.doc_key(out[0]) == "a.pdf#x0"


def test_arm_quota_respects_max_in_cap():
    dense = [_hit(f"d{i}") for i in range(30)]
    fulltext = [_hit(f"f{i}") for i in range(30)]
    fused = [_hit(f"x{i}") for i in range(40)]
    out = strategies._take_with_arm_quota(fused, [("dense", dense), ("fulltext", fulltext)],
                                          top_k=20, max_in=25)
    assert len(out) == 25                      # 上限裁剪


def test_arm_quota_dedupes_and_respects_top_k():
    dense = [_hit("same"), _hit("d2")]
    fulltext = [_hit("same"), _hit("f2")]
    fused = [_hit("same")] + [_hit(f"x{i}") for i in range(10)]
    out = strategies._take_with_arm_quota(fused, [("dense", dense), ("fulltext", fulltext)],
                                          top_k=5, max_in=5)
    assert len(out) == 5
    keys = [strategies.fusion_mod.doc_key(h) for h in out]
    assert len(keys) == len(set(keys))         # 无重复


def test_arm_quota_zero_disables():
    """quota=0 时退化为纯融合序截断（配置可关）。"""
    old = strategies.config.ARM_QUOTA
    try:
        strategies.config.ARM_QUOTA = 0
        fused = [_hit(f"x{i}") for i in range(10)]
        out = strategies._take_with_arm_quota(fused, [("dense", [_hit("d0")])], top_k=3)
        assert [h["chunk_id"] for h in out] == ["x0", "x1", "x2"]
    finally:
        strategies.config.ARM_QUOTA = old
