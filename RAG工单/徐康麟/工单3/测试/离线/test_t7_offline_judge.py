# -*- coding: utf-8 -*-
"""离线级：判分口径自检（§2.1 五步 + 强制负例 N-1/N-2/N-3）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么必须测判分口径本身：验收 3「14 题准确率 ≥ 90%」完全依赖这把尺子。
若尺子坏了（放宽/收紧），准确率数字就失去意义。本文件把 设计/验收标准.md §2.1 的
强制负例逐条钉死，并**如实记录**口径与负例冲突的情况（冲突时判失败并写明依据，
不静默放宽、也不为了转绿而删用例）。
"""

from __future__ import annotations

import pytest

from common import judge_local

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def test_judge_chain_reports_source() -> None:
    """判分链必须标注口径来源（产品 bridge / 工单1 权威 / 本地副本），不得静默换尺子。"""
    ok, reason, source = judge_local.judge("注册资本为5,520万元。", "注册资本为5,520.00万元。")
    assert ok is True, reason
    assert source, "判分来源缺失"
    print(f"判分口径来源：{source}")


def test_mandatory_negative_n2_and_n3() -> None:
    """N-2（缺千分位/小数应判**对**）与 N-3（答「未知。」应判**错**）必须符合设计。"""
    report = {row["case"]: row for row in judge_local.negative_case_report()}
    n2 = next(row for name, row in report.items() if name.startswith("N-2"))
    n3 = next(row for name, row in report.items() if name.startswith("N-3"))
    assert n2["passed"], f"N-2 未通过：{n2}"
    assert n3["passed"], f"N-3 未通过：{n3}"


def test_mandatory_negative_n1_partial_answer_must_be_wrong() -> None:
    """N-1：题 33 只答「82.10%」（缺 97.31/94.84/94.34）**必须判错**。

    ⚠️ 实测（2026-10-04，定稿代码）：无论走工单1 权威 Evaluator 还是本地副本，
    该输入都被判**对** —— 因为 §2.1 第①步是**双向**子串判定，
    「8210%」是参考答案（含四个比重）的子串，于是在第①步就放行了，
    第②步「参考答案中的全部数值都要命中」根本没机会生效。

     ⚠️ 本用例**预期为红，且不得为了让测试转绿而删改** —— 它记录的是**判分尺子的已知特性**：

    * 现象：无论走工单1 权威 Evaluator 还是本地副本，``82.10%`` 都被判**对**
      —— §2.1 第①步是**双向**子串判定，「8210%」是参考答案（含四个比重）的子串，
      于是第①步就放行，第②步「参考答案中全部数值都要命中」根本没机会生效。
    * captain 裁定（2026-10-04）：**判分器不修**。它是与工单2 基线**同口径**的尺子，
      放宽/收紧都会让「优化前后对比」失去意义；尺子偏松对基线与本工单同等作用，不构成不公平。
    * 兜底责任在**产品侧**：枚举型问题必须给全枚举（题 33 先给全 4 个比重），
      由 ``assertions.ENUMERATED_QUESTIONS`` 的**要素齐备硬断言**把关
      （见 ``测试/在线/test_t7_online_qa14.py::test_enumerated_answers_are_complete``），
      **不得**依赖尺子漏放。
    """
    golden = ("报告期内，公司来自军用领域的收入占主营业务收入的比重分别为"
              "82.10%、97.31%、94.84%和94.34%。")
    ok, reason, source = judge_local.judge("82.10%", golden)
    assert ok is False, (f"§2.1 负例 N-1 未生效：只答 82.10% 被判为**正确**（依据：{reason}）；"
                         f"判分口径来源：{source}；根因：第①步双向子串判定先放行，"
                         f"第②步的「全部数值命中」检查被跳过。"
                         f"【性质】判分尺子已知特性 —— captain 裁定不修判分器，"
                         f"产品侧以「枚举完整性自检」兜底")
