# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：数字校验（防幻觉）单元测试

重点覆盖两个工单3 相关的行为：
  1. 表格题允许"用表中两个数一步算出比例"（id 1 需要 1,670 ÷ 6,670）；
  2. 该放行**不能**变成万能挡箭牌 —— 实测踩坑：用上下文全部数字做两两运算时，
     几乎任何两位小数都能"碰巧"被算出来，防幻觉校验被架空。
"""
from __future__ import annotations

from src.rag import _is_derivable, check_numbers


def test_plain_numbers_must_come_from_context():
    _, bad = check_numbers("收入为1,234.56万元", "收入为1,234.56万元。")
    assert bad == []


def test_hallucinated_number_is_flagged():
    _, bad = check_numbers("收入为1,234.56万元", "收入为9,999.99万元。")
    assert "1,234.56" in bad


def test_derivable_ratio_allowed_when_operands_in_answer():
    """id 1：表格给 1,670 与 6,670，答案写出比例与依据的两个数。"""
    ans = "本次发行股数1,670万股，占发行后总股本6,670万股的比例为25.04%。"
    ctx = "| 发行股数 | 1,670万股 |\n| 发行后总股本 | 6,670万股 |"
    _, bad = check_numbers(ans, ctx, allow_direct=True)
    assert bad == []


def test_derivable_ratio_requires_operands_in_same_fragment():
    """分子分母分处两个片段时不得互算（见 _is_derivable 的收敛过程③）。"""
    ans = "本次发行股数占发行后总股本的比例为25.04%。"
    ctx = ("【片段1】发行股数 1,670 万股。\n\n---\n\n"
           "【片段2】发行后总股本 6,670 万股。")
    _, bad = check_numbers(ans, ctx, allow_direct=True)
    assert "25.04" in bad


def test_context_sized_number_pool_does_not_disable_guard():
    """回归：上下文数字多时，"可推导"规则不得被稀释成万能挡箭牌。

    实测踩坑：曾拿上下文**全部**数字做两两运算（129 个数字 ≈ 8000 对 × 7 种运算
    ≈ 5.6 万个候选值），几乎任何两位小数都能"碰巧"落在某个比值上，
    于是不在片段中的数字全被放行。现在只允许用**答案自身写出**的数字做运算。
    """
    ctx = "营业收入 1,918.21 万元、利润 625.12 万元、占比 206.85%、毛利率 31.42%。"
    # 888.88 不在片段中，且无法由答案中的数字算出 → 必须被标记
    ans = "该指标为 888.88。"
    _, bad = check_numbers(ans, ctx, allow_direct=True)
    assert "888.88" in bad


def test_is_derivable_within_fragment_pool():
    pool = [[1670.0, 6670.0]]
    assert _is_derivable(25.04, pool) is True        # 1670/6670*100
    assert _is_derivable(0.2504, pool) is True       # 直接比值
    assert _is_derivable(94.37, pool) is False
    assert _is_derivable(5.0, []) is False
    # 分母必须更大：66.7% 不能由 1670/6670 得出，也不该认任何"倒过来"的比值
    assert _is_derivable(250.4, pool) is False


def test_derivable_requires_same_fragment():
    """两个数分处不同片段时不得相互运算（否则候选空间膨胀、判别力归零）。"""
    pools = [[1670.0], [6670.0]]
    assert _is_derivable(25.04, pools) is False


def test_repeated_big_numbers_flagged():
    """列表型答案（≥3 个大数字）中同一数字重复出现，必是复制/漏抄错误。"""
    _, bad = check_numbers(
        "分别为6,464.51万元、18,780.67万元、6,464.51万元和4,627.14万元。",
        "分别为6,464.51万元、18,780.67万元、14,414.16万元和4,627.14万元。")
    assert "6,464.51" in bad


# ---------------------------------------------------------------------------
# 生成后校验：表格行覆盖 / 书名号名称一致性（工单3 实测失败模式）
# ---------------------------------------------------------------------------
def test_table_row_keys_extracts_first_column():
    from src.rag import table_row_keys
    parent = ("2、不存在控制关系的关联方\n"
              "| 项目 | 内容 |\n|---|---|\n"
              "| 企业名称 | 与本公司关系 |\n"
              "| 融冰投资 | 持有公司股份5%以上的股东 |\n"
              "| 力源贸易 | 同一实际控制人控制的企业 |\n")
    keys = table_row_keys(parent)
    assert "融冰投资" in keys and "力源贸易" in keys
    assert "企业名称" not in keys          # 列标题不是答案行
    assert "项目" not in keys


def test_table_row_keys_skips_pure_numeric_rows():
    from src.rag import table_row_keys
    parent = ("| 项目 | 金额 |\n|---|---|\n| 1 | 100 |\n| 2 | 200 |\n| 营业收入 | 300 |\n")
    assert table_row_keys(parent) == ["营业收入"]


def test_missing_table_rows_detected_for_list_question():
    from src.rag import check_table_rows
    blocks = [{"is_table": True,
               "parent_text": ("表：关联方\n| 项目 | 内容 |\n|---|---|\n"
                               "| 融冰投资 | 5%以上股东 |\n| 力源贸易 | 同一控制 |\n"
                               "| 普芯达 | 近亲属控制 |\n")}]
    ans = "持有公司股份5%以上的股东：融冰投资。"
    miss = check_table_rows(ans, blocks, "不存在控制关系的关联方企业有哪些？")
    assert "力源贸易" in miss and "普芯达" in miss


def test_missing_table_rows_not_checked_for_non_list_question():
    from src.rag import check_table_rows
    blocks = [{"is_table": True, "parent_text": "表：X\n| 项目 | 内容 |\n|---|---|\n| 甲 | 1 |\n"}]
    assert check_table_rows("甲的值为1。", blocks, "甲的值是多少？") == []


def test_quote_mismatch_detected():
    from src.rag import check_quotes
    ctx = "公司参与制定了《某视频指挥系统技术规范（1.0版）》。"
    assert check_quotes("即《某视频技术规范 1.0》。", ctx) == ["某视频技术规范 1.0"]
    assert check_quotes("即《某视频指挥系统技术规范（1.0版）》。", ctx) == []


def test_enumeration_sentence_chosen_by_question_overlap():
    """回归（id 33）：文档里两句结构相同的枚举句，必须挑与问题词面贴合的那句。"""
    from src.rag import complete_numeric_list
    ctx = ("报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、"
           "18,780.67万元和4,627.14万元，占主营业务收入比重分别为82.10%、97.31%、"
           "94.84%和94.34%。"
           "报告期内，公司视频指挥控制类产品的销售收入分别为6,172.41万元、"
           "14,198.94万元、18,687.75万元和4,508.18万元，占主营业务收入比例分别为"
           "78.39%、95.86%、94.37%和91.92%。")
    q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"
    ans = "报告期内，来自军用领域的收入占主营业务收入的比重分别是94.37%、95.86%、78.39%和94.34%。"
    fixed, patched = complete_numeric_list(ans, ctx, q)
    assert patched is True
    assert "82.10%" in fixed and "97.31%" in fixed
    assert "94.37%" not in fixed


def test_no_patch_when_answer_already_matches():
    from src.rag import complete_numeric_list
    ctx = "占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%。"
    ans = "比重分别为82.10%、97.31%、94.84%和94.34%。"
    _, patched = complete_numeric_list(ans, ctx, "比重分别是多少？")
    assert patched is False


def test_question_relevance_guard_blocks_offtopic_retry():
    """回归（id 1）：重试答案"数字更干净"但答到了别的段落，必须被拒绝。

    实测案例：问"本次发行股数…占发行后总股本的比例"，正确答案被一个
    讲注册资本/股权演变的答案顶掉——后者没有可疑数字，纯数字判据拦不住。
    """
    from src.rag import question_relevance
    q = "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"
    good = "本次发行股数1,670万股，占发行后总股本的比例为25.04%。"
    bad = ("2009年11月23日，武汉市商务局出具批复，批准力源有限整体变更为力源信息，"
           "注册资本5,000万元，其中赵马克持有本公司股份2,117.70万股。")
    assert question_relevance(q, good) > question_relevance(q, bad)
    assert question_relevance(q, bad) < question_relevance(q, good) * 0.7


def test_question_relevance_empty_question_is_permissive():
    from src.rag import question_relevance
    assert question_relevance("", "任意答案") == 1.0


def test_enumeration_patch_rejects_low_overlap_sentence():
    """回归（id 1）：与问题词面不贴合的枚举句不得触发替换。

    实测案例：问"本次发行股数…占发行后总股本的比例"，一段讲"注册资本5,000万元…
    赵马克持股42.35%"的枚举句因**问题里的公司名**占了大部分 4-gram 而蒙混过关，
    把正确回答整个替换掉。修复：算重叠度前先剔除公司名，并提高阈值 + 要求显著领先。
    """
    from src.rag import complete_numeric_list
    ctx = ("2009年11月23日，武汉市商务局出具批复，批准力源有限整体变更为力源信息，"
           "注册资本5,000万元，其中赵马克持有本公司股份2,117.70万股，"
           "占本次发行前公司总股本的42.35%。")
    q = "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"
    ans = "本次发行股数1,670万股，占发行后总股本的比例为25.04%。"
    fixed, patched = complete_numeric_list(ans, ctx, q)
    assert patched is False
    assert fixed == ans


# ---------------------------------------------------------------------------
# 沿革/演变类同构表：问"当前值"必须取日期最晚的那张（工单3 实测 id 543）
# ---------------------------------------------------------------------------
def _chrono_blocks():
    return [
        {"is_table": True, "table_caption": "（二）2017 年 7 月公司注册资本增加至 5,225 万元",
         "parent_text": "| 项目 | 内容 |\n|---|---|\n| 注册资本 | 5,225 万元 |"},
        {"is_table": True, "table_caption": "（五）2018年8月公司注册资本增加至5,520万元",
         "parent_text": "| 项目 | 内容 |\n|---|---|\n| 注册资本 | 5,520 万元 |"},
        {"is_table": True, "table_caption": "（一）2017 年 1 月公司注册资本增加至 5,016 万元",
         "parent_text": "| 项目 | 内容 |\n|---|---|\n| 注册资本 | 5,016 万元 |"},
    ]


def test_chronology_flags_non_latest_answer():
    from src.rag import check_chronology
    hints = check_chronology("2017年7月公司注册资本增加至5,225万元。",
                             _chrono_blocks(), "武汉兴图新科电子股份有限公司注册资本是多少？")
    assert hints and "5,520" in hints[0]


def test_chronology_passes_when_latest_used():
    from src.rag import check_chronology
    assert check_chronology("公司注册资本为5,520万元。", _chrono_blocks(),
                            "注册资本是多少？") == []


def test_chronology_skipped_when_question_names_a_date():
    """问题自带时间限定时不做纠正（问的就是历史上某一次）。"""
    from src.rag import check_chronology
    assert check_chronology("2017年1月增加至5,016万元。", _chrono_blocks(),
                            "2017年1月公司注册资本是多少？") == []


def test_chronology_ignores_single_table():
    from src.rag import check_chronology
    blocks = [_chrono_blocks()[1]]
    assert check_chronology("注册资本为1,000万元。", blocks, "注册资本是多少？") == []


def test_caption_template_groups_across_whitespace_and_index():
    from src.rag import _caption_template
    a = _caption_template("（二）2017 年 7 月公司注册资本增加至 5,225 万元")
    b = _caption_template("（五）2018年8月公司注册资本增加至5,520万元")
    assert a == b


def test_caption_value_numbers_excludes_date_parts():
    from src.rag import _caption_value_numbers
    assert _caption_value_numbers("（五）2018年8月公司注册资本增加至5,520万元") == ["5,520"]
