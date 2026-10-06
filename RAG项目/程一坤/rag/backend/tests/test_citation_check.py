"""测试引用校验模块。

任务书 4-B 要求：
- 校验回答里的 [n] 是否都在法源清单范围内
- 编号是否连续
- 是否有"无引用结论"
- 不合规返回问题清单
"""
import pytest
from app.chat.citation_check import (
    CitationError,
    _is_law_name_candidate,
    check_citations,
)


def test_valid_citations():
    """测试合法引用：编号连续，都在范围内"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月[2]。"
    total_sources = 2

    # 应该不抛出异常
    check_citations(answer, total_sources)


def test_citation_out_of_range():
    """测试引用越界：编号超出法源清单范围"""
    answer = "根据劳动合同法[1]，试用期不得超过六个月[3]。"
    total_sources = 2

    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, total_sources)

    error = exc_info.value
    assert "越界" in str(error) or "范围" in str(error)
    assert "[3]" in str(error)


def test_citation_with_skipped_numbers_allowed():
    """跳号允许（v2）：模型按需引用，不必把清单法源全部引一遍。

    真实案例：好答案引了 [1,2,3,4,5,7,8] 因跳过 [6] 被整段拦掉，
    迫使模型硬凑不相关引用或被误杀，故放宽为"在范围内即可"。
    """
    answer = "根据劳动合同法[1]，试用期不得超过六个月[3]。"
    total_sources = 3

    # 不应抛出异常
    check_citations(answer, total_sources)


def test_citation_partial_subset_allowed():
    """只引用部分法源（如 [1,2,3,4,5,7,8] 跳过 6）同样放行。"""
    answer = (
        "拖欠工资可以投诉[2]，也可以申请支付令[1]，"
        "仲裁时效一般为一年[5]，逾期不支付可加付赔偿金[7][8]。"
    )
    check_citations(answer, 8)


def test_conclusion_without_citation():
    """测试无引用结论：实质性结论后没有引用编号"""
    answer = "试用期不得超过六个月。根据劳动合同法[1]，还有其他规定。"
    total_sources = 1

    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, total_sources)

    error = exc_info.value
    assert "无引用" in str(error) or "缺少引用" in str(error)


def test_citation_undefined_source():
    """测试引用清单外法规名：回答中出现清单里没有的法规"""
    answer = "根据《劳动法》[1]和《民法典》，试用期不得超过六个月。"
    total_sources = 1
    available_sources = ["中华人民共和国劳动合同法"]

    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, total_sources, available_sources)

    error = exc_info.value
    assert "民法典" in str(error) or "清单外" in str(error)


def test_empty_answer():
    """测试空回答"""
    answer = ""
    total_sources = 1

    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, total_sources)

    error = exc_info.value
    assert "空" in str(error) or "无内容" in str(error)


def test_answer_without_any_citation():
    """测试完全没有引用的回答"""
    answer = "试用期一般是三到六个月，具体要看合同约定。"
    total_sources = 5

    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, total_sources)

    error = exc_info.value
    assert "引用" in str(error)


def test_valid_answer_with_disclaimer():
    """测试合法回答：带免责声明"""
    answer = """根据劳动合同法[1]，试用期不得超过六个月[2]。

内容仅供法律信息参考，不能替代律师出具的正式法律意见。"""
    total_sources = 2

    # 免责声明不应影响校验
    check_citations(answer, total_sources)


def test_multiple_citations_in_one_sentence():
    """测试一句话多个引用"""
    answer = "根据劳动合同法[1]和实施条例[2]，试用期有明确规定[3]。"
    total_sources = 3

    check_citations(answer, total_sources)


# ==================== 法规名归一化（简称/全称比对）====================


def test_abbreviated_law_name_not_rejected():
    """a) 回答写简称《劳动合同法》、清单是全称 → 不得误判"清单外法规"。

    修复前：mentioned_laws - set(available_sources) 是精确字符串比对，
    《劳动合同法》 != 《中华人民共和国劳动合同法》→ 误拦。
    """
    answer = "根据《劳动合同法》第三十九条[1]，用人单位可以解除劳动合同。"
    check_citations(answer, 1, ["中华人民共和国劳动合同法"])


def test_unknown_law_still_rejected_after_normalization():
    """b) 归一化后仍不匹配的真问题必须照旧报错。"""
    answer = "根据《劳动合同法》[1]和《民法典》第五条，合同应当遵守。"
    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, 1, ["中华人民共和国劳动合同法"])
    error = str(exc_info.value)
    assert "民法典" in error
    # 归一化只用于比对：报错信息仍展示用户可读的原始名称
    assert "《民法典》" in error


def test_implementation_regulation_not_confused_with_law():
    """c)《劳动合同法实施条例》不得与《劳动合同法》混淆（禁止包含关系匹配）。"""
    answer = "根据《劳动合同法实施条例》第六条[1]，视为无固定期限。"
    # 清单里只有劳动合同法、没有实施条例 → 实施条例是清单外，必须报错
    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, 1, ["中华人民共和国劳动合同法"])
    assert "实施条例" in str(exc_info.value)

    # 反向：清单里有实施条例 → 通过，且不因"劳动合同法是其前缀"而误放别的
    check_citations(answer, 1, ["中华人民共和国劳动合同法实施条例"])


def test_labor_law_and_labor_contract_law_are_different():
    """《劳动法》与《劳动合同法》是两部不同的法律，归一化后不得混同。"""
    answer = "根据《劳动法》[1]，工作时间不得超过八小时。"
    with pytest.raises(CitationError):
        # 清单里只有劳动合同法 → 《劳动法》仍是清单外
        check_citations(answer, 1, ["中华人民共和国劳动合同法"])


def test_normalization_handles_fullwidth_and_spaces():
    """全角/半角、空白差异也应归一化（比对层面）。"""
    answer = "根据《 劳动合同法 》[1]，用人单位应当订立书面合同。"
    # 清单用全角空格写法
    check_citations(answer, 1, ["中华人民共和国　劳动合同法"])


# ==================== 批次 30：文书名 vs 法规名（后缀白名单） ====================
# 背景：回答里的书名号不止用于法规，还用于文书名（《道路交通事故认定书》）。文书名本来就不会
# 出现在法源清单里，全量比对必然把它误报成"清单外法规"（真实误报见 20 题评测的 direct-013）。
# 现按 LAW_NAME_SUFFIXES 后缀白名单先筛出"法规名候选"，只有候选才参与清单比对。


def test_document_title_not_reported_as_unknown_law():
    """direct-013 场景：《道路交通事故认定书》是文书名，不得报"清单外法规"。

    修复前：该文书名被 《([^》]+)》 捞出、与清单比对无果 → 误抛 UnknownLawCitationError，
    回答被追加"提到清单外法规"警示。
    """
    answer = (
        "先看《道路交通事故认定书》里的事故责任划分[1]。"
        "单位应当在事故伤害发生之日起30日内提出工伤认定申请[2]。"
    )
    # 不应抛出任何异常：文书名不进比对，两条引用也都在范围内
    check_citations(answer, 2, ["工伤保险条例"])


def test_common_document_titles_not_reported_as_unknown_law():
    """各类文书/材料名（…书 / …证 / …表 / …单 / …通知书）都不参与清单外法规比对。"""
    for title in (
        "工伤认定决定书",
        "解除劳动合同通知书",
        "劳动能力鉴定结论书",
        "工伤认定申请表",
        "工资支付清单",
        "参保缴费证明",
    ):
        answer = f"你可以先准备《{title}》，再对照《工伤保险条例》申请[1]。"
        check_citations(answer, 1, ["工伤保险条例"])


def test_real_law_names_are_still_law_candidates():
    """白名单必须覆盖库里 11 部法规的真实名称形态（含带序号括号的司法解释）。

    《民法典》单列"法典"后缀的原因：末字是"典"不是"法"，只收"法"会把它静默漏判。
    """
    for name in (
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "劳动法",
        "工伤保险条例",
        "工资支付暂行规定",
        "女职工劳动保护特别规定",
        "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）",
        "民法典",
    ):
        assert _is_law_name_candidate(name), f"{name} 应被识别为法规名候选"


def test_document_titles_are_not_law_candidates():
    """文书/材料名不得被识别成法规名候选。"""
    for name in ("道路交通事故认定书", "解除劳动合同通知书", "工资表", "身份证"):
        assert not _is_law_name_candidate(name), f"{name} 不该被识别为法规名候选"


def test_unknown_law_with_serial_suffix_still_reported():
    """带序号括号的真法规仍要报"清单外"——结尾是"）"不代表它是文书名。

    为什么要剥序号再判后缀：库里就有"…适用法律问题的解释（一）"这个正式名称，
    不剥会被当成文书名静默跳过，真法规反而漏判。
    """
    answer = "按《最高人民法院关于审理劳动争议案件适用法律问题的解释（一）》[1]，仲裁时效为一年。"
    with pytest.raises(CitationError) as exc_info:
        check_citations(answer, 1, ["中华人民共和国劳动争议调解仲裁法"])
    assert "解释" in str(exc_info.value)
