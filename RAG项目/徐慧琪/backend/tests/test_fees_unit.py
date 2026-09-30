# 编排：单位闸门（用户裁决追加）。
# 只认字面数字的回查有个跨任务空洞：片段写「1万元至10万元」时，单位正确的
# (10000, 100000) 被拒、单位盲的 (1, 10) 反而通过，而结果里没有单位字段 ——
# 用户看到的「1 ~ 10」比语料少了 4 个数量级。故 unit 随区间一起产出，
# 并与数字受**同一道闸门**：给了就必须能在片段里找到出处。
#
# 本批补上闸门的另一半：**缺省不是无条件的** —— 命中数字紧邻量级词（万/千/百/亿）
# 而模型不报单位时，片段只证明了「1 万」这个数，没证明「1」，同样判拒。
# 只堵「报错单位」而不堵「不报单位」时，后者是同一风险的 fail-open 出口
# （终审实测：真实语料「（1）每件收费：1 万元－50 万元。」→ 渲染成「1 ~ 50」）。
from app.recommend.fees import _unit_has_source, estimate
from tests._fee_fixtures import (FLAT_HIT, HIT, gen, gen_basis, gen_unit, no_log,
                                 search)

# 真实语料里的原句（data/raw/fee/北京智深律师事务所收费标准.txt）：数字与
# 量级词之间被 PDF 文本层塞了一个空格，是「紧邻」在语料里的主体形态
CORPUS_LINE = "（1）每件收费：1 万元－50 万元。"
CORPUS_HIT = {"text": CORPUS_LINE, "source_doc": "北京智深律师事务所收费标准",
              "source_no": "3、", "status": "现行有效"}


def test_unit_with_source_in_the_snippet_is_passed_through():
    """模型照抄片段里的单位（SNIPPET 写的是「万元」）→ 原样透传，渲染方据此
    印出「1 ~ 10 万元」。unit 缺失导致的静默错量级正是这个字段存在的理由。"""
    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen_unit(1, 10, "万元"), log_fn=no_log)
    assert result["status"] == "ok"
    assert result["unit"] == "万元"


def test_unit_without_source_in_the_snippet_is_rejected_and_logged():
    """片段写「万元」而模型说「千元」：数字两端都有出处，但单位是编的 ——
    与编造数字同样危险（静默错一个数量级），必须判拒并留痕。留痕要记下
    **那两个通过了回查的数字**：事后才看得出拒的是单位而不是区间。"""
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen_unit(1, 10, "千元"), log_fn=log)
    assert result["status"] == "rejected"
    assert result["low"] is None and result["high"] is None
    assert result["unit"] is None
    assert calls[-1]["status"] == "rejected"
    assert calls[-1]["range_low"] == 1 and calls[-1]["range_high"] == 10


def test_unit_missing_on_a_magnitude_free_snippet_is_still_ok():
    """模型不给单位在**无量级词**的片段上是允许的（兼容分支）：按件、按百分比的
    口径本来就没有量级可丢，区间照常产出、status 仍是 ok，unit 为 None。

    夹具必须无量级词：原用例拿 HIT（写的是「1万元」）钉 ok，钉的其实是错的行为
    —— 那种片段缺 unit 就是静默错 10000 倍，本批已改判拒（下一条用例）。"""
    result = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
                      generate_fn=gen(1, 10), log_fn=no_log)
    assert result["status"] == "ok"
    assert result["unit"] is None


def test_unit_missing_on_a_magnitude_snippet_is_rejected_and_logged():
    """缺省闸门：片段写「1万元」「10万元」而模型不报单位 → 判拒（与『报了
    没出处的单位』同一条出口）。留痕仍要记下那两个通过了回查的数字，事后
    才分得清「拒的是区间」还是「拒的是量级缺省」——reason 里写明来由。"""
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen(1, 10), log_fn=log)
    assert result["status"] == "rejected"
    assert result["low"] is None and result["high"] is None
    assert "量级词" in result["reason"]
    # 与「单位报了但没出处」分得开：那条的 reason 说的是单位无法指回片段
    assert calls[-1]["status"] == "rejected"
    assert calls[-1]["range_low"] == 1 and calls[-1]["range_high"] == 10


def test_either_end_being_magnitude_attached_is_enough_to_reject():
    """只钉「两端都紧邻量级词」会漏掉单端形态：任一端的数带上量级，整个区间就
    无法按无单位口径印出来（印 low 掉一个量级、印 high 掉另一个）。两个方向各来
    一条 —— 只查 low 或只查 high 的实现会各漏一条（与回查的镜像用例同款教训）。"""
    low_attached = {"text": "标的额超过1万元的按比例；每件另收50元",
                    "source_doc": "X", "source_no": "第一条",
                    "status": "现行有效"}
    refused = estimate("民间借贷纠纷", search_fn=search([low_attached]),
                       generate_fn=gen(1, 50), log_fn=no_log)
    assert refused["status"] == "rejected" and "量级词" in refused["reason"]
    high_attached = {"text": "每件收取5元；标的额超过10万元的部分按比例",
                     "source_doc": "X", "source_no": "第一条",
                     "status": "现行有效"}
    refused = estimate("民间借贷纠纷", search_fn=search([high_attached]),
                       generate_fn=gen(5, 10), log_fn=no_log)
    assert refused["status"] == "rejected" and "量级词" in refused["reason"]


def test_real_corpus_space_form_is_rejected():
    """终审实测的原始标本：真实语料「（1）每件收费：1 万元－50 万元。」→
    1 与 50 都有字面出处、回查放行，缺 unit 时渲染成「1 ~ 50」，与语料差
    10000 倍。数字与量级词之间的空格是 PDF 文本层留下的（语料里 79 处），
    不跳过空白就会在这道闸门的主体形态上失效。"""
    result = estimate("民间借贷纠纷", search_fn=search([CORPUS_HIT]),
                      generate_fn=gen(1, 50), log_fn=no_log)
    assert result["status"] == "rejected" and "量级词" in result["reason"]
    # 同一片段、同一个数，报了「万元」就照过 —— 证明拒的是缺省而不是数本身
    ok = estimate("民间借贷纠纷", search_fn=search([CORPUS_HIT]),
                  generate_fn=gen_unit(1, 50, "万元"), log_fn=no_log)
    assert ok["status"] == "ok" and ok["unit"] == "万元"


def test_thousands_separator_with_magnitude_is_rejected_when_unit_missing():
    """千分位写法「1,000 万元」同样要判拒：分词层把它当成**一个**数（1,000），
    量级判定必须走同一层的词元 —— 直接按字面串搜「1000 万」搜不到，闸门就会
    在这种写法上静默失效（而语料口径明确为 fee_corpus 预置了千分位形态）。"""
    hit = {"text": "标的额超过1,000万元的，按比例交纳", "source_doc": "X",
           "source_no": "第一条", "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([hit]),
                      generate_fn=gen(1000, 1000), log_fn=no_log)
    assert result["status"] == "rejected" and "量级词" in result["reason"]


def test_blank_unit_counts_as_absent_and_non_string_unit_is_rejected():
    """空串是「模型没给单位」的另一种写法（归一成 None）：在无量级词的片段上照
    常 ok；在量级片段上与缺省同判拒（否则空串就成了绕过缺省闸门的后门）。
    非字符串（数字/对象）则是**坏上报**：折不出可用于比对的单位，与编造单位
    同样判拒 —— 顺带挡住 `5 in 片段` 抛 TypeError 逃成故障那条路。"""
    blank = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
                     generate_fn=gen_unit(1, 10, ""), log_fn=no_log)
    assert blank["status"] == "ok" and blank["unit"] is None
    blank_magnitude = estimate("民间借贷纠纷", search_fn=search([HIT]),
                               generate_fn=gen_unit(1, 10, ""), log_fn=no_log)
    assert blank_magnitude["status"] == "rejected"
    bad = estimate("民间借贷纠纷", search_fn=search([HIT]),
                   generate_fn=gen_unit(1, 10, 5), log_fn=no_log)
    assert bad["status"] == "rejected" and bad["unit"] is None


# ---- 编排：单位必须是**独立**出现（子串洞的收口） ----
# `unit in 片段` 有个洞：「元」是「万元」的后半截，于是模型报「元」而语料写
# 「万元」时照样通过 —— 那正好让 unit 字段在它存在的主要理由（量级不得静默
# 错档）上失效。收口口径：至少有一处出现，其前一个字符不是汉字。


def test_unit_that_is_only_a_fragment_of_a_bigger_unit_is_rejected():
    """片段**只有**万元形态（数字两端仍有出处）：万元过（前是 1/0），
    元 拒（前是 万）—— 判拒只能来自单位，正是这条用例要单独钉住的事。"""
    only_wan = {"text": "标的额1万元至10万元的部分，按比例交纳",
                "source_doc": "诉讼费用交纳办法", "source_no": "第一条",
                "status": "现行有效"}
    through = estimate("民间借贷纠纷", search_fn=search([only_wan]),
                       generate_fn=gen_unit(1, 10, "万元"), log_fn=no_log)
    assert through["status"] == "ok" and through["unit"] == "万元"
    refused = estimate("民间借贷纠纷", search_fn=search([only_wan]),
                       generate_fn=gen_unit(1, 10, "元"), log_fn=no_log)
    assert refused["status"] == "rejected" and refused["unit"] is None


def test_standalone_unit_near_the_reported_number_is_still_accepted():
    """收紧不得把合法形态一起拒掉：上报的数就是带「元」的那个数时**必须照过** ——
    即使同一个窗口里还有「元」的坏出现（「1万元」的后半截）。这条钉住判定是
    「存在一处独立出现」，而不是「所有出现都得独立」。

    夹具从 1/10 换成 50/50：1 与 10 在 SNIPPET 里都出自「1万元」「10万元」，
    收紧到「单位必须贴着上报的数」之后，那正是**跨数借单位**的形态（下一条用例
    要拒的），故保留要义、换掉夹具。"""
    near_bad = {"text": "每件交纳50元；不足1万元的按比例交纳", "source_doc": "X",
                "source_no": "第一条", "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([near_bad]),
                      generate_fn=gen_unit(50, 50, "元"), log_fn=no_log)
    assert result["status"] == "ok" and result["unit"] == "元"


def test_percent_and_composite_units_are_not_over_rejected():
    """带符号的单位一并钉住：`%`（片段「按照2.5%交纳」，前是 5）与复合单位
    「元/件」（片段「50元/件」，前是 0）都要能过 —— 收紧的是「更长单位词的
    后半截」，不是把非纯汉字单位一起拒掉。"""
    percent = {"text": "按照2.5%交纳", "source_doc": "X", "source_no": "第一条",
               "status": "现行有效"}
    ok_percent = estimate("民间借贷纠纷", search_fn=search([percent]),
                          generate_fn=gen_unit(2.5, 2.5, "%"), log_fn=no_log)
    assert ok_percent["status"] == "ok" and ok_percent["unit"] == "%"
    composite = {"text": "每件交纳50元/件", "source_doc": "X",
                 "source_no": "第一条", "status": "现行有效"}
    ok_composite = estimate("民间借贷纠纷", search_fn=search([composite]),
                            generate_fn=gen_unit(50, 50, "元/件"), log_fn=no_log)
    assert ok_composite["status"] == "ok" and ok_composite["unit"] == "元/件"


# ---- 编排：单位必须贴着**上报的那个数**（终审复核的误放行，本批收口） ----
# 纯存在性检查（「片段里出现过即放行」）看不出单位属于谁：片段「标的额 100 万元
# 以上的，每件收费 50 元」里 100 出自「100 万元」、「元」来自「50 元」，两关各自
# 都说得通，合起来却把 100 当成无单位的 100，渲染成「100 ~ 100元」（差 10000 倍）。


def test_unit_borrowed_from_another_number_is_rejected():
    """终审复核实测的原句 + 上报 (100, 100, "元")：回查认「100 万元」里的数、单位认
    「50 元」里的元 —— 两个数各自的出处都在，错的是**它们不是同一个数**。"""
    hit = {"text": "标的额 100 万元以上的，每件收费 50 元", "source_doc": "X",
           "source_no": "第一条", "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([hit]),
                      generate_fn=gen_unit(100, 100, "元"), log_fn=no_log)
    assert result["status"] == "rejected" and "单位" in result["reason"]
    # 反方向对照：同一个数字，单位就贴着它写（「100 万元」）→ 照过。缺了这条，
    # 「unit 一律判拒」的实现也能让上面那条绿
    ok = estimate("民间借贷纠纷", search_fn=search([hit]),
                  generate_fn=gen_unit(100, 100, "万元"), log_fn=no_log)
    assert ok["status"] == "ok" and ok["unit"] == "万元"


def test_unit_far_from_the_number_on_the_same_line_is_rejected():
    """窗口是「邻近」而不是「整行」：同一个数字与它后面隔着一整句的「元」不算出处
    （本句间隔 21 个字符）。把窗口写成整行时本用例转红 —— 那条坏法仍能放行
    「标的额 100 万元…最低每件 3000 元」配 (100, 100, "元") 的跨数借单位。"""
    hit = {"text": "标的额 100 万元以上按比例收取，最低每件 3000 元", "source_doc": "X",
           "source_no": "第一条", "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([hit]),
                      generate_fn=gen_unit(100, 100, "元"), log_fn=no_log)
    assert result["status"] == "rejected" and "单位" in result["reason"]


def test_unit_written_once_at_one_end_is_enough():
    """窗口不能小到把「区间只写一次单位」的语料形态拒掉 —— 判定必须是**两端任一**
    邻近即算。两个方向各来一条，与 _magnitude_attached 的镜像用例同款教训：
    只查 low 或只查 high 的实现会各漏一条。

    夹具取真语料原句（昌民「1.一阶段：每件收费20000--100000元。」）：下界 20000
    到单位隔 8 个字符（「--100000」），只有**上界** 100000 够得着。首版夹具用的是
    「普通律师 800~1000 元/小时」——那里的间隔恰好是 6，正好落在窗口边界内，
    于是「只查 low 端」的实现照样全绿（忠实变异实测），换掉才钉得住。"""
    wide = {"text": "1.一阶段：每件收费20000--100000元。", "source_doc": "昌民",
            "source_no": "1.", "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([wide]),
                      generate_fn=gen_unit(20000, 100000, "元"), log_fn=no_log)
    assert result["status"] == "ok" and result["unit"] == "元"
    # 镜像方向：单位只贴着**下界**（上界之后没有单位）→ 同样照过
    low_end = {"text": "按小时计费 800 元起，全案总价 30000 以上另议", "source_doc": "X",
               "source_no": "第一条", "status": "现行有效"}
    mirrored = estimate("民间借贷纠纷", search_fn=search([low_end]),
                        generate_fn=gen_unit(800, 30000, "元"), log_fn=no_log)
    assert mirrored["status"] == "ok" and mirrored["unit"] == "元"


# ---- 编排：计价基础（用户 2026-09-29 裁决，满足 FR-9.4「含计费方式说明」） ----
# 真跑实证 N10：语料「四、计时收费标准：1000元—8000元/有效工作小时。」渲染成
# 「1000 ~ 8000元」，用户会读成一次性收费。计价基础由模型上报、按**宽松存在性**
# 回查 —— 与 unit 的严格 lookbehind **有意不同**：它不承载量级，「小时」出现在
# 「有效工作小时」里、「件」出现在「每件」里都是合法出处，严格匹配反而会拒掉
# 正确上报。上报了但片段里根本没这个词 → 与编造单位同罪。
TIMED_HIT = {"text": "四、计时收费标准：1000元—8000元/有效工作小时。",
             "source_doc": "北京融理律师事务所收费标准", "source_no": "四、",
             "status": "现行有效"}


def test_charge_basis_is_reported_through_with_a_loose_existence_check():
    """「小时」这种短词必须能过 —— 语料里写作「有效工作小时」。第一条断言是**两把
    尺子不同源**的直接证据：unit 那把严格尺子在这里返回 False，若计价基础复用它，
    正确的上报会被拒（这正是终审点名「有意不同」的形态）。"""
    assert _unit_has_source("小时", TIMED_HIT["text"]) is False
    result = estimate("民间借贷纠纷", search_fn=search([TIMED_HIT]),
                      generate_fn=gen_basis(1000, 8000, "小时"), log_fn=no_log)
    assert result["status"] == "ok" and result["charge_basis"] == "小时"


def test_charge_basis_verbatim_phrase_is_accepted_too():
    """提示词要求「逐字照抄片段里的那个词」，故整词形态（有效工作小时）也必须过。"""
    result = estimate("民间借贷纠纷", search_fn=search([TIMED_HIT]),
                      generate_fn=gen_basis(1000, 8000, "有效工作小时"), log_fn=no_log)
    assert result["status"] == "ok" and result["charge_basis"] == "有效工作小时"


def test_charge_basis_missing_or_blank_is_not_rejected():
    """缺省不判拒（与 unit 同口径）：语料本来就有不写计价基础的口径（纯按标的额
    比例），拒掉它们会把合法区间一起挡下。空串是「模型没给」的另一种写法。"""
    for value in (None, "", "   "):
        produced = gen_basis(1, 10, value)
        result = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
                          generate_fn=produced, log_fn=no_log)
        assert result["status"] == "ok" and result["charge_basis"] is None, value


def test_charge_basis_without_source_is_rejected():
    """片段里根本没提过的计价基础 = 编造，与编造单位同罪；reason 要与单位那条分得开
    （留痕表里没有这一列，事后靠 reason 分辨堵在哪一关）。"""
    result = estimate("民间借贷纠纷", search_fn=search([TIMED_HIT]),
                      generate_fn=gen_basis(1000, 8000, "件"), log_fn=no_log)
    assert result["status"] == "rejected" and "计价基础" in result["reason"]
    # 非字符串上报同样判拒（折不出可用于比对的词，与 unit 的非字符串分支同形）
    bad = estimate("民间借贷纠纷", search_fn=search([TIMED_HIT]),
                   generate_fn=gen_basis(1000, 8000, 3), log_fn=no_log)
    assert bad["status"] == "rejected" and "计价基础" in bad["reason"]
