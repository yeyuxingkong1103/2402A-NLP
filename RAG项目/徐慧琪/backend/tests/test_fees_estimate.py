# 费用编排（fees.estimate）的主干分支：命中 → 生成 → 回查 → 判拒/降级，
# 外加两道本批补上的闸门 —— 效力（设计 §七：非现行片段不采用）与
# 缺单位的量级闸门（在 test_fees_unit.py，本文件按主题只放不带单位的那几条）。
#
# 编排的测试全部用注入的假检索/假生成：这里要验的是**分支** —— 有没有命中、
# 回查过不过、故障有没有被误当成「没有依据」。真检索与真模型由 CLI 冒烟覆盖。
#
# 本文件与 test_fees.py（回查本体）、test_fees_unit.py（单位闸门）共同覆盖
# 原先的单文件；夹具见 _fee_fixtures.py
import pytest

from app.recommend.fees import estimate
from tests._fee_fixtures import (FLAT_HIT, FLAT_SNIPPET, HIT, gen, no_log, search)


def test_ok_when_hit_and_range_verifies():
    """命中且两端有出处 → ok，且「依据」指回那段片段。

    夹具用**无量级词**的 FLAT_HIT：HIT 写的是「1万元」，缺 unit 时该被缺省闸门
    拒掉（那个方向归 test_fees_unit.py），混在这一条里会让本用例的失败原因
    指向闸门而不是它要验的 ok 支。"""
    result = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
                      generate_fn=gen(1, 10), log_fn=no_log)
    assert result["status"] == "ok"
    assert result["low"] == 1 and result["high"] == 10
    assert result["basis"] == FLAT_SNIPPET
    assert result["source_doc"] == "诉讼费用交纳办法"
    assert result["source_no"] == "第十三条"


def test_no_corpus_when_search_returns_nothing():
    """未命中是**设计行为**，不是降级：如实说「暂无费用口径依据」，不猜数。"""
    result = estimate("民间借贷纠纷", search_fn=search([]),
                      generate_fn=gen(1, 10), log_fn=no_log)
    assert result["status"] == "no_corpus"
    assert result["low"] is None and result["high"] is None
    assert result["reason"]


def test_rejected_when_range_fails_verification():
    """回查不过 = 模型编了数 → 降级成「暂无依据」，且留痕要记下被拒的区间。"""
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen(1, 30), log_fn=log)
    assert result["status"] == "rejected"
    assert result["low"] is None and result["high"] is None
    assert calls and calls[-1]["status"] == "rejected"
    assert calls[-1]["range_low"] == 1 and calls[-1]["range_high"] == 30
    # 判拒时「依据」类字段必须是 None，而不是把片段原文带上：「依据」指**被采用**
    # 的片段，判拒的区间没有依据。片段仍可由留痕的 snippet_id 追溯，AC-21 不因此
    # 丢失；渲染方若按字段渲染，也不会把没被采用的片段显示成依据（审查实测：改成
    # 片段原文时 28 条全绿，故补此钉）
    assert result["basis"] is None
    assert result["source_doc"] is None and result["source_no"] is None


def test_rejected_when_generator_returns_missing_bounds():
    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen(None, None), log_fn=no_log)
    assert result["status"] == "rejected"


def test_search_failure_propagates_not_downgraded():
    """Milvus 挂了绝不能被答成「暂无费用口径依据」—— 那是 ③a 明令的红线。"""
    def boom(cause):
        raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError):
        estimate("民间借贷纠纷", search_fn=boom, generate_fn=gen(1, 10),
                 log_fn=no_log)


def test_generator_failure_propagates_not_downgraded():
    def boom(snippet, cause):
        raise RuntimeError("llm down")

    with pytest.raises(RuntimeError):
        estimate("民间借贷纠纷", search_fn=search([HIT]),
                 generate_fn=boom, log_fn=no_log)


def test_ok_path_logs_the_range():
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
             generate_fn=gen(1, 10), log_fn=log)
    assert calls[-1]["status"] == "ok"
    assert calls[-1]["range_low"] == 1 and calls[-1]["range_high"] == 10
    assert calls[-1]["cause"] == "民间借贷纠纷"


def test_generator_receives_first_hit_text_and_cause():
    """生成器必须拿到**命中片段的原文**与案由，而不是片段字典或别的字段。
    只喂一个命中时，传错对象（如整条 hits[0]）、取错下标（hits[-1]）都测不出来，
    故这里喂两条命中，再按「结果指回哪段」钉住取的是第一条。"""
    seen = []

    def gen_fn(snippet, cause):
        seen.append((snippet, cause))
        return {"low": 1, "high": 10}

    other = {"text": "其他条款：每件交纳300元",
             "source_doc": "别的文件", "source_no": "第九条",
             "status": "现行有效"}
    result = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT, other]),
                      generate_fn=gen_fn, log_fn=no_log)
    assert seen == [(FLAT_SNIPPET, "民间借贷纠纷")]
    assert result["basis"] == FLAT_SNIPPET and result["source_no"] == "第十三条"


def test_no_corpus_does_not_ask_the_generator():
    """没依据时连模型都不该问：「不猜数」的第一步是「不去猜」。先生成后判
    hits 的实现能让 brief 那条 no_corpus 用例照样绿（生成器返回什么都无所谓），
    只有在生成器上装探针才拦得住。"""
    def gen_boom(snippet, cause):
        raise AssertionError("无命中时不应该调用生成器")

    result = estimate("民间借贷纠纷", search_fn=search([]),
                      generate_fn=gen_boom, log_fn=no_log)
    assert result["status"] == "no_corpus"


def test_string_bounds_from_the_model_are_coerced_to_numbers():
    """模型常把 JSON 数字写成字符串（"low": "1"）。回查只收「数值 | None」，
    而两个字符串端点传进去会走**字典序**比较并逐字命中，静默放行成 ok ——
    结果与留痕里灌进字符串。不折算时本用例在 `result["low"] == 1` 上转红。"""
    result = estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]),
                      generate_fn=gen("1", "10"), log_fn=no_log)
    assert result["status"] == "ok"
    assert result["low"] == 1 and result["high"] == 10


def test_non_numeric_or_boolean_bounds_are_rejected_not_raised():
    """非数值端点（「约五十元」）与布尔（JSON 的 true/false）都是**降级**方向：
    回查无从比对这些值，编排层必须自己判拒（留痕不归本用例，见
    test_rejected_when_range_fails_verification）。不折算时这两个形态一个抛
    TypeError（int 与 str 比大小）被调用方记成「服务不可用」—— 降级冒充故障；
    另一个被回查静默放行 —— 把「是/否」当成 1/0 的金额。两个方向都要钉住。"""
    for low, high in ((1, "约五十元"), (True, 10)):
        result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                          generate_fn=gen(low, high), log_fn=no_log)
        assert result["status"] == "rejected", (low, high)


def test_log_carries_traceability_fields_from_the_generator():
    """AC-21 的「可追溯」要能指到具体某次生成：模型名与 request_id 必须从生成
    结果透传进留痕，片段 id 取命中项的 source_no。这条字段链断掉任何一环，
    事后都说不清某个区间是哪个模型、哪次请求给的。"""
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    def gen_fn(snippet, cause):
        return {"low": 1, "high": 10,
                "model": "deepseek-chat", "request_id": "req-7"}

    estimate("民间借贷纠纷", search_fn=search([FLAT_HIT]), generate_fn=gen_fn,
             log_fn=log)
    assert calls[-1]["model"] == "deepseek-chat"
    assert calls[-1]["request_id"] == "req-7"
    assert calls[-1]["snippet_id"] == "第十三条"


def test_shape_is_constant_with_unit_key_in_every_branch():
    """形状恒定：**四个**分支都带同一组 9 键（unit 与 charge_basis 在 ok 分支带值或
    None，其余 None）—— 渲染方不必先判断键在不在。任一分支漏键或多键，本用例转红。

    四个分支 = ok / no_corpus / 区间判拒 / **效力判拒**。第四支此前不在 cases 里
    （它与区间判拒共用同一个 _rejected 出口，所以「共用」这件事本身没有被钉住）：
    哪天有一个出口改用别的形状（如不带那批 None 键的裸字典），只有到了渲染层才炸。
    """
    expected = {"status", "low", "high", "unit", "charge_basis", "basis",
                "source_doc", "source_no", "reason"}
    stale = dict(FLAT_HIT, status="已废止")
    cases = [(search([FLAT_HIT]), gen(1, 10)), (search([]), gen(1, 10)),
             (search([FLAT_HIT]), gen(1, 30)), (search([stale]), gen(1, 10))]
    for search_fn, generate_fn in cases:
        result = estimate("民间借贷纠纷", search_fn=search_fn,
                          generate_fn=generate_fn, log_fn=no_log)
        assert set(result) == expected, result["status"]
        assert result["unit"] is None and result["charge_basis"] is None


def test_out_of_range_bound_is_rejected_not_raised():
    """端点是超范围大整数（10**400）时，回查折算字面量的 float() 会抛
    OverflowError —— 「模型报了个荒谬的数」是**降级**，不能让它逃出去被上层
    记成「服务不可用」（正是折算层存在的理由的另一面）。故兜在 _as_number 里，
    这里断言的是判拒而不是抛异常。"""
    result = estimate("民间借贷纠纷", search_fn=search([HIT]),
                      generate_fn=gen(1, 10 ** 400), log_fn=no_log)
    assert result["status"] == "rejected"
    assert result["low"] is None and result["high"] is None


# ---- 效力闸门（设计 §七 漏掉的那一行：命中片段非现行 → 不采用） ----
# 检索侧已带 expr 过滤（tools/ingest_fee.search 的主防线），这里是编排层的兜底：
# search_fn 是可注入的，编排层不假定检索方一定过滤了。判据与法条侧同形
# （cite_verify：`article["status"] != "现行有效"`），**缺失同样判拒** ——
# 证明不了现行就不等于现行，放行它的方向正是「静默引用过期价位」。


def test_non_current_status_is_rejected_and_logged_without_asking_the_model():
    """非现行片段要判拒、留痕，且**连模型都不问**：一段不会被采用的语料不值得
    再花一次生成（与 no_corpus 一问就是「不猜数」的反面）。生成器上装探针才拦得住
    「先生成后判效力」的实现 —— 它能让 state 断言照样绿（结果同样是 rejected）。"""
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    def gen_boom(snippet, cause):
        raise AssertionError("非现行片段不该拿去问模型")

    stale = {"text": "不超过1万元的，每件交纳50元", "source_doc": "已废止办法",
             "source_no": "第三条", "status": "已废止"}
    result = estimate("民间借贷纠纷", search_fn=search([stale]),
                      generate_fn=gen_boom, log_fn=log)
    assert result["status"] == "rejected"
    # 「依据」类字段一律为 None：没被采用的片段不能显示成依据（同回查判拒）
    assert result["basis"] is None and result["source_doc"] is None
    # 留痕要能指认「拒的是哪一段」：snippet_id 取命中项，区间与模型都是空
    # （根本没问到模型那一层），status 记 rejected
    assert calls[-1]["status"] == "rejected"
    assert calls[-1]["snippet_id"] == "第三条"
    assert calls[-1]["range_low"] is None and calls[-1]["range_high"] is None
    assert calls[-1]["model"] is None and calls[-1]["request_id"] is None


def test_missing_or_blank_status_is_rejected_too():
    """缺失（键不在）与空串都判拒：skip 掉它们等于给「检索方忘了带 status」留一条
    静默放行的通路 —— 而 search 把缺字段归一成空串（FEE_OUTPUT_FIELDS 的返回口径），
    真实链路上正会出现这种值。三个非现值一起钉，免得只挡 `已废止` 这一种写法。

    夹具必须**无量级词、数字有出处**（FLAT_HIT 的口径）：这样「判拒」只可能来自
    效力这一关。用「1万元」那句时缺省闸门会顺手拒掉它，于是把效力闸门放宽成
    「放行缺失/空串」本用例照样绿（忠实变异 M4 实测）—— 一条永远不会红的断言。
    reason 也一并钉：它写的是效力，而不是缺省闸门那句「量级词」。
    """
    for status in (None, "", "已修改"):
        hit = dict(FLAT_HIT)
        if status is None:
            hit.pop("status")
        else:
            hit["status"] = status
        result = estimate("民间借贷纠纷", search_fn=search([hit]),
                          generate_fn=gen(1, 10), log_fn=no_log)
        assert result["status"] == "rejected", status
        assert "效力" in result["reason"], status


def test_current_status_still_passes_the_gate():
    """收紧不得把现行片段一起拒掉：只是把同一条命中的 status 换成现行，就该走
    ok —— 这条与上面两条配对，缺了它就证明不了判据是「非现行」而不是「一律
    不采用」。用无量级词的 FLAT_HIT：本用例只验效力那一关（HIT 缺 unit 会被
    缺省闸门拦下，见 test_fees_unit.py）"""
    current = dict(FLAT_HIT, status="现行有效")
    result = estimate("民间借贷纠纷", search_fn=search([current]),
                      generate_fn=gen(1, 10), log_fn=no_log)
    assert result["status"] == "ok" and result["basis"] == FLAT_SNIPPET
