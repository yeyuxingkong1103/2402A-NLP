# 数值回查是防「模型凭空报个价」的唯一闸门：区间两端的数字必须能在命中
# 片段里逐字找到。**注意它是必要不充分条件** —— 它保证数字有出处，
# 不保证语义正确（那是提示词与人工冒烟的事）。测试要压住两个方向：
# 有出处的必须过、没出处的必须拒。
#
# 本文件只管**回查本身**（纯函数 numbers_in / verify_range）；编排层的用例
# （命中 → 生成 → 判拒/降级、单位与量级闸门、效力闸门）在 test_fees_estimate.py
# 与 test_fees_unit.py —— 原先是同一个文件，非空行顶到 300 的硬闸门后按主题拆的
from app.recommend.fees import FEE_DISCLAIMER, numbers_in, verify_range
from tests._fee_fixtures import SNIPPET


def test_numbers_in_extracts_plain_integers():
    assert numbers_in("不超过1万元的，每件交纳50元") == {"1", "50"}


def test_numbers_in_handles_decimals():
    assert numbers_in("按照2.5%交纳") == {"2.5"}


def test_numbers_in_handles_empty_and_none():
    assert numbers_in("") == set()
    assert numbers_in(None) == set()


def test_accepts_range_whose_ends_appear_verbatim():
    assert verify_range(1, 10, SNIPPET) is True


def test_accepts_decimal_end():
    assert verify_range(2.5, 2.5, "按照2.5%交纳") is True


def test_rejects_range_with_number_absent_from_snippet():
    """凭空多出来的数字必须被拒 —— 这是这道闸门存在的全部理由。"""
    assert verify_range(1, 30, SNIPPET) is False


def test_rejects_when_only_one_end_is_present():
    """一端有出处、另一端没有，同样算编造。"""
    assert verify_range(1, 99, SNIPPET) is False


def test_rejects_when_only_the_high_end_is_present():
    """上一条的**镜像方向**：low 无出处、high 有出处。两条合起来两端才各有护栏 ——
    只补一条时，「只查高端不查低端」的残缺实现（`_literal(high) in pool`）能让全部
    拒用用例通过，因为原有用例清一色是 low 有出处的方向（实测变异 M6 全绿）。
    配对**必须非倒置**（这里 7 < 10）：倒置对（如 (30, 10)）会被倒置守卫先短路成
    False，与端点检查无关 —— 那样这条就再也证明不了它声称的东西（实测：只查高端的
    实现下倒置对全绿，本行才是它唯一的拦阻者）。"""
    assert verify_range(7, 10, SNIPPET) is False


def test_rejects_missing_bounds():
    assert verify_range(None, 10, SNIPPET) is False
    assert verify_range(1, None, SNIPPET) is False


def test_rejects_equal_ends_without_source():
    """两个端点相等时同样各自要有出处：`(30, 30)` 是「约 30 元」这类**单一值**的
    形态（模型的常见输出），片段里没有 30 就该拒。原有拒用用例清一色两端不同，
    盖不到这个形态 —— 加一条 `or low == high` 就能让它们全绿（实测能过 11 条）。"""
    assert verify_range(30, 30, SNIPPET) is False


def test_rejects_inverted_range():
    """倒置区间（low > high）两端有出处也无效：安全阀自己 fail-closed，不指望编排
    层兜底（技术方案 6.4 硬规矩①的失败方向正是「放行了不该放的数」，且倒置区间
    到渲染层会被读成另一个区间）。"""
    assert verify_range(10, 1, SNIPPET) is False


def test_integer_valued_float_matches_integer_literal():
    """8 与 8.0 是同一个数，不能因为写法不同把正确区间拒掉。"""
    assert verify_range(8.0, 8.0, "收费比例为8%") is True


def test_pool_is_normalized_so_verbatim_copy_still_verifies():
    """片段写「2.50」「50.0」时，模型**逐字照抄**也得能过 —— 归一化要做在 pool 一侧，
    两侧共用同一把尺子。只归一化上报值（单侧）时这两条会静默变成 False，接受方向
    退化成永远拒绝，而安全阀的另一半职责正是「有出处必须过」。"""
    assert numbers_in("按照2.50%交纳，另交50.0元") == {"2.5", "50"}
    assert verify_range(2.5, 2.5, "按照2.50%交纳") is True
    assert verify_range(50, 50, "交纳50.0元") is True


def test_thousands_separator_is_one_number_not_a_source_for_its_digits():
    """「1,000」是**一个**数：分词时不能让它的子串「1」「000」成为可选出处，否则
    「标的额 1000 元」的片段会放行 verify_range(1, 1) —— 又是误放行的方向。分隔符
    在分词层吃掉（不是事后归一化），顺带钉住它仍能被正查命中。"""
    snippet = "标的额超过1,000元的，交纳1.5%"
    assert verify_range(1, 1, snippet) is False
    assert verify_range(1000, 1000, snippet) is True


def test_rejects_when_snippet_is_empty():
    assert verify_range(1, 10, "") is False


def test_disclaimer_constant_is_the_agreed_wording():
    """AC-21 的原话，不得改写（改坏了渲染层也没人拦）。"""
    assert FEE_DISCLAIMER == "参考区间，不构成报价或委托"
