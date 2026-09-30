# 附加区块是「案由 + 律师 + 费用」的组装点，也是**故障降级**的落点：
# 费用检索/模型挂掉时这一块要标成 unavailable，而不是让整条问答崩掉，
# 更不能伪装成「暂无费用口径依据」（③a 红线在 ③b 的延续）。
import random

from app.recommend.attach import build
from app.recommend.cause import GENERIC_FIELD
from app.recommend.fee_log import _COLUMNS
from app.recommend.fees import FEE_DISCLAIMER

# 命中片段的夹具，两处都与闸门口径对齐：
# ①**无量级词**：estimate 的缺省闸门规定「命中数字紧邻万/千/百/亿 而 unit 缺失
#   → 判拒」，而本文件的假生成器一律不报 unit（`_deps` 的 low/high=1）。原先那句
#   「不超过1万元的，每件交纳50元」里的 1 正紧邻「万元」，在本文件的 ok 用例上
#   会被闸门挡下 —— 那是它**该**被挡的情形，夹具换成 1 元口径才对得上「缺 unit 的
#   合法情形」（量级与缺省两向的用例都在 test_fees_unit.py）
# ②status 必给：效力闸门按「非现行/缺失一律不采用」判拒（设计 §七）
HIT = {"text": "每件交纳1元，按件计收", "source_doc": "诉讼费用交纳办法",
       "source_no": "第十三条", "status": "现行有效"}

# 五种 fee 状态共用的字段形状：status / low / high / unit / charge_basis /
# basis / source_doc / source_no / reason。渲染方按固定字段读，缺一个键就是
# KeyError（本批新增 charge_basis，故 _unavailable 也必须同步加上）
FEE_KEYS = {"status", "low", "high", "unit", "charge_basis", "basis",
            "source_doc", "source_no", "reason"}


def _deps(hits=(HIT,), low=1, high=1, boom=False):
    def search_fn(cause):
        if boom:
            raise RuntimeError("milvus down")
        return list(hits)

    return {"search_fn": search_fn,
            "generate_fn": lambda snippet, cause: {"low": low, "high": high},
            "log_fn": lambda **kwargs: None,
            "rng": random.Random(0)}


def test_block_carries_all_required_parts():
    block = build("租房押金不退怎么办", **_deps())
    # 顶层**恰为**这 5 键（Task 6 的设计决定）：多一个键就是一处没想清楚的下游
    # 契约，渲染方与消费方都不知道该不该读它。只逐个查「五个键都在」时，
    # 顺手塞个 fee_status 之类的冗余键照样全绿
    assert set(block) == {"cause", "field", "lawyers", "fee", "disclaimer"}
    assert block["cause"] == "房屋租赁合同纠纷"
    assert block["field"] == "房产与建设工程"
    assert len(block["lawyers"]) == 3
    assert block["fee"]["status"] == "ok"
    assert block["disclaimer"] == FEE_DISCLAIMER


def test_unknown_cause_falls_back_to_generic_without_raising():
    """认不出案由不是错误：领域退「通用」，卡片照给，费用照走检索。"""
    # 案由为 None 时仍要能把费用流程走完（estimate 的 cause 形参是 str，
    # 故编排层必须给个字符串，不能把 None 直接透传 —— 那会在真链路里炸）
    seen = []

    deps = _deps()
    inner = deps["search_fn"]

    def spy(cause):
        seen.append(cause)
        return inner(cause)

    deps["search_fn"] = spy
    block = build("今天天气怎么样", **deps)
    assert block["cause"] is None
    assert block["field"] == GENERIC_FIELD
    assert block["lawyers"]
    assert seen == [""]


def test_fee_failure_is_marked_unavailable_not_no_corpus():
    """红线：服务故障绝不能被说成「没有依据」。"""
    block = build("租房押金不退怎么办", **_deps(boom=True))
    assert block["fee"]["status"] == "unavailable"
    assert block["fee"]["reason"]
    # 故障要能带出去给人排查：只断言 reason 非空时，写成一句固定的
    # 「暂无费用口径依据」也全绿 —— 那正是本用例要挡的伪装方向
    assert "milvus down" in block["fee"]["reason"]
    assert "no_corpus" not in block["fee"]["reason"]
    # 费用这一块坏掉不该拖垮律师卡片（其余部件一并钉住：只断言 lawyers 时，
    # 「费用一挂就整块退化成空壳」的实现照样全绿）
    assert block["field"] == "房产与建设工程"
    assert len(block["lawyers"]) == 3
    assert block["disclaimer"] == FEE_DISCLAIMER


def test_no_corpus_is_reported_as_such():
    block = build("租房押金不退怎么办", **_deps(hits=()))
    assert block["fee"]["status"] == "no_corpus"


def test_rejected_range_is_reported_as_such():
    block = build("租房押金不退怎么办", **_deps(low=1, high=99))
    assert block["fee"]["status"] == "rejected"


def test_disclaimer_is_present_on_every_fee_status():
    """AC-21：带费用区块时 100% 带固定提示 —— 含没拿到区间的那几种情形。"""
    # 把四种状态各跑一遍并按「状态 → 提示」逐条对：只断言 disclaimer 时，
    # 某一种状态下提前 return（漏掉 disclaimer）仍可全绿
    for deps, status in ((_deps(), "ok"), (_deps(hits=()), "no_corpus"),
                         (_deps(low=1, high=99), "rejected"),
                         (_deps(boom=True), "unavailable")):
        block = build("租房押金不退怎么办", **deps)
        assert block["fee"]["status"] == status
        assert block["disclaimer"] == FEE_DISCLAIMER


def test_booming_logger_does_not_break_the_block():
    """留痕写库失败不该让用户看不到费用信息 —— 审计是旁路，不是主路。"""
    deps = _deps()

    def boom_log(**kwargs):
        raise RuntimeError("mysql down")

    deps["log_fn"] = boom_log
    block = build("租房押金不退怎么办", **deps)
    assert block["fee"]["status"] == "ok"
    # 不只是「没崩」：费用结果本身必须完好 —— 把留痕失败也折成 unavailable
    # 或把区间清空的实现（故障归属搞混）要在这里红
    assert block["fee"]["low"] == 1 and block["fee"]["high"] == 1
    assert block["fee"]["basis"] == HIT["text"]


def test_booming_logger_warns_instead_of_disappearing(caplog):
    """Task 6 交接单点名、Task 9 漏做的信号（本批补上，与 answer.py 那条同族）：
    留痕失败不改变用户体验是对的，但**不许无声** —— AC-21 的证据在 MySQL 里，
    写不进去时全链路没有任何红灯，「连续几小时零留痕」与「本来就没有问答」
    表现完全一样。删掉 guarded_log 里那行 logging.warning 时本用例必须红。

    日志只带留痕字段与异常：fields 里是案由/片段 id/区间/模型/请求 id/状态，
    不含用户问句（AC-19 的口径同样适用于日志）。
    """
    deps = _deps()

    def boom_log(**kwargs):
        raise RuntimeError("mysql down")

    deps["log_fn"] = boom_log
    with caplog.at_level("WARNING", logger="app.recommend.attach"):
        block = build("租房押金不退怎么办", **deps)
    assert block["fee"]["status"] == "ok"
    assert "留痕" in caplog.text and "mysql down" in caplog.text
    assert "租房押金不退怎么办" not in caplog.text


def test_cards_get_the_mapped_field_and_the_injected_rng():
    """卡片必须按**映射后的领域**生成，且随机源来自注入的 rng。

    两条都是「透传」类接口，坏了不会报错：领域传错（比如直接拿 GENERIC_FIELD）
    时卡片照给 3 张、只是领域不对；rng 丢掉时卡片每次不同、张数照旧是 3。
    故按「同一 seed 两次结果逐张相同」与「每张卡片的 field == block['field']」
    分别钉住 —— 只数张数是拦不住这两种坏法的。
    """
    first = build("租房押金不退怎么办", **_deps())
    again = build("租房押金不退怎么办", **_deps())
    assert first["lawyers"] == again["lawyers"]
    assert all(c["field"] == "房产与建设工程" for c in first["lawyers"])


def test_fee_result_keeps_one_shape_across_all_four_statuses():
    """四种状态共用同一套 fee 字段（交接 1：unavailable 少一个 unit 就是 KeyError）。

    渲染方按固定字段读区间（含 `fee["unit"]`），形状不齐时只在故障那一种状态上炸 ——
    而故障态恰是最没人手工测到的一支。故这里按**键集合**逐状态比对，而不是只看
    `status`：去掉 `_unavailable` 里的 `unit` 键，本用例红。
    """
    for deps in (_deps(), _deps(hits=()), _deps(low=1, high=99), _deps(boom=True)):
        fee = build("租房押金不退怎么办", **deps)["fee"]
        assert set(fee) == FEE_KEYS
        # 无区间的三种状态里，这些字段必须是 None 而不是缺席（同上：渲染方读键）
        if fee["status"] != "ok":
            assert fee["low"] is None and fee["high"] is None
            assert fee["unit"] is None and fee["basis"] is None


def test_log_fields_are_forwarded_verbatim_to_the_logger():
    """交接 2：留痕字段名必须与 fee_log._COLUMNS 逐字一致。

    `fee_log.record` 对不认识的 kwargs **静默丢弃**（不报错），所以字段名写错
    只会变成一格 NULL 的证据 —— 留痕当场失效却没有任何红灯。故这里按 ok 与
    rejected 两条路径分别比对**字段名集合与取值**，别让组装层把名字改掉。
    """
    calls = []

    def log(**kwargs):
        calls.append(kwargs)

    deps = _deps()
    deps["log_fn"] = log
    build("租房押金不退怎么办", **deps)
    assert set(calls[-1]) == set(_COLUMNS)
    assert calls[-1]["status"] == "ok"
    assert calls[-1]["cause"] == "房屋租赁合同纠纷"
    assert calls[-1]["snippet_id"] == "第十三条"
    assert (calls[-1]["range_low"], calls[-1]["range_high"]) == (1, 1)

    rej = _deps(low=1, high=99)
    rej["log_fn"] = log
    build("租房押金不退怎么办", **rej)
    # 拒绝也要留痕（AC-21 的「拒绝过什么」），且同样走组装层转发的字段名
    assert set(calls[-1]) == set(_COLUMNS)
    assert calls[-1]["status"] == "rejected"
    assert (calls[-1]["range_low"], calls[-1]["range_high"]) == (1, 99)
