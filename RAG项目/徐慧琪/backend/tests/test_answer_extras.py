# ③b-1：附加区块接入问答编排的那一段。
#
# 为什么独立成文件：原先这几个用例留在 test_answer.py 末尾，而那个文件的非空行
# 已顶到「单文件 ≤ 300」的硬闸门（本批要往这里加一条「故障信号」的用例，就只能
# 先按主题拆）。这里只验「什么时候调、什么时候不调、挂了怎么办」——区块内容由
# test_attach.py 负责，渲染由 tools/tests/test_ask.py 负责。
from app.generation.answer import Answerer
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC
from tests._fakes import BAD_QUOTE, Boom, FakeLLM, _answerer, _json_answer


def _extras_spy():
    calls = []

    def fn(question):
        calls.append(question)
        return {"cause": "房屋租赁合同纠纷", "field": "房产与建设工程",
                "lawyers": [], "fee": {"status": "no_corpus"},
                "disclaimer": "参考区间，不构成报价或委托"}

    return fn, calls


def test_public_ok_gets_extras():
    fn, calls = _extras_spy()
    result = _answerer(FakeLLM([_json_answer()]), extras_fn=fn).answer("租房押金不退怎么办",
                                                                      SIDE_PUBLIC)
    assert result.status == "ok"
    assert result.extras is not None
    assert result.extras["cause"] == "房屋租赁合同纠纷"
    assert calls == ["租房押金不退怎么办"]


def test_internal_never_gets_extras():
    """律师侧永不经过这条路（数据不出域，且推荐律师是公众侧功能）。"""
    fn, calls = _extras_spy()
    result = _answerer(FakeLLM([_json_answer()]), extras_fn=fn) \
        .answer("第五百八十四条规定了什么", SIDE_INTERNAL)
    assert result.status == "ok"
    assert result.extras is None and calls == []


def test_no_extras_fn_means_no_extras():
    """默认关闭：不传就不调 —— ③a 的既有行为完全不变。"""
    answerer = _answerer(FakeLLM([_json_answer()]))
    assert answerer.answer("租房押金不退怎么办", SIDE_PUBLIC).extras is None


def test_public_out_of_scope_gets_no_extras():
    """与法律无关的两条出口不给区块（第四批起仅限**不含问价词**的）—— 判据里
    唯一被排除的状态仍是 out_of_scope。

    两条路由必须分开走：拒答在**检索之前**就返回（连模型都没问），模型自报的
    out_of_scope 走的是**正常出口**（`bounded.status`）。判据若只挂在早退那一条
    上，第二条会漏 —— 而「今天天气怎么样」正是用户 2026-09-29 裁决里点名的标本。
    这两条标本都不含问价词，所以走的是「问价意图」那半边的否支。
    """
    fn, calls = _extras_spy()
    refused = _answerer(FakeLLM([]), extras_fn=fn).answer("这个官司我能赢吗", SIDE_PUBLIC)
    assert refused.status == "out_of_scope"
    reply = _json_answer(status="out_of_scope", answer="这不是法律问题。", citations=False)
    answered = _answerer(FakeLLM([reply]), extras_fn=fn) \
        .answer("今天天气怎么样", SIDE_PUBLIC)
    assert answered.status == "out_of_scope"
    assert refused.extras is None and answered.extras is None
    assert calls == [], "被排除的出口连 extras_fn 都不该碰（省下的正是它的成本）"


def test_price_intent_wins_over_out_of_scope():
    """本批存在的理由：命中问价词的 out_of_scope **照给**区块。

    标本就是实测推翻上一版判据的那一条 —— 模型把它判成 out_of_scope（理由：
    「这属于律师服务收费的市场行情问题，不是由上面这些法条来规定的内容」），
    于是「状态非 out_of_scope」的判据下用户什么都没拿到。断言 calls 逐字相等：
    恰好一次，且喂给 extras_fn 的**是用户原问句**（费用检索拿它当查询串）。
    """
    fn, calls = _extras_spy()
    reply = _json_answer(status="out_of_scope", answer="这属于市场行情问题。",
                         citations=False)
    result = _answerer(FakeLLM([reply]), extras_fn=fn).answer("律师费大概多少", SIDE_PUBLIC)
    assert result.status == "out_of_scope"
    assert result.extras is not None
    assert calls == ["律师费大概多少"]


def test_price_intent_on_refusal_exit_gets_extras():
    """拒答出口（检索之前的规则早退）也走同一条判据 —— 问价就照给。

    这一条把 `answer()` 里拒答那行的 `extras=` 从「形状一致性」变成**真护栏**：
    删掉那一行，本用例当场变红（上一批同位置的用例是删了也不会红的，代码注释里
    当时明写了这一点）。判据必须是**一个**，不能因为「拒答在检索之前返回」就少
    走一步 —— 用户在这个问句里问的正是费用。
    """
    fn, calls = _extras_spy()
    result = _answerer(FakeLLM([]), extras_fn=fn) \
        .answer("帮我写一份起诉状，律师费大概多少", SIDE_PUBLIC)
    assert result.status == "out_of_scope"
    assert result.extras is not None
    assert calls == ["帮我写一份起诉状，律师费大概多少"]


def test_price_intent_does_not_open_internal_side():
    """问价词只放宽状态这一维，**不放宽侧别**：律师侧问了价也不给区块。

    红线（数据不出域 + 推荐律师是公众侧功能）在第四批的这条新路径上同样成立 ——
    若判据写成 `or is_price_intent(...)` 时把侧别检查包进去，两个出口都会漏。
    """
    fn, calls = _extras_spy()
    reply = _json_answer(status="out_of_scope", answer="这属于市场行情问题。",
                         citations=False)
    answered = _answerer(FakeLLM([reply]), extras_fn=fn) \
        .answer("律师费大概多少", SIDE_INTERNAL)
    refused = _answerer(FakeLLM([]), extras_fn=fn) \
        .answer("帮我写一份起诉状，律师费大概多少", SIDE_INTERNAL)
    assert answered.extras is None and refused.extras is None
    assert calls == [], "律师侧一个出口都不许碰 extras_fn"


def _non_ok_exit_samples(extras_fn, side=SIDE_PUBLIC):
    """解耦后**法律相关但非 ok** 的出口各给一个标本（出口名 → QAResult）。

    为什么按出口穷举而不是只测一条：判据（侧别 + 状态 + 问价意图）散在八条
    返回路径上，漏掉哪一条的 extras=，就只有那一条出口没有区块 —— 在用户侧的
    观感与「这条本来就不给区块」一模一样，只有逐出口走一遍才看得见。
    六个标本恰好覆盖六条不同的返回语句：追问走正常出口、无召回走早退、
    两次校验失败走循环末尾、三条 error 各有各的早退（生成中故障 / 检索故障 /
    LLM 客户端没建起来）。
    """
    modified = _json_answer(quote=BAD_QUOTE)

    def broken_retrieve(question):
        raise RuntimeError("Milvus 连接中断")

    def never_retrieve(question):
        raise AssertionError("LLM 客户端都没建起来，不该走到检索")

    def broken_factory(s):
        raise RuntimeError("密钥缺失")

    follow_up = _json_answer(status="need_more_info", answer="还需要签订时间。",
                             citations=False)
    return {
        "need_more_info": _answerer(FakeLLM([follow_up]), extras_fn=extras_fn) \
            .answer("押金不退怎么办", side),
        "abstain_no_blocks": _answerer(FakeLLM([]), blocks=(), exact_nos=(),
                                       extras_fn=extras_fn) \
            .answer("毫不相关的问题", side),
        "abstain_two_failures": _answerer(FakeLLM([modified, modified]),
                                          extras_fn=extras_fn) \
            .answer("押金不退怎么办", side),
        "error_llm": _answerer(Boom(), extras_fn=extras_fn) \
            .answer("押金不退怎么办", side),
        "error_retrieval": Answerer(retrieve_fn=broken_retrieve,
                                    corpus=type("C", (), {})(),
                                    llm_factory=lambda s: FakeLLM([]),
                                    extras_fn=extras_fn).answer("押金不退怎么办", side),
        "error_llm_factory": Answerer(retrieve_fn=never_retrieve,
                                      corpus=type("C", (), {})(),
                                      llm_factory=broken_factory,
                                      extras_fn=extras_fn).answer("押金不退怎么办", side),
    }


def test_law_related_non_ok_public_states_get_extras():
    """解耦（用户 2026-09-29 裁决）：**法律相关但非 ok** 的回答同样组装区块。

    理由（裁决原文）：枚举里只有 out_of_scope 真意味着「与法律无关」，而追问/
    降级/故障恰恰是「用户问法条问不出、但可能正在问价」的情形 —— 问「律师费大概
    多少」时法条本来就答不出，那正是费用区块最有用的时刻。
    出口名与状态一起断言：标本若哪天漂到别的出口上（比如 need_more_info 的回包
    没走到正常出口），本用例会先在这里红，而不是让「这条出口有区块」变成一句
    声称而已。
    """
    fn, calls = _extras_spy()
    samples = _non_ok_exit_samples(fn)
    assert {name: result.status for name, result in samples.items()} == {
        "need_more_info": "need_more_info", "abstain_no_blocks": "abstain",
        "abstain_two_failures": "abstain", "error_llm": "error",
        "error_retrieval": "error", "error_llm_factory": "error"}
    for name, result in samples.items():
        assert result.extras is not None, f"{name} 出口没有组装附加区块"
        assert result.extras["cause"] == "房屋租赁合同纠纷"
    # 次数逐条钉住：每次问答**恰好一次** —— 解耦的代价是每个出口多一次费用检索
    # 与一次生成调用，多调几次不会有别的红灯，只会静默变贵
    assert calls == ["押金不退怎么办", "毫不相关的问题", "押金不退怎么办",
                     "押金不退怎么办", "押金不退怎么办", "押金不退怎么办"]


def test_internal_non_ok_exits_never_get_extras():
    """律师侧的红线在非 ok 出口上同样成立 —— 解耦只放给公众侧。

    只在 ok 路径上验律师侧是不够的：解耦新开了五条会调用 extras_fn 的路径，
    数据不出域的红线必须在**每一条**上都成立（律师侧本来就连 DeepSeek 客户端
    都不该构造）。
    """
    fn, calls = _extras_spy()
    samples = _non_ok_exit_samples(fn, SIDE_INTERNAL)
    assert all(result.extras is None for result in samples.values())
    assert calls == [], "律师侧一个出口都不许碰 extras_fn"


def test_extras_fn_failure_keeps_the_main_answer():
    """区块是装饰、不是答案的一部分：它挂掉不能连累已算好的回答。Task 9 起
    extras_fn 接的是真 fee_corpus 检索/生成 —— 最容易运行期挂的那部分。"""
    def boom(question):
        raise RuntimeError("milvus down")

    result = _answerer(FakeLLM([_json_answer()]), extras_fn=boom) \
        .answer("押金不退怎么办", SIDE_PUBLIC)
    assert result.status == "ok" and result.extras is None
    assert result.answer == "应当赔偿。"


def test_extras_failure_is_not_silent(caplog):
    """Task 7 交接单点名、Task 9 漏做的信号（本批补上）：只丢区块是对的，但无声地
    丢会让「区块连续消失」与「本来就没有区块」不可区分 —— 主流程全绿，故障只在
    用户看到的输出里少一块。故删掉 answer.py 那行 logging.warning 时本用例必须红。

    顺带钉住日志里**不含用户问句**：AC-19 的「不记用户文本」不只在留痕表上成立，
    日志同样不例外（异常与侧别足够定位问题）。
    """
    def boom(question):
        raise RuntimeError("milvus down")

    with caplog.at_level("WARNING", logger="app.generation.answer"):
        result = _answerer(FakeLLM([_json_answer()]), extras_fn=boom) \
            .answer("押金不退怎么办", SIDE_PUBLIC)
    assert result.status == "ok" and result.extras is None
    assert "附加区块" in caplog.text and "milvus down" in caplog.text
    assert "押金不退怎么办" not in caplog.text
