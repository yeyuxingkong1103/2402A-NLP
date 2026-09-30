# 编排层的测试全部用假 LLM 与假检索：这里要验的是**分支**——校验失败会不会重生成、
# 重生成仍失败会不会降级、LLM 挂了是不是 error 而不是 abstain；真模型真检索归 Task 13 冒烟。
import pytest

from app.generation import boundaries
from app.generation.answer import RETRY_HINT, Answerer, QAResult
from app.generation.chain import build_prompt
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC, system_prompt
from app.generation.schema import Answer, answer_parser
from app.retrieval.pipeline import RetrievalResult
from tests._fakes import (BAD_QUOTE, BLOCK_584, Boom, FakeLLM, Flaky, _answerer,
                          _json_answer)

def test_good_answer_returns_ok_with_sources():
    result = _answerer(FakeLLM([_json_answer()])).answer("第五百八十四条", "internal")
    assert isinstance(result, QAResult)
    assert result.status == "ok"
    assert result.attempts == 1
    assert result.sources == [BLOCK_584], "原文要带出来供前端分区渲染"


def test_result_carries_retrieval_for_evaluation():
    # 评测脚本要靠它算 Recall@k，否则每道题都要重跑一次检索
    result = _answerer(FakeLLM([_json_answer()])).answer("第五百八十四条", "internal")
    assert result.retrieval is not None
    assert result.retrieval.exact_nos == [584]


def test_verification_failure_triggers_one_retry():
    # 第一次故意给改写过的 quote，第二次给正确的
    rewritten = _json_answer(quote=BAD_QUOTE)
    llm = FakeLLM([rewritten, _json_answer()])
    result = _answerer(llm).answer("第五百八十四条", "internal")
    assert result.attempts == 2
    assert result.status == "ok"
    assert len(llm.prompts) == 2


def test_retry_prompt_carries_the_failure_reason():
    modified = _json_answer(quote=BAD_QUOTE)
    llm = FakeLLM([modified, _json_answer()])
    _answerer(llm).answer("第五百八十四条", "internal")
    second = str(llm.prompts[1])
    assert RETRY_HINT in second
    assert "原文" in second, "要把失败原因带给模型，否则它会重犯同一个错"


def test_two_failures_lead_to_abstain():
    modified = _json_answer(quote=BAD_QUOTE)
    result = _answerer(FakeLLM([modified, modified])).answer("第五百八十四条", "internal")
    assert result.status == "abstain"
    assert result.attempts == 2
    assert result.citations == []
    assert "未找到相关依据" in result.answer
    assert any("原文" in reason for reason in result.failures), \
        "降级原因要留在结果里：④期排查「为什么答不出来」只有这一条线索"


def test_unparseable_json_triggers_retry_then_abstain():
    # 解析失败与校验失败走同一条重生成通路（设计文档 4.4）
    result = _answerer(FakeLLM(["我不是 JSON", "我还是不是 JSON"])).answer("q", "internal")
    assert result.status == "abstain"
    assert result.attempts == 2


def test_empty_retrieval_abstains_without_calling_llm():
    llm = FakeLLM([])  # 脚本为空：一旦被调用就会抛 AssertionError
    result = _answerer(llm, blocks=(), exact_nos=()).answer("毫不相关的问题", "internal")
    assert result.status == "abstain"
    assert result.attempts == 0
    assert llm.prompts == []


def test_refusal_short_circuits_before_retrieval():
    # FR-8.3：拒答在检索之前判定，一次检索与一次生成都省下
    llm = FakeLLM([])
    result = _answerer(llm).answer("这个官司我能赢吗", "internal")
    assert result.status == "out_of_scope"
    assert result.attempts == 0
    assert llm.prompts == []


def test_public_answer_gets_disclaimer_injected():
    result = _answerer(FakeLLM([_json_answer()])).answer("第五百八十四条", "public")
    assert "仅供参考" in result.disclaimer


def test_llm_exception_yields_error_not_abstain():
    # 模型不可用 ≠ 没有依据；混淆两者会让 ④期把服务故障答成"查无此条"
    result = _answerer(Boom()).answer("第五百八十四条", "internal")
    assert result.status == "error"
    assert result.answer != "未找到相关依据"


def test_llm_factory_error_yields_error(monkeypatch):
    def factory(side):
        raise RuntimeError("密钥缺失")

    result = Answerer(retrieve_fn=lambda q: RetrievalResult(question=q),
                      corpus=type("C", (), {})(), llm_factory=factory).answer("q", "public")
    assert result.status == "error"


def test_model_disclaimer_is_overwritten_by_code():
    # AC-18：免责由代码注入并**覆盖**模型输出，模型漏写或改写都不影响结果
    result = _answerer(FakeLLM([_json_answer().replace('"disclaimer": ""',
                                                       '"disclaimer": "随便写写"')])) \
        .answer("第五百八十四条", "public")
    assert result.disclaimer == boundaries.PUBLIC_DISCLAIMER


def test_need_more_info_passes_through_without_citations():
    # 追问补齐不是失败：模型给了 need_more_info 就直接返回，不进重生成
    reply = _json_answer(status="need_more_info", answer="还需要合同签订时间。",
                         citations=False)
    llm = FakeLLM([reply])
    result = _answerer(llm).answer("第五百八十四条", "internal")
    assert result.status == "need_more_info"
    assert result.attempts == 1
    assert len(llm.prompts) == 1


def test_model_out_of_scope_passes_through():
    reply = _json_answer(status="out_of_scope", answer="这不是法律问题。",
                         citations=False)
    result = _answerer(FakeLLM([reply])).answer("今天天气怎么样", "public")
    assert result.status == "out_of_scope"
    assert result.attempts == 1
    assert result.citations == []


@pytest.mark.parametrize("status", ["need_more_info", "out_of_scope"])
def test_non_ok_status_drops_unverified_citations(status):
    # AC-6「输出中无未校验引用」：非 ok 的答案不过四关（verify_answer 对它们
    # 一律放行），出口若不清空引用，模型编造的条号会原样带出——终审实测
    # need_more_info + 第9999条 四关全「过」并出现在 CLI 输出里。
    # 该分支此前零测试：旧用例刻意让非 ok 答案不带引用，正好绕开了这条通路
    reply = _json_answer(status=status, answer="还需要补充信息。",
                         article="第九千九百九十九条")
    result = _answerer(FakeLLM([reply])).answer("第五百八十四条", "internal")
    assert result.status == status
    assert result.citations == [], "非 ok 状态的引用未经四关校验，绝不许带出出口"


def test_verify_dependency_failure_is_error_not_abstain():
    # Task 8 的交接：verify_answer 不吞异常，DB / Milvus 挂了会向上抛。
    # 这正是设计要的——校验依赖故障 ≠ 查无此条，降级成 abstain 会让 ④期
    # 把"校验库挂了"答成"没有依据"，运维排查时也看不出真相
    class BrokenCorpus:
        def article(self, no):
            raise RuntimeError("MySQL 连接中断")

        def para_index(self, no):
            return {"paragraphs": set(), "items": set()}

    result = _answerer(FakeLLM([_json_answer()]), corpus=BrokenCorpus()) \
        .answer("第五百八十四条", "internal")
    assert result.status == "error"
    assert result.attempts == 1, "服务故障不重试：重试打不穿故障，只烧一次调用"
    assert "未找到相关依据" not in result.answer


def test_public_abstain_and_error_carry_disclaimer():
    # AC-18 裁决：公众侧所有出口都带免责，降级与故障出口也不例外。
    # 这两条出口不经过 boundaries.apply_boundaries，是 unified 出口要钉住的部分
    abstain = _answerer(FakeLLM([_json_answer(quote=BAD_QUOTE)]
                                * 2)).answer("第五百八十四条", "public")
    error = _answerer(Boom()).answer("第五百八十四条", "public")
    assert abstain.status == "abstain" and error.status == "error"
    for result in (abstain, error):
        assert result.disclaimer == boundaries.PUBLIC_DISCLAIMER
        # 失败路径也要带回召回集：前端要在"没有依据"旁边列出找到的条文，
        # 评测（Task 12）靠 retrieval 算 Recall，缺了就得重跑一次检索
        assert result.sources == [BLOCK_584] and result.retrieval is not None


def test_public_refusal_carries_disclaimer():
    # 拒答也要免责：它同样是对公众输出的法律相关内容
    result = _answerer(FakeLLM([])).answer("这个官司我能赢吗", "public")
    assert result.status == "out_of_scope"
    assert result.disclaimer == boundaries.PUBLIC_DISCLAIMER


def test_internal_exits_carry_no_disclaimer():
    modified = _json_answer(quote=BAD_QUOTE)
    cases = [_answerer(FakeLLM([modified, modified])).answer("第五百八十四条", SIDE_INTERNAL),
             _answerer(Boom()).answer("第五百八十四条", SIDE_INTERNAL),
             _answerer(FakeLLM([])).answer("这个官司我能赢吗", SIDE_INTERNAL)]
    assert [c.status for c in cases] == ["abstain", "error", "out_of_scope"]
    assert [c.disclaimer for c in cases] == ["", "", ""]


def test_disclaimer_rule_agrees_with_boundaries():
    # 免责判定在 answer 的统一出口与 boundaries 各有一份，分叉就是合规口径分叉；
    # 断言把两份钉在一起：任一边改了规则而另一边没改，这里必红
    modified = _json_answer(quote=BAD_QUOTE)
    for side, expected in ((SIDE_PUBLIC, boundaries.PUBLIC_DISCLAIMER),
                           (SIDE_INTERNAL, "")):
        lowered = _answerer(FakeLLM([modified, modified])).answer("第五百八十四条", side)
        assert lowered.status == "abstain"
        assert lowered.disclaimer == expected
        assert boundaries.apply_boundaries(Answer(), side).disclaimer == expected


def test_unknown_side_fails_closed_on_every_exit():
    # 未知侧别不能在任一出口 fail-open 成空免责（与 profiles.system_prompt、
    # llm_router.get_llm、boundaries.apply_boundaries 同向）。能用未知侧真走到的出口
    # 只有三个：拒答（在 get_llm 之前就返回，指望不上它兜底）、无召回降级、建链故障
    # （system_prompt 当场抛 ValueError，被"准备请求"那个 try 收下）——生成期那条
    # error 出口带着未知侧**到不了**（建链先炸），所以下面不收它
    cases = {"refusal": (_answerer(FakeLLM([])), "这个官司我能赢吗"),
             "abstain": (_answerer(FakeLLM([]), blocks=(), exact_nos=()), "毫不相关的问题"),
             "build_error": (_answerer(FakeLLM([_json_answer()])), "第五百八十四条")}
    for answerer, question in cases.values():
        with pytest.raises(ValueError):
            answerer.answer(question, "lawyer")


def test_build_failure_is_error_at_once_not_retry(monkeypatch):
    # 钉住"建链故障落在哪个出口"：修复前 system_prompt 的 ValueError 被当成解析失败
    # →重试→降级（attempts==2、abstain），只因最后仍会抛才没暴露。换成记录型替身
    # 就看得到 QAResult 真身
    seen = []
    monkeypatch.setattr("app.generation.answer._side_disclaimer",
                        lambda side: seen.append(side) or "")
    result = _answerer(FakeLLM([_json_answer()])).answer("第五百八十四条", "lawyer")
    assert result.status == "error", "建链失败是故障，不是「没有依据」"
    assert result.attempts == 0, "准备请求阶段就失败了，不该进重生成"
    assert seen == ["lawyer"], "该出口仍要过统一入口（此处被替身记录）"


@pytest.mark.parametrize("broken", ["article", "para_index"])
def test_verify_valueerror_is_error_not_abstain(broken):
    # ValueError 正是被误捕获的那个类型：校验依赖抛它（corpus 里 int() 撞上畸形
    # Milvus 值就是这条路）时，宽捕获会当成"解析失败"→重试→降级成"未找到相关依据"，
    # 于是"校验库挂了"被答成"查无此条"
    class BrokenCorpus:
        def article(self, no):
            if broken == "article":
                raise ValueError("invalid literal for int()")
            return {"text": BLOCK_584["text"], "status": "现行有效", "law_id": "minfadian"}

        def para_index(self, no):
            raise ValueError("invalid literal for int() with base 10: ''")

    result = _answerer(FakeLLM([_json_answer()]), corpus=BrokenCorpus()) \
        .answer("第五百八十四条", "internal")
    assert result.status == "error"
    assert result.attempts == 1, "服务故障不重试"
    assert "未找到相关依据" not in result.answer


def test_malformed_block_is_error_not_crash():
    # 召回块缺字段是**数据故障**：拼载荷撞 KeyError 要收敛成 error，不能让异常
    # 直接穿出 answer()——④期接口层会把它变成 500，与"服务故障"语义脱节
    bad_block = {k: v for k, v in BLOCK_584.items() if k != "path"}
    result = _answerer(FakeLLM([_json_answer()]), blocks=(bad_block,)) \
        .answer("第五百八十四条", "internal")
    assert result.status == "error"
    assert "生成服务不可用" in result.answer


def test_retry_service_error_keeps_first_failure():
    # 首轮校验没过、次轮服务挂掉：结果是 error，但首轮失败原因不能丢（排查的唯一现场）
    result = _answerer(Flaky(_json_answer(quote=BAD_QUOTE))).answer("第五百八十四条", "internal")
    assert result.status == "error"
    assert result.attempts == 2
    assert any("原文" in reason for reason in result.failures)


def test_retrieval_failure_yields_error_with_disclaimer():
    # 检索挂掉（Milvus / MySQL 不可用）是**服务故障**，不是"没有依据"；
    # 这条出口除了本例外没有任何用例走到，公众侧同样要带免责
    def broken_retrieve(question):
        raise RuntimeError("Milvus 连接中断")

    result = Answerer(retrieve_fn=broken_retrieve, corpus=type("C", (), {})(),
                      llm_factory=lambda side: FakeLLM([])).answer("q", "public")
    assert result.status == "error"
    assert result.attempts == 0, "还没生成就挂了，不该报告重生成次数"
    assert result.disclaimer == boundaries.PUBLIC_DISCLAIMER


def test_prompt_keeps_format_instructions_verbatim():
    # 裁决①的落地：格式说明含 JSON schema（成片花括号），照直拼进 ChatPromptTemplate
    # 会抛 ValueError: Invalid format specifier。转义后必须**逐字**还原，
    # 不能因为"能跑了"就把 schema 洗掉——字段契约靠它，模型全靠它写对字段名
    messages = build_prompt(SIDE_PUBLIC).format_messages(payload="载荷")
    assert answer_parser().get_format_instructions() in messages[0].content
    assert messages[0].content.startswith(system_prompt(SIDE_PUBLIC))
    assert messages[1].content == "载荷", "human 的 {payload} 是变量，不能被一起转义"


def test_special_token_pollution_is_cleaned_before_parsing():
    # Task 7 真连的原始标本：DeepSeek 的 json 模式会把 <|OPENAI|> 漏进 content
    # 并顶掉一个空格（实测 '{"ok":<|OPENAI|>true}'），json.loads 直接失败。
    # 不洗的话它走"解析失败→重生成→降级"，用户拿到"未找到相关依据"，
    # 而真实原因（模型漏 token）在任何日志里都看不见
    polluted = _json_answer().replace(": ", ":<|OPENAI|> ", 1)
    result = _answerer(FakeLLM([polluted])).answer("第五百八十四条", "internal")
    assert result.status == "ok"
    assert result.attempts == 1, "洗掉 token 后一次就该过，不该白烧一次重生成"
    assert result.answer == "应当赔偿。"
