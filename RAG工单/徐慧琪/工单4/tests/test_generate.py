# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from unittest.mock import MagicMock

import pytest

from rag04.config import get_settings
from rag04.schema import Hit
from rag04.generate.prompt import (
    detect_lang, build_context, build_messages, build_citations,
)
from rag04.generate.llm import (
    LLMClient, detect_answer_refusal, is_refusal_question, parse_answerability,
)


def _h(cid, text, page=1, bt="text", page_id="s1"):
    return Hit(chunk_id=cid, doc_id="招股说明书2", page=page, block_type=bt,
               source_id=page_id, text=text, score=1.0)


# ---------- prompt ----------

def test_detect_lang():
    assert detect_lang("武汉力源信息技术股份有限公司") == "zh"
    assert detect_lang("What is the company name?") == "en"


def test_messages_require_answer_in_question_language():
    msgs = build_messages("Who is the legal representative?", [_h("a", "程家明")])
    joined = " ".join(m["content"] for m in msgs)
    assert "English" in joined or "same language" in joined.lower()


def test_messages_chinese_forces_chinese_answer():
    msgs = build_messages("法定代表人是谁", [_h("a", "程家明")])
    joined = " ".join(m["content"] for m in msgs)
    assert "中文" in joined


def test_context_includes_page_and_type_for_traceability():
    ctx = build_context([_h("a", "内容A", page=38, bt="image")])
    assert "38" in ctx
    assert "image" in ctx or "图像" in ctx
    assert "内容A" in ctx


def test_context_marks_image_source():
    ctx = build_context([_h("a", "组织结构图描述", page=38, bt="image")])
    assert "图像" in ctx or "image" in ctx


def test_citations_carry_page_type_and_id():
    cits = build_citations([_h("a", "x", page=38, bt="image", page_id="image#38#fig0")])
    assert cits[0]["page"] == 38
    assert cits[0]["block_type"] == "image"
    assert cits[0]["source_id"] == "image#38#fig0"


def test_messages_include_completeness_instruction_for_lists():
    msgs = build_messages("组织结构图中销售部有几个部门", [_h("a", "x")])
    joined = " ".join(m["content"] for m in msgs)
    assert "完整" in joined or "全部" in joined, "列表型问题必须要求完整枚举"


# ---------- llm ----------

def test_is_refusal_question_detects_offtopic():
    assert is_refusal_question("今天天气怎么样")
    assert is_refusal_question("")


def test_is_refusal_question_false_for_domain():
    assert not is_refusal_question("本次发行股数是多少")


# ---------- C1（终审）：离题门不得把真实题集判为离题 ----------

def test_no_real_question_is_classified_offtopic():
    """离题门是「明显离题」的粗筛，不是领域白名单：16 道中文题与 16 道英文题
    都必须是域内。原实现把中文白名单套到所有语言，实测英文题 8/16 被判离题
    （id 2/4/95/957/795/543/531/207），在进入 LLM 前就短路成固定拒答。"""
    from rag04.eval.questions import QUESTIONS

    bad_zh = [q.qid for q in QUESTIONS if is_refusal_question(q.question)]
    bad_en = [q.qid for q in QUESTIONS if is_refusal_question(q.question_en)]
    assert bad_zh == [], f"中文题不得被判离题：{bad_zh}"
    assert bad_en == [], f"英文题不得被判离题：{bad_en}"


@pytest.mark.parametrize("question", [
    "北京今天天气如何？",
    "What's the weather in Beijing?",
    "给我讲个笑话",
    "Tell me a joke",
])
def test_offtopic_questions_still_refused_in_both_languages(question):
    """离题门仍须拦下真正的离题问题（中英各自判定，不得因放宽而失效）。"""
    assert is_refusal_question(question) is True


def test_offtopic_gate_uses_question_language_hints():
    """显式传入语言时按该语言的提示词表判定（generate() 走的就是这条路径）。"""
    assert is_refusal_question("本次发行股数是多少", lang="zh") is False
    assert is_refusal_question("How many shares are being issued?", lang="en") is False
    assert is_refusal_question("北京今天天气如何？", lang="zh") is True
    assert is_refusal_question("What's the weather in Beijing?", lang="en") is True


def test_generate_english_question_reaches_llm_not_canned_refusal():
    """英文题必须真正调用 LLM（原实现 8/16 道英文题在此短路成固定拒答）。"""
    s = get_settings()
    cli = _fake_llm("The company plans to invest in four projects.")
    c = LLMClient(s, client=cli)
    ans = c.generate(
        "Which projects does Wuhan P&S Information Technology Co., Ltd. plan to "
        "invest the raised funds in?", [_h("a", "募集资金投资项目", page=22)])
    assert cli.chat.completions.create.call_count == 1, "英文域内题必须调用 LLM"
    assert ans.llm_backend == "deepseek" and ans.refused is False


def _fake_llm(content="答案内容"):
    cli = MagicMock()
    cli.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")]
    )
    return cli


def test_llm_generate_returns_answer_with_citations():
    s = get_settings()
    # 偏差说明：brief 原文为 "2,000"（千分位），与下方断言 "2000" 不匹配；
    # 断言本意是校验模型输出原样透传，故去掉千分位保持断言不变。
    c = LLMClient(s, client=_fake_llm("武汉力源本次发行2000万股"))
    ans = c.generate("本次发行股数是多少", [_h("a", "发行2000万股", page=10)])
    assert "2000" in ans.answer
    assert ans.citations
    assert ans.citations[0]["page"] == 10
    assert ans.lang == "zh"


def test_llm_falls_back_to_ollama_on_api_failure(monkeypatch):
    s = get_settings()
    bad = MagicMock()
    bad.chat.completions.create.side_effect = RuntimeError("api down")

    called = {"ollama": False}

    def fake_ollama(self, question, hits, deadline=None):
        called["ollama"] = True
        return "本地兜底答案"

    monkeypatch.setattr(LLMClient, "_generate_ollama", fake_ollama)
    c = LLMClient(s, client=bad)
    ans = c.generate("本次发行股数是多少", [_h("a", "发行2000万股", page=10)])
    assert called["ollama"]
    assert ans.llm_backend == "ollama"


def test_llm_strips_thinking_when_backend_rejects_it():
    """Task6 沿用项：服务端不识别 thinking 参数时剥离重试，不得中断生成。"""
    s = get_settings()
    err = RuntimeError("Unrecognized request argument supplied: extra_body")
    err.status_code = 400
    ok = MagicMock(choices=[MagicMock(message=MagicMock(content="剥离后答案"))])
    cli = MagicMock()
    cli.chat.completions.create.side_effect = [err, ok]
    c = LLMClient(s, client=cli)
    ans = c.generate("本次发行股数是多少", [_h("a", "发行2000万股", page=10)])
    assert ans.answer == "剥离后答案"
    assert ans.llm_backend == "deepseek"
    assert cli.chat.completions.create.call_count == 2


def test_llm_returns_retrieved_text_when_all_backends_fail(monkeypatch):
    """双后端都挂时返回检索原文，绝不空手而归。"""
    s = get_settings()
    bad = MagicMock()
    bad.chat.completions.create.side_effect = RuntimeError("api down")
    monkeypatch.setattr(LLMClient, "_generate_ollama",
                        lambda self, q, h, deadline=None:
                        (_ for _ in ()).throw(RuntimeError("ollama down")))
    c = LLMClient(s, client=bad)
    ans = c.generate("本次发行股数是多少", [_h("a", "发行2000万股", page=10)])
    assert ans.answer
    assert "2000" in ans.answer
    assert ans.llm_backend == "retrieval_only"


def test_llm_refuses_offtopic_question():
    s = get_settings()
    c = LLMClient(s, client=_fake_llm())
    ans = c.generate("今天天气怎么样", [_h("a", "x")])
    assert ans.refused


# ---------- RC6：文内拒答检测（id531 假阳性的真正入口） ----------

def test_in_domain_refusal_answer_marked_refused():
    """检索/生成失败时模型会答「根据文档内容无法确定」，必须置 refused，
    否则 coverage 的字面命中会把拒答算作答对（id531：命中「程家明」= 1.0）。"""
    s = get_settings()
    c = LLMClient(s, client=_fake_llm(
        "根据文档内容无法确定。\n\n检索片段中出现了多个不同的法定代表人姓名"
        "（陈爱民、程家明、胡家望），但均未明确指向该公司。引用来源：…"))
    ans = c.generate("武汉兴图新科电子股份有限公司法定代表人是谁？",
                     [_h("a", "程家明", page=22)])
    assert ans.refused is True
    assert "无法确定" in ans.answer, "拒答文本保留原样，不做替换"


def test_real_answer_with_hedge_later_is_not_refused():
    """先给出答案、后文补充说明「某细节无法确定」不算拒答。"""
    s = get_settings()
    c = LLMClient(s, client=_fake_llm(
        "根据检索片段，报告期内公司来自军用领域的收入占比分别为82.10%、97.31%、"
        "94.84%；其中2019年上半年数据未提供，无法确定全年口径。"))
    ans = c.generate("军用领域收入占比分别是多少？", [_h("a", "82.10%", page=129)])
    assert ans.refused is False


@pytest.mark.parametrize("text,expected,why", [
    # —— 真拒答（修复后真实评估里被标记的 7 题均为「根据文档内容无法确定」开头）——
    ("根据文档内容无法确定。\n\n检索片段中出现了多个不同的法定代表人姓名。",
     True, "id531 原型"),
    ("根据检索片段，无法确定该金额对应的科目。", True, "套话+无法确定"),
    ("未能找到与注册资本相关的记录。", True, "谓词+答案动词（无套话）"),
    ("无法给出确切的项目清单。", True, "谓词+答案动词（无套话）"),
    # —— 正确答语（复核指出的误判）——
    ("没有控制关系的关联方包括：融冰投资、听音投资、联众聚源。",
     False, "负向问句的正常答法：没有+名词，不是拒答"),
    ("无法回避的是，报告期内公司收入增长较快。",
     False, "「无法回避」是行文套语，不是拒答谓词"),
    ("根据检索片段，报告期内公司来自军用领域的收入占比分别为82.10%、97.31%。",
     False, "先给答案"),
    ("没有控制关系的关联方企业有融冰投资、听音投资。", False, "负向答语的另一写法"),
    # —— Fix pass 2：套式里的裸否定、`未` 开头的高频财报词、裸「未+答案动词」——
    ("根据检索片段，没有控制关系的关联方包括：融冰投资、听音投资。",
     False, "套式+没有+名词：仍是正常答法，不是拒答"),
    ("根据文档内容，未分配利润为1,000万元，占净资产的比例为5%。",
     False, "未分配利润：财报术语，不是「未+答案动词」"),
    ("根据招股意向书，未经审计的财务数据显示收入为500万元。",
     False, "未经审计：财报术语，不是拒答"),
    ("未找到明确指明该公司法定代表人的片段，检索片段中出现陈爱民、程家明、胡家望。",
     True, "裸「未+答案动词」（无根据套话）同样是拒答，否则 coverage 字面命中又成假阳性"),
    # —— Fix pass 3：否定词与答案动词之间的有界修饰语不得漏判 ——
    ("根据文档内容，无法准确判断。", True, "无法+准确+判断（套式）"),
    ("根据检索片段，不能完全确定该金额对应的科目。", True, "不能+完全+确定（套式）"),
    ("根据文档内容，没有直接披露该事项。", True, "没有+直接+披露（套式）"),
    ("未能立即找到该记录。", True, "未能+立即+找到（无套话）"),
    ("无法准确判断该公司的法定代表人是谁，检索片段中出现陈爱民、程家明、胡家望等多名姓名。",
     True, "RC6 原始假阳性的插入修饰语变体：不得因一词之差放行"),
])
def test_refusal_detector_table(text, expected, why):
    """判定表：裸否定（没有/无法+非答案动词）不得触发拒答。"""
    assert detect_answer_refusal(text) is expected, why


def test_english_refusal_answer_marked_refused():
    s = get_settings()
    c = LLMClient(s, client=_fake_llm(
        "Based on the provided excerpts, I cannot determine the legal representative."))
    # C1 后这里用真实评估题（原为回避离题门而加了 "company" 的改写问句）
    ans = c.generate("Who is the legal representative of Wuhan Xingtu Xinke "
                     "Electronics Co., Ltd.?", [_h("a", "x")])
    assert ans.refused is True
    assert ans.llm_backend == "deepseek", "域内英文题必须走到文本判定而非离题短路"


def test_llm_english_question_answered_in_english():
    s = get_settings()
    c = LLMClient(s, client=_fake_llm("The company raised 20 million shares."))
    ans = c.generate("How many shares were issued?", [_h("a", "2000万股", page=10)])
    assert ans.lang == "en"


# ---------- RC7：LLM 结构化「证据是否充分」字段（重建第二阶段） ----------

def test_messages_require_structured_answerability_field():
    for q in ("本次发行股数是多少", "How many shares are being issued by the company?"):
        joined = " ".join(m["content"] for m in build_messages(q, [_h("a", "x")]))
        assert "answerable" in joined, "系统提示必须要求结构化 answerable 字段"


@pytest.mark.parametrize("text,expected_bool,expected_answer", [
    ('{"answerable": false}\n根据文档内容无法确定。', False, "根据文档内容无法确定。"),
    ('{"answerable": true}\n本次发行股数为1,670万股。', True, "本次发行股数为1,670万股。"),
    ('```json\n{"answerable": true}\n```\n答案是1,670万股。', True, "答案是1,670万股。"),
    ('answerable: true\n本次发行股数为1,670万股。', True, "本次发行股数为1,670万股。"),
    ('证据充分性：不足\n根据文档内容无法确定。', False, "根据文档内容无法确定。"),
])
def test_parse_answerability_recognizes_marker_forms(text, expected_bool,
                                                    expected_answer):
    val, answer = parse_answerability(text)
    assert val is expected_bool
    assert answer == expected_answer, "标记行必须从答案正文中剥离"
    assert "answerable" not in answer and "证据充分性" not in answer


def test_parse_answerability_absent_returns_none_and_keeps_text():
    val, answer = parse_answerability("本次发行股数为1,670万股。")
    assert val is None and answer == "本次发行股数为1,670万股。"


def test_generate_structured_field_is_primary_judgement():
    """有结构化字段时以它为主判据：正文没有拒答套话也不会被误判为拒答。"""
    s = get_settings()
    c = LLMClient(s, client=_fake_llm('{"answerable": true}\n本次发行股数为1,670万股。'))
    ans = c.generate("本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    assert ans.refused is False
    assert ans.answerable is True
    assert ans.refusal_source == "structured"
    assert "answerable" not in ans.answer, "结构化标记不得进入引用/答案正文"


def test_generate_structured_false_marks_refused():
    s = get_settings()
    c = LLMClient(s, client=_fake_llm('{"answerable": false}\n根据文档内容无法确定。'))
    ans = c.generate("本次发行股数是多少", [_h("a", "x", page=22)])
    assert ans.refused is True and ans.answerable is False
    assert ans.refusal_source == "structured"


def test_generate_regex_fallback_when_marker_absent():
    """非 JSON 后端（模型未产出标记）仍走 RC6 正则兜底。"""
    s = get_settings()
    c = LLMClient(s, client=_fake_llm("根据文档内容无法确定。"))
    ans = c.generate("本次发行股数是多少", [_h("a", "x", page=22)])
    assert ans.refused is True and ans.answerable is None
    assert ans.refusal_source == "regex"


def test_generate_regex_still_nets_prose_refusal_when_field_claims_true():
    """单向兜底：字段称可答、正文却是拒答时必须仍判拒答，不得放回 id531 假阳性。"""
    s = get_settings()
    c = LLMClient(s, client=_fake_llm(
        '{"answerable": true}\n根据文档内容无法确定，片段中出现陈爱民、程家明、胡家望。'))
    ans = c.generate("武汉兴图新科电子股份有限公司法定代表人是谁？",
                     [_h("a", "程家明", page=22)])
    assert ans.refused is True
    assert ans.refusal_source == "structured+regex"


def test_generate_retrieval_only_backend_marks_source(monkeypatch):
    """双后端全挂 → 返回检索原文，判定来源如实记为 retrieval_only（正则口径）。

    C5（终审）刻意改判：兜底原文**不是答案**——原始片段里恰好含要点时
    coverage 会把它判对（id531 缺陷类），故源头即记 refused=True（不计正确）。
    本测试原先断言 refused is False，此处是有意更新而非删除该断言。
    """
    s = get_settings()
    bad = MagicMock()
    bad.chat.completions.create.side_effect = RuntimeError("api down")
    monkeypatch.setattr(LLMClient, "_generate_ollama",
                        lambda self, q, h, deadline=None:
                        (_ for _ in ()).throw(RuntimeError("ollama down")))
    c = LLMClient(s, client=bad)
    ans = c.generate("本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    assert ans.llm_backend == "retrieval_only"
    assert ans.answerable is None and ans.refusal_source == "retrieval_only"
    assert ans.refused is True, "检索原文兜底不得计为答案（可能被 coverage 字面判对）"
    assert "1,670万股" in ans.answer, "仍不空手而归：原文照旧返回"


# ---------- C4（终审）：降级链时限 ----------

def _slow_client(delay: float, *, answer: str = "慢但可用的答案"):
    """模拟「按 timeout 阻塞」的后端：timeout 不足则抛超时，否则睡够后返回。"""
    cli = MagicMock()

    def create(**kwargs):
        tmo = kwargs.get("timeout")
        if tmo is not None and tmo < delay:
            raise TimeoutError(f"timeout {tmo:.3f}s < {delay:.3f}s")
        import time as _t
        _t.sleep(delay)
        return MagicMock(choices=[MagicMock(message=MagicMock(content=answer))])

    cli.chat.completions.create.side_effect = create
    return cli


def test_hung_backend_degrades_within_chain_deadline(monkeypatch):
    """挂死的后端必须在链时限内降级到检索原文，不得阻塞 50 倍预算。

    原实现：API 30s + Ollama 120s = 挂死窗口 ~150s。现改为共享 deadline
    （min(llm_timeout_s, _CHAIN_DEADLINE_S)）。
    """
    import time as _t

    import rag04.generate.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_CHAIN_DEADLINE_S", 0.4)
    s = get_settings()
    c = LLMClient(s, client=_slow_client(5.0))       # 永远比剩余预算慢
    t0 = _t.perf_counter()
    ans = c.generate("本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    elapsed = _t.perf_counter() - t0

    assert elapsed < 2.0, f"链时限未被遵守，阻塞 {elapsed:.2f}s"
    assert ans.llm_backend == "retrieval_only"
    assert ans.refused is True


def test_slow_but_working_backend_still_answers(monkeypatch):
    """时限内返回的慢后端照常作答（收紧不得误杀正常慢响应）。"""
    import rag04.generate.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_CHAIN_DEADLINE_S", 5.0)
    s = get_settings()
    c = LLMClient(s, client=_slow_client(0.3))
    ans = c.generate("本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    assert ans.llm_backend == "deepseek"
    assert "慢但可用" in ans.answer


def test_ollama_timeout_is_bounded_by_remaining_budget(monkeypatch):
    """API 用尽预算后不再进入 Ollama（不叠加第二个超时窗口）。"""
    s = get_settings()
    seen = {"ollama": 0}

    def fake_ollama(self, question, hits, deadline=None):
        seen["ollama"] += 1
        return "本地兜底答案"

    monkeypatch.setattr(LLMClient, "_generate_ollama", fake_ollama)
    bad = MagicMock()
    bad.chat.completions.create.side_effect = RuntimeError("api down")
    c = LLMClient(s, client=bad)
    ans = c.generate("本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    # 正常路径（API 快速失败）仍走 Ollama：剩余预算充足
    assert seen["ollama"] == 1 and ans.llm_backend == "ollama"

    # 模拟 API 耗尽预算：deadline 已过 → 跳过 Ollama，直接检索原文
    import time as _t

    import rag04.generate.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_CHAIN_DEADLINE_S", 0.01)
    calls = {"n": 0}

    def slow_fail(**kwargs):
        _t.sleep(0.2)
        raise RuntimeError("api hang")

    bad2 = MagicMock()
    bad2.chat.completions.create.side_effect = slow_fail
    monkeypatch.setattr(LLMClient, "_generate_ollama",
                        lambda self, q, h, deadline=None: calls.__setitem__("n", calls["n"] + 1))
    ans2 = LLMClient(s, client=bad2).generate(
        "本次发行股数是多少", [_h("a", "1,670万股", page=22)])
    assert calls["n"] == 0, "预算耗尽后不得再进 Ollama"
    assert ans2.llm_backend == "retrieval_only"
