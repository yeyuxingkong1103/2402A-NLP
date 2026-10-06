# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
工单05 的纯函数测试。

【原则与 tests/test_pipeline.py 一致】**不碰 Milvus、不碰 Ollama**：
Query 理解的规则层是纯字符串变换，会话存储是内存结构 —— 都能离线断言。
唯一需要"模型"的两条用假 client 注入（验的是**调用契约**，不是模型质量）。

跑法：pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.doc_profiles import canonical_entity                    # noqa: E402
from app.core.query_understanding import (                             # noqa: E402
    LLMRewriter, QueryUnderstanding, compose, detect_subject_only_ellipsis,
    find_entity_span, has_pronoun, intent_template_of, needs_llm_fallback,
    next_focus_entity, replace_pronouns, resolve_rules, slotize)
from app.core.session import COMPANY_SLOT, SessionStore, Turn          # noqa: E402

QUESTION_FILE = ROOT / "eval" / "questions.json"

# 工单05 原文 5 轮剧本
Q1 = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
Q2 = "他参与的哪个工程荣获了国家科技进步一等奖？"
Q3 = "这个公司的法定代表人是谁？"
Q4 = "那武汉力源信息技术股份有限公司呢？"
Q5 = ("武汉力源信息技术股份有限公司组织结构图中，"
      "哪个销售部的销售处最多？有哪些销售处？")

XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"
# 逐字取自单轮题库（gold 与它们相同 ⇒ 检索路径逐字节等价）
ID_795 = "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"
ID_531 = "武汉兴图新科电子股份有限公司法定代表人是谁？"


def _run_script(questions, store=None):
    """按会话顺序跑规则层，返回 [(method, rewritten)]，并逐轮写回会话。"""
    store = store or SessionStore()
    sid = store.get_or_create(None).session.id   # 必须先建会话，否则 append 无处可写
    out = []
    for q in questions:
        rr = resolve_rules(q, focus_entity=store.focus_entity(sid),
                           intent_template=store.intent_template(sid))
        assert rr is not None, f"规则弃权：{q}"
        out.append((rr.method, rr.rewritten))
        doc_key = find_entity_span(rr.rewritten)
        store.append(sid, Turn(
            question=q, rewritten=rr.rewritten,
            focus_entity=next_focus_entity(resolved_entity=rr.resolved_entity,
                                           doc_key=doc_key.doc_key if doc_key else "",
                                           previous=store.focus_entity(sid)),
            intent_template=intent_template_of(rr.rewritten)))
    return out


# ======================================================================
# T1 —— 回归闸门：既有 16 题在多轮语境下**一个字都不许改**
# ======================================================================
def test_no_rewrite_for_all_delivered_questions():
    qs = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))["questions"]
    assert qs, "题库为空"
    for q in qs:
        rr = resolve_rules(q["question"])
        assert rr is not None and rr.method == "none", f"id={q['id']} 被改写"
        assert rr.rewritten == q["question"], f"id={q['id']} 文本被改动"


# ======================================================================
# T2/T3/T4/T5 —— 5 轮剧本的改写结果
# ======================================================================
def test_five_turn_script_method_and_rewrites():
    got = _run_script([Q1, Q2, Q3, Q4, Q5])
    assert [m for m, _ in got] == ["none", "rule-coref", "rule-coref", "rule-switch", "none"]
    assert got[1][1] == ID_795, "「他」应消解成 id=795 的题面（逐字）"
    assert got[2][1] == ID_531, "「这个公司」应消解成 id=531 的题面（逐字）"
    assert got[3][1] == f"{LIYUAN}法定代表人是谁？", "Q4 应继承上一轮问点并换公司"
    assert got[4][1] == Q5, "完整问题必须原样不动"


def test_topic_switch_reports_carried_intent():
    store = SessionStore()
    sid = store.get_or_create(None).session.id
    for q in (Q1, Q2, Q3):
        rr = resolve_rules(q, focus_entity=store.focus_entity(sid),
                           intent_template=store.intent_template(sid))
        store.append(sid, Turn(question=q, rewritten=rr.rewritten,
                               focus_entity=next_focus_entity(
                                   resolved_entity=rr.resolved_entity, doc_key="xingtu"),
                               intent_template=intent_template_of(rr.rewritten)))
    rr = resolve_rules(Q4, focus_entity=store.focus_entity(sid),
                       intent_template=store.intent_template(sid))
    assert rr.method == "rule-switch"
    assert rr.carried_intent == "法定代表人是谁？"
    assert rr.resolved_entity == LIYUAN


# ======================================================================
# T6 —— 省略检测器：只认「剥完只剩空」，不误判完整问题
# ======================================================================
@pytest.mark.parametrize("q,expect_none", [
    (Q4, False), (Q2, True), (Q3, True), (Q1, True), (Q5, True),
    ("武汉力源信息技术股份有限公司呢？", False),
    ("那力源呢？", False),
])
def test_ellipsis_detector(q, expect_none):
    got = detect_subject_only_ellipsis(q)
    assert (got is None) == expect_none, f"{q!r} → {got!r}"


def test_ellipsis_detector_ignores_bare_company_only():
    """只点名公司、不带其它内容才算省略；带内容的不继承问点。"""
    assert detect_subject_only_ellipsis(Q4) == "liyuan"
    assert detect_subject_only_ellipsis("那力源呢？") == "liyuan"
    assert detect_subject_only_ellipsis(Q5) is None


# ======================================================================
# T7/T8 —— 槽位模板（英文介词不被削掉；实体匹配容忍空格）
# ======================================================================
def test_slotize_keeps_english_preposition():
    q = "What is the registered capital of Wuhan Xingtu Xinke Electronics Co.,Ltd.?"
    tmpl, matched = slotize(q)
    assert matched == "Wuhan Xingtu Xinke Electronics Co.,Ltd."
    assert tmpl == f"What is the registered capital of {COMPANY_SLOT}?"
    # 代入另一家 → 语法完整（abstract_query 的残句法会削成 "…of?" 悬空）
    out = compose(tmpl, "Wuhan P&S Information Technology Co.,Ltd.")
    assert out.endswith("of Wuhan P&S Information Technology Co.,Ltd.?")


def test_entity_match_tolerates_whitespace():
    """配置里是 'Co.,Ltd.'，前端示例写作 'Co., Ltd.' —— 必须都能匹配。"""
    span = find_entity_span("What about Wuhan P&S Information Technology Co., Ltd.?")
    assert span is not None and span.doc_key == "liyuan"


def test_english_topic_switch_via_slot_template():
    store = SessionStore()
    got = _run_script([
        "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co.,Ltd.?",
        "What about Wuhan P&S Information Technology Co.,Ltd.?"], store)
    assert got[0][0] == "none"
    assert got[1][0] == "rule-switch"
    assert got[1][1] == ("Who is the legal representative of "
                         "Wuhan P&S Information Technology Co.,Ltd.?")


# ======================================================================
# T9 —— 实体 + 代词同现的歧义句：不猜，原样透传
# ======================================================================
def test_entity_with_pronoun_is_left_alone():
    q = "武汉兴图新科电子股份有限公司和武汉力源信息技术股份有限公司，他的法定代表人是谁？"
    rr = resolve_rules(q, focus_entity=XINGTU, intent_template="")
    assert rr is not None and rr.method == "none" and rr.rewritten == q


# ======================================================================
# T10/T11 —— 代词防误伤
# ======================================================================
@pytest.mark.parametrize("text", ["其中营业收入为100万元", "其他应收款", "尤其重要",
                                  "及其子公司", "其实", "其余"])
def test_qi_pronoun_does_not_break_words(text):
    assert replace_pronouns(text, XINGTU) == text, "「其」的固定搭配被误替换了"


@pytest.mark.parametrize("text", ["他人不得干预", "其它应收款"])
def test_ta_pronoun_does_not_match_others(text):
    assert replace_pronouns(text, XINGTU) == text


def test_real_pronouns_are_replaced():
    assert replace_pronouns("它的法定代表人是谁？", XINGTU) == f"{XINGTU}法定代表人是谁？"
    assert has_pronoun("该公司注册资本是多少？") is True
    assert has_pronoun("报告期内公司收入是多少？") is False


def test_pronoun_eats_following_de():
    """代词后的「的」要一起吃掉 —— 否则与题面差一个字，查询串就变了。"""
    assert replace_pronouns("他的法定代表人是谁？", LIYUAN) == f"{LIYUAN}法定代表人是谁？"
    assert replace_pronouns("这个公司的法定代表人是谁？", LIYUAN) == f"{LIYUAN}法定代表人是谁？"


# ======================================================================
# T12/T13/T14 —— 焦点实体
# ======================================================================
def test_short_alias_detected_but_not_in_doc_profiles():
    """短别名只在 resolver 内部识别 —— **不写进** doc_profiles.entity_names。

    写进去会改变 `abstract_query()` 的行为（抽象掉更多字）→ 改变已交付 16 题的
    查询视角 → 直接威胁「16/16 无回归」。所以这里既断言它能被识别，
    也断言 doc_profiles 的实体表里没有它。
    """
    span = find_entity_span("那力源呢？")
    assert span is not None and span.doc_key == "liyuan"
    from app.core.doc_profiles import doc_ids_by_entity
    assert "力源" not in doc_ids_by_entity()
    assert "兴图新科" in doc_ids_by_entity()   # 这个是原有的，不算新增


def test_canonical_entity_is_full_name():
    assert canonical_entity("xingtu") == XINGTU
    assert canonical_entity("liyuan") == LIYUAN
    assert canonical_entity("liyuan", english=True) == \
        "Wuhan P&S Information Technology Co.,Ltd."
    assert canonical_entity("xingtu", english=True) == \
        "Wuhan Xingtu Xinke Electronics Co.,Ltd."


def test_next_focus_prefers_resolved_then_doc_then_previous():
    assert next_focus_entity(resolved_entity=LIYUAN, doc_key="xingtu") == LIYUAN
    assert next_focus_entity(doc_key="xingtu", previous=LIYUAN) == XINGTU
    # 一句话点了两家公司 → 路由不过滤（doc_key=""）→ 焦点保持不动
    assert next_focus_entity(doc_key="", previous=XINGTU) == XINGTU


def test_focus_entity_after_script_is_last_company():
    store = SessionStore()
    sid = store.get_or_create(None).session.id
    focus = ""
    for q in (Q1, Q3, Q4):
        rr = resolve_rules(q, focus_entity=focus, intent_template=store.intent_template(sid))
        focus = next_focus_entity(resolved_entity=rr.resolved_entity,
                                  doc_key=(find_entity_span(rr.rewritten).doc_key),
                                  previous=focus)
        store.append(sid, Turn(question=q, rewritten=rr.rewritten, focus_entity=focus,
                               intent_template=intent_template_of(rr.rewritten)))
    assert focus == LIYUAN, "话题切到力源后，焦点必须变成力源的规范全称"


# ======================================================================
# T15/T16 —— LLM 兜底
# ======================================================================
class _FakeClient:
    def __init__(self, reply='{"rewritten": "改写后的问题", "reason": "代词"}', boom=False):
        self.reply, self.boom, self.calls = reply, boom, []

    async def chat(self, messages, **kw):
        self.calls.append(kw)
        if self.boom:
            raise RuntimeError("模型不可用")
        return self.reply


def test_llm_failed_falls_back_to_original():
    import asyncio
    rw = LLMRewriter(client=_FakeClient(boom=True))
    res = asyncio.run(rw.rewrite("他说的对吗？", focus_entity=XINGTU))
    assert res.method == "llm-failed"
    assert res.rewritten == "他说的对吗？", "兜底失败必须回退原问题，不能静默丢弃"


def test_llm_rewrite_call_contract():
    """契约：JSON 格式 + 温度 0，且**不得**传 num_ctx（改它会触发模型重载 7.2s）。"""
    import asyncio
    c = _FakeClient()
    res = asyncio.run(LLMRewriter(client=c).rewrite("他说的对吗？", focus_entity=XINGTU))
    assert res.method == "llm" and res.rewritten == "改写后的问题"
    kw = c.calls[0]
    assert kw.get("fmt") == "json"
    assert kw.get("temperature") == 0.0
    assert "num_ctx" not in kw or kw["num_ctx"] is None


def test_llm_bad_json_is_reported_not_silently_used():
    import asyncio
    res = asyncio.run(LLMRewriter(client=_FakeClient(reply="我不太确定")).rewrite("他呢？"))
    assert res.method == "llm-failed" and res.rewritten == "他呢？"


def test_rules_first_never_calls_llm():
    """规则能解时不许调模型 —— 这是「规则优先」的硬约束。"""
    import asyncio
    c = _FakeClient()
    qu = QueryUnderstanding(llm_enabled=True, rewriter=LLMRewriter(client=c))
    res = asyncio.run(qu.rewrite(Q2, focus_entity=XINGTU))
    assert res.method == "rule-coref"
    assert c.calls == [], "规则命中却调了模型"


# ======================================================================
# T17/T18/T19 —— 会话存储
# ======================================================================
def test_session_ttl_expiry_is_reported():
    st = SessionStore(ttl=0.4)
    lk = st.get_or_create("a")
    assert lk.reason == "new" and not lk.expired
    time.sleep(0.6)
    again = st.get_or_create("a")
    assert again.expired and again.reason == "expired", \
        "过期必须是**可观测**的（演示 TTL 靠它）"


def test_session_lru_is_bounded():
    st = SessionStore(max_sessions=2)
    for sid in ("a", "b", "c"):
        st.get_or_create(sid)
    assert st.stats()["n_sessions"] == 2
    assert st.get("a") is None, "最旧的会话应被 LRU 逐出"


def test_history_respects_prompt_budget():
    st = SessionStore()
    sid = st.get_or_create(None).session.id
    st.append(sid, Turn(question="q", rewritten="x" * 100, focus_entity=XINGTU))
    msgs, dropped = st.history_messages(sid, budget_chars=2600)
    assert msgs == [] and dropped is True, "超预算必须丢轮并如实上报"
    msgs2, dropped2 = st.history_messages(sid, budget_chars=0)
    assert len(msgs2) == 1 and dropped2 is False


def test_history_uses_rewritten_question():
    st = SessionStore()
    sid = st.get_or_create(None).session.id
    st.append(sid, Turn(question=Q2, rewritten=ID_795, focus_entity=XINGTU))
    msgs, _ = st.history_messages(sid)
    assert msgs[0]["content"] == ID_795, "history 里必须是改写后的问题，否则模型看不懂「他」"
    assert msgs[0]["content"] != Q2


def test_history_drops_whole_turns_not_half():
    """丢轮要整轮丢 —— 只丢 user 留下 assistant 会拼出没有提问的回答。"""
    st = SessionStore()
    sid = st.get_or_create(None).session.id
    st.append(sid, Turn(question="q1", rewritten="a" * 50, answer="A" * 50, focus_entity=XINGTU))
    st.append(sid, Turn(question="q2", rewritten="b" * 50, answer="B" * 50, focus_entity=XINGTU))
    msgs, dropped = st.history_messages(sid, budget_chars=2600, with_answers=True)
    assert dropped is True
    assert [m["role"] for m in msgs] == [] or [m["role"] for m in msgs][0] == "user"


def test_intent_template_lookback_skips_slotless_turns():
    st = SessionStore()
    sid = st.get_or_create(None).session.id
    st.append(sid, Turn(question="q1", rewritten=ID_531, intent_template="⟦公司⟧法定代表人是谁？"))
    st.append(sid, Turn(question="q2", rewritten="今年收入多少？", intent_template=""))
    assert st.intent_template(sid, lookback=2) == "⟦公司⟧法定代表人是谁？"
    assert st.intent_template(sid, lookback=1) == ""


def test_session_isolation_between_ids():
    st = SessionStore()
    a = st.get_or_create("A").session.id
    b = st.get_or_create("B").session.id
    st.append(a, Turn(question="q", rewritten="r", focus_entity=XINGTU))
    assert st.focus_entity(a) == XINGTU
    assert st.focus_entity(b) == "", "不同会话之间不许串上下文"


# ======================================================================
# T20/T21 —— schemas 的字段同步（专治 pydantic extra="ignore" 静默丢字段）
# ======================================================================
def test_chat_schemas_declare_multiturn_fields():
    from app.schemas import ChatResponse, ChatRequest, EvalItemOut
    assert "session_id" in ChatRequest.model_fields
    for f in ("session_id", "turn_index", "rewritten_question", "rewrite_method",
              "rewrite_evidence", "focus_entity", "carried_intent",
              "session_expired", "history_chars", "history_dropped", "rewrite_ms"):
        assert f in ChatResponse.model_fields, f"ChatResponse 缺 {f}（会被静默丢弃）"
    # 前端在渲染、但没有与 EvalItem 同步的字段（工单05 顺手补的）
    for f in ("reference_answer", "evidence_page", "error", "no_rag_rule_hit",
              "routed_doc", "route_matched", "route_fallback"):
        assert f in EvalItemOut.model_fields, f"EvalItemOut 缺 {f}"


# ======================================================================
# T22 —— 规则弃权与 LLM 触发条件
# ======================================================================
def test_needs_llm_fallback_conditions():
    assert needs_llm_fallback(Q2) is True          # 有代词、无焦点 → 值得兜底
    assert needs_llm_fallback(Q1) is False         # 点名了公司 → 规则已定
    assert needs_llm_fallback(Q5) is False
    assert needs_llm_fallback("那它呢？") is True
    assert needs_llm_fallback("什么是招股说明书？") is False


def test_resolve_rules_returns_none_when_abstaining():
    assert resolve_rules(Q2, focus_entity="") is None, "无焦点实体时代词消解应弃权"
    assert resolve_rules(Q2, focus_entity=XINGTU) is not None


def test_empty_question_is_safe():
    for q in ("", "   ", None):
        rr = resolve_rules(q or "")
        assert rr is not None and rr.rewritten == (q or "")


# ======================================================================
# T23 —— 判据的「同义写法」（`|`），工单05 顺手修掉的判据缺陷
# ======================================================================
def test_keyword_variants_accept_both_writings():
    from app.core.evaluator import keyword_hits, rule_hit
    kws = ["4个部门|四个部门", "6个销售处"]
    assert rule_hit("销售部由4个部门构成，大客户销售部有6个销售处构成。", kws, "all")[0]
    assert rule_hit("销售部由…和国际贸易部四个部门构成。…共6个销售处构成。", kws, "all")[0]
    # 漏答一项仍必须判失败（不能因为支持同义写法就放松）
    ok, _ = rule_hit("大客户销售部有6个销售处构成。", kws, "all")
    assert ok is False
    assert keyword_hits("四个部门", ["4个部门|四个部门"]) == ["4个部门|四个部门"]
