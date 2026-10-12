# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""L2/L3/L4：受约束改写、校验器、规则兜底。

关键：注入假 LLM，把两个**实测失败模式**固化成回归测试：
  · 3b 把「他」换成「这个公司」——代词换代词的无效替换
  · 0.5b 把指令原样吐回——复述指令
"""
import pytest

from src import coref, dialogue
from src.coref import Span


XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"


def _state_with(*questions) -> dialogue.DialogueState:
    st = dialogue.DialogueState()
    for i, q in enumerate(questions, 1):
        dialogue.append_turn(st, dialogue.Turn(idx=i, raw_question=q))
    return st


def _fake_llm(reply: str):
    def _f(messages, **kw):
        return {"answer": reply}
    return _f


def _four_turn_state() -> dialogue.DialogueState:
    """工单 4 轮验收序列的原始问题；末轮点名武汉力源 → focus_company 为力源。"""
    return _state_with(
        f"报告期内，{XINGTU}来自军用领域的收入分别是多少？",
        "他参与的哪个工程荣获了国家科技进步一等奖？",
        "这个公司的法定代表人是谁？",
        f"那{LIYUAN}呢？",
    )


# --- L0 直通 ---------------------------------------------------------------
def test_resolve_passthrough_does_not_call_llm():
    st = dialogue.DialogueState()
    called = []
    r = coref.resolve(f"{XINGTU}的法定代表人是谁？", st,
                      llm_fn=lambda m, **k: called.append(m) or {"answer": "x"})
    assert r.mode == "passthrough"
    assert r.resolved == f"{XINGTU}的法定代表人是谁？"
    assert called == [], "直通轮不得调用 LLM（延迟预算的关键）"


# --- L4 规则兜底（无 LLM）---------------------------------------------------
def test_rule_coref_replaces_pronoun():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("他参与的哪个工程获奖了？", st)[0]
    assert coref.rule_fallback("他参与的哪个工程获奖了？", span, st) == \
        f"{XINGTU}参与的哪个工程获奖了？"


def test_rule_ellipsis_uses_last_aspect():
    """Q4 的规则兜底路径：用上一轮的 aspect 补全。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？",
                     "这个公司的法定代表人是谁？")
    assert st.last_aspect == "法定代表人"
    span = coref.detect_spans(f"那{LIYUAN}呢？", st)[0]
    assert coref.rule_fallback(f"那{LIYUAN}呢？", span, st) == \
        f"{LIYUAN}的法定代表人是谁？"


# --- 校验器：两个实测失败模式 ------------------------------------------------
def test_validator_rejects_pronoun_for_pronoun():
    """实测失败模式 1：3b 把「他」换成「这个公司」。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    ok, why = coref.validate_rewrite("这个公司参与的哪个工程获奖了？", q, span, st)
    assert ok is False
    assert "残留代词" in why or "实体" in why


def test_validator_rejects_instruction_paraphrase_via_containment():
    """实测失败模式 2 的**释义版**：措辞不同、4-gram 重合仅 0.357，靠实体包含判据拒绝。

    此例pin的是「改写必须含候选实体」这条防线，不是 n-gram 复述规则
    （n-gram 规则由下面的 test_validator_rejects_instruction_echo 专门 pin）。
    """
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    echo = "把下面括号里的代词替换成正确的公司全称，其余字不变，只输出结果"
    ok, why = coref.validate_rewrite(echo, q, span, st)
    assert ok is False
    assert "实体" in why, f"该释义版应由实体包含判据拒绝，实际：{why}"


def test_validator_rejects_instruction_echo():
    """实测失败模式 2：0.5b 原样吐回指令。

    取 `_JUDGE_SYS_ZH` 前缀并**尾接公司全称**，使实体包含判据放行——
    这样唯一能拒绝它的就是 n-gram 复述规则；若该守卫被删除或阈值失效，本测试必红。
    """
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    echo = coref._JUDGE_SYS_ZH[:80] + XINGTU
    assert len(echo) <= coref.config.REWRITE_MAX_CHARS, "样本不能先被超长判据拦下"
    ok, why = coref.validate_rewrite(echo, q, span, st)
    assert ok is False
    assert "疑似复述指令" in why, f"必须由 n-gram 复述规则拒绝，实际：{why}"


def test_validator_rejects_empty_and_overlong():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    assert coref.validate_rewrite("", q, span, st)[0] is False
    assert coref.validate_rewrite("他" + "啊" * 500, q, span, st)[0] is False


def test_validator_accepts_good_rewrite():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    ok, _ = coref.validate_rewrite(f"{XINGTU}参与的哪个工程获奖了？", q, span, st)
    assert ok is True


# --- L2/L3 走 LLM（注入假模型）---------------------------------------------
def test_validator_rejects_non_minimal_edit_with_hallucinated_entity():
    """Task9 实测失败模式：英文问句被塞进一个**不存在的公司名**（Microsoft）。

    该输出「含候选实体 + 无残留代词 + 长度合规」，旧校验全票放行；实测后果是检索
    答不出（三方评测里唯一输给单轮基线的一轮）。受约束改写只允许替换代词，
    其余文字必须原样保留（_GEN_SYS 规则 1），故非最小编辑必须被拒绝。
    """
    st = _state_with(f"{XINGTU}的法定代表人是谁？")
    q = ("Which project did he participate in that won the National Science "
         "and Technology Progress Award?")
    span = coref.detect_spans(q, st)[0]
    assert span.kind == "person"
    bad = ("Which project did Microsoft participate in that won t"
           f"{XINGTU} National Science and Technology Progress Award?")
    ok, why = coref.validate_rewrite(bad, q, span, st)
    assert ok is False, "塞入无关实体且吃掉原文的改写必须被拒绝"
    assert "最小编辑" in why
    good = (f"Which project did {XINGTU} participate in that won the National "
            "Science and Technology Progress Award?")
    assert coref.validate_rewrite(good, q, span, st)[0] is True


def test_resolve_coref_llm_accepts_valid_output():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    r = coref.resolve("他参与的哪个工程荣获了国家科技进步一等奖？", st,
                      llm_fn=_fake_llm(f"{XINGTU}参与的哪个工程荣获了国家科技进步一等奖？"))
    assert r.mode == "coref_llm"
    assert XINGTU in r.resolved
    assert r.degraded is False


def test_resolve_falls_back_when_llm_returns_junk():
    """LLM 坏输出 → 必须降级到规则，绝不能把坏改写放行。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    r = coref.resolve("他参与的哪个工程荣获了国家科技进步一等奖？", st,
                      llm_fn=_fake_llm("这个公司参与的哪个工程荣获了国家科技进步一等奖？"))
    assert r.degraded is True
    assert r.mode == "coref_rule"
    assert r.resolved.startswith(XINGTU)


def test_resolve_falls_back_when_llm_raises():
    def _boom(messages, **kw):
        raise TimeoutError("llm timeout")
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    r = coref.resolve("他参与的哪个工程获奖了？", st, llm_fn=_boom)
    assert r.degraded is True
    assert r.resolved.startswith(XINGTU)


def test_resolve_ellipsis_with_llm():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？",
                     "这个公司的法定代表人是谁？")
    r = coref.resolve(f"那{LIYUAN}呢？", st,
                      llm_fn=_fake_llm(f"{LIYUAN}的法定代表人是谁？"))
    assert LIYUAN in r.resolved
    assert "法定代表人" in r.resolved


def test_resolve_ellipsis_falls_back_to_rule():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？",
                     "这个公司的法定代表人是谁？")
    r = coref.resolve(f"那{LIYUAN}呢？", st, llm_fn=_fake_llm("他不理解"))
    assert r.degraded is True
    assert r.resolved == f"{LIYUAN}的法定代表人是谁？"


def test_resolve_no_history_no_guess():
    """指代无法解析时必须**不猜**（硬性要求 8）。"""
    st = dialogue.DialogueState()
    r = coref.resolve("他的法定代表人是谁？", st, llm_fn=_fake_llm("x"))
    assert r.mode == "unresolved"
    assert r.resolved == ""


def test_resolve_use_llm_false_is_pure_rules():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    r = coref.resolve("他参与的哪个工程获奖了？", st, use_llm=False)
    assert r.mode == "coref_rule"
    assert r.resolved.startswith(XINGTU)


# --- 无可继承模板的省略句：不猜 ----------------------------------------------
def test_resolve_ellipsis_without_template_is_unresolved():
    """硬性要求：省略句没有 last_aspect 可继承时，LLM 不得凭空编出信息点。"""
    st = dialogue.DialogueState()          # 首轮，last_aspect 为空
    r = coref.resolve(f"那{LIYUAN}呢？", st,
                      llm_fn=_fake_llm(f"{LIYUAN}的注册资本是多少？"))
    assert r.mode == "unresolved"
    assert r.resolved == ""


# --- Fix wave 1：L4 候选选择必须「焦点优先」 ---------------------------------
def test_pronoun_rule_path_prefers_focus_company():
    """4 轮序列末轮点名力源 → 无 LLM 的规则兜底必须解析到力源，而非最早提过的兴图。"""
    st = _four_turn_state()
    assert st.companies_seen() == [XINGTU, LIYUAN]
    assert st.focus_company == LIYUAN
    r = coref.resolve("他参与的哪个工程获奖了？", st, use_llm=False)
    assert r.mode == "coref_rule"
    assert r.resolved.startswith(LIYUAN)
    assert XINGTU not in r.resolved


def test_pronoun_exception_path_prefers_focus_company():
    """LLM 异常降级到 L4 的同一条路径也必须落在当前焦点公司上。"""
    st = _four_turn_state()

    def _boom(messages, **kw):
        raise TimeoutError("llm timeout")

    r = coref.resolve("他参与的哪个工程获奖了？", st, llm_fn=_boom)
    assert r.mode == "coref_rule" and r.degraded is True
    assert r.resolved.startswith(LIYUAN)
    assert XINGTU not in r.resolved


# --- Fix wave 1：改写调用受 ≤3s 预算约束 -------------------------------------
def test_default_llm_forwards_rewrite_budget(monkeypatch):
    """_default_llm 必须把改写超时与短生成长度透传给 llm.chat。"""
    from src import llm

    seen = {}

    def _capture(messages, **kw):
        seen.update(kw)
        return {"answer": "x"}

    monkeypatch.setattr(llm, "chat", _capture)
    coref._default_llm([{"role": "user", "content": "x"}], model="m")
    assert seen["timeout"] == coref.config.REWRITE_TIMEOUT_S
    assert seen["num_predict"] == coref.config.REWRITE_NUM_PREDICT
    assert seen["num_predict"] <= 128, "改写只输出一句话，必须限制生成长度"


# --- Fix wave 1：跨 kind 的残留代词同样要拒绝 --------------------------------
def test_validator_rejects_cross_kind_leftover_pronoun():
    """person span 的输出里残留「这个公司」仍是上下文依赖的坏改写。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    q = "他参与的哪个工程获奖了？"
    span = coref.detect_spans(q, st)[0]
    assert span.kind == "person"
    bad = f"{XINGTU}的子公司，这个公司参与的哪个工程获奖了？"
    ok, why = coref.validate_rewrite(bad, q, span, st)
    assert ok is False
    assert "残留代词" in why


# --- Fix wave 1：L2 判定路径（多候选 + 候选注入 prompt） ---------------------
def test_resolve_judge_path_picks_from_candidate_list():
    """两候选时 L2 判定器的选择生效，且 prompt 里带上了完整候选列表。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？",
                     f"{LIYUAN}的法定代表人是谁？")
    calls = []

    def _judge_then_gen(messages, **kw):
        calls.append((messages, kw))
        return {"answer": f"{XINGTU}参与的哪个工程获奖了？"}

    r = coref.resolve("他参与的哪个工程获奖了？", st, llm_fn=_judge_then_gen)
    assert r.mode == "coref_llm"
    assert XINGTU in r.resolved
    assert len(calls) >= 2, "多候选时应先 L2 判定再 L3 生成"
    judge_msgs, judge_kw = calls[0]
    judge_prompt = judge_msgs[0]["content"]
    assert XINGTU in judge_prompt and LIYUAN in judge_prompt
    assert judge_kw["model"] == coref.config.REWRITE_MODEL_JUDGE


# --- Fix wave 1：RewriteResult.reason 覆盖 -----------------------------------
def test_rewrite_result_reason_reports_degradation_cause():
    """降级原因必须可见（Task 9 的 degraded 率要靠它诊断）。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    # 成功路径：无降级原因
    ok = coref.resolve("他参与的哪个工程获奖了？", st,
                       llm_fn=_fake_llm(f"{XINGTU}参与的哪个工程获奖了？"))
    assert ok.mode == "coref_llm" and ok.reason == ""
    # LLM 坏输出（含公司名但残留代词）→ 降级，reason 说明是哪条校验没过
    bad = coref.resolve("他参与的哪个工程获奖了？", st,
                        llm_fn=_fake_llm(f"{XINGTU}的子公司，这个公司参与的哪个工程获奖了？"))
    assert bad.degraded is True and bad.reason
    assert "残留代词" in bad.reason
    # 无历史 → 不猜，reason 说明原因
    empty = coref.resolve("他的法定代表人是谁？", dialogue.DialogueState(),
                          llm_fn=_fake_llm("x"))
    assert empty.mode == "unresolved" and empty.reason
