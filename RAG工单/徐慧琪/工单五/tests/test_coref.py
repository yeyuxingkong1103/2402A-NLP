# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""L1 检测层：指代与省略的 span 识别。"""
import pytest

from src import coref, dialogue


XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"


def _state_with(*questions) -> dialogue.DialogueState:
    st = dialogue.DialogueState()
    for i, q in enumerate(questions, 1):
        dialogue.append_turn(st, dialogue.Turn(idx=i, raw_question=q))
    return st


# --- 语言判定 ---------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    (f"{XINGTU}的法定代表人是谁？", "zh"),
    ("Which project did he participate in?", "en"),
    ("what about 武汉力源信息技术股份有限公司?", "en"),
])
def test_detect_language(text, expected):
    assert coref.detect_language(text) == expected


# --- L0 直通 -----------------------------------------------------------------
def test_standalone_when_full_company_named():
    st = dialogue.DialogueState()
    assert coref.is_standalone(f"{XINGTU}的法定代表人是谁？", st) is True
    assert coref.is_standalone(f"那{LIYUAN}呢？", st) is False


def test_standalone_does_not_fire_on_pronoun():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    assert coref.is_standalone("他参与的哪个工程获奖了？", st) is False


def test_standalone_agrees_with_zero_anaphora_span():
    """L0 直通判定必须与 detect_spans 同源：零指代输入不得被判为可直通。

    否则任何以 is_standalone 作为「跳过改写」闸门的调用方，都会静默丢掉
    零指代消解（设计测试集 B：历史点名公司后「注册资本是多少？」须回指该公司）。
    """
    st = _state_with(f"{XINGTU}注册资本是多少？")
    assert coref.is_standalone("注册资本是多少？", st) is False
    assert coref.detect_spans("注册资本是多少？", st)[0].kind == "zero"


# --- 人称代词 ---------------------------------------------------------------
def test_person_pronoun_to():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("他参与的哪个工程荣获了国家科技进步一等奖？", st)[0]
    assert span.kind == "person"
    assert span.text == "他"
    assert span.start == 0


def test_person_pronoun_english():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("Which project did he participate in?", st)[0]
    assert span.kind == "person"
    assert span.text.lower() == "he"


# --- 单字「他」的构词保护 -----------------------------------------------------
@pytest.mark.parametrize("q", [
    "其他业务收入是多少？",
    "他人是谁？",
    "他日如何？",
    "他乡在哪？",
    "他国如何？",
])
def test_ta_compounds_are_not_person(q):
    """构词里的「他」不是代词（中文无词边界，须靠正则排除）。"""
    st = dialogue.DialogueState()
    span = coref.detect_spans(q, st)[0]
    assert span.kind != "person"
    assert span.text != "他"


def test_other_ye_question_with_history_is_not_person():
    """有历史时「其他业务收入是多少？」可以是零指代，但绝不能是 person。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("其他业务收入是多少？", st)[0]
    assert span.kind == "zero"


def test_person_pronoun_ta_men_still_wins_as_longest():
    """多字「他们」必须仍命中：同起点按 (start, -len) 压过单字「他」。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("他们参与的哪个工程获奖了？", st)[0]
    assert span.kind == "person"
    assert span.text == "他们"
    assert span.start == 0
    assert span.end == 2


def test_person_pronoun_ta_single_still_matches():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("他参与的哪个工程获奖了？", st)[0]
    assert span.kind == "person"
    assert span.text == "他"
    assert span.start == 0


# --- 单字「它」的构词保护（与「他」同类） ------------------------------------
@pytest.mark.parametrize("q", [
    "其它业务收入是多少？",
])
def test_ta_two_compounds_are_not_person(q):
    """「其它」是最常见的含「它」构词，单字「它」不得误命中为代词。"""
    st = dialogue.DialogueState()
    span = coref.detect_spans(q, st)[0]
    assert span.kind != "person"
    assert span.text != "它"


def test_qita_question_with_history_is_not_person():
    """有历史时「其它业务收入是多少？」可以是零指代，但绝不能是 person。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("其它业务收入是多少？", st)[0]
    assert span.kind == "zero"


def test_person_pronoun_ta_men_it_still_wins_as_longest():
    """多字「它们」必须仍命中：同起点按 (start, -len) 压过单字「它」。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("它们参与的哪个工程获奖了？", st)[0]
    assert span.kind == "person"
    assert span.text == "它们"
    assert span.start == 0
    assert span.end == 2


def test_person_pronoun_ta_single_it_still_matches():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("它参与的哪个工程获奖了？", st)[0]
    assert span.kind == "person"
    assert span.text == "它"
    assert span.start == 0


# --- 指示代词 ---------------------------------------------------------------
def test_demonstrative_zhege_gongsi():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans("这个公司的法定代表人是谁？", st)[0]
    assert span.kind == "demonstrative"
    assert span.text == "这个公司"
    assert span.start == 0


def test_demonstrative_gai_gongsi_takes_priority_over_qi():
    """「该公司」必须整体命中，不能被更短的「其」抢先。"""
    st = _state_with(f"{XINGTU}注册资本是多少？")
    span = coref.detect_spans("该公司的法定代表人是谁？", st)[0]
    assert span.text == "该公司"


def test_demonstrative_english_this_company():
    st = _state_with(f"{XINGTU}注册资本是多少？")
    span = coref.detect_spans("Who is the legal representative of this company?", st)[0]
    assert span.kind == "demonstrative"


# --- 单字「其」的构词保护（走 detect_spans 全链路） ---------------------------
@pytest.mark.parametrize("q", [
    "其中提到的收入是多少？",
    "尤其以军用领域为主？",
    "其实公司的主营业务是什么？",
])
def test_qi_compounds_are_not_demonstrative(q):
    """构词里的「其」不是指示代词——保护必须在 detect_spans 全链路上生效。"""
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？")
    span = coref.detect_spans(q, st)[0]
    assert span.kind != "demonstrative"
    assert span.text != "其"


@pytest.mark.parametrize("q", [
    "极其重要的事项是什么？",
    "与其相关的收入是多少？",
])
def test_qi_compounds_ji_yu_are_not_demonstrative(q):
    """「极其」「与其」是 _QI_RE 的漏网构词。"""
    st = dialogue.DialogueState()
    span = coref.detect_spans(q, st)[0]
    assert span.kind != "demonstrative"
    assert span.text != "其"


# --- 输入健壮性 -------------------------------------------------------------
@pytest.mark.parametrize("q", ["", None, "   "])
def test_empty_or_none_input_does_not_raise(q):
    st = dialogue.DialogueState()
    assert coref.detect_spans(q, st)[0].kind == "passthrough"
    assert coref.is_standalone(q, st) is True
    assert coref.detect_language(q) == "zh"


# --- 省略 -------------------------------------------------------------------
def test_ellipsis_na_x_ne():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？", "这个公司的法定代表人是谁？")
    span = coref.detect_spans(f"那{LIYUAN}呢？", st)[0]
    assert span.kind == "ellipsis"
    assert span.entity == LIYUAN, "省略句里的新实体必须被解析出来"


def test_ellipsis_short_x_ne():
    st = _state_with(f"{XINGTU}的法定代表人是谁？")
    span = coref.detect_spans(f"{LIYUAN}呢？", st)[0]
    assert span.kind == "ellipsis"
    assert span.entity == LIYUAN


def test_ellipsis_what_about_english():
    st = _state_with(f"{XINGTU}的法定代表人是谁？")
    span = coref.detect_spans(f"What about {LIYUAN}?", st)[0]
    assert span.kind == "ellipsis"
    assert span.entity == LIYUAN


def test_not_ellipsis_when_question_is_complete():
    """完整问题不能被误判为省略。"""
    st = _state_with(f"{XINGTU}的法定代表人是谁？")
    spans = [s for s in coref.detect_spans(f"{LIYUAN}的注册资本是多少？", st) if s.kind == "ellipsis"]
    assert spans == []


# --- 零指代 -----------------------------------------------------------------
def test_zero_anaphora():
    """前文有公司、本句无实体且以疑问结尾 → 零指代。"""
    st = _state_with(f"{XINGTU}的法定代表人是谁？")
    span = coref.detect_spans("注册资本是多少？", st)[0]
    assert span.kind == "zero"
    assert span.entity == XINGTU


def test_no_zero_anaphora_without_history():
    st = dialogue.DialogueState()
    spans = [s for s in coref.detect_spans("注册资本是多少？", st) if s.kind == "zero"]
    assert spans == []


# --- 候选集 -----------------------------------------------------------------
def test_candidates_come_from_history_and_focus():
    st = _state_with(f"{XINGTU}来自军用领域的收入是多少？",
                     "这个公司的法定代表人是谁？",
                     f"那{LIYUAN}呢？")
    span = coref.detect_spans("他的注册资本是多少？", st)[0]
    assert LIYUAN in span.candidates
    assert XINGTU in span.candidates
