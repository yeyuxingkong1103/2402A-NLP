# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""对话状态：焦点公司、aspect 抽取、窗口裁剪、会话表。"""
import pytest

from src import config
from src import dialogue
from src.dialogue import Turn


XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"


def test_extract_companies_full_names():
    assert dialogue.extract_companies(f"报告期内，{XINGTU}来自军用领域的收入是多少？") == [XINGTU]
    assert dialogue.extract_companies(f"那{LIYUAN}呢？") == [LIYUAN]


def test_extract_companies_aliases():
    assert dialogue.extract_companies("武汉力源的法定代表人是谁？") == [LIYUAN]
    assert dialogue.extract_companies("兴图新科注册资本是多少？") == [XINGTU]


def test_extract_companies_none():
    assert dialogue.extract_companies("他参与的哪个工程获奖了？") == []


def test_extract_companies_dedup_and_order():
    q = f"{XINGTU}和{LIYUAN}对比，{XINGTU}的收入是多少？"
    assert dialogue.extract_companies(q) == [XINGTU, LIYUAN]


@pytest.mark.parametrize("q,expected", [
    (f"{XINGTU}的法定代表人是谁？", "法定代表人"),
    (f"{LIYUAN}注册资本是多少？", "注册资本"),
    (f"那{LIYUAN}呢？", ""),
])
def test_extract_aspect(q, expected):
    assert dialogue.extract_aspect(q) == expected


def test_update_focus_switches_company():
    st = dialogue.DialogueState()
    assert dialogue.update_focus(st, f"{XINGTU}来自军用领域的收入是多少？") == XINGTU
    assert st.focus_company == XINGTU
    # 代词轮不改焦点
    assert dialogue.update_focus(st, "他参与的哪个工程获奖了？") == XINGTU
    # 点名另一家 → 焦点切换
    assert dialogue.update_focus(st, f"那{LIYUAN}呢？") == LIYUAN
    assert st.focus_company == LIYUAN


def test_append_turn_backfills_focus_and_aspect():
    st = dialogue.DialogueState()
    t1 = Turn(idx=1, raw_question=f"{XINGTU}的法定代表人是谁？")
    dialogue.append_turn(st, t1)
    assert t1.focus_company == XINGTU
    assert t1.aspect == "法定代表人"
    assert st.last_aspect == "法定代表人"


def test_append_turn_keeps_previous_aspect_for_pronoun_turn():
    """代词/省略轮自己没有 aspect 时，last_aspect 必须保持上一轮的值（Q4 省略补全的模板来源）。"""
    st = dialogue.DialogueState()
    dialogue.append_turn(st, Turn(idx=1, raw_question=f"{LIYUAN}的注册资本是多少？"))
    assert st.last_aspect == "注册资本"

    # 这一轮自己抽不出 aspect（省略句式），必须沿用上一轮的
    dialogue.append_turn(st, Turn(idx=2, raw_question=f"那{LIYUAN}呢？"))
    assert st.turns[-1].aspect == "", "该轮本身不应抽出 aspect，否则测的不是沿用路径"
    assert st.last_aspect == "注册资本", "代词/省略轮必须沿用上一轮的 aspect"


def test_history_window_truncation():
    st = dialogue.DialogueState(history_window=3)
    for i in range(1, 8):
        dialogue.append_turn(st, Turn(idx=i, raw_question=f"{XINGTU}的第{i}个问题是什么？"))
    assert len(st.turns) == 7              # 全量轮次保留（评测要逐轮留痕）
    assert len(dialogue.recent_turns(st, 3)) == 3
    assert [t.idx for t in dialogue.recent_turns(st, 3)] == [5, 6, 7]
    assert dialogue.recent_turns(st, 0) == []      # n=0 必须请求 0 轮，而不是回退到默认窗口


def test_summary_generated_beyond_window():
    st = dialogue.DialogueState(history_window=3)
    for i in range(1, 8):
        dialogue.append_turn(st, Turn(idx=i, raw_question=f"{XINGTU}的第{i}个问题是什么？"))
    assert st.summary, "超出窗口的轮次必须被摘要，否则历史无界增长"
    assert len(st.summary) <= config.DIALOGUE_SUMMARY_MAX_CHARS


def test_max_turns_hard_cap():
    st = dialogue.DialogueState()
    for i in range(1, 60):
        dialogue.append_turn(st, Turn(idx=i, raw_question=f"问题{i}是什么？"))
    assert len(st.turns) <= 50


def test_session_registry_lru():
    dialogue.reset_all()
    for i in range(70):
        dialogue.get_state(f"s{i}")
    assert len(dialogue._SESSIONS) <= 64


def test_reset_state_clears():
    dialogue.reset_all()
    st = dialogue.get_state("a")
    dialogue.append_turn(st, Turn(idx=1, raw_question=f"{XINGTU}注册资本是多少？"))
    dialogue.reset_state("a")
    assert dialogue.get_state("a").turns == []
    assert dialogue.get_state("a").focus_company is None
