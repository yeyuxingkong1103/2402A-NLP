# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

工单5 新增模块的**离线单元测试**（不联网、不调用 LLM、不需要索引）。

覆盖三块：
    1. ``app/core/subject_binding.py``  —— 主体绑定（同页多主体互污染的通用解法）
    2. ``app/core/query_rewriter.py``   —— Query 改写五条规则 R0~R4
    3. ``app/core/dialogue_state.py``   —— 会话状态（焦点实体 / 槽位 / 轮次裁剪）

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_multiturn_core.py -q
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

#: 测试用临时目录**必须落在仓库内**：本机沙箱只允许写工作区，
#: 系统临时目录（%TEMP%）与 pytest 的 ``tmp_path``（仓库根 ``.pytest_tmp``）都会
#: 触发 WinError 5 / PermissionError，故统一自建在 ``研发/data/runtime/_pytest_tmp``。
_TMP_BASE = REPO_ROOT / "研发" / "data" / "runtime" / "_pytest_tmp"


def _mk_tmp(prefix: str) -> Path:
    """建一个仓库内的临时目录。"""
    _TMP_BASE.mkdir(parents=True, exist_ok=True)
    path = _TMP_BASE / f"{prefix}{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path

from app.core.dialogue_state import DialogueState, DialogueStateStore, TurnRecord  # noqa: E402
from app.core.query_rewriter import QueryRewriter                                  # noqa: E402
from app.core.subject_binding import (bind_subject, extract_person_value,          # noqa: E402
                                      is_person_field, issuer_in_body, window_orgs)

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

#: 语料里真实出现过的两段文本（节选），用于主体绑定回归
LIYUAN = "武汉力源信息技术股份有限公司"
XINGTU = "武汉兴图新科电子股份有限公司"
P23 = (f"{LIYUAN} 招股意向书 第三节 本次发行概况 一、发行人的基本情况 "
       "1、公司名称：武汉力源信息技术股份有限公司 2、英文名称：Wuhan P&S "
       "3、注册资本：人民币5,000 万元 4、法定代表人：赵马克 5、公司成立日期：2001 年8 月9 日 "
       "6、整体变更为股份有限公司日期：2009 年12 月8 日")
P345 = (f"{LIYUAN} 招股意向书 二、保荐机构(主承销商)声明 "
        "项目协办人： 王 平 保荐代表人： 刘兴华 王英娜 法定代表人： 何 如 国信证券股份有限公司 2010 年 月 日")
P25 = (f"{LIYUAN} 招股意向书 三、本次发行的有关当事人 "
       "(一)发行人：武汉力源信息技术股份有限公司 1、法定代表人：赵马克 2、办公地址：武汉市洪山区珞瑜路424号洪山创业大厦 "
       "3、联系人：王晓东、廖莉华 4、电话：027-87526790 5、传真：027-87526551 "
       "(二)保荐机构(主承销商)：国信证券股份有限公司 1、法定代表人：何如 2、办公地址：深圳市红岭中路1012号")
P52 = (f"{LIYUAN} 招股意向书 上海博润简要财务数据如下： (单位：元) "
       "项目 2010 年6 月30 日 总资产 62,505,097.29 净资产 20,000,000.00 "
       "该公司法定代表人胡桃荣持有 20% 股权")


def _rw() -> QueryRewriter:
    """构造一个带语料实体表的改写器（实体表形态与索引自动统计一致）。"""
    return QueryRewriter(corpus_issuers={"pdf1": [XINGTU], "pdf2": [LIYUAN]})


# ============================================================ 1. 主体绑定
class TestSubjectBinding:
    def test_正确块被接受(self):
        """p23「4、法定代表人：赵马克」→ 接受。"""
        v = bind_subject(P23, P23.find("法定代表人"), issuer=LIYUAN)
        assert v.ok, f"应接受，实际 reason={v.reason}"

    def test_保荐机构签章块被拒收(self):
        """p345 的「法定代表人：何如」属于国信证券，必须拒收。"""
        v = bind_subject(P345, P345.find("法定代表人"), issuer=LIYUAN)
        assert not v.ok and v.reason == "foreign_org"
        assert "国信证券" in v.foreign_org

    def test_同一块里的第二个主体被拒收(self):
        """p25 里 (一)发行人的赵马克接受、(二)保荐机构的何如拒收。"""
        first = P25.find("法定代表人")
        second = P25.rfind("法定代表人")
        assert first != second
        assert bind_subject(P25, first, issuer=LIYUAN).ok
        assert not bind_subject(P25, second, issuer=LIYUAN).ok

    def test_句子片段不算机构名(self):
        """『整体变更为股份有限公司』是句片而非机构，不得触发 foreign_org。"""
        assert bind_subject(P23, P23.find("法定代表人"), issuer=LIYUAN).foreign_org == ""

    def test_页眉不算正文(self):
        """页眉每页都印发行人全称，只能骗过块级判断，骗不过 issuer_in_body。"""
        assert issuer_in_body(P23, LIYUAN) is True          # 正文里确实有
        assert issuer_in_body(P52, LIYUAN) is False         # p52 只有页眉
        assert issuer_in_body(P25, LIYUAN) is True

    def test_窗口内机构名剔除后缀残片(self):
        """窗口切片落在公司名中间时，『股份有限公司』这类残片必须剔除。"""
        assert window_orgs(P23, P23.find("法定代表人")) == []

    @pytest.mark.parametrize("text,expect", [
        ("：赵马克 5、公司成立日期", "赵马克"),
        ("：程家明先生", "程家明"),
        ("：武汉力源信息技术股份有限公司", ""),   # 后面还跟着汉字 → 不是人名
        ("： 何 如 国信证券", ""),                # 字间有空格 → 提取不到（由绑定层拒收）
        ("：如下", ""),
    ])
    def test_人名抽取(self, text, expect):
        assert extract_person_value(text) == expect

    def test_人名型字段识别(self):
        assert is_person_field("这个公司的法定代表人是谁？", ["法定代表人"]) is True
        assert is_person_field("注册资本是多少？", ["注册资本"]) is False


# ============================================================ 2. Query 改写
class TestQueryRewriter:
    def test_R0_空输入(self):
        for bad in ["", "   ", "？？？", "\t\n"]:
            r = _rw().rewrite(bad, DialogueState(session_id="s"))
            assert r.strategy == "empty_input", bad

    def test_R4_带实体带槽位原样(self):
        q = f"报告期内，{XINGTU}来自军用领域的收入分别是多少？"
        st = DialogueState(session_id="s")
        r = _rw().rewrite(q, st)
        assert r.strategy == "none" and r.rewritten == q
        assert r.focus_entity_after == XINGTU

    def test_R2_指代消解_他(self):
        st = DialogueState(session_id="s", focus_entity=XINGTU)
        r = _rw().rewrite("他参与的哪个工程荣获了国家科技进步一等奖？", st)
        assert r.strategy == "pronoun"
        assert r.rewritten == f"{XINGTU}参与的哪个工程荣获了国家科技进步一等奖？"
        assert r.resolved_refs[0]["surface"] == "他"

    def test_R2_指代消解_这个公司(self):
        st = DialogueState(session_id="s", focus_entity=XINGTU)
        r = _rw().rewrite("这个公司的法定代表人是谁？", st)
        assert r.strategy == "pronoun"
        assert r.rewritten == f"{XINGTU}的法定代表人是谁？"
        assert r.slot == "法定代表人是谁？"

    def test_R1_省略式换主体_继承槽位(self):
        """核心用例：「那 X 呢？」= 显式实体 + 无槽位 → 继承上一轮槽位。"""
        st = DialogueState(session_id="s", focus_entity=XINGTU, last_slot="法定代表人是谁？")
        r = _rw().rewrite(f"那{LIYUAN}呢？", st)
        assert r.strategy == "ellipsis_entity_switch"
        assert r.rewritten == f"{LIYUAN}的法定代表人是谁？"
        assert r.inherited_slot == "法定代表人是谁？"
        assert r.focus_entity_after == LIYUAN
        # 换主体后不得再把旧主体带进来（跨实体泄漏）
        assert XINGTU not in r.rewritten

    def test_R1_省略式实体不含前导那字(self):
        """「那」是省略标记，不得被并进实体名（实测踩过的坑）。"""
        st = DialogueState(session_id="s", focus_entity=XINGTU, last_slot="法定代表人是谁？")
        r = _rw().rewrite(f"那{LIYUAN}呢？", st)
        assert r.focus_entity_after == LIYUAN
        assert not r.focus_entity_after.startswith("那")

    def test_R3_光问字段补主体(self):
        st = DialogueState(session_id="s", focus_entity=XINGTU)
        r = _rw().rewrite("法定代表人是谁？", st)
        assert r.strategy == "topic_carry"
        assert r.rewritten == f"{XINGTU}的法定代表人是谁？"

    def test_无历史时指代不硬猜(self):
        st = DialogueState(session_id="s")
        r = _rw().rewrite("这个公司的法定代表人是谁？", st)
        assert r.strategy == "none" and r.changed is False

    def test_无历史时省略式不硬猜(self):
        st = DialogueState(session_id="s")
        r = _rw().rewrite(f"那{LIYUAN}呢？", st)
        assert r.strategy == "none"

    def test_其他里的他不是指代词(self):
        """『其他』里的『他』不是指代词，不得被替换成公司名。"""
        rw = _rw()
        assert rw.detect_anaphora("还有其他的关联方吗？") == []
        # 但它是「光问字段」的追问 → 走 topic_carry 补主体，且副词前不加「的」
        st = DialogueState(session_id="s", focus_entity=XINGTU)
        r = rw.rewrite("还有其他的关联方吗？", st)
        assert r.strategy == "topic_carry"
        assert r.rewritten == f"{XINGTU}还有其他的关联方吗？"
        assert r.resolved_refs == [] or r.resolved_refs[0]["kind"] == "topic_carry"

    def test_独立问句不被污染(self):
        st = DialogueState(session_id="s", focus_entity=XINGTU)
        r = _rw().rewrite("集成电路行业的市场规模是多少？", st)
        assert r.strategy == "none" and r.rewritten == "集成电路行业的市场规模是多少？"


# ============================================================ 3. 会话状态
class TestDialogueState:
    def _turn(self, q, a="答", focus=None, slot=None):
        return TurnRecord(question=q, rewritten_query=q, answer=a, focus_entity=focus, slot=slot)

    def test_轮次上限裁剪(self):
        st = DialogueStateStore(max_turns=3)
        for i in range(5):
            st.append("s", self._turn(f"q{i}"))
        h = st.history("s")
        assert len(h) == 3
        assert [t.question for t in h] == ["q2", "q3", "q4"]
        assert st.get("s").total_turns == 5          # 累计轮次不受裁剪影响

    def test_焦点实体与实体栈(self):
        st = DialogueStateStore()
        st.append("s", self._turn("a", focus=XINGTU))
        st.append("s", self._turn("b", focus=LIYUAN))
        state = st.get("s")
        assert state.focus_entity == LIYUAN
        assert state.entity_stack == [XINGTU, LIYUAN]
        # 重复实体去重后置顶
        st.append("s", self._turn("c", focus=XINGTU))
        assert st.get("s").entity_stack == [LIYUAN, XINGTU]

    def test_槽位继承(self):
        st = DialogueStateStore()
        st.append("s", self._turn("a", slot="法定代表人是谁？"))
        assert st.get("s").last_slot == "法定代表人是谁？"

    def test_会话隔离(self):
        st = DialogueStateStore()
        st.append("s1", self._turn("a", focus=XINGTU))
        st.append("s2", self._turn("b", focus=LIYUAN))
        assert st.get("s1").focus_entity == XINGTU
        assert st.get("s2").focus_entity == LIYUAN

    def test_reset_清空(self):
        st = DialogueStateStore()
        st.append("s", self._turn("a", focus=XINGTU))
        st.reset("s")
        assert st.get("s").focus_entity is None
        assert st.history("s") == []

    def test_落盘与回读(self):
        """落盘后由新 store 回读（模拟进程重启后续聊）。"""
        d = _mk_tmp("state_")
        try:
            st = DialogueStateStore(persist_dir=d)
            st.append("s", self._turn("a", focus=LIYUAN, slot="法定代表人是谁？"))
            # 新进程视角：新建 store 从磁盘回读
            st2 = DialogueStateStore(persist_dir=d)
            state = st2.get("s")
            assert state.focus_entity == LIYUAN
            assert state.last_slot == "法定代表人是谁？"
            assert state.total_turns == 1
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_损坏会话文件不阻断(self):
        """会话文件损坏时必须降级为新会话，绝不抛异常阻断对话。"""
        d = _mk_tmp("broken_")
        try:
            (d / "s.json").write_text("{ 这不是合法 JSON", encoding="utf-8")
            st = DialogueStateStore(persist_dir=d)
            assert st.get("s").focus_entity is None
        finally:
            shutil.rmtree(d, ignore_errors=True)
