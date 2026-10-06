# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
tests/test_conversation_engine.py —— 工单五 多轮对话引擎单元测试（新增文件）

覆盖：会话管理、指代消解（代词/那X呢/实体继承）、问句改写、历史记录。
不依赖 RAGEngineV4（mock 掉），纯测会话与消解逻辑。
"""
import pytest

from src.conversation_engine import (ConversationEngine, ConversationSession,
                                     _detect_company, _COMPANIES)

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化任务"


# ================= 工单五：实体识别 =================
class TestDetectCompany:
    def test_full_name(self):
        assert _detect_company("武汉兴图新科电子股份有限公司来自军用领域") == \
            "武汉兴图新科电子股份有限公司"

    def test_short_name(self):
        assert _detect_company("兴图新科的收入") == "兴图新科"

    def test_liyuan(self):
        assert _detect_company("力源信息本次发行") == "力源信息"

    def test_no_company(self):
        assert _detect_company("法定代表人是谁") is None

    def test_longest_match(self):
        """工单五：兴图 与 武汉兴图新科电子股份有限公司 同现时取最长"""
        q = "武汉兴图新科电子股份有限公司和兴图新科"
        assert _detect_company(q) == "武汉兴图新科电子股份有限公司"


# ================= 工单五：会话管理 =================
class TestSession:
    def test_create_session(self):
        eng = ConversationEngine()
        s = eng.get_session()
        assert s.session_id and len(s.session_id) == 12
        assert s.history == []
        assert s.current_entity is None

    def test_get_same_session(self):
        eng = ConversationEngine()
        s1 = eng.get_session("abc123")
        s2 = eng.get_session("abc123")
        assert s1 is s2

    def test_add_turn_updates_entity(self):
        s = ConversationSession("t1")
        s.add_turn("问题", "答案", entity="兴图新科", doc="招股说明书1")
        assert s.current_entity == "兴图新科"
        assert s.current_doc == "招股说明书1"
        assert len(s.history) == 2  # user + assistant

    def test_reset_session(self):
        eng = ConversationEngine()
        eng.get_session("rst")
        eng.reset_session("rst")
        assert "rst" not in eng._sessions


# ================= 工单五：指代消解 =================
class TestCorefResolution:
    def setup_method(self):
        self.eng = ConversationEngine()

    def _session_with_entity(self, entity="武汉兴图新科电子股份有限公司",
                             intent=None):
        s = self.eng.get_session("test_sid")
        s.current_entity = entity
        s.current_doc = _COMPANIES.get(entity)
        if intent:
            s.last_intent = intent
        return s

    def test_direct_with_entity(self):
        """工单五：问句含完整公司名 → direct，实体正确"""
        s = self.eng.get_session("d1")
        r = self.eng.resolve_coref(
            "武汉兴图新科电子股份有限公司的收入是多少？", s)
        assert r["strategy"] == "direct"
        assert r["entity"] == "武汉兴图新科电子股份有限公司"
        assert r["doc_id"] == "招股说明书1"
        assert r["is_followup"] is False

    def test_pronoun_ta(self):
        """工单五：他 → 当前实体（兴图新科）"""
        s = self._session_with_entity()
        r = self.eng.resolve_coref(
            "他参与的哪个工程荣获了国家科技进步一等奖？", s)
        assert r["strategy"] == "pronoun_resolution"
        assert r["entity"] == "武汉兴图新科电子股份有限公司"
        assert "武汉兴图新科电子股份有限公司" in r["resolved_query"]
        assert r["is_followup"] is True

    def test_pronoun_this_company(self):
        """工单五：这个公司 → 当前实体"""
        s = self._session_with_entity()
        r = self.eng.resolve_coref("这个公司的法定代表人是谁？", s)
        assert r["strategy"] == "pronoun_resolution"
        assert r["entity"] == "武汉兴图新科电子股份有限公司"
        # 工单五：消解后不应出现"公司公司"
        assert "公司公司" not in r["resolved_query"]

    def test_switch_entity_reuse_intent(self):
        """工单五：那武汉力源信息技术股份有限公司呢？→ 切换实体 + 复用问法"""
        s = self._session_with_entity(intent="的法定代表人是谁")
        r = self.eng.resolve_coref("那武汉力源信息技术股份有限公司呢？", s)
        assert r["strategy"] == "switch_entity_reuse_intent"
        assert r["entity"] == "武汉力源信息技术股份有限公司"
        assert r["doc_id"] == "招股说明书2"
        assert "法定代表人" in r["resolved_query"]
        assert r["is_followup"] is True

    def test_inherit_entity(self):
        """工单五：无代词无新实体 → 继承当前实体"""
        s = self._session_with_entity()
        r = self.eng.resolve_coref("营业收入是多少？", s)
        assert r["strategy"] == "inherit_entity"
        assert r["entity"] == "武汉兴图新科电子股份有限公司"

    def test_no_context_pronoun(self):
        """工单五：代词但无历史实体 → 不报错，实体为 None"""
        s = self.eng.get_session("no_ctx")
        r = self.eng.resolve_coref("他的收入是多少？", s)
        # 无实体可消解，entity 为 None（由 RAG 层兜底）
        assert r["entity"] is None or r["strategy"] != "pronoun_resolution"


# ================= 工单五：问法提取 =================
class TestExtractIntent:
    def test_extract_legal_representative(self):
        eng = ConversationEngine()
        intent = eng._extract_intent("这个公司的法定代表人是谁？")
        assert "法定代表人" in intent

    def test_extract_military_revenue(self):
        eng = ConversationEngine()
        intent = eng._extract_intent(
            "武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
        assert "军用" in intent or "收入" in intent

    def test_short_query_no_intent(self):
        eng = ConversationEngine()
        assert eng._extract_intent("你好") is None


# ================= 工单五：chat 端到端（mock RAG） =================
class TestChatMock:
    def test_chat_records_history(self):
        """工单五：chat 后会话历史应更新"""
        eng = ConversationEngine()

        class FakeRAG:
            def ask(self, query, **kw):
                return {"answer": f"回答：{query}", "latency_ms": 100,
                        "route": {}, "references": [], "retrieved_text_chunks": [],
                        "retrieved_tables": [], "retrieved_images": [],
                        "breakdown": {}, "token_usage": {}}

        eng._rag = FakeRAG()
        r = eng.chat("武汉兴图新科电子股份有限公司的收入？")
        assert r["answer"].startswith("回答")
        hist = eng.get_history(r["session_id"])
        assert len(hist["history"]) == 2
        assert hist["current_entity"] == "武汉兴图新科电子股份有限公司"

    def test_chat_multiturn_coref(self):
        """工单五：连续两轮 —— 第二轮代词应消解为第一轮实体"""
        eng = ConversationEngine()

        class FakeRAG:
            def ask(self, query, **kw):
                return {"answer": f"OK:{query[:20]}", "latency_ms": 100,
                        "route": {}, "references": [], "retrieved_text_chunks": [],
                        "retrieved_tables": [], "retrieved_images": [],
                        "breakdown": {}, "token_usage": {}}

        eng._rag = FakeRAG()
        r1 = eng.chat("武汉兴图新科电子股份有限公司的法定代表人是谁？")
        sid = r1["session_id"]
        r2 = eng.chat("他的收入是多少？", session_id=sid)
        assert r2["entity"] == "武汉兴图新科电子股份有限公司"
        assert r2["coref_strategy"] == "pronoun_resolution"
