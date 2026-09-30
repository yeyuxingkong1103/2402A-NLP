"""
test_integration.py — 集成测试（unittest）

覆盖需求文档 4.14 要求的三类场景：
    完整流程     输入 → 意图识别 → 检索 → 生成 → 返回
    多用户隔离   不同用户的记忆与知识库检索互不干扰
    多角色切换   不同领域的问题路由到不同角色

需要调用大模型的用例在未配置 API Key 时自动跳过：

    python -m unittest tests.test_integration -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  （必须先执行，才能安全导入业务模块）

import chat  # noqa: E402
import config  # noqa: E402
import domains  # noqa: E402
import intent  # noqa: E402
import long_term  # noqa: E402
import session_memory  # noqa: E402
import vector_store  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402

HAS_API_KEY = bool(config.get_llm_config()["api_key"])
SKIP_REASON = "未配置 API Key（DEEPSEEK_API_KEY），跳过需要大模型的用例"


def clear_memory(user_id: str) -> None:
    """清空某个用户的短期与长期记忆。"""
    session_memory.clear_short_term(user_id)
    session_memory.clear_rounds(user_id)
    long_term.clear_long_term(user_id)


def seed_knowledge_base() -> None:
    """往三个领域各塞一点测试语料，保证检索有内容可召回。"""
    kb = KnowledgeBase()
    kb.clear("legal")
    kb.clear("medical")
    kb.clear("english")

    kb.import_texts(
        [
            "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。劳动者在试用期内提前三日通知用人单位，可以解除劳动合同。",
            "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，支付半个月工资。",
            "休息日安排劳动者工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬。法定休假日安排工作的，支付不低于工资的百分之三百的工资报酬。",
        ],
        domain="legal",
        source="integration_legal.txt",
    )
    kb.import_texts(
        [
            "普通感冒多由病毒引起，具有自限性，病程通常五至七天。抗生素对病毒感染无效，不应常规使用。",
            "成年人推荐睡眠时长为七到九小时。规律作息、限制午睡时长、睡前避免咖啡因是改善失眠的基础措施。",
        ],
        domain="medical",
        source="integration_medical.txt",
    )
    kb.import_texts(
        [
            "一般过去时表示过去特定时间发生的动作，常与 yesterday、last week 等时间状语连用。现在完成时强调对现在的影响，结构为 have/has 加过去分词。",
            "被动语态的基本结构为 be 加过去分词，执行者需要强调时用 by 引出。",
        ],
        domain="english",
        source="integration_english.txt",
    )


class TestMultiUserIsolation(unittest.TestCase):
    """多用户隔离（纯本地，不需要大模型）。"""

    @classmethod
    def setUpClass(cls):
        vector_store.reset_store()
        seed_knowledge_base()

    def test_short_term_memory_isolated(self):
        clear_memory("alice")
        clear_memory("bob")

        session_memory.save_short_term("alice", "我是爱丽丝", "你好爱丽丝")
        session_memory.save_short_term("bob", "我是鲍勃", "你好鲍勃")

        alice = session_memory.get_short_term("alice")
        bob = session_memory.get_short_term("bob")

        self.assertEqual(len(alice), 2)
        self.assertIn("爱丽丝", alice[0]["content"])
        self.assertNotIn("鲍勃", str(alice))
        self.assertIn("鲍勃", bob[0]["content"])

    def test_long_term_memory_isolated(self):
        long_term.clear_long_term("alice")
        long_term.clear_long_term("bob")

        long_term.store_long_term("alice", "爱丽丝是一名律师，擅长劳动法。", fact_type="profile")
        long_term.store_long_term("bob", "鲍勃是一名医生，在儿科工作。", fact_type="profile")

        alice_hits = long_term.search_long_term("alice", "律师 劳动法", top_k=3)
        bob_hits = long_term.search_long_term("bob", "医生 儿科", top_k=3)

        self.assertTrue(alice_hits, "爱丽丝的长期记忆没有召回")
        self.assertTrue(bob_hits, "鲍勃的长期记忆没有召回")
        self.assertTrue(all(hit.get("user_id") == "alice" for hit in alice_hits))
        self.assertTrue(all(hit.get("user_id") == "bob" for hit in bob_hits))

        alice_text = " ".join(hit.get("text", "") for hit in alice_hits)
        self.assertIn("爱丽丝", alice_text)
        self.assertNotIn("鲍勃", alice_text)

    def test_clear_does_not_affect_others(self):
        clear_memory("carol")
        session_memory.save_short_term("carol", "问题", "回答")
        clear_memory("dave")
        session_memory.save_short_term("dave", "问题", "回答")

        clear_memory("carol")

        self.assertEqual(session_memory.get_short_term("carol"), [])
        self.assertEqual(len(session_memory.get_short_term("dave")), 2)

    def test_long_term_filtered_by_domain(self):
        """同一用户的记忆也要按领域隔离，否则法律问题会召回医疗记忆。"""
        uid = "domain_user"
        clear_memory(uid)
        long_term.store_long_term(
            uid, "用户问题：劳动合同试用期一般多久", fact_type="summary", domain="legal"
        )

        self.assertTrue(long_term.search_long_term(uid, "劳动合同试用期一般多久", domain="legal"))
        self.assertEqual(
            long_term.search_long_term(uid, "劳动合同试用期一般多久", domain="medical"), []
        )


class TestKnowledgeBaseRouting(unittest.TestCase):
    """知识库按领域隔离（纯本地，不需要大模型）。"""

    @classmethod
    def setUpClass(cls):
        vector_store.reset_store()
        seed_knowledge_base()

    def test_domains_are_separated(self):
        kb = KnowledgeBase()
        legal = kb.search("试用期辞职", "legal", top_k=3)
        medical = kb.search("试用期辞职", "medical", top_k=3)

        self.assertTrue(any("劳动合同" in doc["text"] for doc in legal))
        # 医疗库里不该有劳动法的内容
        self.assertFalse(any("劳动合同" in doc["text"] for doc in medical))

    def test_each_domain_retrievable(self):
        kb = KnowledgeBase()
        cases = {
            "legal": "经济补偿怎么算",
            "medical": "感冒要不要吃抗生素",
            "english": "被动语态怎么构成",
        }
        for domain, question in cases.items():
            with self.subTest(domain=domain):
                results = kb.search(question, domain, top_k=2)
                self.assertGreater(len(results), 0, f"{domain} 领域检索不到内容")

    def test_domain_class_retrieves_own_collection(self):
        """领域类检索的是自己的集合，不会串库。"""
        docs = domains.get_domain("legal").retrieve("经济补偿怎么算", top_k=2)
        self.assertTrue(docs, "领域类检索不到内容")
        self.assertTrue(all("legal" in (d.get("source") or "") for d in docs))


class TestMemoryInPrompt(unittest.TestCase):
    """长期记忆召回必须真正进入提示词，不能只存不查。"""

    @classmethod
    def setUpClass(cls):
        vector_store.reset_store()
        seed_knowledge_base()

    def test_recalled_memory_lands_in_prompt(self):
        uid = "prompt_user"
        clear_memory(uid)
        long_term.store_long_term(
            uid,
            "用户问题：劳动合同试用期一般多久\n回答要点：三个月到六个月",
            fact_type="summary",
            domain="legal",
        )

        # 固定意图识别结果，避免用例依赖模型的单次判断
        with mock.patch.object(intent, "detect_intent", return_value="legal"):
            ctx = chat.prepare(uid, "劳动合同试用期一般多久")

        self.assertEqual(ctx["domain"], "legal")
        self.assertTrue(ctx["memory_hits"], "长期记忆没有召回，检查阈值与 domain 过滤")
        self.assertIn("三个月到六个月", ctx["system_prompt"])
        self.assertIn("【历史对话参考】", ctx["system_prompt"])
        # 知识库那一段也要在，两路检索是并行的
        self.assertIn("【知识库检索】", ctx["system_prompt"])

    def test_empty_memory_still_renders_section(self):
        uid = "prompt_empty_user"
        clear_memory(uid)

        with mock.patch.object(intent, "detect_intent", return_value="legal"):
            ctx = chat.prepare(uid, "劳动合同试用期一般多久")

        self.assertIn("【历史对话参考】", ctx["system_prompt"])
        self.assertIn("暂无历史记忆", ctx["system_prompt"])
        self.assertIsInstance(ctx["sources"], list)


@unittest.skipUnless(HAS_API_KEY, SKIP_REASON)
class TestFullPipeline(unittest.TestCase):
    """完整链路：输入 → 意图识别 → 检索 → 生成 → 返回。"""

    @classmethod
    def setUpClass(cls):
        vector_store.reset_store()
        seed_knowledge_base()

    def setUp(self):
        clear_memory("integration_user")

    def test_chat_returns_expected_fields(self):
        result = chat.chat("integration_user", "经济补偿金怎么计算？")

        self.assertIn("reply", result)
        self.assertTrue(result["reply"].strip(), "模型回复为空")
        self.assertEqual(result["domain"], "legal")
        self.assertEqual(result["role"], "法律顾问")
        self.assertIsInstance(result["sources"], list)

    def test_stream_event_sequence(self):
        events = list(chat.chat_stream("integration_user", "感冒需要吃抗生素吗？"))

        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "meta", "首个事件应为 meta")
        self.assertEqual(kinds[-1], "done", "末个事件应为 done")
        self.assertIn("delta", kinds, "未产生任何正文增量")
        self.assertNotIn("error", kinds, f"流程报错：{events}")

        meta = events[0]
        self.assertEqual(meta["domain"], "medical")
        self.assertEqual(meta["role"], "医疗咨询")

        reply = "".join(e["content"] for e in events if e["type"] == "delta")
        self.assertGreater(len(reply.strip()), 10)

    def test_role_switching_by_domain(self):
        cases = {
            "劳动合同到期不续签要赔偿吗？": ("legal", "法律顾问"),
            "孩子发烧39度需要去医院吗？": ("medical", "医疗咨询"),
            "现在完成时和一般过去时有什么区别？": ("english", "英语学习助手"),
        }
        for question, (domain, role) in cases.items():
            with self.subTest(question=question):
                result = chat.chat("integration_user", question)
                self.assertEqual(result["domain"], domain)
                self.assertEqual(result["role"], role)

    def test_memory_persisted_after_chat(self):
        chat.chat("integration_user", "休息日加班工资怎么算？")

        history = session_memory.get_short_term("integration_user")
        self.assertEqual(len(history), 2, "短期记忆应记下用户提问与模型回答")
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

        stored = long_term.list_long_term("integration_user")
        self.assertGreater(len(stored), 0, "长期记忆应写入本轮内容")
        self.assertEqual(stored[0]["fact_type"], "summary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
