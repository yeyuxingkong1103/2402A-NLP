"""tests/test_roleplay.py —— backend/roleplay.py 的单元测试（角色、会话、降级）。

在链路中的位置：
    纯单元测试。**全部外部依赖都被 mock 掉** ——
    不连 Milvus、不调 Ollama、不碰真实数据库（用临时目录里的 SQLite）。

关键设计：测试用临时数据库，且必须重置模块级全局
    roleplay.py 把数据库连接存成模块级全局 _DB，
    并把库路径从环境变量读进 ROLEPLAY_DB。测试在 setUp 里：
        1. 建一个临时目录
        2. 把 roleplay._DB 置为 None（丢弃可能已存在的连接）
        3. 把 roleplay.ROLEPLAY_DB 指向临时库
    这样下一次调 _db() 时会用新路径重新建库 ——
    与真实开发库完全隔离，测试不会污染 data/roleplay.sqlite3，
    也不会因为开发库里的历史数据而失败。
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


class RoleplayTest(unittest.TestCase):
    """覆盖角色管理、会话历史、以及模型不可用时的降级行为。"""

    def setUp(self):
        """每个测试前：准备隔离的临时数据库。"""
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "roleplay.sqlite3"
        import roleplay
        # 重置全局连接与库路径 —— 顺序上必须先清连接、再改路径，
        # 否则旧连接仍指向原库（模块级单例的手工重置方式）。
        #
        # 拆包后 _DB / ROLEPLAY_DB 定义在 roleplay.db 里，所以必须改写 roleplay.db 的模块全局：
        # 写 roleplay._DB 只会改掉包入口上的那个再导出名，而 _db() 读的是自己模块里的变量，
        # 结果就是"看似重置了、实际没有"，测试会连到 data/roleplay.sqlite3 生产库上。
        roleplay.db._DB = None
        roleplay.db.ROLEPLAY_DB = self.db_path
        self.roleplay = roleplay

    def tearDown(self):
        """每个测试后：关连接、删临时目录。

        显式 close 并把 _DB 置回 None：
            不关的话，Windows 上临时目录会因文件被占用而删不掉，
            TemporaryDirectory.cleanup() 会抛异常。
            同时置 None 是为了让下一个测试的 setUp 从干净状态开始。
        """
        if self.roleplay.db._DB is not None:
            self.roleplay.db._DB.close()
            self.roleplay.db._DB = None
        self.tmp.cleanup()

    def test_default_roles_and_custom_role(self):
        """验证内置角色已初始化，且可以新建/读取自定义角色。

        assertGreaterEqual(len(roles), 5) 用 >= 而不是 ==：
            内置角色清单会随迭代增删，硬等一个数字会让测试在
            合理的产品变更下误报失败。>= 5 表达的是"基础角色集不少于五个"这个真实约束。

        保存后立刻用 get_role 读回并核对 name：
            验证的是"写进去的能读出来"，而不是只验证 save_role 的返回值 ——
            后者可能只是回显入参，读回来才证明真的落库了。
        """
        roles = self.roleplay.list_roles()
        self.assertGreaterEqual(len(roles), 5)
        saved = self.roleplay.save_role({"id": "test_role", "name": "测试角色", "persona": "你是测试角色。"})
        self.assertEqual(saved["id"], "test_role")
        self.assertEqual(self.roleplay.get_role("test_role")["name"], "测试角色")

    def test_session_and_history(self):
        """验证会话创建与消息读写，以及**会话按用户隔离**。

        最后两条断言是重点：
            list_sessions("user-a") == 1 条
            list_sessions("user-b") == []（空）
        这验证了会话查询确实按 user_id 过滤 ——
        如果实现里漏掉了过滤条件，user-b 会看到 user-a 的会话，
        这是隐私问题，也正是这条断言要守住的东西。
        """
        session = self.roleplay.create_session("user-a", "friend")
        self.roleplay.append_message(session, "user", "你好")
        self.roleplay.append_message(session, "assistant", "你好呀")
        self.assertEqual(len(self.roleplay.history(session["id"])), 2)
        self.assertEqual(len(self.roleplay.list_sessions("user-a")), 1)
        self.assertEqual(self.roleplay.list_sessions("user-b"), [])

    # patch 目标必须指向**实际使用这些函数**的模块（拆包后是 roleplay.chat）：
    #     chat() 定义在 roleplay/chat.py，它调用的是自己模块里 import 进来的
    #     call_llm / retrieve_with_trace / retrieve_long_term / remember_long_term。
    #     若 patch 到包入口（如 "roleplay.call_llm"），替换的只是 __init__ 上的再导出名，
    #     chat.py 里的引用不受影响 —— mock 不生效，测试会去调真实的本地模型。
    @patch("roleplay.chat.remember_long_term", return_value=False)
    @patch("roleplay.chat.retrieve_long_term", return_value=[])
    @patch("roleplay.chat.retrieve_with_trace", return_value={"context_docs": [], "results": [], "trace": []})
    @patch("roleplay.chat.call_llm", side_effect=RuntimeError("model offline"))
    def test_chat_falls_back_without_model(self, _llm, _retrieve, _long, _remember):
        """验证本地模型不可用时，对话仍能完成并如实标记为降级。

        这是本项目"单点失败不能拖垮整条链路"的验证：
            Milvus 不可用（retrieve_with_trace 被替换成返回空）
            + Ollama 不可用（call_llm 抛异常）
            => 用户仍然拿到一条回答（fallback_answer），会话也正常落库。

        三层断言分别验证：
            model_source == "fallback"   如实标记了这次是降级结果，
                                         而不是伪装成模型生成 —— 前端和评测据此区分
            session_id 为真              会话被正确创建/复用
            history 长度为 2             用户消息与助手回答都落库了（一轮完整对话）

        四个 @patch 装饰器的执行顺序是**自下而上**，
        所以测试方法的参数顺序与之相反：
            最下面的 call_llm           -> 第一个参数 _llm
            retrieve_with_trace         -> 第二个参数 _retrieve
            retrieve_long_term          -> 第三个参数 _long
            最上面的 remember_long_term -> 第四个参数 _remember
        写错顺序会导致 mock 张冠李戴（把 remember 的 mock 当成 call_llm 用），
        测试可能仍"通过"但实际没验证到想验的东西。

        remember_long_term 被 mock 成 False：
            表示"没能存进长期记忆"（Milvus 不可用的情形），
            同时避免测试真的去连 Milvus。
        """
        result = self.roleplay.chat("user-a", "friend", "你好")
        self.assertEqual(result["model_source"], "fallback")
        self.assertTrue(result["session_id"])
        self.assertEqual(len(self.roleplay.history(result["session_id"])), 2)


if __name__ == "__main__":
    # 支持 python tests/test_roleplay.py 直接运行
    unittest.main()
