# -*- coding: utf-8 -*-
"""tests/test_vector_store.py —— backend/vector_store.py 的单元测试。

在链路中的位置：
    纯单元测试。**不需要真实 Milvus** ——
    涉及写入的用例用 unittest.mock.patch 把 ensure_collection 替换成假对象，
    只验证"我们构造出的数据长什么样"，不验证 Milvus 的行为。

测的两个东西，恰好是这一层最容易出问题的地方：
    过滤表达式的转义与拼装（写错了会删错数据或过滤失效）
    payload 摊平（写错了 Milvus 会因字段层级不对而写入失败）
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import vector_store


class VectorStoreTest(unittest.TestCase):
    """覆盖过滤表达式构造与 upsert 的数据整形。"""

    def test_equal_filter_escapes_string(self):
        """验证字符串值被正确转义、数字值不加引号。

        第一条断言是安全测试：
            传 'a"b' 期望得到 'source == "a\\"b"'
            —— 那个反斜杠证明双引号被转义了。
            不转义的话字面量会提前闭合，后面的内容被当成表达式解析，
            过滤条件被改写（最坏情况是删除条件失效，误删数据）。

        第二条断言是类型处理测试：
            传数字 3 期望 'page == 3'（**不带引号**）。
            如果数字被当成字符串加上引号，Milvus 会按字符串比较，
            结果恒为假 —— 表现为"过滤条件写了但一个都匹配不到"，
            这种 bug 很难从现象反推到原因。
        """
        self.assertEqual(vector_store.equal_filter("source", 'a"b'), 'source == "a\\"b"')
        self.assertEqual(vector_store.equal_filter("page", 3), "page == 3")

    def test_and_filter_joins_conditions(self):
        """验证多条件用 and 连接、且每个条件都被括号包裹。

        括号是重点：
            and/or 在过滤表达式里有优先级，不加括号时
            "(a and b) or c" 可能被解析成 "a and (b or c)"，语义完全不同。
            断言里显式写出带括号的期望值，把这个约定固定下来。

        第二条断言验证空串条件被过滤掉：
            只传空串和一个真实条件时，结果不该出现多余的 " and ()" 之类的残渣。
            这保证调用方能放心地传"可能为空的可选条件"。
        """
        self.assertEqual(vector_store.and_filter("user_id == \"u\"", "role_id == \"r\""), '(user_id == "u") and (role_id == "r")')
        self.assertEqual(vector_store.and_filter("", "role_id == \"r\""), '(role_id == "r")')

    # patch 目标必须是**实际调用它**的那个模块：
    #   upsert_vectors 定义在 vector_store/ops.py，它调用的是自己模块内的 ensure_collection。
    #   若 patch 到包入口 "vector_store.ensure_collection"，只是替换了包上的再导出名，
    #   ops.py 里的引用不受影响 —— mock 不生效，测试会去连真实 Milvus。
    #   这是"模块拆成包"之后必须同步调整的一类引用（本次重构唯一改动的测试行）。
    @patch("vector_store.ops.ensure_collection")
    def test_upsert_vectors_flattens_payload(self, ensure_collection):
        """验证 upsert 把 payload 摊平后写入、并返回写入条数。

        输入记录是嵌套结构：
            {"id": ..., "vector": ..., "payload": {"source": ..., "page": ...}}
        期望写入 Milvus 的是摊平结构：
            {"id": ..., "vector": ..., "source": ..., "page": ...}

        为什么必须摊平：
            集合开启了动态字段（enable_dynamic_field=True），
            业务字段必须与 id/vector 处于同一层才能被存进去。
            嵌套着写的话，Milvus 只会看到 id 和 vector，
            payload 里那些"来源/页码/正文"全部丢失 ——
            现象是"检索能命中但引用是空的"，排查起来很绕。

        assert_called_once_with 的验证强度：
            它不仅检查"调用过"，还精确比对调用参数，
            同时顺带证明**只调用了一次**（重复写入会造成数据重复）。
            参数里 id 是字符串 "point-1"，与输入一致，说明没有意外类型转换。
        """
        client = ensure_collection.return_value
        count = vector_store.upsert_vectors(
            "rag_docs_v6",
            [{"id": "point-1", "vector": [0.1, 0.2], "payload": {"source": "a.pdf", "page": 1}}],
        )
        self.assertEqual(count, 1)
        client.upsert.assert_called_once_with(
            collection_name="rag_docs_v6",
            data=[{"id": "point-1", "vector": [0.1, 0.2], "source": "a.pdf", "page": 1}],
        )


if __name__ == "__main__":
    # 支持 python tests/test_vector_store.py 直接运行
    unittest.main()
