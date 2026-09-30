"""tests/test_core.py —— backend 主线核心函数的单元测试。

在链路中的位置：
    纯单元测试，直接调用 backend/pipeline.py 与 backend/retrieval.py 的函数。
    不依赖 Milvus、Ollama 或任何外部服务 —— 因此这组测试在任何机器上都能跑。

测试对象是整条链路里"最容易出错、又最适合单测"的四个纯函数：
    normalize_text   术语归一化（错了会导致检索永远命中不上）
    chunk_pages      分块（错了会导致页码/章节引用失真）
    rewrite_query    查询改写（错了会导致召不回目标片段）
    rrf_merge        融合排序（错了会导致排序结果不符合设计）

为什么只测这些：
    它们是"确定性逻辑"——同样输入必得同样输出。检索链路的其余部分
    （向量相似度、模型生成）本质是概率性的，断言其精确输出没有意义，
    那部分靠 eval/ 的指标评测来验证。分层选择测试手段，是这个测试文件的取舍。
"""
import sys
import unittest
from pathlib import Path

# 把 backend/ 加入路径，才能直接 import 老主线的模块（其模块内部用裸导入互相引用）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pipeline import chunk_pages, normalize_text
from retrieval import rrf_merge, rewrite_query


class CorePipelineTest(unittest.TestCase):
    """覆盖解析后处理与检索排序这两段纯逻辑。"""

    def test_normalize_standard_terms(self):
        """验证标准号与 SF6 的写法归一化。

        输入特意用了 PDF 里实际出现过的三种"脏"写法：
            "SF 6"          下标 6 被解析器提成了普通文本
            "GB / T"        斜杠两侧被插入了空格
            "44653 - 2024"  连字符两侧被插入了空格
        这三种形态如果不清洗，用户按规范写法搜索时就永远命不中。
        断言用 assertIn（包含）而不是 assertEqual（全等）：
            归一化不负责去掉"气体；"这些正常内容，只要求包含规范化后的术语。
        """
        text = normalize_text("SF 6 气体；GB / T 44653 - 2024")
        self.assertIn("SF6", text)
        self.assertIn("GB/T 44653-2024", text)

    def test_chunk_pages_keeps_page_and_section(self):
        """验证分块后仍保留页码与章节信息。

        这是"答案可溯源"的基础：chunk 若丢了 page/section，
        答案末尾就无法标注"第 N 页"，引用也就失去了核对价值。

        输入里的 "1 范围" 是国标正文的起始标记，用来验证章节识别。
        第一条 chunk 的 section 断言为 "正文"：
            因为它由标题 "1 范围" 之前的缓冲内容 + 该标题行组成，
            此时 section 还停留在初始值 "正文" —— 这是当前实现的既有行为，
            测试把它固定下来，将来若有改动会被立刻发现。
        """
        chunks = chunk_pages([{"page": 3, "text": "1 范围\n这是正文内容。\n2 要求\n设备应安全可靠。"}])
        self.assertEqual(chunks[0]["page"], 3)
        self.assertEqual(chunks[0]["section"], "正文")
        self.assertTrue(all(chunk["chunk_id"].startswith("chunk-") for chunk in chunks))

    def test_rewrite_expands_domain_terms(self):
        """验证查询改写确实补上了文档里的术语。

        这题的背景是 V2 迭代里被救回来的"功能清单类"问题：
        用户问"净化装置有哪些功能"，而文档写的是"过滤/干燥/吸附"等具体功能，
        纯语义检索会漏召回。改写规则把术语补全后，BM25 路才能命中。

        断言"过滤"（规则表补的词）与"六氟化硫"（SF6 的同义词）都在结果里，
        并确认 changed=True（确实发生了改写）。
        前两条验证改写内容对不对，changed 验证"改写这件事被正确标记了"——
        前端要靠它决定要不要显示"系统替你补了什么"。
        """
        result = rewrite_query("SF6 净化装置有哪些功能？")
        self.assertIn("过滤", result["rewritten_query"])
        self.assertIn("六氟化硫", result["rewritten_query"])
        self.assertTrue(result["changed"])

    def test_rrf_applies_keyword_weight(self):
        """验证 RRF 融合中"关键词路权重更高"的设计。

        输入：
            向量路排名  [1, 2, 3]
            关键词路排名 [3, 1, 4]
        期望：
            3 排第一 —— 它在关键词路排第 1（贡献 2.5/61），在向量路排第 3（贡献 1/63），
            合计最高。这条断言正是"关键词路权重 2.5"的验证：
            若权重为 1，1 在向量路第 1、关键词路第 2 会胜出。
        另外断言 1 和 4 都在结果里：
            4 只被关键词路召回（向量路没有它），
            验证"只命中一路的有价值片段不会被丢掉"这个设计意图。
        """
        merged = rrf_merge([1, 2, 3], [3, 1, 4])
        self.assertEqual(merged[0], 3)
        self.assertIn(1, merged)
        self.assertIn(4, merged)


if __name__ == "__main__":
    # 支持 python tests/test_core.py 直接运行（不依赖 pytest）
    unittest.main()
