import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from answer_test import build_context, extract_fallback_answer


class AnswerTestTests(unittest.TestCase):
    def test_extract_fallback_answer_for_contraindication(self):
        contexts = [
            {
                "id": "abc",
                "section_path": "5 高血压治疗 > 5.4.3药物治疗方案",
                "chunk_type": "table_row",
                "text": "表4基层常用降压药物 | 分类: C(二氢吡啶类钙拮抗剂） | 名称: 氨氯地平 | 禁忌证: 相对禁忌：快速性心律失常；慢性心力衰竭 | 主要不良反应: 头痛",
            }
        ]

        answer = extract_fallback_answer("氨氯地平的禁忌证是什么？", contexts)

        self.assertIn("快速性心律失常", answer)
        self.assertIn("慢性心力衰竭", answer)
        self.assertIn("5 高血压治疗", answer)

    def test_build_context_includes_rank_and_source(self):
        contexts = [
            {
                "id": "abc",
                "section_path": "章节路径",
                "chunk_type": "table_row",
                "rerank_score": 0.9,
                "text": "正文内容",
            }
        ]

        context = build_context(contexts)

        self.assertIn("[1]", context)
        self.assertIn("章节路径", context)
        self.assertIn("正文内容", context)


if __name__ == "__main__":
    unittest.main()
