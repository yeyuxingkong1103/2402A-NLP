import unittest
from unittest.mock import patch

from app.rag_main import classify_document


class KnowledgeCatalogTests(unittest.TestCase):
    def test_fallback_uses_document_content(self):
        with patch("app.rag_main.settings.llm_enabled", False):
            result = classify_document("guide.pdf", "家庭血压测量指南\n测量前安静休息五分钟")
        self.assertEqual(result["title"], "家庭血压测量指南")
        self.assertEqual(len(result["questions"]), 3)


if __name__ == "__main__":
    unittest.main()
