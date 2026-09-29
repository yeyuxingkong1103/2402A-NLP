import unittest

from app.rag_main import chunk_text, clean_text


class ChunkingTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("甲  \t乙\n\n\n丙"), "甲 乙\n\n丙")

    def test_chunk_limits(self):
        text = "。".join(["高血压健康教育内容" * 8 for _ in range(20)]) + "。"
        chunks = chunk_text(text, size=200, overlap=30)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.summary) <= 120 for chunk in chunks))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("内容", size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
