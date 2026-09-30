import unittest
from pathlib import Path
from unittest.mock import patch

from app import rag_main


class SamplePdfTests(unittest.TestCase):
    def test_sample_pdf_has_extractable_text(self):
        path = Path("output/pdf/medical_knowledge_sample.pdf")
        if not path.exists():
            self.skipTest("Run scripts/generate_sample_pdf.py first")
        with patch.object(rag_main.settings, "paddleocr_vl_enabled", False):
            text = rag_main.extract_pdf_text(path)
        self.assertIn("高血压", text)
        self.assertIn("国家卫生健康委员会", text)


if __name__ == "__main__":
    unittest.main()
