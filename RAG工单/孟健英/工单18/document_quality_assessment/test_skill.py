# 工单编号：人工智能 NLP-RAG-实现文档质量评估 Skill 并集成至智能体工作流工单
"""DocumentQualityAssessmentSkill 单元测试"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from skill import DocumentQualityAssessmentSkill

BASE = "/mnt/d/专高六 工单/工单18"
CONFIG = os.path.join(BASE, "document_quality_assessment", "assessment_config.yaml")
DATA = os.path.join(BASE, "data", "documents")


class TestDocumentQualityAssessmentSkill(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.skill = DocumentQualityAssessmentSkill(CONFIG)

    def test_01_config_loaded(self):
        """配置加载正常"""
        self.assertIn("scan_detection", self.skill.config)
        self.assertIn("length_distribution", self.skill.config)
        self.assertIn("duplicate_detection", self.skill.config)
        self.assertIn("sensitive_detection", self.skill.config)

    def test_02_format_distribution(self):
        """格式分布统计"""
        files = [Path(f) for f in os.listdir(DATA)[:50]]
        files = [Path(os.path.join(DATA, f)) for f in files if os.path.isfile(os.path.join(DATA, f))]
        result = self.skill._format_distribution(files)
        self.assertIn("total", result)
        self.assertIn("by_format", result)
        self.assertEqual(result["total"], len(files))
        print(f"[TEST] 格式分布: {result['by_format'][:3]}")

    def test_03_pdf_inspection(self):
        """PDF 类型识别"""
        pdf_files = [f for f in os.listdir(DATA) if f.endswith(".pdf")][:5]
        for fn in pdf_files:
            info = self.skill._inspect_file(Path(os.path.join(DATA, fn)))
            self.assertIn(info["pdf_type"], ["Text_PDF", "Scan_PDF", "Mixed_PDF"])
            print(f"[TEST] {fn}: {info['pdf_type']}, 字符数 {info['length']}")

    def test_04_length_distribution(self):
        """长度分布"""
        lengths = [100, 500, 1000, 5000, 20000, 50000, 80000, 120000]
        result = self.skill._length_distribution(lengths)
        self.assertIn("percentiles", result)
        self.assertIn("P50", result["percentiles"])
        print(f"[TEST] 长度分位数: {result['percentiles']}")

    def test_05_duplicate_detection(self):
        """重复检测"""
        files = [Path(os.path.join(DATA, f)) for f in os.listdir(DATA)[:30]]
        result = self.skill._duplicate_detection(files)
        self.assertIn("md5_duplicates", result)
        self.assertIn("simhash_pending_pairs", result)
        print(f"[TEST] MD5 重复组: {result['md5_duplicate_groups']}")

    def test_06_sensitive_detection(self):
        """敏感信息检测"""
        # 用临时文件测
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False, encoding="utf-8") as f:
            f.write("联系人手机号 13812345678，邮箱 test@example.com，身份证 110101199001011234")
            tmp = f.name
        try:
            result = self.skill._sensitive_detection([Path(tmp)], [])
            self.assertGreaterEqual(result["total"], 3)
            types = result["summary"].keys()
            self.assertIn("phone", types)
            self.assertIn("email", types)
            self.assertIn("id_card", types)
            print(f"[TEST] 敏感信息: {result['summary']}")
        finally:
            os.remove(tmp)

    def test_07_assess_folder_small(self):
        """端到端：小规模"""
        report = self.skill.assess_folder(DATA, max_files=20)
        self.assertEqual(report["meta"]["total_files"], 20)
        self.assertIn("format_distribution", report)
        self.assertIn("pdf_type_distribution", report)
        self.assertIn("length_distribution", report)
        self.assertIn("duplicate_detection", report)
        self.assertIn("sensitive_detection", report)
        print(f"[TEST] 端到端 20 文件: {report['pdf_type_distribution']['summary']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
