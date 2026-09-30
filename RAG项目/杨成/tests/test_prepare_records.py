import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_records import prepare_jsonl, normalize_source_file


class PrepareRecordsTests(unittest.TestCase):
    def test_normalize_source_file_keeps_document_name_and_parentheses(self):
        self.assertEqual(
            normalize_source_file("data/parsed/高血压营养和运动指导原则（2024年版）/高血压营养和运动指导原则（2024年版）.md"),
            "高血压营养和运动指导原则（2024年版）",
        )
        self.assertEqual(
            normalize_source_file("data/parsed/国家基层高血压防治管理指南2025版.full.md"),
            "国家基层高血压防治管理指南2025版",
        )

    def test_adds_document_metadata_and_stable_id(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.jsonl"
            output = Path(temp_dir) / "prepared.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "child_id": "child_00001",
                        "text": "内容",
                        "vector": [0.1] * 1024,
                        "source_file": "data/parsed/guide.md",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            count = prepare_jsonl(source, output, "hypertension_guideline_2025", "2025")
            record = json.loads(output.read_text(encoding="utf-8").strip())

            self.assertEqual(count, 1)
            self.assertEqual(record["document_id"], "hypertension_guideline_2025")
            self.assertEqual(record["document_version"], "2025")
            self.assertEqual(record["child_id"], "child_00001")
            self.assertEqual(record["source_file"], "guide")
            self.assertEqual(len(record["id"]), 32)

    def test_rejects_wrong_dense_dimension(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.jsonl"
            output = Path(temp_dir) / "prepared.jsonl"
            source.write_text(json.dumps({"child_id": "c1", "text": "x", "vector": [0.1]}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "向量维度异常"):
                prepare_jsonl(source, output, "hypertension_guideline_2025", "2025")

    def test_rejects_duplicate_ids_in_same_output(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.jsonl"
            output = Path(temp_dir) / "prepared.jsonl"
            records = [
                {"child_id": "child_00001", "text": "a", "vector": [0.1] * 1024},
                {"child_id": "child_00001", "text": "b", "vector": [0.2] * 1024},
            ]
            source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "重复 id"):
                prepare_jsonl(source, output, "hypertension_guideline_2025", "2025")


if __name__ == "__main__":
    unittest.main()
