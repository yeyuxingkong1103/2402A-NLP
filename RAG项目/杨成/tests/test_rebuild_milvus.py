import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from insert_to_milvus import insert_batches, to_milvus_row
from rebuild_milvus import load_prepared_records, rebuild_milvus


class RebuildMilvusTests(unittest.TestCase):
    def test_refuses_to_drop_without_explicit_flag(self):
        with self.assertRaisesRegex(RuntimeError, "--drop-existing"):
            rebuild_milvus(None, "hypertension_rag", [], drop_existing=False)

    def test_load_prepared_records_rejects_duplicate_ids(self):
        class FakePath:
            def __init__(self, rows):
                self.rows = rows

            def exists(self):
                return True

            def open(self, *args, **kwargs):
                from io import StringIO
                import json

                return StringIO("\n".join(json.dumps(row) for row in self.rows))

            def __str__(self):
                return "fake.jsonl"

        rows = [
            {
                "id": "same",
                "text": "a",
                "vector": [0.1] * 1024,
                "document_id": "doc",
                "document_version": "v1",
                "source_file": "guide.md",
                "section_path": "section",
                "chunk_type": "text",
            },
            {
                "id": "same",
                "text": "b",
                "vector": [0.2] * 1024,
                "document_id": "doc",
                "document_version": "v1",
                "source_file": "guide.md",
                "section_path": "section",
                "chunk_type": "text",
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "重复 id"):
            load_prepared_records([FakePath(rows)])
    def test_insert_row_uses_prepared_stable_id_and_document_metadata(self):
        row = to_milvus_row(
            {
                "id": "stable-id",
                "text": "正文",
                "vector": [0.1] * 1024,
                "document_id": "doc",
                "document_version": "v1",
                "source_file": "guide.md",
                "section_path": "section",
                "chunk_type": "text",
            }
        )

        self.assertEqual(row["id"], "stable-id")
        self.assertEqual(row["document_id"], "doc")
        self.assertEqual(row["document_version"], "v1")

    def test_insert_row_normalizes_source_file_before_milvus(self):
        row = to_milvus_row(
            {
                "id": "stable-id",
                "text": "正文",
                "vector": [0.1] * 1024,
                "document_id": "doc",
                "document_version": "v1",
                "source_file": "data/parsed/高血压营养和运动指导原则（2024年版）/高血压营养和运动指导原则（2024年版）.md",
                "section_path": "section",
                "chunk_type": "text",
            }
        )

        self.assertEqual(row["source_file"], "高血压营养和运动指导原则（2024年版）")

        class FakeClient:
            def __init__(self):
                self.insert_calls = 0
                self.upsert_calls = 0
                self.flush_calls = 0

            def insert(self, collection_name, data):
                self.insert_calls += 1
                return {"insert_count": len(data)}

            def upsert(self, collection_name, data):
                self.upsert_calls += 1
                return {"upsert_count": len(data)}

            def flush(self, collection_name):
                self.flush_calls += 1

        records = [{"id": "stable-id"}]
        insert_client = FakeClient()
        upsert_client = FakeClient()

        insert_batches(insert_client, "hypertension_rag", records, batch_size=50, upsert=False)
        insert_batches(upsert_client, "hypertension_rag", records, batch_size=50, upsert=True)

        self.assertEqual(insert_client.insert_calls, 1)
        self.assertEqual(insert_client.upsert_calls, 0)
        self.assertEqual(upsert_client.insert_calls, 0)
        self.assertEqual(upsert_client.upsert_calls, 1)


if __name__ == "__main__":
    unittest.main()
