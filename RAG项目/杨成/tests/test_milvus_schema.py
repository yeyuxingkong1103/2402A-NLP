import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import init_milvus


class MilvusSchemaTests(unittest.TestCase):
    def test_schema_includes_dense_bm25_and_document_metadata(self):
        schema = init_milvus.build_schema()
        field_names = {field.name for field in schema.fields}

        self.assertEqual(
            {
                "id",
                "vector",
                "text",
                "section_path",
                "chunk_type",
                "source_file",
                "document_id",
                "document_version",
                "sparse_vector",
            },
            field_names,
        )
        self.assertEqual("BM25", schema.functions[0].type.name)
        self.assertEqual(["text"], schema.functions[0].input_field_names)
        self.assertEqual(["sparse_vector"], schema.functions[0].output_field_names)
        text_field = next(field for field in schema.fields if field.name == "text")
        self.assertEqual({"type": "chinese"}, json.loads(text_field.analyzer_params))

    def test_index_params_include_dense_and_sparse_indexes(self):
        index_params = init_milvus.build_index_params()
        fields = {item["field_name"]: item for item in index_params}

        self.assertEqual(fields["vector"]["index_params"]["index_type"], "HNSW")
        self.assertEqual(fields["vector"]["index_params"]["metric_type"], "COSINE")
        self.assertEqual(fields["sparse_vector"]["index_params"]["index_type"], "SPARSE_INVERTED_INDEX")
        self.assertEqual(fields["sparse_vector"]["index_params"]["metric_type"], "BM25")


if __name__ == "__main__":
    unittest.main()
