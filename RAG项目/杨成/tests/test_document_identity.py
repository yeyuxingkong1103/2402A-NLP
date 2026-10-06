import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from document_identity import make_primary_key, normalize_document_id


class DocumentIdentityTests(unittest.TestCase):
    def test_primary_key_is_independent_of_source_file_path(self):
        first = make_primary_key("hypertension_guideline_2025", "child_00001")
        second = make_primary_key("hypertension_guideline_2025", "child_00001")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_different_document_ids_do_not_collide(self):
        first = make_primary_key("hypertension_guideline_2025", "child_00001")
        second = make_primary_key("hypertension_nutrition_exercise_2024", "child_00001")
        self.assertNotEqual(first, second)

    def test_document_id_rejects_empty_or_path_like_values(self):
        with self.assertRaises(ValueError):
            normalize_document_id("")
        with self.assertRaises(ValueError):
            normalize_document_id("../guide")


if __name__ == "__main__":
    unittest.main()
