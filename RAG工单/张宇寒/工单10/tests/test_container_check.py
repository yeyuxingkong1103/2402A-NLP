import unittest

from container_check import validate_results


class ContainerCheckTests(unittest.TestCase):
    def test_accepts_ready_service_and_target_metrics(self):
        summary = validate_results(
            {"ready": True},
            {"count": 4, "average_context_precision": 0.81, "average_context_recall": 0.91, "under_3_seconds": True},
        )
        self.assertEqual(summary["status"], "passed")

    def test_rejects_unready_or_below_target_service(self):
        with self.assertRaisesRegex(RuntimeError, "索引"):
            validate_results({"ready": False}, {"count": 4})
        with self.assertRaisesRegex(RuntimeError, "Context Precision"):
            validate_results({"ready": True}, {"count": 4, "average_context_precision": 0.79, "average_context_recall": 0.95, "under_3_seconds": True})
        with self.assertRaisesRegex(RuntimeError, "3秒"):
            validate_results({"ready": True}, {"count": 4, "average_context_precision": 0.9, "average_context_recall": 0.95, "under_3_seconds": False})


if __name__ == "__main__":
    unittest.main()
