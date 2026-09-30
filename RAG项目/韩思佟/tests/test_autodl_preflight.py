"""Protect the deployment checks against silently accepting partial uploads."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "deploy/autodl/preflight.py"
spec = importlib.util.spec_from_file_location("preflight", MODULE)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def write_tensor(path, size=16):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [size // 4], "data_offsets": [0, size]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(size))


class PreflightTests(unittest.TestCase):
    def test_truncated_nonempty_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            tensor = Path(folder) / "model.safetensors"
            write_tensor(tensor)
            self.assertEqual(preflight.safetensors_size(tensor), 16)
            tensor.write_bytes(tensor.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete model file"):
                preflight.safetensors_size(tensor)

    def test_model_missing_index_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (model / name).write_text("{}")
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 32},
                "weight_map": {"a": "first.safetensors", "b": "second.safetensors"},
            }))
            write_tensor(model / "first.safetensors")
            with self.assertRaisesRegex(ValueError, "Missing or empty file"):
                preflight.check_model(model)
            write_tensor(model / "second.safetensors")
            preflight.check_model(model)

    def test_cpu_mode_never_calls_gpu_check(self):
        args = [str(MODULE), "--model-path", "unused", "--api-python", "base", "--vllm-python", "vllm"]
        with patch("sys.argv", args), patch.object(preflight, "check_project"), patch.object(preflight, "check_model"), patch.object(preflight, "check_environment"), patch.object(preflight, "check_gpu") as gpu:
            self.assertEqual(preflight.main(), 0)
            gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
