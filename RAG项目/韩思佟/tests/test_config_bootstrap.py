"""Regression tests for configuration loading before third-party imports."""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("RAG_TEST_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
SOURCE = PROJECT_ROOT / "app" / "single_app.py"
SENTINEL = "RAG_BOOTSTRAP_ORDER_TEST"
MISSING = object()


class ObservingMilvusLoader(importlib.abc.Loader):
    """Act like a dependency that loads a stale cwd .env during import."""

    def __init__(self):
        self.observed_before_stale_load = None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.observed_before_stale_load = os.environ.get(SENTINEL)
        os.environ.setdefault(SENTINEL, "stale-third-party-value")
        module.MilvusClient = type("StubMilvusClient", (), {})


class MilvusFinder(importlib.abc.MetaPathFinder):
    def __init__(self, loader):
        self.loader = loader

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymilvus":
            return importlib.util.spec_from_loader(fullname, self.loader)
        return None


class ConfigBootstrapTests(unittest.TestCase):
    def import_case(self, file_value, shell_value=MISSING):
        with tempfile.TemporaryDirectory() as tempdir:
            env_file = Path(tempdir) / "project.env"
            env_file.write_text(f"{SENTINEL}={file_value}\n", encoding="utf-8")
            tracked_env = ("RAG_ENV_FILE", SENTINEL)
            saved_env = {name: os.environ.get(name, MISSING) for name in tracked_env}
            tracked_modules = ("pymilvus", "sentence_transformers")
            saved_modules = {name: sys.modules.get(name, MISSING) for name in tracked_modules}
            module_name = f"single_app_bootstrap_{uuid.uuid4().hex}"
            loader = ObservingMilvusLoader()
            finder = MilvusFinder(loader)
            try:
                os.environ["RAG_ENV_FILE"] = str(env_file)
                if shell_value is MISSING:
                    os.environ.pop(SENTINEL, None)
                else:
                    os.environ[SENTINEL] = shell_value
                sys.modules.pop("pymilvus", None)
                sys.modules["sentence_transformers"] = types.SimpleNamespace(
                    CrossEncoder=type("StubCrossEncoder", (), {}),
                    SentenceTransformer=type("StubSentenceTransformer", (), {}),
                )
                sys.meta_path.insert(0, finder)
                spec = importlib.util.spec_from_file_location(module_name, SOURCE)
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return loader.observed_before_stale_load, module.setting(SENTINEL)
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)
                sys.modules.pop(module_name, None)
                for name, value in saved_modules.items():
                    if value is MISSING:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
                for name, value in saved_env.items():
                    if value is MISSING:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_project_env_is_loaded_before_dependency_can_set_stale_value(self):
        observed, final_value = self.import_case("project-file-value")
        self.assertEqual("project-file-value", observed)
        self.assertEqual("project-file-value", final_value)

    def test_explicit_shell_environment_still_overrides_project_file(self):
        observed, final_value = self.import_case("project-file-value", "shell-value")
        self.assertEqual("shell-value", observed)
        self.assertEqual("shell-value", final_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
