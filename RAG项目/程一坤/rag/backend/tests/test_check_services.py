"""check_services.py 自检脚本本身的回归测试（批次 32）。

背景：report() 的 docstring 承诺"失败时登记到 failures"，但实现没有 append，
导致"打印 ❌ 但退出码 0"的假绿灯（LLM 探针空内容就是真实案例）。
本文件锁三件事：
1. report() 失败时必须登记 failures（key 参数与默认 name 两种）；
2. 失败登记后 main() 退出码必须非 0；
3. 外部 API 探针的"非异常失败"（Reranker 空 results / LLM 空 content）
   也必须登记——这两条正是修复前的假绿灯路径。

加载方式：check_services.py 在 scripts/ 下、不在包里，用 importlib 按路径加载，
模块级 failures 列表随用随清。
"""

import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_services.py"


@pytest.fixture()
def cs(monkeypatch):
    """按路径加载 check_services 模块，并隔离环境（不读真实 .env、不发真实网络请求）。"""
    monkeypatch.setattr(sys, "argv", ["check_services.py"])
    spec = importlib.util.spec_from_file_location("check_services_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.failures.clear()
    return module


def _fake_urlopen(dispatch: dict[str, dict]):
    """按 URL 前缀返回伪造响应体；不在表内的 URL 直接抛错（模拟网络不可达）。"""

    class FakeResp:
        def __init__(self, body: dict):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self._body).encode()

    def fake_urlopen(req, timeout=30):
        url = req.full_url
        for prefix, body in dispatch.items():
            if url.startswith(prefix):
                return FakeResp(body)
        raise OSError(f"unreachable: {url}")

    return fake_urlopen


def _fake_embedding_module(monkeypatch):
    """注入伪造的 app.models.embedding，避免真实 import 拖起整条后端依赖链。"""
    fake_pkg = types.ModuleType("app")
    fake_pkg.__path__ = []
    fake_models = types.ModuleType("app.models")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts):
            return [[0.1] * 4]

    fake_embedding = types.ModuleType("app.models.embedding")
    fake_embedding.SiliconFlowEmbeddingClient = FakeClient
    for name, mod in (
        ("app", fake_pkg),
        ("app.models", fake_models),
        ("app.models.embedding", fake_embedding),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def test_report_failure_registers_key(cs):
    """失败必须登记 failures：显式 key 与默认 name 两条路都要落；成功不登记。"""
    cs.report("展示名甲", False, key="k1")
    cs.report("展示名乙", False)
    cs.report("成功项", True)
    assert cs.failures == ["k1", "展示名乙"]


def test_main_exit_code_nonzero_when_failures(cs, monkeypatch, capsys):
    """有失败登记时 main() 必须返回 1（退出码契约）。"""
    monkeypatch.setattr(cs, "load_env", lambda: None)
    monkeypatch.setattr(
        cs,
        "check_docker",
        lambda: cs.report("docker 命令可用", False, "缺失", "启动 Docker", key="docker") or False,
    )
    monkeypatch.setattr(cs, "check_redis", lambda: True)
    monkeypatch.setattr(cs, "check_mysql", lambda: None)
    monkeypatch.setattr(cs, "check_milvus", lambda counts: True)
    assert cs.main() == 1
    assert "docker" in cs.failures
    capsys.readouterr()  # 丢弃打印输出


def test_main_exit_code_zero_when_all_pass(cs, monkeypatch, capsys):
    """全绿时 main() 必须返回 0——反向锁死"退出码 0 只属于真全绿"。"""
    monkeypatch.setattr(cs, "load_env", lambda: None)
    monkeypatch.setattr(cs, "check_docker", lambda: True)
    monkeypatch.setattr(cs, "check_redis", lambda: True)
    monkeypatch.setattr(cs, "check_mysql", lambda: {"document_chunks": 1})
    monkeypatch.setattr(cs, "check_milvus", lambda counts: True)
    assert cs.main() == 0
    assert cs.failures == []
    capsys.readouterr()


def test_llm_empty_content_registers_failure(cs, monkeypatch):
    """场景：LLM 返回空 content（修复前的真实假绿灯）→ 必须登记 llm 失败。"""
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://embed.test/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_API_BASE_URL", "http://rerank.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    _fake_embedding_module(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _fake_urlopen(
            {
                "http://rerank.test/": {"results": [{"index": 0, "relevance_score": 0.9}]},
                "http://llm.test/": {"choices": [{"message": {"content": ""}}]},
            }
        ),
    )
    cs.check_external_api()
    assert "llm" in cs.failures


def test_reranker_empty_results_registers_failure(cs, monkeypatch):
    """场景：Reranker 返回空 results（非异常的假绿灯路径）→ 必须登记 reranker 失败。"""
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://embed.test/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_API_BASE_URL", "http://rerank.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    _fake_embedding_module(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _fake_urlopen(
            {
                "http://rerank.test/": {"results": []},
                "http://llm.test/": {"choices": [{"message": {"content": "正常"}}]},
            }
        ),
    )
    cs.check_external_api()
    assert "reranker" in cs.failures
