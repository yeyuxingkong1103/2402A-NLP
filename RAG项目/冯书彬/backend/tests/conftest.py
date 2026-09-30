import os

import pytest
from fastapi.testclient import TestClient

# 单元和 API 测试默认不加载真实 Milvus/BGE 模型，避免本地 .env 影响回归稳定性。
os.environ.setdefault("MILVUS_URI", "")

from backend.app.main import app


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="GPU 真实模型测试需显式设置 RUN_GPU_TESTS=1")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def test_secrets(monkeypatch):
    # 所有测试使用固定测试密钥，生产配置仍必须显式提供真实密钥。
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")


@pytest.fixture
def client():
    # 复用同一个 FastAPI 应用实例，避免测试内重复构建路由。
    return TestClient(app)
