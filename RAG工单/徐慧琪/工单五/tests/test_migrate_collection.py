# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""scripts/migrate_collection.py 的分支逻辑测试。

只测**决策分支**，不加载真实 embedder、不读 119MB 索引、不联网：
- Qdrant 走 tmp_path 本地嵌入式模式（不碰 data/qdrant）；
- embedder 维度用 stub 固定为 4（真实模型加载在 Windows 主线程会崩栈且慢）。
"""
import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_migrate_module():
    """按路径加载脚本（scripts/ 不是包，不能直接 import）。"""
    path = _ROOT / "scripts" / "migrate_collection.py"
    spec = importlib.util.spec_from_file_location("migrate_collection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_migrate_module()

OLD, NEW = "zhaogu_old_test", "zhaogu_new_test"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """隔离的本地 Qdrant（临时目录）。"""
    from src import config, vector_store as vs
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path))
    monkeypatch.setattr(config, "QDRANT_MODE", "local")
    vs.reset_client_for_tests()
    yield vs
    vs.reset_client_for_tests()


@pytest.fixture(autouse=True)
def stub_embedder_dim(monkeypatch):
    """不加载真实模型：维度固定 4。"""
    monkeypatch.setattr(mc.embedder, "get_dim", lambda: 4)


def _seed(vs, name, n, source="招股说明书1.pdf", dim=4):
    """造 n 个点（与源库同构的最小 payload）。"""
    vs.ensure_collection(dim=dim, name=name, recreate=True)
    chunks = [{"chunk_id": f"c{i:06d}", "text": f"文本 {i}", "source": source}
              for i in range(n)]
    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]
    vs.upsert_chunks(chunks, vectors, name=name, progress=False)
    return vs.count(name)


def test_migrates_when_destination_missing(store):
    """目标 collection 不存在 → 走迁移路径，点数与源一致。"""
    assert _seed(store, OLD, 3) == 3
    assert store.count(NEW) == 0

    rc = mc.migrate(OLD, NEW)

    assert rc == 3
    assert store.count(NEW) == 3
    assert store.count(OLD) == 3          # 源库只读，迁移不动它
    assert {p["source"] for p in store.scroll_all(NEW)} == {"招股说明书1.pdf"}


def test_noop_success_when_destination_complete(store, monkeypatch):
    """目标已存在且点数一致 → 直接返回成功，不重写、不碰 embedder。"""
    _seed(store, OLD, 3)
    _seed(store, NEW, 3, source="已完成.pdf")

    def _boom():
        raise AssertionError("点数据完整时不应加载 embedder")

    monkeypatch.setattr(mc.embedder, "get_dim", _boom)

    rc = mc.migrate(OLD, NEW)

    assert rc == 3
    # 目标内容仍是原来的（未被重新 upsert 覆盖）
    assert {p["source"] for p in store.scroll_all(NEW)} == {"已完成.pdf"}


def test_fails_when_destination_partial(store, capsys, monkeypatch):
    """迁移中断后目标点数不全 → 必须非 0 失败并告警，绝不静默成功。"""
    _seed(store, OLD, 5)
    _seed(store, NEW, 2, source="半成品.pdf")

    def _boom():
        raise AssertionError("点数不符时应立即报错，不应触碰 embedder")

    monkeypatch.setattr(mc.embedder, "get_dim", _boom)

    rc = mc.migrate(OLD, NEW)

    out, err = capsys.readouterr()
    assert rc != 0
    assert "点数不符" in err
    assert "跳过" not in out
    assert "迁移完成" not in out
    assert store.count(NEW) == 2          # 不擅自续写/覆盖
    assert store.count(OLD) == 5


def test_rebuild_recovers_partial_destination(store, capsys):
    """--rebuild：目标点数不全时清空重建并完整迁移。

    回归用例：本地模式下 recreate=True 会走 _purge_local_collection()，
    其第一步 close() 会把 migrate() 开头缓存的 client 关闭并置空；
    若不重新 get_client()，循环里的 scroll/upsert 会抛
    "QdrantLocal instance is closed"，且此时目标已被清空（0 点）。
    """
    assert _seed(store, OLD, 3) == 3
    _seed(store, NEW, 1, source="半成品.pdf")

    rc = mc.migrate(OLD, NEW, rebuild=True)

    out, err = capsys.readouterr()
    assert rc == 3
    assert "删除并重建" in out
    assert store.count(NEW) == 3
    # 重建后不得残留半成品数据，且内容与源一致
    assert {p["source"] for p in store.scroll_all(NEW)} == {"招股说明书1.pdf"}
    assert store.count(OLD) == 3          # 源库只读，重建不动它
