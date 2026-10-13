# 工单编号：人工智能NLP-RAG-混合检索任务
"""组织结构图增量入库的**目标库一致性**回归测试。

背景（2026-10-13 实测踩坑）：
  `build_orgchart.ingest_orgchart` 里，删除/计数曾写死 `config.COLLECTION_OPT`，
  而 `vector_store.upsert_chunks(chunks, vectors)` 不传 name、走
  `default_name()`（跟随**活动嵌入模型**）。于是执行 README 里记载的
  `python build_index.py --embedding-model bge-m3-ft-v1` 时：
      · 删除目标 = 主库 zhaogu_v6_hybrid（真的删掉了 30 个 orgchart 点）
      · 写入目标 = 活动模型的库 zhaogu_v6_hybrid__bge-m3-ft-v1
  → 主库 8010 → 7980，id5（组织结构图题）的证据块丢失。

本测试钉住不变式：入库目标必须等于 upsert 的缺省库（default_name）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_orgchart  # noqa: E402
from src import config, embedder, vector_store  # noqa: E402


def test_target_collection_follows_active_embedding_model(monkeypatch):
    """切模型后，入库目标必须跟着切（而不是停在默认库）。"""
    monkeypatch.setattr(embedder, "active_embedding_model", lambda: "bge-m3-ft-v1")
    assert build_orgchart._target_collection_name() == config.collection_name("bge-m3-ft-v1")
    assert build_orgchart._target_collection_name() != config.COLLECTION_OPT

    monkeypatch.setattr(embedder, "active_embedding_model", lambda: "bge-m3")
    assert build_orgchart._target_collection_name() == config.COLLECTION_OPT


def test_target_collection_matches_upsert_default(monkeypatch):
    """与 upsert 的缺省库严格一致 —— 这正是 bug 的根因（两个名字曾经分叉）。"""
    for model in ("bge-m3", "bge-m3-ft-v1", "miniLM-L6"):
        monkeypatch.setattr(embedder, "active_embedding_model", lambda m=model: m)
        assert build_orgchart._target_collection_name() == vector_store.default_name()
