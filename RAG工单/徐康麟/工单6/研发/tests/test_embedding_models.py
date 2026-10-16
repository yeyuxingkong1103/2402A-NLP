# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-混合检索任务（工单6）

工单编号：人工智能NLP-RAG-混合检索任务

``app/core/embedding_models.py`` 的**离线单元测试**（``probe=False``，绝不访问 Ollama）：

1. ``KNOWN_MODELS`` 至少含 bge-m3（1024 维）与 m3e-base（768 维）；
2. ``resolve_model`` 接受短 id / Ollama 标签 / 省略 ``:latest`` 的标签 / 索引 slug / 空串（默认 bge-m3）；
3. ``resolve_model`` 对未知模型**抛 ValueError**（不静默回退，避免悄悄换模型）；
4. ``list_models`` 的 id / slug / 索引状态标注；``index_dir_for`` 与 ``EmbeddingModelSpec`` 契约。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_embedding_models.py -q
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.embedding_models import (DEFAULT_MODEL_ID, KNOWN_MODELS,  # noqa: E402
                                       EmbeddingModelSpec, describe_models, embed_texts_with_model,
                                       index_dir_for, list_models, resolve_model)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


@pytest.fixture()
def tmp_index_root():
    """工作区内的临时索引根目录（跑完删除）。

    **不用** pytest 的 ``tmp_path`` / ``tempfile``：本机沙箱下系统临时目录不可写
    （既有测试 ``test_dialogue_cache.py`` 已实测记录该坑），故统一建在仓库内。
    """
    path = REPO_ROOT / ".pytest_runtime" / f"embedding_models_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# ① 注册表
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id,expected_dim,expected_tag", [
    ("bge-m3", 1024, "bge-m3:latest"),
    ("m3e-base", 768, "turingdance/m3e-base:latest"),
], ids=["bge-m3", "m3e-base"])
def test_known_models_contains_required_models(model_id: str, expected_dim: int,
                                               expected_tag: str) -> None:
    spec = resolve_model(model_id, probe=False)
    assert spec.id == model_id
    assert spec.dim == expected_dim
    assert spec.ollama_model == expected_tag
    assert KNOWN_MODELS[model_id]["dim"] == expected_dim
    assert spec.label and spec.slug == f"{model_id}_{expected_dim}"


def test_known_models_all_have_required_metadata() -> None:
    assert {"bge-m3", "m3e-base"} <= set(KNOWN_MODELS)
    for model_id, meta in KNOWN_MODELS.items():
        assert meta["ollama_model"] and isinstance(meta["ollama_model"], str)
        assert int(meta["dim"]) > 0
        assert meta["label"]


# ---------------------------------------------------------------------------
# ② resolve_model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alias", [
    "m3e-base", "turingdance/m3e-base:latest", "turingdance/m3e-base", "m3e-base_768", "M3E-BASE",
], ids=["短id", "完整标签", "省略latest", "索引slug", "大写"])
def test_resolve_model_accepts_every_alias_form(alias: str) -> None:
    spec = resolve_model(alias, probe=False)
    assert spec.id == "m3e-base" and spec.dim == 768


@pytest.mark.parametrize("raw", ["", "   ", None], ids=["空串", "空白", "None"])
def test_resolve_model_empty_falls_back_to_default(raw: object) -> None:
    spec = resolve_model(raw, probe=False)  # type: ignore[arg-type]
    assert spec.id == DEFAULT_MODEL_ID == "bge-m3"
    assert spec.dim == 1024


@pytest.mark.parametrize("bad", ["不存在的模型", "bge", "m3e", "gpt-4", "bge-m3_512"],
                         ids=["中文未知", "bge", "m3e", "gpt-4", "维度不符的slug"])
def test_resolve_model_unknown_raises_value_error(bad: str) -> None:
    """未知模型必须报错：静默回退会让「换了模型但没换索引」变成错误检索结果。"""
    with pytest.raises(ValueError) as excinfo:
        resolve_model(bad, probe=False)
    assert "未知嵌入模型" in str(excinfo.value)


def test_resolve_model_marks_index_state_from_index_root(tmp_index_root: Path) -> None:
    spec = resolve_model("m3e-base", index_root=tmp_index_root, probe=False)
    assert Path(spec.index_dir) == tmp_index_root / "m3e-base_768"
    assert spec.index_built is False, "目录里还没有 vectors.npy"
    (tmp_index_root / "m3e-base_768").mkdir(parents=True, exist_ok=True)
    (tmp_index_root / "m3e-base_768" / "vectors.npy").write_bytes(b"")
    again = resolve_model("m3e-base", index_root=tmp_index_root, probe=False)
    assert again.index_built is True


# ---------------------------------------------------------------------------
# ③ list_models / describe_models / index_dir_for
# ---------------------------------------------------------------------------
def test_list_models_without_probe_lists_all_known_models() -> None:
    specs = list_models(probe=False)
    assert [spec.id for spec in specs] == list(KNOWN_MODELS)
    assert all(spec.dim == int(KNOWN_MODELS[spec.id]["dim"]) for spec in specs)
    assert all(spec.slug == f"{spec.id}_{spec.dim}" for spec in specs)
    assert all(spec.available is True for spec in specs), "probe=False 视为本机可用（不做网络探测）"


def test_list_models_marks_index_built_only_when_vectors_exist(tmp_index_root: Path) -> None:
    before = {spec.id: spec.index_built for spec in list_models(index_root=tmp_index_root, probe=False)}
    assert set(before.values()) == {False}
    (tmp_index_root / "bge-m3_1024").mkdir(parents=True, exist_ok=True)
    (tmp_index_root / "bge-m3_1024" / "vectors.npy").write_bytes(b"")
    after = {spec.id: spec.index_built for spec in list_models(index_root=tmp_index_root, probe=False)}
    assert after["bge-m3"] is True
    assert after["m3e-base"] is False
    assert all(Path(spec.index_dir) == tmp_index_root / spec.slug
               for spec in list_models(index_root=tmp_index_root, probe=False))


def test_describe_models_returns_serialisable_rows() -> None:
    rows = describe_models(probe=False)
    assert rows and all(isinstance(row, dict) for row in rows)
    assert {"id", "ollama_model", "dim", "label", "slug", "index_built"} <= set(rows[0])


def test_index_dir_for_and_spec_to_dict_contract() -> None:
    spec = resolve_model("bge-m3", probe=False)
    assert index_dir_for(spec, "研发/data/index") == Path("研发/data/index") / "bge-m3_1024"
    payload = spec.to_dict()
    assert payload["id"] == "bge-m3" and payload["dim"] == 1024
    assert payload["slug"] == "bge-m3_1024"
    assert set(payload) == {"id", "ollama_model", "dim", "label", "note", "available",
                            "index_built", "index_dir", "slug"}


# ---------------------------------------------------------------------------
# ④ 空输入嵌入（不触发任何 HTTP）
# ---------------------------------------------------------------------------
def test_embed_texts_with_model_empty_input_returns_zero_rows() -> None:
    spec = resolve_model("m3e-base", probe=False)
    result = embed_texts_with_model([], spec)
    assert result.count == 0
    assert result.dim == 768
    assert result.vectors.shape == (0, 768)
    assert result.model == "turingdance/m3e-base:latest"
    assert result.backend == "ollama"
    assert result.notes == ["空输入"]
    assert result.to_dict()["count"] == 0


def test_embedding_model_spec_is_a_plain_dataclass() -> None:
    spec = EmbeddingModelSpec(id="demo", ollama_model="demo:latest", dim=8, label="演示")
    assert spec.slug == "demo_8"
    assert spec.available is True and spec.index_built is False and spec.index_dir == ""
