# -*- coding: utf-8 -*-
"""工单6 多嵌入模型注册表 —— 支持 **bge / m3e** 等多模型可配置切换。

工单编号：人工智能NLP-RAG-混合检索任务

工单原文要求
------------
「使用嵌入模型将文档片段和用户 Query 转为向量……**支持多种嵌入模型（如 bge、m3e），
至少可配置一种**。」

本机事实（2026-10 实测，``GET http://127.0.0.1:11434/api/tags``）
----------------------------------------------------------------
Ollama 已装且可直接用于嵌入的模型（**不是**猜测，是接口返回的 ``details.embedding_length``）：

======================  ==========  ==========================================
Ollama 模型标签           维度        说明
======================  ==========  ==========================================
``bge-m3:latest``        1024        工单4 既有索引所用模型（**默认**，多语、8k 上下文）
``turingdance/m3e-base`` 768         m3e-base（工单点名要求的 m3e 家族）
``bge-large:latest``     1024        bge-large-zh
``nomic-embed-text``     768         英文向
``all-minilm:l6-v2``     384         轻量英文向
======================  ==========  ==========================================

设计要点
--------
* **维度即索引身份**：向量索引目录为 ``研发/data/index/{slug}_{dim}/``，
  例如 ``bge-m3_1024``（既有）、``m3e-base_768``（可由 ``scripts/build_index_multi.py`` 构建）。
  切换嵌入模型 = 切换到该模型的索引目录，**检索代码不变**。
* **不同模型的向量不可混用**（不同语义空间/维度）⇒ ``resolve_index_dir`` 强制按
  ``模型 → 目录`` 映射，找不到索引时**显式报错或回退默认模型**，绝不静默跨模型检索。
* 嵌入走 Ollama ``/api/embed``，**复用 工单4 ``embedder._http_json`` 的同一调用口径**
  （超时、批量、错误处理一致），只把 ``model`` 参数化。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

#: 默认嵌入模型（= 工单4 既有索引所用；不改动既有产物）
DEFAULT_MODEL_ID = "bge-m3"


@dataclass(slots=True)
class EmbeddingModelSpec:
    """一个可配置的嵌入模型。"""

    id: str                      # 短 id（界面/配置用），如 "bge-m3"
    ollama_model: str            # Ollama 模型标签，如 "bge-m3:latest"
    dim: int                     # 向量维度（= 索引身份的一部分）
    label: str                   # 中文名（界面展示）
    note: str = ""               # 说明
    available: bool = True       # 探测结果（本机是否已装）
    index_built: bool = False    # 本机是否已建好对应向量索引
    index_dir: str = ""

    @property
    def slug(self) -> str:
        """索引目录名：``{id}_{dim}``。"""
        return f"{self.id}_{self.dim}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "ollama_model": self.ollama_model, "dim": self.dim,
            "label": self.label, "note": self.note, "available": self.available,
            "index_built": self.index_built, "index_dir": self.index_dir, "slug": self.slug,
        }


#: 已知模型清单（``dim`` 取自 Ollama ``/api/tags`` 的 ``details.embedding_length``，实测值）
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "bge-m3": {
        "ollama_model": "bge-m3:latest", "dim": 1024,
        "label": "BGE-M3（1024 维，多语，默认）",
        "note": "工单4 既有索引所用模型；中英文均可，8k 上下文",
    },
    "m3e-base": {
        "ollama_model": "turingdance/m3e-base:latest", "dim": 768,
        "label": "M3E-Base（768 维，中文）",
        "note": "工单点名要求的 m3e 家族；中文语义强，512 上下文",
    },
    "bge-large": {
        "ollama_model": "bge-large:latest", "dim": 1024,
        "label": "BGE-Large（1024 维，中文）",
        "note": "中文向大模型；512 上下文",
    },
    "nomic-embed": {
        "ollama_model": "nomic-embed-text:latest", "dim": 768,
        "label": "Nomic-Embed-Text（768 维，英文向）",
        "note": "英文语料表现好；2k 上下文",
    },
    "all-minilm": {
        "ollama_model": "all-minilm:l6-v2", "dim": 384,
        "label": "All-MiniLM-L6-v2（384 维，轻量）",
        "note": "最快、最省内存；英文向，中文弱",
    },
}


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """极简 JSON HTTP（标准库；与 工单4 ``embedder._http_json`` 同口径）。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def probe_ollama_models(base_url: str = "http://127.0.0.1:11434",
                        timeout: float = 2.0) -> list[dict[str, Any]]:
    """探测 Ollama 上**可用于嵌入**（``capabilities`` 含 ``embedding``）的模型。"""
    try:
        data = _http_json(f"{base_url}/api/tags", None, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for item in data.get("models") or []:
        caps = item.get("capabilities") or []
        if caps and "embedding" not in caps:
            continue
        details = item.get("details") or {}
        out.append({
            "name": str(item.get("name") or ""),
            "dim": int(details.get("embedding_length") or 0),
            "parameter_size": details.get("parameter_size"),
        })
    return out


def list_models(*, index_root: Path | str | None = None,
                base_url: str = "http://127.0.0.1:11434",
                probe: bool = True, timeout: float = 2.0) -> list[EmbeddingModelSpec]:
    """列出全部**已知**模型，并标注本机「是否已装」「是否已建索引」。"""
    installed: dict[str, int] = {}
    if probe:
        for item in probe_ollama_models(base_url, timeout):
            installed[str(item["name"])] = int(item.get("dim") or 0)
    root = Path(index_root) if index_root is not None else None
    specs: list[EmbeddingModelSpec] = []
    for mid, meta in KNOWN_MODELS.items():
        tag = str(meta["ollama_model"])
        is_installed = tag in installed if probe else True
        dim = installed.get(tag) or int(meta["dim"])
        spec = EmbeddingModelSpec(id=mid, ollama_model=tag, dim=int(dim),
                                  label=str(meta["label"]), note=str(meta["note"]),
                                  available=bool(is_installed))
        if root is not None:
            candidate = root / spec.slug
            spec.index_dir = str(candidate)
            spec.index_built = (candidate / "vectors.npy").is_file()
        specs.append(spec)
    return specs


def resolve_model(model_id: str | None, *, index_root: Path | str | None = None,
                  base_url: str = "http://127.0.0.1:11434", probe: bool = False,
                  timeout: float = 2.0) -> EmbeddingModelSpec:
    """把用户输入的模型标识解析成 ``EmbeddingModelSpec``。

    接受：短 id（``m3e-base``）、Ollama 标签（``turingdance/m3e-base:latest``）、
    省略 ``:latest`` 的标签。未命中时抛 ``ValueError``（**不静默回退**，避免悄悄换模型）。
    """
    raw = str(model_id or "").strip()
    if not raw:
        raw = DEFAULT_MODEL_ID
    specs = list_models(index_root=index_root, base_url=base_url, probe=probe, timeout=timeout)
    lookup: dict[str, EmbeddingModelSpec] = {}
    for spec in specs:
        lookup[spec.id.lower()] = spec
        lookup[spec.ollama_model.lower()] = spec
        lookup[spec.ollama_model.split(":")[0].lower()] = spec
        lookup[spec.slug.lower()] = spec
    found = lookup.get(raw.lower())
    if found is None:
        raise ValueError(
            f"未知嵌入模型：{raw!r}（可选：{', '.join(s.id for s in specs)}）"
        )
    return found


# ---------------------------------------------------------------------------
# 嵌入调用（模型参数化）
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ModelEmbeddingResult:
    """一次模型化嵌入的结果。"""

    vectors: np.ndarray
    dim: int
    model: str
    count: int
    elapsed_ms: float
    backend: str = "ollama"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "dim": self.dim, "count": self.count,
                "elapsed_ms": self.elapsed_ms, "backend": self.backend,
                "notes": list(self.notes)}


def embed_texts_with_model(texts: Sequence[str], spec: EmbeddingModelSpec, *,
                           base_url: str = "http://127.0.0.1:11434",
                           batch_size: int = 16, timeout: float = 120.0,
                           logger: Any = None) -> ModelEmbeddingResult:
    """用**指定模型**批量嵌入（Ollama ``/api/embed``）。

    维度自检：返回向量的实际维度必须等于 ``spec.dim``，否则抛 ``ValueError``
    —— 维度不符意味着「模型换了但索引没换」，必须立刻失败而不是产出错误检索结果。
    """
    items = [str(t) if str(t).strip() else "（空块）" for t in texts]
    started = time.perf_counter()
    if not items:
        return ModelEmbeddingResult(vectors=np.zeros((0, spec.dim), dtype=np.float32),
                                    dim=spec.dim, model=spec.ollama_model, count=0,
                                    elapsed_ms=0.0, notes=["空输入"])
    vectors: list[list[float]] = []
    dim = 0
    for start in range(0, len(items), max(1, int(batch_size))):
        batch = list(items[start:start + max(1, int(batch_size))])
        payload = {"model": spec.ollama_model, "input": batch}
        try:
            data = _http_json(f"{base_url}/api/embed", payload, float(timeout))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Ollama 嵌入失败（model={spec.ollama_model}, offset={start}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        got = data.get("embeddings") or []
        if len(got) != len(batch):
            raise RuntimeError(
                f"Ollama 返回条数不符：期望 {len(batch)}，实际 {len(got)}（model={spec.ollama_model}）"
            )
        for vec in got:
            if dim == 0:
                dim = len(vec)
            elif len(vec) != dim:
                raise RuntimeError(f"同一批次内维度不一致：{dim} vs {len(vec)}")
        vectors.extend(got)
    array = np.asarray(vectors, dtype=np.float32)
    if array.size and int(array.shape[1]) != int(spec.dim):
        raise ValueError(
            f"模型 {spec.ollama_model} 实际维度 {array.shape[1]} 与注册维度 {spec.dim} 不符"
            f"（请检查 index 目录是否与该模型匹配）"
        )
    result = ModelEmbeddingResult(vectors=array, dim=int(array.shape[1]) if array.size else 0,
                                  model=spec.ollama_model, count=int(array.shape[0]),
                                  elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    if logger is not None:
        logger.log_event("embedding_models.embed", model=spec.ollama_model, count=result.count,
                         dim=result.dim, elapsed_ms=result.elapsed_ms)
    return result


def embed_query_with_model(text: str, spec: EmbeddingModelSpec, *,
                           base_url: str = "http://127.0.0.1:11434",
                           timeout: float = 60.0, logger: Any = None) -> np.ndarray:
    """用指定模型嵌入单条查询，返回 ``(dim,)`` float32 向量。"""
    result = embed_texts_with_model([text], spec, base_url=base_url, batch_size=1,
                                    timeout=timeout, logger=logger)
    if result.count != 1:
        raise RuntimeError(f"查询嵌入条数异常：{result.count}（model={spec.ollama_model}）")
    return result.vectors[0]


def index_dir_for(spec: EmbeddingModelSpec, index_root: Path | str) -> Path:
    """该模型对应的向量索引目录 ``{index_root}/{slug}``。"""
    return Path(index_root) / spec.slug


def describe_models(*, index_root: Path | str | None = None,
                    base_url: str = "http://127.0.0.1:11434",
                    probe: bool = True) -> list[dict[str, Any]]:
    """给界面/文档用的模型清单（含可用性与索引状态）。"""
    return [s.to_dict() for s in list_models(index_root=index_root, base_url=base_url, probe=probe)]


__all__ = [
    "WORK_ORDER", "DEFAULT_MODEL_ID", "KNOWN_MODELS", "EmbeddingModelSpec",
    "ModelEmbeddingResult", "probe_ollama_models", "list_models", "resolve_model",
    "embed_texts_with_model", "embed_query_with_model", "index_dir_for", "describe_models",
]
