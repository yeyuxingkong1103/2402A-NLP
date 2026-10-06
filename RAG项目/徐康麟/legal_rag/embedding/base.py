# -*- coding: utf-8 -*-
"""向量化接口与工厂。

约定：所有后端都实现 embed_texts / embed_query，返回等长浮点向量。
真实后端（BGE-m3）与零依赖兜底实现（offline）可互换。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

__all__ = ["Embedder", "build_embedder", "resolve_ollama_embed_model",
           "DEFAULT_OLLAMA_EMBED_MODEL"]


class Embedder(ABC):
    """向量化模型接口。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（文档侧）。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def ensure_dim(self) -> int:
        """返回（必要时实测）向量维度。

        **维度不写死**：能声明就声明（``self.dim``），声明不了就真跑一次嵌入量长度。
        真实后端（``OllamaEmbedder``）会覆盖本方法，从**第一次真实响应**取维度并缓存；
        Milvus 侧在建表/写入前会调它做校验。
        """
        current = int(getattr(self, "dim", 0) or 0)
        if current > 0:
            return current
        vectors = self.embed_texts(["维度探测 probe"])
        if not vectors or not vectors[0]:
            raise RuntimeError(
                f"无法确定 {self.name} 的向量维度：探测响应为空。\n"
                f"提示：检查后端是否可用（Ollama: curl http://127.0.0.1:11434/api/tags）。")
        self.dim = len(vectors[0])
        return self.dim


#: Ollama 官方库里的嵌入模型名（不带命名空间的短名）
DEFAULT_OLLAMA_EMBED_MODEL = "bge-m3"


def _is_hf_style_model_name(name: str) -> bool:
    """是不是「HuggingFace 仓库名」风格（如 ``BAAI/bge-m3``）。

    Ollama 只认 ``bge-m3`` / ``bge-m3:latest`` 这类短名；把 HF 仓库名直接发给
    Ollama 一定 404。这里用「含斜杠且不是 ``localhost``/``主机:端口`` 形式」判断，
    避免把合法的 registry 地址误判成 HF 名。
    """
    text = (name or "").strip()
    if "/" not in text:
        return False
    # host:port/name 形态（自建 registry）不算 HF 仓库名
    head = text.split("/", 1)[0]
    return ":" not in head and "." not in head


def resolve_ollama_embed_model(explicit: str = "", config=None) -> str:
    """决定请求 Ollama ``/api/embed`` 时用的模型名。

    背景（t7 F-t2-02 blocker）
    -------------------------
    ``EMBEDDING_MODEL`` 这个环境变量被 **两个含义不同的字段**解释：

    * ``config.embedding_model`` —— 语义是 HuggingFace 仓库名，默认 ``BAAI/bge-m3``；
    * ``config.ollama.embedding_model`` —— 语义是 Ollama 模型名，默认 ``bge-m3``。

    engine 以前把前者当 Ollama 模型名传进来（还挡住了后者），于是实际发的是
    ``model="BAAI/bge-m3"``，Ollama 只有 ``bge-m3:latest`` → **HTTP 404**。

    本函数把优先级钉死：

    1. ``explicit`` 且**看起来是 Ollama 能认的名字** → 用 explicit（真正的显式覆盖）；
    2. 否则用 ``config.ollama.embedding_model``（本项目里正确的那一份）；
    3. 再否则用 :data:`DEFAULT_OLLAMA_EMBED_MODEL`。

    ``explicit`` 若是 HF 风格（含 ``/``）则**一律忽略**，因为那不是合法的 Ollama 模型名；
    但若它恰好等于配置里的名字，也直接采用（调用方显式传了同样的值，无歧义）。
    """
    candidate = (explicit or "").strip()
    cfg = getattr(config, "ollama", None) if config is not None else None
    configured = str(getattr(cfg, "embedding_model", "") or "").strip()

    if candidate and candidate == configured:
        return candidate
    if candidate and not _is_hf_style_model_name(candidate):
        if _is_hf_style_model_name(configured):
            # config 里那份是 HF 仓库名、不可用；尊重显式传入的 Ollama 短名
            return candidate
        if candidate != configured:
            return candidate            # 显式覆盖（且确实是 Ollama 短名）
    return configured or DEFAULT_OLLAMA_EMBED_MODEL


def build_embedder(provider: str, model: str = "", dim: int = 256,
                   config=None, **kwargs) -> Embedder:
    """按配置构建向量化后端。

    ==========  ==================================================================
    provider    说明
    ==========  ==================================================================
    offline     零依赖哈希 TF（离线模式 / 单测默认）
    ollama      **本阶段默认**：Ollama ``/api/embed``（bge-m3 = 1024 维，免 torch）
    bge_m3      重型本地（torch + sentence-transformers，需 requirements-full）
    ==========  ==================================================================

    ``model`` 是**显式覆盖**：ollama 分支下优先取 ``config.ollama.embedding_model``，
    传入 HF 风格的名字（如 ``BAAI/bge-m3``）会被忽略，见 :func:`resolve_ollama_embed_model`。

    真实后端依赖缺失时抛出带修复提示的 RuntimeError，
    而不是把 ImportError 堆栈直接甩给调用方。
    """
    key = (provider or "offline").strip().lower()
    if key in ("offline", "hash", "fallback"):
        from .offline import OfflineEmbedder

        return OfflineEmbedder(dim=dim)
    if key in ("bge_m3", "bge-m3", "bgem3"):
        from .bge_m3 import BgeM3Embedder

        return BgeM3Embedder(model_name=model or "BAAI/bge-m3")
    if key in ("ollama", "ollama_bge_m3", "ollama-bge-m3"):
        from .ollama_embedder import OllamaEmbedder

        cfg = config
        if cfg is None:
            from ..config import RagConfig

            cfg = RagConfig.from_env()
        ollama_model = resolve_ollama_embed_model(model, cfg)
        if model and ollama_model != model.strip():
            logging.getLogger(__name__).info(
                "ollama 嵌入模型名取 config.ollama.embedding_model=%r，"
                "已忽略传入的 %r（EMBEDDING_MODEL 在该 provider 下会被解释为 Ollama 模型名，"
                "HF 仓库名发给 Ollama 会 404）", ollama_model, model)
        return OllamaEmbedder(ollama_model, config=cfg, **kwargs)
    raise ValueError(
        f"未知的 embedding_provider: {provider!r}（可选 offline | bge_m3 | ollama）")
