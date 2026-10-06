# -*- coding: utf-8 -*-
"""领域数据模型。

Chunk 是知识库的最小检索单元，字段对齐项目设计文档里的
「Milvus Collection 建议字段」：id / vector / text（供 BM25）/ 创建时间 /
修改时间 / 文档来源 / 摘要，另外预留 role_id 以便后续按角色分区隔离。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    parent_id: str | None = None
    is_parent: bool = False
    role_id: str = ""
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    vector: list[float] | None = None

    def to_metadata(self) -> dict[str, Any]:
        """向量库中随向量一起保存的标量字段。"""
        data = asdict(self)
        data.pop("vector", None)
        data.pop("text", None)
        data["text"] = self.text          # 原文保留，供 BM25 使用
        return data

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    #: 被 LLM 列表式选择器挑中（该条要置顶且豁免词面证据闸门，见 retrieve/selector.py）
    selected: bool = False

    @property
    def source(self) -> str:
        return self.chunk.source


@dataclass
class Role:
    role_id: str
    name: str
    domain: str
    persona: str
    default_provider: str = ""


@dataclass
class Message:
    role: str          # user | assistant | system
    content: str
    created_at: float = field(default_factory=time.time)
    #: 助手消息**当时的引用来源**（与 `/chat` 响应的 `citations` 同源、同字段）。
    #: 用户裁决：历史里要能看见依据（刷新后仍可复核）。**默认空列表** ——
    #: 旧消息（本次改动之前写进 Redis / 内存的）没有这个键，读出来就是"无引用来源"，
    #: 不得因此报错，也不改变既有行为。
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    role_id: str = ""
    session_id: str = ""
    provider: str = ""
    #: 本轮回答是否走了**兜底**（t120 F1）：为 True 时 ``provider`` 就是兜底后端
    #: （例如 ``mock``），接口层必须把它显式暴露出来，不能让它看起来像正常回答。
    degraded: bool = False
    #: 兜底原因（**可定位**口径，见 ``llm_base.failure_reason``）：
    #: missing_api_key / connect_failed / http_5xx / http_4xx / timeout / …
    degraded_reason: str = ""
