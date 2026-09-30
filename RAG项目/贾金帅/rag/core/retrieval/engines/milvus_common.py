# -*- coding: utf-8 -*-
"""Milvus 连接公共部分：字段映射解析 + 地址安全校验。

自 ``vector_store.py`` 拆出。这里只依赖 ``src.config`` 与 pymilvus，
不含检索逻辑，单独成文便于阅读与单独测试。
"""
from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class MilvusFieldMap:
    """从 collection 真实 schema 解析出的字段映射。"""

    id: str
    vector: str
    text: str | None
    title: str | None
    chunk_id: str | None
    parent_id: str | None
    document_id: str | None
    chunk_type: str | None
    source_path: str | None
    cleaned_path: str | None
    section_path: str | None
    metadata: str | None
    output_fields: tuple[str, ...]


def _is_vector_field(field) -> bool:
    """兼容不同 pymilvus 版本判断向量字段类型。"""
    dtype = getattr(field, "dtype", None)
    dtype_name = getattr(dtype, "name", str(dtype)).upper()
    return "VECTOR" in dtype_name or dtype in (100, 101, 102, 103)


def resolve_milvus_field_map(schema_fields) -> MilvusFieldMap:
    """按照 collection 的真实字段解析在线检索契约。"""
    fields = list(schema_fields)
    actual = {field.name for field in fields}

    def pick(*candidates: str) -> str | None:
        return next((name for name in candidates if name and name in actual), None)

    id_field = pick(config.MILVUS_ID_FIELD, "id", "chunk_id", "pk")
    vector_field = pick(config.MILVUS_VECTOR_FIELD, "vector", "embedding")
    if vector_field is None:
        vector_field = next(
            (field.name for field in fields if _is_vector_field(field)), None
        )
    if id_field is None:
        id_field = next(
            (field.name for field in fields if getattr(field, "is_primary", False)),
            "id",
        )
    if vector_field is None:
        raise ValueError("Milvus collection 中找不到向量字段")

    return MilvusFieldMap(
        id=id_field,
        vector=vector_field,
        text=pick(config.MILVUS_TEXT_FIELD, "text", "document", "content", "chunk_text"),
        title=pick(config.MILVUS_NAME_FIELD, "title", "name", "drug_name"),
        chunk_id=pick("chunk_id"),
        parent_id=pick("parent_id"),
        document_id=pick("document_id", "doc_id"),
        chunk_type=pick("chunk_type"),
        source_path=pick("source_path", "source", "file_path"),
        cleaned_path=pick("cleaned_path"),
        section_path=pick("section_path", "section"),
        metadata=pick("metadata", "meta"),
        output_fields=tuple(
            field.name for field in fields if not _is_vector_field(field)
        ),
    )


def _decode_json(value):
    """安全解码 Milvus VARCHAR 中保存的 JSON。"""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

# 拒绝的地址（SSRF 防护）：环回/链路本地/保留/多播
_BLOCKED_IP = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
)

# Milvus 连接超时（秒）：Milvus 不可达时快速标记不可用，避免阻塞首查
_MILVUS_CONNECT_TIMEOUT = float(os.getenv("MILVUS_CONNECT_TIMEOUT", "3.0"))


def _lazy_import_pymilvus():
    try:
        from pymilvus import (  # type: ignore
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )

        return Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    except ImportError:
        return None


def validate_milvus_address(host: str, port: str) -> tuple[str, str]:
    """校验 Milvus 地址：host 拒绝 localhost/环回/保留/多播。

    允许 RFC1918 私有段（192.168.x 等内网部署场景，用户显式配置）；
    仅当 config.MILVUS_ALLOW_LOCAL=true 时放行环回地址（本地 Docker Milvus
    开发场景，用户显式开启；默认关闭保持严格校验）。

    Raises
    ------
    ValueError
        host 解析到被禁止的地址时抛出。
    """
    host = (host or "").strip()
    port = str(port or "").strip()
    lowered = host.lower()
    if not host:
        raise ValueError("Milvus host 为空")
    allow_local = str(getattr(config, "MILVUS_ALLOW_LOCAL", False)).lower() \
        in {"1", "true", "yes", "on"}
    if not allow_local and lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError(f"拒绝访问本地地址: {host}")
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        ip = None  # 域名（DNS 解析在出口层防护）
    if ip is not None and not allow_local:
        for net in _BLOCKED_IP:
            if ip in net:
                raise ValueError(f"拒绝访问受限地址: {host}")
        if ip.is_loopback or ip.is_link_local or ip.is_reserved \
                or ip.is_multicast:
            raise ValueError(f"拒绝访问受限地址: {host}")
        # 私有段（RFC1918）放行：内网部署场景（同事的 Milvus 服务）
    return host, port
