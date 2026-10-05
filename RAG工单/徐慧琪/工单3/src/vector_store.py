# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：向量库（Qdrant）

【环境适配说明】
工单原方案为 Docker 部署 qdrant/qdrant。本机实测：
  - `docker` 不在 PATH，`C:\\Program Files\\Docker` 不存在；
  - 没有 qdrant 可执行文件；已安装 `qdrant-client`（1.18）。
因此使用 qdrant-client 的 **本地嵌入式模式**（QdrantClient(path=...)）：
同一套 API、同样的 collection/payload 结构，无需服务端进程即可持久化。
若将来起了 Qdrant 服务，把 config.QDRANT_MODE 改为 "server" 并填 QDRANT_URL 即可切换。

payload 约定（工单02 元数据要求全部落库）：
  text / page_idx / heading_path / source / block_type
  parent_id / parent_text / parent_page_idx / chunk_id / seq
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import hashlib
import json
import os
import shutil
import threading
import time
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src import config

_client: QdrantClient | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------
def get_client() -> QdrantClient:
    """获取（并缓存）Qdrant 客户端。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        if config.QDRANT_MODE == "server":
            _client = QdrantClient(url=config.QDRANT_URL, timeout=60)
            print(f"[qdrant] 已连接服务端 {config.QDRANT_URL}", flush=True)
        else:
            os.makedirs(config.QDRANT_LOCAL_PATH, exist_ok=True)
            _client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
            print(f"[qdrant] 已启用本地嵌入式存储 {config.QDRANT_LOCAL_PATH}", flush=True)
        return _client


def reset_client_for_tests() -> None:
    """测试辅助：释放缓存客户端（便于指向临时目录）。"""
    close()


def close() -> None:
    """关闭客户端（本地模式必须显式关闭，否则存储锁不释放）。"""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


# ---------------------------------------------------------------------------
# collection 管理
# ---------------------------------------------------------------------------
def collection_exists(name: str | None = None) -> bool:
    name = name or config.COLLECTION_OPT
    try:
        return name in [c.name for c in get_client().get_collections().collections]
    except Exception:
        return False


def _local_collection_dir(name: str) -> str:
    return os.path.join(config.QDRANT_LOCAL_PATH, "collection", name)


def _purge_local_collection(name: str) -> None:
    """彻底铲除本地 collection（目录 + meta.json 条目）。

    【踩坑记录，移植自工单1】本地嵌入式模式下 `delete_collection()` 只摘除
    meta.json 条目，`collection/<name>/storage.sqlite` 仍在；随后同名
    create_collection() 会复用该目录，**旧点被原样带回**（重建后 count 偏大、
    检索出现已删除的脏数据）。故本地模式重建必须显式删目录并同步 meta.json。
    """
    close()
    path = _local_collection_dir(name)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    meta_path = os.path.join(config.QDRANT_LOCAL_PATH, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("collections", {}).pop(name, None) is not None:
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False)
        except Exception:  # pragma: no cover - 兜底
            pass


def ensure_collection(dim: int, name: str | None = None,
                      recreate: bool = False) -> None:
    """创建 collection；recreate=True 时先删除重建。"""
    name = name or config.COLLECTION_OPT

    if recreate and config.QDRANT_MODE == "local":
        _purge_local_collection(name)

    client = get_client()
    if collection_exists(name):
        if recreate:
            client.delete_collection(name)
        else:
            return

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=int(dim),
            distance=qm.Distance[config.VECTOR_DISTANCE.upper()],
        ),
    )
    # payload 索引：source（多文档过滤）/ block_type（块类型过滤，如只看 table_row）/
    # table_id（按表聚合，答案定位报告用）
    for field, schema in (("source", qm.PayloadSchemaType.KEYWORD),
                          ("block_type", qm.PayloadSchemaType.KEYWORD),
                          ("table_id", qm.PayloadSchemaType.KEYWORD)):
        try:
            client.create_payload_index(collection_name=name, field_name=field,
                                        field_schema=schema)
        except Exception:
            pass  # 本地模式部分版本不支持，忽略
    print(f"[qdrant] 已创建 collection: {name}（size={dim}, "
          f"distance={config.VECTOR_DISTANCE}）", flush=True)


def count(name: str | None = None) -> int:
    name = name or config.COLLECTION_OPT
    try:
        return int(get_client().count(name, exact=True).count)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------
def _stable_id(key: str) -> int:
    """chunk 唯一键 → 稳定正整数 ID（Qdrant 要求 int 或 UUID）。

    【工单3 踩坑修正】传入的必须是**全局唯一键**（`source#chunk_id`），不能只用
    chunk_id：两个文档的 chunk_id 各自从 c000001 开始，单文档建库时相安无事，
    多文档时会**互相覆盖**——实测 7968 个点入库后集合里只剩 5024 个（招股2 覆盖
    了招股1 的前 2944 条）。

    用 blake2b 而非内置 hash()：内置 hash 对 str 是随机加盐的，进程重启后 ID 变化，
    会造成"同名点重复入库"。
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1      # 63 位，避开有符号溢出


def _to_point(chunk: dict, vector: Sequence[float]) -> qm.PointStruct:
    """chunk + 向量 → Qdrant 点（payload 含工单要求的全部元数据）。"""
    payload = {
        # --- 工单02 要求：page_idx / heading_path / source / block_type ---
        "text": chunk["text"],
        "page_idx": int(chunk.get("page_idx", 0)),
        "heading_path": list(chunk.get("heading_path") or []),
        "source": chunk.get("source", config.SOURCE_NAME),
        "block_type": chunk.get("block_type", "text"),
        # --- 父子块（小块检索 + 大块上下文）---
        "parent_id": chunk.get("parent_id", ""),
        "parent_text": chunk.get("parent_text", ""),
        "parent_page_idx": int(chunk.get("parent_page_idx",
                                         chunk.get("page_idx", 0))),
        # --- 溯源与排序 ---
        "chunk_id": chunk["chunk_id"],
        "seq": int(chunk.get("seq", 0)),
    }
    # --- 工单3：表格 chunk 的 8 项元数据（表头/行号/HTML 等）全部落库 ---
    for key in ("table_id", "table_html", "table_header", "row_index",
                "table_caption", "table_quality", "table_n_rows", "from_ocr"):
        if key in chunk:
            payload[key] = chunk[key]
    unique_key = f"{payload['source']}#{chunk['chunk_id']}"
    return qm.PointStruct(id=_stable_id(unique_key),
                          vector=list(vector), payload=payload)


def upsert_chunks(chunks: Sequence[dict], vectors: Sequence[Sequence[float]],
                  batch_size: int | None = None, name: str | None = None,
                  progress: bool = True) -> int:
    """批量写入向量，返回写入点数。"""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
    name = name or config.COLLECTION_OPT
    batch_size = batch_size or config.UPSERT_BATCH_SIZE
    client = get_client()

    total = len(chunks)
    written = 0
    t0 = time.time()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = [_to_point(chunks[i], vectors[i]) for i in range(start, end)]
        client.upsert(collection_name=name, points=points, wait=True)
        written += len(points)
        if progress and (written % (batch_size * 10) == 0 or written == total):
            rate = written / max(time.time() - t0, 1e-6)
            print(f"  入库 {written}/{total}（{rate:.0f} 点/秒）", flush=True)
    return written


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def search(vector: Sequence[float], top_k: int | None = None,
           name: str | None = None,
           query_filter: qm.Filter | None = None) -> list[dict]:
    """向量检索，返回 [{score, text, page_idx, parent_text, ...}]。"""
    name = name or config.COLLECTION_OPT
    top_k = top_k or config.VECTOR_TOP_K
    client = get_client()

    if hasattr(client, "query_points"):
        res = client.query_points(collection_name=name, query=list(vector),
                                  limit=top_k, query_filter=query_filter,
                                  with_payload=True)
        hits = res.points
    else:  # pragma: no cover - 旧版兼容
        hits = client.search(collection_name=name, query_vector=list(vector),
                             limit=top_k, query_filter=query_filter,
                             with_payload=True)

    out = []
    for h in hits:
        payload = dict(h.payload or {})
        payload["score"] = float(h.score)
        out.append(payload)
    return out


# BM25 语料构建时排除的重字段：table_html 每个表格行块都带一份（工单要求
# "每个表格 chunk 必须携带 table_html"），全量 scroll 会把几十 MB HTML 拉进内存，
# 而 BM25 只用 text 字段。落库保留、读取排除，两头兼顾。
_HEAVY_PAYLOAD_FIELDS = ["table_html"]


def scroll_all(name: str | None = None, limit: int = 100000,
               page: int = 10000, with_html: bool = False) -> list[dict]:
    """取出全部 payload（BM25 语料构建用），分页拉取避免静默截断。"""
    name = name or config.COLLECTION_OPT
    client = get_client()
    selector: object = True
    if not with_html:
        try:
            selector = qm.PayloadSelectorExclude(exclude=_HEAVY_PAYLOAD_FIELDS)
        except Exception:  # noqa: BLE001 —— 旧版客户端不支持时退化为全量
            selector = True

    out: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(collection_name=name, limit=min(page, limit),
                                        offset=offset, with_payload=selector,
                                        with_vectors=False)
        out.extend(dict(r.payload or {}) for r in records)
        if offset is None or not records or len(out) >= limit:
            break
    return out[:limit]


def info() -> dict:
    """向量库状态，供界面展示。"""
    return {
        "mode": config.QDRANT_MODE,
        "path": (config.QDRANT_LOCAL_PATH if config.QDRANT_MODE == "local"
                 else config.QDRANT_URL),
        "collection": config.COLLECTION_OPT,
        "points": count(),
    }
