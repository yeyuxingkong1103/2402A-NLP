# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：向量库（Qdrant）

【环境适配说明】
工单原方案为 Docker 部署 qdrant/qdrant。本机实测：
  - `docker` 不在 PATH，`C:\\Program Files\\Docker` 不存在；
  - 没有 qdrant 可执行文件；
  - 但已安装 `qdrant-client`。
因此改用 qdrant-client 的 **本地嵌入式模式**（QdrantClient(path=...)）：
同一套 API、同样的 collection/payload 结构，无需任何服务端进程即可持久化。
若将来起了 Qdrant 服务，把 config.QDRANT_MODE 改为 "server" 并填 QDRANT_URL 即可切换，
本模块其余代码无需改动。

collection 约定（对齐工单第三步要求）：
  - VectorParams(size=embedding维度, distance=COSINE)
  - payload: text / page_idx / source（另附 chunk_id、heading_path、type 便于溯源）
  - 批量写入 batch_size=100
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401

import json
import os
import shutil
import threading
import time
from typing import Iterable, Sequence

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
    name = name or config.COLLECTION_NAME
    try:
        return name in [c.name for c in get_client().get_collections().collections]
    except Exception:
        return False


def _local_collection_dir(name: str) -> str:
    """本地嵌入式模式下 collection 的落盘目录。"""
    return os.path.join(config.QDRANT_LOCAL_PATH, "collection", name)


def _purge_local_collection(name: str) -> None:
    """彻底铲除本地 collection（目录 + meta.json 条目）。

    【踩坑记录】本地嵌入式模式下 `delete_collection()` 只把条目从 meta.json
    里摘掉，`collection/<name>/storage.sqlite` 原封不动。随后同名
    `create_collection()` 会复用这个目录，**上一轮的点会被原样带回来**：
    重建后 count 比实际写入数多出旧点的数量，检索里会出现已删除的脏数据。
    因此本地模式重建必须显式删目录，并同步 meta.json 保持一致。
    """
    close()  # 先释放 sqlite 文件锁，否则 Windows 上删不掉

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
                print(f"[qdrant] 已清除本地 collection 数据目录: {path}", flush=True)
        except Exception as exc:  # pragma: no cover - 兜底
            print(f"[qdrant] 清理 meta.json 失败（忽略）：{exc}", flush=True)


def ensure_collection(dim: int, name: str | None = None,
                      recreate: bool = False) -> None:
    """创建 collection；recreate=True 时先删除重建。"""
    name = name or config.COLLECTION_NAME

    if recreate and config.QDRANT_MODE == "local":
        # 本地模式走物理清除（见 _purge_local_collection 的踩坑记录）
        _purge_local_collection(name)

    client = get_client()

    if collection_exists(name):
        if recreate:
            client.delete_collection(name)
            print(f"[qdrant] 已删除旧 collection: {name}", flush=True)
        else:
            return

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=int(dim),
            distance=qm.Distance[config.VECTOR_DISTANCE.upper()],
        ),
    )
    # 为 source 建 payload 索引，便于将来多文档过滤
    try:
        client.create_payload_index(collection_name=name, field_name="source",
                                    field_schema=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name=name, field_name="type",
                                    field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass  # 本地模式的部分版本不支持，忽略即可
    print(f"[qdrant] 已创建 collection: {name}（size={dim}, "
          f"distance={config.VECTOR_DISTANCE}）", flush=True)


def count(name: str | None = None) -> int:
    name = name or config.COLLECTION_NAME
    try:
        return int(get_client().count(name, exact=True).count)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------
def _to_point(chunk: dict, vector: Sequence[float]) -> qm.PointStruct:
    """把 chunk + 向量组装成 Qdrant 点。"""
    return qm.PointStruct(
        id=_stable_id(chunk["chunk_id"]),
        vector=list(vector),
        payload={
            # --- 工单要求的三个核心字段 ---
            "text": chunk["text"],
            "page_idx": int(chunk.get("page_idx", 0)),
            "source": chunk.get("source", config.SOURCE_NAME),
            # --- 便于溯源与展示的附加字段 ---
            "chunk_id": chunk["chunk_id"],
            "type": chunk.get("type", "text"),
            "heading_path": list(chunk.get("heading_path") or []),
            "raw_text": chunk.get("raw_text", chunk["text"]),
        },
    )


def _stable_id(chunk_id: str) -> int:
    """把 c00012 形式的 chunk_id 映射为稳定的正整数 ID（Qdrant 要求 int 或 UUID）。"""
    digits = "".join(ch for ch in chunk_id if ch.isdigit())
    return int(digits) if digits else abs(hash(chunk_id)) % (10 ** 12)


def upsert_chunks(chunks: Sequence[dict], vectors: Sequence[Sequence[float]],
                  batch_size: int | None = None,
                  name: str | None = None,
                  progress: bool = True) -> int:
    """批量写入向量。返回写入点数。"""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
    name = name or config.COLLECTION_NAME
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
           source_filter: str | None = None) -> list[dict]:
    """向量检索，返回 [{score, text, page_idx, source, ...}]。"""
    name = name or config.COLLECTION_NAME
    top_k = top_k or config.VECTOR_TOP_K
    client = get_client()

    query_filter = None
    if source_filter:
        query_filter = qm.Filter(must=[qm.FieldCondition(
            key="source", match=qm.MatchValue(value=source_filter))])

    # 兼容不同版本：新版用 query_points，旧版用 search
    hits = []
    if hasattr(client, "query_points"):
        res = client.query_points(collection_name=name, query=list(vector),
                                  limit=top_k, query_filter=query_filter,
                                  with_payload=True)
        hits = res.points
    else:  # pragma: no cover
        hits = client.search(collection_name=name, query_vector=list(vector),
                             limit=top_k, query_filter=query_filter,
                             with_payload=True)

    out = []
    for h in hits:
        payload = dict(h.payload or {})
        payload["score"] = float(h.score)
        out.append(payload)
    return out


def scroll_all(name: str | None = None, limit: int = 100000,
               page: int = 10000) -> list[dict]:
    """取出全部 payload（BM25 索引构建用）。

    分页拉取：早期版本只取一页就把 offset 丢掉，点数超过单页上限时 BM25 语料
    会**静默变少**，与向量库不一致（当前 1615 点远未触顶，但多文档扩充后会踩到）。
    """
    name = name or config.COLLECTION_NAME
    client = get_client()

    out: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(collection_name=name, limit=min(page, limit),
                                        offset=offset, with_payload=True,
                                        with_vectors=False)
        out.extend(dict(r.payload or {}) for r in records)
        if offset is None or not records or len(out) >= limit:
            break
    return out[:limit]


def info() -> dict:
    """返回向量库状态，供界面展示。"""
    return {
        "mode": config.QDRANT_MODE,
        "path": config.QDRANT_LOCAL_PATH if config.QDRANT_MODE == "local" else config.QDRANT_URL,
        "collection": config.COLLECTION_NAME,
        "points": count(),
    }
