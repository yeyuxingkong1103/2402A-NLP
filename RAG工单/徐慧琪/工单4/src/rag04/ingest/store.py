# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""Qdrant 三库存储与连接池封装。

三个 collection 分离的原因：图像块用 CLIP 512 维、文本/表格用 bge-m3 1024 维，
语义空间与维度都不同，混在一起会互相污染，且无法分别设置阈值与融合权重。

支持两种模式：
  - 本地嵌入式（默认，path=...）：免装服务，开箱即用
  - 服务端（url=...）：满足硬性要求4「Qdrant 连接池」的生产形态
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag04.config import Settings
from rag04.retrieve.embed import DIM
from rag04.schema import Chunk, Hit

logger = logging.getLogger("rag04.store")

COLL_TEXT = "text_chunks"
COLL_TABLE = "table_chunks"
COLL_IMAGE = "image_chunks"

_CLIP_DIM = 512

# chunk_id -> 点 id 的确定性命名空间。Qdrant 只接受 uint64 或 UUID 形式的字符串点 id，
# 而 Task 8 产出的 chunk_id 是 16 位 md5 十六进制串，故此处做一次确定性映射。
_POINT_NS = uuid.uuid5(uuid.NAMESPACE_URL, "rag04/chunk_id")


def _point_id(chunk_id: str) -> str:
    """由 chunk_id 确定性派生点 id（保持「点 id 只由 chunk_id 决定、绝不用 figure_id」）。

    同一 chunk_id 恒等映射到同一点 id，因此重复 upsert 是覆盖而非新增；
    chunk_id 原值仍写入 payload，检索时优先从 payload 还原。
    """
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(_POINT_NS, chunk_id))


def collection_for(block_type: str) -> str:
    return {"text": COLL_TEXT, "table": COLL_TABLE, "image": COLL_IMAGE}[block_type]


class VectorStore:
    """三库封装。连接池参数在服务端模式下生效。"""

    def __init__(self, settings: Settings, path: Path | None = None,
                 url: str | None = None, max_connections: int = 16) -> None:
        self.settings = settings
        self.max_connections = max_connections
        if url:
            # pool_size 即 httpx 连接池的 max_connections，对应硬性要求4「Qdrant 连接池」。
            self.client = QdrantClient(url=url, timeout=30, pool_size=max_connections)
            self.mode = "server"
        else:
            p = Path(path or settings.qdrant_path)
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))
            self.mode = "embedded"

    def ensure_collections(self) -> None:
        """建齐三库（幂等）。"""
        specs = [
            (COLL_TEXT, DIM),
            (COLL_TABLE, DIM),
            (COLL_IMAGE, _CLIP_DIM),
        ]
        for name, dim in specs:
            try:
                if not self.client.collection_exists(name):
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    logger.info("已创建 collection：%s（%d 维）", name, dim)
            except Exception as e:
                logger.error("创建 collection %s 失败：%s", name, e)
                raise

    def clear_collections(self) -> dict[str, int]:
        """清空三库全部点（RC-2 重建安全，硬性要求）。返回清空前的点数快照。

        分块改动（RC5 行块合并 / RC2 图描述文本块 / RC1-in 入库侧过滤）会改变
        chunk_id，旧点**不会被覆盖**：直接 upsert 会让新旧块共存并污染检索。
        故全量重建前必须先清空三库；调用方应先备份 data/qdrant。

        不用 delete_collection + create_collection：嵌入式（local 模式）下
        ``delete_collection`` 对已被 sqlite 持有的集合目录只做
        ``shutil.rmtree(..., ignore_errors=True)``，Windows 上静默删不掉，
        随后 create_collection 会**重新打开旧库**（实测删除后 count 仍为原值，
        重开客户端也一样）。按空过滤器删除点则立即生效（实测 count→0）。
        """
        from qdrant_client.models import Filter, FilterSelector

        self.ensure_collections()
        before = self.counts()
        for name in (COLL_TEXT, COLL_TABLE, COLL_IMAGE):
            try:
                self.client.delete(
                    collection_name=name,
                    points_selector=FilterSelector(filter=Filter()),
                    wait=True,
                )
            except Exception as e:
                logger.error("清空 collection %s 失败：%s", name, e)
                raise
        logger.info("已清空三库：%s", before)
        return before

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """按 block_type 路由写入。相同 chunk_id 覆盖，保证幂等。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 数量不匹配：{len(chunks)} vs {len(vectors)}")

        by_coll: dict[str, list[PointStruct]] = {}
        for c, v in zip(chunks, vectors):
            coll = collection_for(c.block_type)
            payload = c.to_payload()
            payload["chunk_id"] = c.chunk_id
            by_coll.setdefault(coll, []).append(
                PointStruct(id=_point_id(c.chunk_id), vector=list(v), payload=payload)
            )

        n = 0
        for coll, points in by_coll.items():
            self.client.upsert(collection_name=coll, points=points, wait=True)
            n += len(points)
        return n

    def search(self, coll: str, vector: list[float], k: int = 30) -> list[Hit]:
        """向量检索。"""
        try:
            res = self.client.query_points(
                collection_name=coll, query=list(vector), limit=k, with_payload=True,
            ).points
        except Exception as e:
            logger.error("检索失败 %s：%s", coll, e)
            return []

        hits: list[Hit] = []
        for p in res:
            pl = p.payload or {}
            hits.append(Hit(
                chunk_id=pl.get("chunk_id", str(p.id)),
                doc_id=pl.get("doc_id", ""),
                page=int(pl.get("page", 0)),
                block_type=pl.get("block_type", "text"),
                source_id=pl.get("source_id", ""),
                text=pl.get("text", ""),
                score=float(p.score),
                channel="dense",
                image_path=(pl.get("extra") or {}).get("image_path", ""),
                dedup_key=(pl.get("extra") or {}).get("dedup_key", ""),
            ))
        return hits

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in (COLL_TEXT, COLL_TABLE, COLL_IMAGE):
            try:
                out[c] = self.client.count(c, exact=True).count
            except Exception:
                out[c] = 0
        return out

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
