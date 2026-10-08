"""Milvus 只读访问。**本包内唯一碰向量库的模块。**

---

⚠️ **本模块 MUST 只读。**

`backend/index/store.py` 的对应约束是"本包内唯一**会写**库的模块"，这里方向相反：
本包只读，而结构保证正是"只有 store 持有连接、且它不写"。这样切分让 FR-021 的
「检索不写库」成为**结构上的事实**而不是一句约定。

任何 `insert` / `delete` / `upsert` / `create_collection` / `drop_collection`
出现在本文件里都是缺陷 —— `tasks.md` T036 用 grep 钉这一条。

`pymilvus` 采用**惰性导入**：CLI 的 `selfcheck` 子命令不连库，在依赖尚未安装时
也该能跑（与 `backend/index/store.py` 同一取向）。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import (
    CORPUS_FIELDS,
    DEFAULT_URI,
    DIM,
    EXIT_DEP,
    EXIT_DATA,
    INDEX_MANIFEST,
    PRIMARY_FIELD,
    QUERY_LIMIT,
    RetrievalError,
    TOKEN_ENV,
)
from .models import CorpusChunk

__all__ = [
    "load_pymilvus",
    "connect",
    "fetch_corpus",
    "search_semantic",
    "manifest_total_chunks",
    "manifest_collection",
]


def load_pymilvus():
    """惰性导入；缺依赖时报退出码 3 并给出可复制的安装命令。"""

    try:
        import pymilvus
    except ModuleNotFoundError as exc:
        raise RetrievalError(
            EXIT_DEP,
            "缺少依赖 pymilvus。安装：\n"
            "  D:/zg6_Project/9/med_rag/rag/python.exe -m pip install -r requirements.txt",
        ) from exc
    return pymilvus


def connect(uri: str = DEFAULT_URI, token: str | None = None):
    """建立 Milvus 连接。失败一律归为「外部依赖不可用」（退出码 3）。

    `token` 缺省时读环境变量 —— 这是本模块**唯一**读环境变量的地方，且只为了
    取凭据。constitution 原则 III 要求凭据只走环境变量；检索的其余配置一律由
    启动期配置对象传入（FR-024）。
    """

    import os

    pm = load_pymilvus()
    if token is None:
        token = os.environ.get(TOKEN_ENV) or None

    try:
        return pm.MilvusClient(uri=uri, token=token)
    except Exception as exc:  # noqa: BLE001 —— 连接层失败一律归为外部依赖
        # ⚠️ 错误信息里 MUST NOT 回显 token（constitution 原则 III：日志脱敏）。
        raise RetrievalError(
            EXIT_DEP,
            "无法连接向量库 %s —— %s\n"
            "  自检：docker ps --format '{{.Image}}' 应含 milvusdb/milvus:v2.6.9"
            % (uri, _brief(exc)),
        ) from exc


def manifest_total_chunks(manifest_path: Path | str = INDEX_MANIFEST) -> int | None:
    """读 `index_manifest.json` 里的 `total_chunks`。拿不到时返回 `None`。

    返回 `None` 而不是抛异常：这个值只用于**比对告警**，不是检索的前提。
    为一个对比数字让检索起不来是不划算的 —— 当然，拿不到时调用方要如实报告
    "无法比对"，MUST NOT 把它当成"一致"。
    """

    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 见上：只用于比对
        return None
    value = data.get("total_chunks") if isinstance(data, dict) else None
    return value if isinstance(value, int) else None


def manifest_collection(manifest_path: Path | str = INDEX_MANIFEST) -> str | None:
    """读清单里的 collection 名。用途与 `manifest_total_chunks` 相同。"""

    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    value = data.get("collection") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def fetch_corpus(client, collection: str) -> list[CorpusChunk]:
    """拉回全量 chunk 的展示字段，作为两路共用的语料（D1 / FR-009）。

    ⚠️ **两条 MUST，各挡一类静默失效：**

    1. `limit` **显式给出**，MUST NOT 依赖 Milvus 的默认上限。默认值会让语料被
       **静默截断** —— 关键词路悄悄少掉一部分文档，而语义路不受影响（向量检索
       由 Milvus 自己管），两路从此不一致，且没有任何报错。
       与 `backend/index/store.py` 的 `QUERY_LIMIT = 16384` 同一理由。

    2. 拉回条数**等于** limit 时**告警**。相等说明"可能还有更多" —— 这是上面
       那条唯一能被观测到的地方。MUST NOT 静默放过：一份被截断的语料看起来
       完全正常（BM25 照样返回结果，只是少了些）。

    条数与 `index_manifest.json` 的比对由调用方做（`service.build_index`）——
    本模块只负责取数，不负责判断该不该告警。
    """

    try:
        rows = client.query(
            collection_name=collection,
            filter="",
            output_fields=list(CORPUS_FIELDS),
            limit=QUERY_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(
            EXIT_DEP,
            "拉取语料失败（collection=%s）—— %s" % (collection, _brief(exc)),
        ) from exc

    if not isinstance(rows, list):
        raise RetrievalError(EXIT_DATA, "语料返回值不是列表：%r" % type(rows).__name__)

    return [_to_chunk(row) for row in rows if isinstance(row, dict)]


def search_semantic(
    client, collection: str, query_vector: list[float], limit: int
) -> list[tuple[str, float]]:
    """语义路检索，返回 `[(chunk_id, 余弦)]` 按余弦降序。

    ⚠️ `output_fields` **只取主键**，MUST NOT 用 `["*"]` 或包含 `vector`。
    语料已经全量在内存里（D1），从检索结果再拉一遍正文是白花带宽；而把 1024 维
    向量拉回来则会让响应体大出两个数量级 —— Milvus 已经在服务端按 COSINE
    排好序了，分数在 `distance` 里，不需要客户端再算一遍。

    ⚠️ 维度不符时 MUST 抛错而不是继续查。S8 的指纹门禁应已在启动期挡住这种情况，
    这里是**第二道**：真发生了说明门禁被绕过或索引被换过，此时"少一条结果"
    比"用错维度的向量查到一堆看似合理的垃圾"好得多。
    """

    if len(query_vector) != DIM:
        raise RetrievalError(
            EXIT_DATA,
            "查询向量维度不符：期望 %d，实际 %d。"
            "（若刚重新入库过，请重启服务以重新通过指纹门禁）"
            % (DIM, len(query_vector)),
        )

    try:
        results = client.search(
            collection_name=collection,
            data=[query_vector],
            limit=limit,
            output_fields=[PRIMARY_FIELD],
            search_params={"metric_type": "COSINE"},
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(
            EXIT_DEP, "向量检索失败（collection=%s）—— %s" % (collection, _brief(exc))
        ) from exc

    if not results:
        return []

    hits: list[tuple[str, float]] = []
    for hit in results[0]:
        # `hit` 是 dict（pymilvus 的 MilvusClient 返回结构与 ORM 版不同）：
        # 主键在 `id`，也可能在 `entity` 里 —— 两种都取，取不到就跳过该条。
        chunk_id = hit.get("id") if isinstance(hit, dict) else None
        if chunk_id is None and isinstance(hit, dict):
            entity = hit.get("entity") or {}
            chunk_id = entity.get(PRIMARY_FIELD)
        if chunk_id is None:
            continue

        distance = hit.get("distance") if isinstance(hit, dict) else None
        hits.append((str(chunk_id), float(distance) if distance is not None else 0.0))

    return hits


def _to_chunk(row: dict) -> CorpusChunk:
    """把一行 Milvus 记录转成语料条目。

    缺失的可选字段用默认值兜底，**必填字段缺失则保留空值** —— 由 `corpus`
    子命令负责报告这些空值，而不是在这里抛错。理由：一条缺 `file_name` 的记录
    不该让**整个**关键词索引建不起来；它该被报告出来、然后由人决定怎么处理。
    """

    return CorpusChunk(
        chunk_id=str(row.get("chunk_id") or ""),
        text=str(row.get("text") or ""),
        file_name=str(row.get("file_name") or ""),
        page_start=int(row.get("page_start") or 0),
        page_end=int(row.get("page_end") or 0),
        section=row.get("section"),
        block_type=str(row.get("block_type") or ""),
    )


def _brief(exc: BaseException) -> str:
    """异常的可读摘要：**类型 + 消息，不含堆栈**。

    堆栈既无用（使用者改不了 Milvus 客户端内部）又可能带内部路径与连接串
    （constitution 原则 III：日志脱敏）。
    """

    text = str(exc).strip().replace("\n", " ")
    if len(text) > 300:
        text = text[:300] + "…"
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__
