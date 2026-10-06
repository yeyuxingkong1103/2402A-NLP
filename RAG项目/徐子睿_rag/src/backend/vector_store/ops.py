# -*- coding: utf-8 -*-
"""vector_store/ops.py —— 记录级的查询、检索、写入与删除。

在链路中的位置：
    backend/pipeline.py（写入） / backend/retrieval.py（读取） → vector_store 包 → 【本文件】 → Milvus

本文件是真正对 Milvus 发起读写的地方。四类操作：
    query_vectors    按条件取记录（不带向量检索）—— 知识库列表、加载全量语料建 BM25 都用它
    search_vectors   向量相似度检索 —— 语义召回的落地点
    upsert_vectors   批量写入（存在则覆盖）—— 保证同名文档重复构建时幂等
    delete_vectors   按条件删除 —— 强制要求非空过滤条件，避免手滑清库

依赖：
    .client.ensure_collection   保证集合已建好并 load
    .filters._payload_from_row  把 Milvus 的返回结构摊平成业务字典
"""
from __future__ import annotations

from typing import Any, Iterable

from .client import COLLECTION, ensure_collection
from .filters import _payload_from_row

def query_vectors(
    collection_name: str = COLLECTION,
    *,
    filter_expression: str = "",
    limit: int = 16384,
) -> list[dict[str, Any]]:
    """按条件查询记录（不带向量检索，纯条件过滤）。

    参数：
        collection_name: 集合名
        filter_expression: 过滤表达式，空串表示取全部
        limit: 最多返回多少条，默认 16384
    返回：
        [{"id": 主键, "payload": {业务字段}}, ...]

    用途：
        知识库总览、chunk 列表、以及 retrieval 加载全量语料建 BM25 索引都走这里。

    limit 被 clamp 到 [1, 16384]：
        16384 是 Milvus 单次查询的实际上限，传更大没有意义、传 0 或负数会直接报错。
    """
    client = ensure_collection(collection_name)
    rows = client.query(
        collection_name=collection_name,
        filter=filter_expression,
        output_fields=["*"],
        limit=max(1, min(int(limit), 16384)),
    )
    return [{"id": point_id, "payload": payload} for point_id, payload in (_payload_from_row(row) for row in rows)]


def search_vectors(
    collection_name: str,
    vector: list[float],
    *,
    limit: int,
    filter_expression: str = "",
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """向量相似度检索。

    参数：
        collection_name: 集合名
        vector: 查询向量（1024 维，必须与入库时同一个模型编码）
        limit: 返回条数上限
        filter_expression: 可选的元数据过滤，如只搜某一份文档
        score_threshold: 最低相似度。None 表示不过滤
    返回：
        [{"id", "payload", "score"}, ...]，score 为余弦相似度。

    这是"语义召回"的落地点：
        它理解"回充设备"和"气体重新充入装置"是同义表达 —— 字面不同但向量接近。
        这是纯关键词检索做不到、也是本项目需要向量路的原因。

    为什么把阈值过滤放在这一层（而不是交给上层）：
        少往上传无用的低分候选，能减少后续融合与精排的数据量；
        同时保证上层拿到的 score 都是"及格"的，简化上层判断。

    注意搜索参数用 COSINE：
        必须与建索引时的 metric_type 一致，否则 Milvus 会直接拒绝请求。
    """
    client = ensure_collection(collection_name)
    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "data": [vector],           # 接口要求批量传，这里一次只查一个向量
        "anns_field": "vector",     # 指定对哪个向量字段做检索
        "limit": max(1, int(limit)),
        "output_fields": ["*"],     # 把业务字段一并带回来，省一次按主键回表的查询
        "search_params": {"metric_type": "COSINE", "params": {}},
    }
    if filter_expression:
        kwargs["filter"] = filter_expression
    raw_hits = client.search(**kwargs)
    hits = raw_hits[0] if raw_hits else []  # 批量查询的返回是"每个查询向量一个列表"，取第一个
    results = []
    for hit in hits:
        point_id, payload = _payload_from_row(hit)
        # 兼容字段名差异：新版本叫 distance，老版本叫 score
        score = float(hit.get("distance", hit.get("score", 0.0)))
        if score_threshold is not None and score < score_threshold:
            continue
        results.append({"id": point_id, "payload": payload, "score": score})
    return results


def upsert_vectors(collection_name: str, records: Iterable[dict[str, Any]]) -> int:
    """批量写入（存在则覆盖）向量记录。

    参数：
        collection_name: 集合名
        records: [{"id": 主键, "vector": [...], "payload": {...}}, ...]
    返回：
        实际写入的记录数；records 为空时返回 0。

    为什么要把 payload 摊平进 row：
        集合开启了动态字段，业务字段必须和 id/vector 平铺在同一层才能被存进去。
        所以这里把 payload 的内容 update 到 row 上 —— 这也意味着 payload 里
        不能出现名为 id 或 vector 的键，否则会覆盖主键和向量。

    upsert 而非 insert：
        同名文档重复构建时，uuid5 生成的主键相同，upsert 正好实现幂等覆盖，
        不会因为主键冲突而整批写入失败。
    """
    data = []
    for record in records:
        payload = dict(record.get("payload") or {})
        row = {"id": str(record["id"]), "vector": list(record["vector"])}
        row.update(payload)
        data.append(row)
    if not data:
        return 0  # 空批次不调接口，避免 Milvus 报参数错误
    ensure_collection(collection_name).upsert(collection_name=collection_name, data=data)
    return len(data)


def delete_vectors(collection_name: str, filter_expression: str) -> int:
    """按条件删除记录。

    参数：
        collection_name: 集合名
        filter_expression: 过滤表达式
    返回：
        删除条数（Milvus 未返回该字段时给 0）。

    强制要求非空过滤条件：
        空条件在 Milvus 里等价于"匹配全部"，会一次删光整个集合。
        这里直接抛 ValueError 把它挡下来 —— 这是本项目最危险的一个操作，
        宁可让调用方明确写条件，也不接受一个"手滑就清库"的默认行为。
    """
    if not filter_expression:
        raise ValueError("删除向量必须提供过滤条件")
    result = ensure_collection(collection_name).delete(collection_name=collection_name, filter=filter_expression)
    if isinstance(result, dict):
        return int(result.get("delete_count", 0))
    return 0
