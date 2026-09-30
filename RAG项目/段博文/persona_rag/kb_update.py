# -*- coding: utf-8 -*-
"""
Milvus 知识库动态更新模块（多集合版）

功能：
    1. 按来源删除：删除某文件的所有 chunk（文档被废弃时）
    2. 按来源更新：删除旧 chunk -> 重新切分 -> 重新写入（文档修改后）
    3. 列出来源：查看知识库里有哪些文件、每个文件多少条
    4. 全量重建：删集合 -> 重新入库（模型/维度变了时）

所有操作都接受 collection_name 参数，支持多集合架构

本模块在系统中的位置：
    上游：main.py 的「知识库管理」接口（GET /kb/sources、GET /kb/sources/{source}、
          DELETE /kb/sources/{source}、PUT /kb/update、POST /kb/rebuild）调用这里；
          batch_ingest_pdfs.py 首次灌库后也依赖同样的「写入后刷新 BM25」约定。
    下游：通过 db_milvus 提供的单例（Milvus 原生客户端 + LangChain 向量库封装）操作数据，
          切分复用 text_splitter，配置来自 config。

关键设计取舍：
    1. 一律复用 db_milvus 里的单例：客户端和向量模型都很重，绝不能在本模块里重复创建。
    2. 写库按 10 条一批：单条写入网络往返太多，整篇一次写入又容易超时/超包，10 是折中值。
    3. 「更新」是逻辑删除 + 重新插入，不是原地改；所以调用方必须在更新后重建 BM25 索引，
       否则关键词检索会命中已被删除的旧文本。
    4. 集合名统一走参数传入；为兼容旧代码，传 None 时回退到 config 的默认集合 MILVUS_COLLECTION。
"""

import uuid  # 生成主键
from datetime import datetime  # 时间戳
from typing import List, Dict, Optional  # 类型标注

from langchain_core.documents import Document  # 统一文档结构

from db_milvus import (  # 复用已建好的单例
    get_embedding,  # 向量模型
    get_vectorstore,  # LangChain Milvus 封装
    get_milvus_client,  # 原生客户端
)
from config import MILVUS_COLLECTION, MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, EMBED_DIM
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger


# ==================== 1. 按来源删除 ====================

def delete_by_source(source: str, collection_name: str = None) -> int:
    """删除指定来源文件的所有 chunk

    参数：
        source         来源名，即入库时写进 metadata 的 source（一般是文件名），必须完全一致。
        collection_name 目标集合名；传 None 时回退到 config.MILVUS_COLLECTION（兼容单集合的老调用）。
    返回：实际删除的 chunk 条数；集合不存在或该来源不存在时返回 0。
    异常与降级：集合不存在只记 warning 后返回 0，不抛异常，让调用方（如更新流程）能继续往下走。
    注意：Milvus 的删除是「逻辑删除 + 后续 compaction 才真正回收空间」，
          所以删完立刻查总量可能还看到旧条数，但查询结果已经查不到这些数据了。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()  # 复用的单例客户端
    if not client.has_collection(collection_name):
        logger.warning(f"集合 {collection_name} 不存在，无法删除")
        return 0

    # 先用 query 把该来源的主键全查出来：Milvus 的 delete 需要显式给主键
    rows = client.query(
        collection_name=collection_name,
        filter=f'source == "{source}"',  # 表达式里 source 必须用双引号包裹，且 source 本身不能带双引号
        output_fields=["pk"],  # 只取主键，避免把整段文本拉回来
        limit=10000,  # 单个来源的块数上限；超过 1 万块的文件需要分页删，教学场景用不到
    )
    pk_list = [row["pk"] for row in rows]
    if not pk_list:
        logger.info(f"来源 '{source}' 在集合 {collection_name} 中不存在")
        return 0

    client.delete(  # 按主键批量删除
        collection_name=collection_name,
        pks=pk_list,
    )
    logger.info(f"已删除来源 '{source}' 的 {len(pk_list)} 条 chunk（集合 {collection_name}）")
    return len(pk_list)  # 返回删除条数，调用方据此判断是否命中了来源


# ==================== 2. 按来源更新 ====================

def update_by_source(source: str, new_text: str, is_markdown: bool = False,
                     strategy: str = "paragraph", collection_name: str = None) -> Dict:
    """更新某个来源的文档：删旧 -> 重新切分 -> 重新写入

    参数：
        source      来源名，必须与入库时一致，否则删不到旧数据（表现为「凭空多了一份」）。
        new_text    新的整篇文本，不是增量补丁。
        is_markdown 是否按 Markdown 标题层级切分（由调用方按文件后缀判断后传入）。
        strategy    切分策略，默认 paragraph；is_markdown=True 时生效的是 markdown 策略。
        collection_name 目标集合名，None 时回退到默认集合。
    返回：
        正常：{"source","deleted"（删掉的旧块数）,"inserted"（写入的新块数）,
               "corpus_total"（写入后集合总条数）,"collection"}
        新文本切分为空：{"source","deleted","inserted": 0,
               "warning"（提示旧数据已删但没有新数据写入）}
    异常与降级：切分结果为空不抛异常，而是返回带 warning 的字典——此时旧数据已经删掉了，
                调用方（接口层）应当把 warning 透出给用户，让用户重新提交内容。
    注意：本函数内部不做 BM25 刷新，刷新由接口层统一调用，避免重复重建索引的开销。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    from text_splitter import split_text  # 延迟导入：避免本模块被导入时就拖入切分依赖

    deleted_count = delete_by_source(source, collection_name)  # 1. 先删旧数据，保证不会新旧混杂
    logger.info(f"更新 '{source}'：已删除旧 chunk {deleted_count} 条")

    docs = split_text(  # 2. 用同一套切分逻辑重新切，保证 metadata（source/section/chunk_index）结构一致
        raw_text=new_text,
        source=source,
        strategy=strategy,
        is_markdown=is_markdown,
    )
    if not docs:
        logger.warning(f"更新 '{source}'：新文本切分结果为空")
        return {
            "source": source,
            "deleted": deleted_count,
            "inserted": 0,
            "warning": "新文本切分为空，旧数据已删除但未写入新数据",
        }

    vs = get_vectorstore(collection_name)  # 3. 拿到集合对应的向量库封装（内部会自动把文本向量化）
    for i in range(0, len(docs), 10):  # 每 10 条一批：批量小了网络往返多，批量大了单次请求体过大
        batch = docs[i:i + 10]
        batch_ids = [str(uuid.uuid4()) for _ in batch]  # 主键必须自己生成，uuid 保证不冲突
        vs.add_documents(documents=batch, ids=batch_ids)

    client = get_milvus_client()
    # 4. 回读集合总条数；集合被删掉等异常情况下退化为「本次写入条数」，保证响应结构稳定
    total = client.get_collection_stats(collection_name).get("row_count", 0) if client.has_collection(collection_name) else len(docs)
    logger.info(f"更新 '{source}' 完成：删除 {deleted_count} 条 -> 写入 {len(docs)} 条，集合 {collection_name} 总计 {total} 条")
    return {
        "source": source,
        "deleted": deleted_count,
        "inserted": len(docs),
        "corpus_total": total,
        "collection": collection_name,
    }


# ==================== 3. 列出所有来源 ====================

def list_sources(collection_name: str = None) -> List[Dict]:
    """列出指定集合中所有文档来源及其 chunk 数量

    参数：collection_name 目标集合名，None 时回退到默认集合。
    返回：[{"source"（来源名）,"chunks"（该来源的块数）}, ...]，按块数从多到少排序。
          集合不存在时返回空列表。
    实现说明：Milvus 不支持 group by / count group，只能按 offset 分页把 source 字段全捞回来，
              在 Python 里用字典累加计数。数据量大时这个函数会变慢，是教学项目的简化取舍。
    并发与耗时：page_size 取 500，是「单次返回体积」和「网络往返次数」之间的折中。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return []

    source_counts: Dict[str, int] = {}  # 来源名 -> 累计块数
    offset = 0
    page_size = 500  # 每页条数

    while True:
        rows = client.query(
            collection_name=collection_name,
            filter="",  # 空表达式表示不过滤，取全部
            output_fields=["source"],  # 只要 source 字段，减少传输量
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            src = row.get("source", "未知来源")  # 老数据可能没有 source 字段，给个占位名
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(rows) < page_size:
            break  # 不足一页说明已经取完，少发一次请求（比再查一次空页更省）
        offset += page_size  # Milvus 的 offset 是「跳过前 N 条」，每次加上页大小

    result = [{"source": k, "chunks": v} for k, v in source_counts.items()]
    result.sort(key=lambda x: x["chunks"], reverse=True)  # 块数多的排前面，前端一眼看到大文件
    logger.info(f"集合 {collection_name} 共 {len(result)} 个来源，{sum(source_counts.values())} 条 chunk")
    return result


# ==================== 4. 获取来源详情 ====================

def get_source_detail(source: str, collection_name: str = None, limit: int = 100) -> Dict:
    """查看某个来源的 chunk 列表

    参数：
        source      来源名，与入库时一致。
        collection_name 目标集合名，None 时回退到默认集合。
        limit       最多返回多少条，默认 100：防止一个大文件把前端页面撑爆。
    返回：{"source","count","chunks": [{"pk","text"（截取前 200 字预览）,
          "section","chunk_index"}, ...],"collection"}；集合不存在时返回 count=0 的空结构。
    说明：这里只用于「人工核对入库效果」，所以文本截断到 200 字即可，不需要完整原文。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return {"source": source, "count": 0, "chunks": []}

    rows = client.query(
        collection_name=collection_name,
        filter=f'source == "{source}"',
        output_fields=["pk", "text", "section", "chunk_index"],
        limit=limit,
    )
    chunks = [
        {
            "pk": row.get("pk", ""),
            "text": row.get("text", "")[:200],  # 只取前 200 字做预览
            "section": row.get("section", ""),
            "chunk_index": row.get("chunk_index", 0),  # 0 表示老数据没有块序号
        }
        for row in rows
    ]
    return {
        "source": source,
        "count": len(rows),  # 注意是「本次返回条数」，受 limit 限制，不等于该来源的总块数（总数看 list_sources）
        "chunks": chunks,
        "collection": collection_name,
    }


# ==================== 5. 全量重建 ====================

def rebuild_collection(docs: List[Document], collection_name: str = None) -> Dict:
    """删除旧集合 -> 创建新集合 -> 全量写入

    参数：
        docs        要重新写入的文档列表，通常由接口层先用 fetch_all_text 把现有文本捞出来再传进来。
        collection_name 目标集合名，None 时回退到默认集合。
    返回：{"message","deleted_old"（是否删过旧集合，恒为 True）,"inserted"（写入条数）,
          "corpus_total"（重建后总条数）,"collection"}
    使用场景：切分策略大改、换向量模型或维度变化（EMBED_DIM 变了）时，旧集合的 schema
              已经和新代码不匹配，只能整体重建；重建期间集合不可用，建议避开使用高峰。
    注意：pymilvus 的建表类在这里才 import，是为了让本模块被导入时不强依赖 pymilvus 版本细节。
    """
    if collection_name is None:
        collection_name = MILVUS_COLLECTION

    from pymilvus import CollectionSchema, FieldSchema, DataType  # 延迟导入，只在本函数用到

    client = get_milvus_client()

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)  # 破坏性操作：先删掉整个集合（含已建索引）
        logger.warning(f"已删除旧集合 {collection_name}（全量重建）")

    # 重新建 schema：字段必须和写入端（db_milvus）以及查询端用的字段名完全对齐，否则写入/检索会报错
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True),  # 主键，存 uuid 字符串
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # 块正文；Milvus 单个 VARCHAR 上限就是 65535
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),  # 向量列，维度必须等于 EMBED_DIM（bge-m3 为 1024）
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),  # 来源名，按来源删除/查询就靠它
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),  # 章节标题，用于展示引用出处
        FieldSchema(name="chunk_index", dtype=DataType.INT64),  # 块在原文档中的序号
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)  # 关闭动态字段：字段固定，写入更严格也更省空间
    client.create_collection(collection_name, schema=schema)

    client.create_index(  # 向量索引必须显式创建，否则查询会走全表扫或直接报错
        collection_name=collection_name,
        field_name="vector",
        index_type=MILVUS_INDEX_TYPE,
        metric_type=MILVUS_METRIC_TYPE,  # 与向量模型 normalize + COSINE 的配置保持配套
    )

    if docs:
        vs = get_vectorstore(collection_name)
        for i in range(0, len(docs), 10):  # 同样按 10 条一批写，避免单次请求过大
            batch = docs[i:i + 10]
            batch_ids = [str(uuid.uuid4()) for _ in batch]
            vs.add_documents(documents=batch, ids=batch_ids)

    total = client.get_collection_stats(collection_name).get("row_count", 0)  # 回读重建后的条数
    logger.info(f"全量重建完成：集合 {collection_name}，共 {total} 条")
    return {
        "message": "全量重建完成",
        "deleted_old": True,  # 走到这里说明上面的 drop 分支一定执行过（或集合本来就不存在）
        "inserted": len(docs),
        "corpus_total": total,
        "collection": collection_name,
    }
