"""src/offline/metadata_store.py —— 文档与分块的元数据存储（SQLAlchemy 侧）。

在链路中的位置：
    写入：src/offline/pipeline.py → 【本文件】 → 关系库（MySQL 或降级为 SQLite）
    读取：src/api/routers/knowledge.py 的知识库管理接口

为什么向量存 Milvus、元数据却存关系库：
    Milvus 擅长"按向量找相似"，但不擅长"列出某角色下所有文档、按更新时间排序"
    这类关系型查询。两者各用所长：
        Milvus     存向量，负责语义检索
        关系库     存文档登记与分块文本，负责管理、列表、审计

    KnowledgeDoc   —— 文档登记（一份文档一行，含状态和构建统计）
    KnowledgeChunk —— 分块明细（一个块一行，含正文、页码、父块号）

状态字段 status 的取值约定（pipeline 与查询侧靠它协调）：
    building —— 正在构建（查不到或正在写）
    ready    —— 构建完成、数据完整可用
    failed   —— 构建失败（可在 metadata 里附错误信息）

多租户：每个函数都带 tenant_id，所有查询都按它过滤，保证租户之间互不可见。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

from src.models.database import KnowledgeChunk, KnowledgeDoc, db_session


def find_document(role_id: str, tenant_id: str, source: str) -> KnowledgeDoc | None:
    """按（角色, 租户, 文档来源）查文档登记。

    参数：
        role_id: 角色 id
        tenant_id: 租户 id
        source: 文档来源（文件路径）
    返回：
        KnowledgeDoc 对象；不存在返回 None。

    三元组定位一份文档：
        同一份文件可以被多个角色各自构建一份（内容可能用不同分块策略），
        所以"来源路径"本身不足以唯一确定一行，必须带上角色和租户。
    """
    with db_session() as db:
        return db.scalar(select(KnowledgeDoc).where(KnowledgeDoc.role_id == role_id, KnowledgeDoc.tenant_id == tenant_id, KnowledgeDoc.doc_source == source))


def create_document(role_id: str, tenant_id: str, source: str, metadata: dict[str, Any] | None = None) -> KnowledgeDoc:
    """创建或重置一份文档登记（幂等）。

    参数：
        role_id / tenant_id / source: 文档的三元组标识
        metadata: 附加元数据（如解析器名称）
    返回：
        落库后的 KnowledgeDoc 对象（含自增 id）。

    为什么是"创建或重置"而不是纯新建：
        同名文档重复构建时若每次都新建，登记表里会堆出多条同源记录。
        这里改成：已存在就把状态重置回 building —— 表达"这份文档又进入构建中了"，
        列表页也能正确显示"构建中"而不是显示上一次的旧统计。

    status 一律置为 building：
        这是与查询侧的契约 —— 只要状态是 building，就说明数据还不完整可信。
        构建成功后由 mark_document 改成 ready。

    db.refresh(doc) 是必需的：
        doc.id 是数据库自增生成的主键，commit 前在对象上是 None。
        refresh 会把数据库里的最终值读回来，调用方才能拿到用于关联 Milvus 的 doc_id。
    """
    with db_session() as db:
        doc = db.scalar(select(KnowledgeDoc).where(KnowledgeDoc.role_id == role_id, KnowledgeDoc.tenant_id == tenant_id, KnowledgeDoc.doc_source == source))
        if doc:
            doc.status = "building"
            doc.metadata_json = metadata or {}
            doc.updated_at = datetime.utcnow()
        else:
            doc = KnowledgeDoc(role_id=role_id, tenant_id=tenant_id, doc_source=source, metadata_json=metadata or {}, status="building")
            db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc


def mark_document(doc_id: int, status: str, metadata: dict[str, Any] | None = None) -> KnowledgeDoc | None:
    """更新文档状态，并合并附加元数据。

    参数：
        doc_id: 文档 id
        status: 新状态（ready / failed / building）
        metadata: 要与已有元数据合并的内容（如 {"chunks": 128, "strategy": "semantic"}）
    返回：
        更新后的对象；文档不存在返回 None。

    合并而不是覆盖（{**旧, **新}）：
        调用方只想补充几个字段（比如补上分块数），
        直接覆盖会把之前记录的解析器信息等丢掉。

    注意本函数**不校验 tenant_id**：
        它是内部函数，doc_id 由 pipeline 在同一次构建流程中持有，不来自外部输入。
        相反，下面 list_documents / get_document / delete_document_metadata
        这些可能被接口层直接调用的函数，都强制带 tenant_id 过滤。
    """
    with db_session() as db:
        doc = db.get(KnowledgeDoc, doc_id)
        if not doc:
            return None
        doc.status = status
        if metadata:
            doc.metadata_json = {**(doc.metadata_json or {}), **metadata}
        doc.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(doc)
        return doc


def replace_chunks(doc_id: int, role_id: str, tenant_id: str, chunks: list[dict[str, Any]]) -> int:
    """整体替换一份文档的分块明细（先删后插）。

    参数：
        doc_id / role_id / tenant_id: 归属信息
        chunks: 分块列表，每项含 content/summary/parent_id/page/metadata
    返回：
        写入的块数。

    为什么是"先删后插"而不是"增量更新"：
        重新构建会换分块策略、或文档内容变了，块与块之间没有稳定的对应关系，
        逐条比对更新既复杂又容易出错。整体替换的逻辑只有一句、
        且天然保证"元数据与当前这次构建结果一致"，不会残留上一次的块。

    删除条件用 (doc_id, tenant_id) 而不是只用 doc_id：
        租户过滤是本项目所有数据操作的统一约束。
    """
    with db_session() as db:
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc_id, KnowledgeChunk.tenant_id == tenant_id))
        for item in chunks:
            db.add(KnowledgeChunk(doc_id=doc_id, role_id=role_id, tenant_id=tenant_id, content=item["content"], summary=item.get("summary", ""), parent_id=item.get("parent_id"), page=item.get("page", -1), metadata_json=item.get("metadata", {})))
        db.commit()
    return len(chunks)


def list_documents(role_id: str | None = None, tenant_id: str = "default") -> list[KnowledgeDoc]:
    """列出文档，可按角色过滤。

    参数：
        role_id: 角色 id；传 None 表示列出该租户下的全部文档
        tenant_id: 租户 id，默认 "default"
    返回：
        按更新时间倒序的文档列表（最近变动的最靠前）。

    注意是"先按租户过滤，再按需追加角色过滤"的构建式查询：
        租户条件是强制的，角色条件是可选追加的 —— 这样即使调用方不传 role_id，
        也不会跨租户泄露数据。
    """
    with db_session() as db:
        query = select(KnowledgeDoc).where(KnowledgeDoc.tenant_id == tenant_id)
        if role_id:
            query = query.where(KnowledgeDoc.role_id == role_id)
        return list(db.scalars(query.order_by(KnowledgeDoc.updated_at.desc())).all())


def get_document(doc_id: int, tenant_id: str = "default") -> KnowledgeDoc | None:
    """按 id 取文档，并校验租户归属。

    参数：
        doc_id: 文档 id
        tenant_id: 租户 id
    返回：
        文档对象；不存在或不属于该租户时返回 None。

    为什么不属于本租户要返回 None 而不是抛 403：
        doc_id 是自增整数，很容易被猜到。
        统一返回"不存在"，攻击者无法通过响应差异判断某个 id 是否真实存在 ——
        这比明确区分"无权访问"和"不存在"更安全。
    """
    with db_session() as db:
        doc = db.get(KnowledgeDoc, doc_id)
        return doc if doc and doc.tenant_id == tenant_id else None


def delete_document_metadata(doc_id: int, tenant_id: str = "default") -> bool:
    """删除文档登记及其全部分块明细。

    参数：
        doc_id: 文档 id
        tenant_id: 租户 id
    返回：
        删除成功 True；文档不存在或不属于该租户返回 False。

    先删子表（chunks）再删主表（doc）：
        分块通过 doc_id 关联文档。先删主表的话，
        如果外键约束开着会直接报错；即使没有外键约束，也会留下孤儿分块数据。

    注意：本函数只清理关系库。
        Milvus 里的向量需要调用方另外调 milvus_store.delete_document 删除 ——
        接口层的删除操作必须把两边都做掉，否则会留下"元数据没了但向量还在"
        的幽灵数据（用户会看到搜到却打不开的引用）。
    """
    with db_session() as db:
        doc = db.get(KnowledgeDoc, doc_id)
        if not doc or doc.tenant_id != tenant_id:
            return False
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc_id, KnowledgeChunk.tenant_id == tenant_id))
        db.delete(doc)
        db.commit()
        return True
