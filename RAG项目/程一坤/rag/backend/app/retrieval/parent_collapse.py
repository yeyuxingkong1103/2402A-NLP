"""检索候选的父块解析与归并：把子块命中升级为父块视角，控制 top_k 多样性。

两个入口、两个时机：
- load_keyword_parent_items（关键词路召回后、RRF 融合前）：把命中的子块正文
  替换为父块正文（MySQL 反查），让引用展示整条法条；chunk_key 仍保留子块
  （溯源精确到子块），分数仍保留子块的 BM25 分。
- collapse_parent_items（重排之后、上下文组装前）：同一法条常被多个子块命中
  而重复占坑，按父块归并（同父块保留最高分）后截断 top_n，避免 top_k 被
  同一法条挤满、多样性塌掉。

数据流（链路位置）：
  关键词召回 hits ──load_keyword_parent_items──▶ 子块升级父块的 RankedItem 列表
  融合+重排后的候选 ──collapse_parent_items──▶ 按父块去重的前 top_n 条

本模块不做多路融合打分（那是 fusion.py 的职责）；排序依据（_rank_score）
与分数合并（_prefer_score）是归并专用的私有工具。
归并正确性依赖两个前提：
1) parent_chunk_key 在 load 阶段已正确指向父块（或为 None 表示父块级命中）；
2) _rank_score 的优先级与重排器分数语义一致（越大越相关）。
术语与 docs/CONTEXT.md 一致：条号 article_number / 款 paragraph_number /
项 item_number。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from app.db.sql_models import Document, DocumentChunk, DocumentVersion, Law, LawVersion
from app.retrieval.context_builder import date_to_timestamp, resolve_current_status
from app.retrieval.fusion import RankedItem


def load_keyword_parent_items(
    session_factory: Any,
    hits: Sequence[Any],
) -> list[RankedItem]:
    """把关键词命中的子块替换为父块正文，并保留 BM25 分。

    Args: session_factory（MySQL 会话工厂，反查子块的父块与元数据）；
        hits（关键词路命中列表，元素需有 chunk_key/score）
    Returns: 展示字段取父块（引用展示整条法条），chunk_key 保留命中的
        子块 key（溯源精确到子块），分数保留子块的 BM25 分

    为什么展示字段取父块而 chunk_key 保留子块：引用要展示整条法条
    （父块完整语境），溯源却要精确到实际命中的子块；两者粒度不同，
    必须拆开保留，不能只留其一。
    失败模式防御：hits 为空直接返回；DB 行缺失（子块已被删）时不抛错——
    _fetch_rows 返回行数可能少于 hits，缺失的 chunk_key 静默不进输出。
    """
    if not hits:
        return []
    # hit_scores：chunk_key → BM25 分，输出构造时回填；float() 统一 Decimal 等类型
    hit_scores = {hit.chunk_key: float(hit.score) for hit in hits}
    # 记录命中顺序：DB 反查后必须还原原排名，RRF 的 rank 才正确
    hit_order = {hit.chunk_key: index for index, hit in enumerate(hits)}
    # 同一会话内完成子块 + 父块两次反查：一次连接拿全数据，避免逐条查库
    with session_factory() as session:
        # hit_rows：子块行（含父块指针），此时仍是 DB 返回序
        hit_rows = _fetch_rows(session, list(hit_scores))
        # parent_keys：所有命中子块的父块键，去重后批量反查父块行
        parent_keys = {row.parent_chunk_key for row in hit_rows if row.parent_chunk_key}
        parent_rows = _fetch_rows(session, list(parent_keys))
    # 反查结果按命中时的原始顺序还原（RRF 按 rank 计分，顺序不能乱）
    hit_rows.sort(key=lambda row: hit_order[row.chunk_key])
    # parent_by_key：父块行索引，O(1) 查找
    parent_by_key = {row.chunk_key: row for row in parent_rows}

    # 逐条把 DB 行升级为父块视角的 RankedItem
    items = []
    for row in hit_rows:
        # 有父块用父块（整条语境），无父块（父块级命中）退回自身
        parent = parent_by_key.get(row.parent_chunk_key) if row.parent_chunk_key else None
        source_row = parent or row
        # 字段映射规则：
        # - chunk_key=子块(溯源)、parent_chunk_key=父块(后续归并的分组键)
        # - content/title/条款项号=父块优先(整条语境)
        # - 分数=子块 BM25(命中证据)、source 恒为 keyword
        items.append(
            RankedItem(
                chunk_key=row.chunk_key,
                parent_chunk_key=source_row.chunk_key,
                score=hit_scores[row.chunk_key],
                source="keyword",
                sources=("keyword",),
                content=source_row.content,
                document_title=source_row.title,
                # 条号优先取父块的（整条语境），父块没有再用子块的
                article_number=source_row.article_number or row.article_number,
                source_url=source_row.source_url,
                keyword_score=hit_scores[row.chunk_key],
                paragraph_number=source_row.paragraph_number,
                item_number=source_row.item_number,
                document_type=source_row.document_type,
                jurisdiction=source_row.jurisdiction,
                effective_date=date_to_timestamp(source_row.effective_date, unknown=0),
                expiration_date=date_to_timestamp(source_row.expiration_date),
                issuing_authority=source_row.issuing_authority,
                is_current=resolve_current_status(source_row.status, source_row.expiration_date),
                document_id=source_row.document_key,
            )
        )
    return items


def collapse_parent_items(items: Sequence[RankedItem], top_n: int) -> list[RankedItem]:
    """按父块归并，保留最高重排分并汇总两路来源分。

    Args: items（融合/打分后的候选列表，同一父块可能对应多个子块命中）；
        top_n（归并后最多保留的父块数）
    Returns: 每个父块一条、按排序分（重排分 > 融合分 > 原始分）降序的前 top_n 条
    归并放在重排之后：同一法条被多个子块命中会重复占坑，不归并会让 top_k 被同一法条挤满、多样性塌掉。

    与 fuse_results 的分工：融合只合并多路排序、不管内容是否重复；
    归并按 parent_chunk_key 聚合，专治"同一条法条的多子块重复占坑"。
    """
    # 单遍扫描：每个条目要么占坑、要么挑战已有胜者，O(n) 完成归并
    # grouped：parent_key → 胜出条目（父块视角），一个父块最终只留一条
    grouped: dict[str, RankedItem] = {}
    for item in items:
        # 无父块信息时（父块级命中/已归并条目）以自身为分组键
        parent_key = item.parent_chunk_key or item.chunk_key
        existing = grouped.get(parent_key)
        if existing is None:
            # 该父块首次出现：转成父块视角占坑
            grouped[parent_key] = _as_parent_item(item, parent_key)
            continue

        # 同父块重复命中：只留"排序分"更高的一条，另一条只贡献来源与分数信息
        # （胜负判定用 _rank_score：重排分 > 融合分 > 原始分，见函数定义）
        winner = item if _rank_score(item) > _rank_score(existing) else existing
        grouped[parent_key] = _as_parent_item(
            winner,
            parent_key,
            # 来源取并集：两路都命中同一父块时，融合分来源信息不能丢
            sources=tuple(sorted(set(existing.sources) | set(item.sources))),
            # 分数取较大者：归并时保留最强信号，不因重复命中被稀释
            vector_score=_prefer_score(existing.vector_score, item.vector_score),
            keyword_score=_prefer_score(existing.keyword_score, item.keyword_score),
        )

    # 归并完统一按排序分降序，截断到 top_n
    # （top_n 限制的是归并后的父块数，不是子块数——这是归并的意义所在）
    # 注意：归并不改写 content——同一父块的多个子块正文不同，
    # 但父块正文已在 load_keyword_parent_items 阶段统一填好
    ordered = sorted(grouped.values(), key=_rank_score, reverse=True)
    return ordered[:top_n]

def _fetch_rows(session: Any, chunk_keys: list[str]) -> list[Any]:
    """按 chunk_key 批量取分块行及其父块/法规元数据（一次联查代替逐条查）。"""
    if not chunk_keys:
        return []
    # select 列分四组，返回 Row 按属性访问：
    # 1) 分块自身字段：键、父块指针、正文与条款项号
    # 2) 文档级字段：标题、来源链接、文档标识
    # 3) 法规身份字段：类型/辖区/发布机关
    # 4) 版本级字段：生效/失效日期与状态（is_current 判定输入）
    # 返回 Row 列表：字段顺序与 select 列一致，调用方按属性访问、不依赖下标
    return session.execute(
        select(
            DocumentChunk.chunk_key,
            DocumentChunk.parent_chunk_key,
            DocumentChunk.content,
            DocumentChunk.article_number,
            DocumentChunk.paragraph_number,
            DocumentChunk.item_number,
            Document.title,
            Document.source_url,
            Document.document_key,
            Law.document_type,
            Law.jurisdiction,
            Law.issuing_authority,
            LawVersion.effective_date,
            LawVersion.expiration_date,
            LawVersion.status,
        )
        # 分块挂在版本下：经 DocumentVersion 找到所属 Document
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        # outerjoin：历史文档可能没挂法规身份，联不上时元数据为 NULL，不能丢命中
        .outerjoin(LawVersion, LawVersion.document_version_id == DocumentVersion.id)
        .outerjoin(Law, Law.id == LawVersion.law_id)
        # in_ 批量过滤：子块与父块各一次往返，行数与命中数同量级
        .where(DocumentChunk.chunk_key.in_(chunk_keys))
    ).all()


def _as_parent_item(
    item: RankedItem,
    parent_key: str,
    *,
    sources: tuple[str, ...] | None = None,
    vector_score: float | None = None,
    keyword_score: float | None = None,
) -> RankedItem:
    """把条目转成"父块视角"：chunk_key 换成父块，分数/来源按入参覆盖。

    归并的分组与胜负判定都基于原始条目，这里负责在最后一步改写视角：
    未显式传入的来源/分数字段沿用原条目，显式传入的（并集来源、较大分数）
    才覆盖——调用方借此把重复命中的信息合并进胜出条目。
    注意：改写视角不改动分数本身，score/fusion_score/rerank_score 原样保留。
    逐字段显式复制而非 **vars(item)：杜绝未来新增字段被无声漏拷。
    """
    return RankedItem(
        chunk_key=parent_key,
        parent_chunk_key=None,  # 已是父块视角，不再有上级
        score=item.score,
        source=item.source,
        sources=sources if sources is not None else item.sources,
        content=item.content,
        document_title=item.document_title,
        article_number=item.article_number,
        source_url=item.source_url,
        vector_score=vector_score if vector_score is not None else item.vector_score,
        keyword_score=keyword_score if keyword_score is not None else item.keyword_score,
        recall_score=item.recall_score,
        fusion_score=item.fusion_score,
        rerank_score=item.rerank_score,
        paragraph_number=item.paragraph_number,
        item_number=item.item_number,
        document_type=item.document_type,
        jurisdiction=item.jurisdiction,
        effective_date=item.effective_date,
        expiration_date=item.expiration_date,
        issuing_authority=item.issuing_authority,
        is_current=item.is_current,
        document_id=item.document_id,
    )


def _rank_score(item: RankedItem) -> float:
    """排序依据：重排分 > 融合分 > 原始分（越靠后的链路分越可信）。

    重排是链路最后一环、直接面向"与问题的相关度"，故存在时绝对优先；
    融合分（RRF）与原始召回分只作退化兜底，保证任一阶段中断仍有合理排序。
    返回值仅用于归并排序与胜负判定，不写回条目自身。
    """
    if item.rerank_score is not None:
        return item.rerank_score
    if item.fusion_score is not None:
        return item.fusion_score
    return item.score


def _prefer_score(first: float | None, second: float | None) -> float | None:
    """取两分数中的较大者（都为 None 返回 None）——归并时保留最强信号。"""
    # None 表示该路从未命中，不能当作 0 参与比较，必须保持 None 语义
    values = [value for value in (first, second) if value is not None]
    return max(values) if values else None
