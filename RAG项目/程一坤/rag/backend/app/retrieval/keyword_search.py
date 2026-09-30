"""关键词检索：jieba 分词 + BM25，并对"法条号"做精确命中加分。

为什么需要它（已经向量检索了）：
- 向量检索擅长语义相似，但对"第四十七条"这类**编号**不敏感，
  编号在向量空间里几乎没有语义，容易被同义表述盖过去；
- 用户问"劳动合同法第四十条怎么规定"时，必须**精确命中**该条，
  这时关键词检索（尤其编号精确匹配）比向量更可靠。

实现取舍（见 docs/开发路线图.md 任务 3.1）：
首期知识库 ≤1 万份文档，直接在进程内建 BM25 索引即可，
不引入 MySQL FULLTEXT、也不引入 Elasticsearch，减少组件与运维成本。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.db.law_status import EXPIRED_STATUSES
from app.db.sql_models import DocumentChunk, DocumentVersion, Law, LawVersion
from app.db.version_status import APPROVED

# 中文条号形态："第四十七条""第九十九条之一"
ARTICLE_NUMBER_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?")

# 分词时丢弃的纯符号与空白
NOISE_TOKEN_PATTERN = re.compile(r"^[\W_]+$")


class KeywordSearchError(RuntimeError):
    """关键词检索失败（索引为空、词表构建失败等）。"""


@dataclass(frozen=True)
class KeywordHit:
    """一条关键词检索结果。"""

    chunk_key: str
    score: float
    # 命中的法条号（若查询里带了条号并精确匹配上，用于排序加分与调试）
    matched_article_number: str | None = None


def tokenize(text: str) -> list[str]:
    """分词：中文用 jieba，英文与数字保留原样，过滤纯符号。"""
    # 去掉 HTML 实体与多余空白，避免把 "&nbsp;" 切成一堆垃圾词
    normalized = text.replace("&nbsp;", " ").strip()
    tokens = []
    for token in jieba.lcut(normalized):
        candidate = token.strip()
        # 过滤空白、纯标点、单字符的标点类 token
        if not candidate or NOISE_TOKEN_PATTERN.match(candidate):
            continue
        tokens.append(candidate.lower())
    return tokens


def extract_article_numbers(text: str) -> list[str]:
    """从查询文本里抽出条号（可能多个），用于精确命中判断。"""
    return [match.group(0) for match in ARTICLE_NUMBER_PATTERN.finditer(text)]


class KeywordSearcher:
    """基于内存 BM25 的关键词检索器。

    使用方式：构造后可反复 search；数据变化后调用 refresh() 重建索引。
    """

    def __init__(
        self,
        session_factory,
        *,
        article_number_boost: float = 5.0,
        query_expander: Any | None = None,
        expansion_weight: float = 0.5,
    ) -> None:
        # session_factory：取 MySQL 会话（只为读分块文本与条号）
        self.session_factory = session_factory
        # 条号精确命中时的加分倍数：编号是硬匹配，应当显著抬高排序
        self.article_number_boost = article_number_boost
        # 同义术语扩写器（批次 13，可选）：只对关键词路生效，
        # 命中替代表达就把法条用语以 OR 方式追加进 BM25 查询（原词保留）
        self.query_expander = query_expander
        # 批次 13-收尾：扩展词得分权重（<1 降权）。
        # 泛化扩展词（如"解除劳动合同"）df 极高，全权打分会在召回里挤占原词名额；
        # 降权后"只命中扩展词"的分块分数减半，"原词+扩展词都命中"不受影响。
        self.expansion_weight = expansion_weight

        self._chunk_keys: list[str] = []
        self._article_numbers: list[str | None] = []
        self._metadata: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def indexed_chunk_count(self) -> int:
        """已建索引的分块数（0 表示还没 refresh）。"""
        return len(self._chunk_keys)

    def refresh(self) -> int:
        """从 MySQL 重新加载全部分块并重建 BM25 索引，返回分块数。"""
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    DocumentChunk.chunk_key,
                    DocumentChunk.retrieval_text,
                    DocumentChunk.article_number,
                    Law.document_type,
                    Law.jurisdiction,
                    LawVersion.effective_date,
                    LawVersion.expiration_date,
                    LawVersion.status,
                )
                .outerjoin(
                    LawVersion,
                    LawVersion.document_version_id == DocumentChunk.document_version_id,
                )
                .outerjoin(Law, Law.id == LawVersion.law_id)
                # 阶段6（6.3②）检索侧兜底：关键词索引只收录已审核发布的版本，
                # 未审核内容不进入 BM25 索引（与向量检索的兜底过滤同一原则）
                .join(
                    DocumentVersion,
                    DocumentVersion.id == DocumentChunk.document_version_id,
                )
                .where(DocumentVersion.version_status == APPROVED)
            ).all()

        if not rows:
            # 空库不是异常，但要明确告知调用方"索引里没有东西"，
            # 否则检索会静默返回空结果，看起来像"没有问题匹配"
            raise KeywordSearchError("没有任何分块可供建立关键词索引")

        self._chunk_keys = [row.chunk_key for row in rows]
        self._article_numbers = [row.article_number for row in rows]
        self._metadata = [
            {
                "document_type": row.document_type,
                "jurisdiction": row.jurisdiction,
                "effective_date": row.effective_date,
                "expiration_date": row.expiration_date,
                "is_current": _resolve_current(row.status, row.expiration_date),
            }
            for row in rows
        ]

        # 用检索文本（标题+条号+正文）建索引，而不是只用正文：法条检索里"法规名 + 条号"本身是强信号
        corpus = [tokenize(row.retrieval_text or "") for row in rows]
        self._bm25 = BM25Okapi(corpus)
        return len(self._chunk_keys)

    def expansion_phrases(self, query: str) -> list[str]:
        """本查询会追加哪些法条用语（未开扩展时为空），供统计与诊断。"""
        if self.query_expander is None:
            return []
        return list(self.query_expander.expand(query))

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        jurisdiction: str | None = None,
        document_types: list[str] | None = None,
        as_of_date: str | date | None = None,
        only_current: bool = False,
    ) -> list[KeywordHit]:
        """检索关键词，在截取 top_k 前应用与向量路等价的法律过滤。"""
        if self._bm25 is None:
            raise KeywordSearchError("索引尚未建立，请先调用 refresh()")

        tokens = tokenize(query)
        # 批次 13：同义术语扩展——口语表达追加对应的法条用语（原词保留），
        # 扩展词对 BM25 而言等价于 OR 子句：只命中扩展词的分块也能拿到分数。
        # 批次 13-收尾：扩展词单独成通道打分后乘权重（默认 0.5），
        # 避免 df 极高的泛化扩展词在召回里挤占原词名额。向量路与重排输入一律不动。
        expansion_tokens: list[str] = []
        for phrase in self.expansion_phrases(query):
            for token in tokenize(phrase):
                if token not in tokens and token not in expansion_tokens:
                    expansion_tokens.append(token)
        if not tokens:
            return []

        if expansion_tokens:
            base_scores = self._bm25.get_scores(tokens)
            expansion_scores = self._bm25.get_scores(expansion_tokens)
            scores = [
                base + self.expansion_weight * extra
                for base, extra in zip(base_scores, expansion_scores)
            ]
        else:
            scores = self._bm25.get_scores(tokens)

        # 查询里出现条号时，给"该条号完全一致"的分块额外加分：
        # 编号是硬匹配条件，不应被 BM25 的词频统计淹没
        wanted_article_numbers = set(extract_article_numbers(query))
        boosts: dict[int, float] = {}
        if wanted_article_numbers:
            for index, article_number in enumerate(self._article_numbers):
                if article_number in wanted_article_numbers:
                    boosts[index] = self.article_number_boost

        # 组装结果：过滤掉得分为 0 的条目（一个词都没命中，没有参考价值）
        candidates: list[KeywordHit] = []
        for index, base_score in enumerate(scores):
            total_score = float(base_score) + boosts.get(index, 0.0)
            if total_score <= 0:
                continue
            metadata = self._metadata[index] if index < len(self._metadata) else {}
            if not _matches_filters(
                metadata,
                jurisdiction=jurisdiction,
                document_types=document_types,
                as_of_date=as_of_date,
                only_current=only_current,
            ):
                continue
            candidates.append(
                KeywordHit(
                    chunk_key=self._chunk_keys[index],
                    score=total_score,
                    matched_article_number=self._article_numbers[index]
                    if index in boosts
                    else None,
                )
            )

        candidates.sort(key=lambda hit: hit.score, reverse=True)
        return candidates[:top_k]


def build_keyword_searcher_rows(
    rows: Sequence[tuple[str, str, str | None]],
    *,
    metadata: Sequence[dict[str, Any]] | None = None,
    query_expander: Any | None = None,
    expansion_weight: float = 0.5,
) -> KeywordSearcher:
    """用内存行构造检索器，供纯逻辑测试复用。"""
    searcher = KeywordSearcher(
        session_factory=None,
        query_expander=query_expander,
        expansion_weight=expansion_weight,
    )
    searcher._chunk_keys = [row[0] for row in rows]
    searcher._article_numbers = [row[2] for row in rows]
    searcher._metadata = list(metadata or [{} for _ in rows])
    searcher._bm25 = BM25Okapi([tokenize(row[1]) for row in rows])
    return searcher


def _matches_filters(
    metadata: dict[str, Any],
    *,
    jurisdiction: str | None,
    document_types: list[str] | None,
    as_of_date: str | date | None,
    only_current: bool,
) -> bool:
    if jurisdiction and metadata.get("jurisdiction") != jurisdiction:
        return False
    if document_types and metadata.get("document_type") not in document_types:
        return False
    if only_current and metadata.get("is_current") is False:
        return False
    target_date = _coerce_date(as_of_date)
    effective_date = metadata.get("effective_date")
    expiration_date = metadata.get("expiration_date")
    if target_date and effective_date and effective_date > target_date:
        return False
    if target_date and expiration_date and expiration_date <= target_date:
        return False
    return True


def _coerce_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _resolve_current(status: str | None, expiration_date: date | None) -> bool | None:
    """判断该版本是否现行有效；证据不足时返回 None（未知）而不是 False。

    区分两者才有意义：元数据未补齐时不该被 only_current 误杀，已失效则必须过滤。
    失效取值取自 app/db/law_status.py（唯一定义处）；不适用（案例材料）不算失效。
    """
    if status is None and expiration_date is None:
        return None
    return status not in EXPIRED_STATUSES and (
        expiration_date is None or expiration_date > date.today()
    )
