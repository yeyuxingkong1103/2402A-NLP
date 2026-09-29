"""法源清单（citation 事件）组装（批次 37 抽取）。

批次 37 之前，sources 字典在 chat/service.py 与 chat/streaming.py 各写一份逐字平行实现
（护栏阶梯的教训：两处平行，漏改一处就是行为分叉）。本模块把组装收敛成一处。

summary（条文一句话摘要）由检索层随正文一起取出（vector_search._fetch_chunks 外连接
chunk_summaries 表），挂在 RetrievedArticle.summary 上——本模块只做纯组装、不碰数据库，
单元测试因此保持零外部依赖。

注意：sources 与 context_excerpts 必须始终同源（都从 retrieval_result.articles 出），
本模块只负责 sources 一侧。
"""

from collections.abc import Sequence
from typing import Any


def build_citation_sources(articles: Sequence[Any]) -> list[dict[str, Any]]:
    """把检索结果组装成 citation 事件的 sources 列表（含 summary）。

    字段与批次 37 前完全一致（chunk_id/law_name/article_number/paragraph_number/page），
    仅新增 summary 键——旧前端多拿一个未知键不影响既有展示；
    未生成摘要的条文显式返回 None。
    """
    return [
        {
            "chunk_id": art.chunk_key,
            "law_name": art.document_title,
            "article_number": art.article_number,
            "paragraph_number": art.paragraph_number,
            # 首期采集与索引没有页码字段，明确返回 null，不用其他字段代替。
            "page": None,
            "summary": getattr(art, "summary", None),
        }
        for art in articles
    ]
