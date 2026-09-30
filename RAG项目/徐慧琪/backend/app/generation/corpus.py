"""校验用的只读语料适配器。

存在的理由：校验器本身要保持纯函数（可离线单测），但真实运行时要查两个库。
把"查库"收进本模块、把"判定"留给 cite_verify，是让安全阀既能被穷举测试
又能跑在真数据上的唯一办法。

本模块只读，不写任何表——安全阀不该有副作用。
"""
from __future__ import annotations

from app.db.milvus import COLLECTION
from app.retrieval.article_lookup import DEFAULT_LAW_ID, fetch_articles


class LawCorpus:
    """MySQL 出条原文与效力，Milvus 出款/项编号。"""

    def __init__(self, conn, client, law_id: str = DEFAULT_LAW_ID):
        self._conn = conn
        self._client = client
        self._law_id = law_id

    def article(self, no: int) -> dict | None:
        """取条原文与效力。取不到返回 None（校验第一关据此判失败）。

        必须走 Task 2 的 fetch_articles、不在本模块另写 SQL：那条 SQL 的
        列绑定（效力读 law_version.status 而非 article.status）由
        test_article_lookup 的 SQL 文本断言守门，另写一份就等于绕开那道断言。
        """
        rows = fetch_articles(self._conn, [no], self._law_id)
        if not rows:
            return None
        row = rows[0]
        return {"text": row["text"], "status": row["status"], "law_id": row["law_id"]}

    def para_index(self, no: int) -> dict:
        """数该条有几款、有哪些项。

        查全量而非本轮召回的子块：召回集里没有某一款，不等于该款不存在，
        按召回集判会把正确引用误杀（设计文档 4.3）。

        款数只能数 chunk_type == "paragraph" 的行、按 1..N 归一，不能拿行的
        paragraph_no 当款号：那个字段是**解析行号**，项行也占号——真库实测
        第 395 条 2 款 + 7 项，款行落在行号 1 与 9、项行占 2..8，照行号收进
        集合会让第 3~9 款全部判「存在」，第③关在这 91 条含项法条上等于失效
        （第 584 条无项、款行号恰好连续，所以旧夹具没暴露这个真形态）。
        """
        expr = (f'article_no == {int(no)} and '
                f'chunk_type in ["paragraph", "item"]')
        rows = self._client.query(COLLECTION, filter=expr,
                                  output_fields=["chunk_type", "item_no"])
        paragraph_rows = [r for r in rows if r["chunk_type"] == "paragraph"]
        items = {r["item_no"] for r in rows if r.get("item_no")}
        return {"paragraphs": set(range(1, len(paragraph_rows) + 1)), "items": items}
