"""条款号精确通路：把条号变成块。

存在的理由：AC-2 要求"输入第X条返回该条"准确率 100%，向量路给不了这个保证。
本模块只做一件事——按条号取回条原文，并与 Milvus 召回结果的父块**保持同一形状**，
这样下游（精排、提示词、校验）不必区分数据来自哪条路。

注意 MySQL 的 article_no 存的是阿拉伯数字字符串（附录 C 定的 VARCHAR(16)），
而 Milvus 的 article_no 是 INT64，两处口径不同，转换放在本模块出口。
"""
from __future__ import annotations

from app.ingest.load_mysql import LAW_ID

DEFAULT_LAW_ID = LAW_ID

# 效力取 version.status 而非 article.status：设计文档 4.3 第②关定的判据是
# 「该 law_version 的 status = 现行有效」，版本表才是效力的判据来源，条表那份
# 是入库时抄下来的副本，法规更新后两列会分叉。所以 JOIN 是必需的不是装饰——
# 召回侧虽已按 status 过滤，精确块绕过了 Milvus，得自己带上效力供校验关使用。
ARTICLE_SQL = """
SELECT a.article_no, a.article_no_cn, a.path, a.text, v.status, a.law_id
FROM article a
JOIN law_version v ON v.law_id = a.law_id AND v.law_version = a.law_version
WHERE a.law_id = %s AND a.article_no IN ({placeholders})
"""


def _row_to_block(row: tuple) -> dict:
    """把一行 SQL 结果转成父块形状，与 Milvus 召回后的父块键名一致。"""
    article_no, article_no_cn, path, text, status, law_id = row
    return {
        # 精确块没有 Milvus 块 id：置顶去重按 article_no 走，不靠 chunk_id
        "chunk_id": None,
        "article_no": int(article_no),
        "article_no_cn": article_no_cn,
        "paragraph_no": None,
        "item_no": None,
        "path": path,
        "chunk_type": "father",
        "parent_id": None,
        "text": text,
        "source": "exact",
        "rank_index": None,
        "rerank_score": None,
        # 效力随块带出，供 cite_verify 第②关使用；不额外查一次库
        "status": status,  # 来自 v.status（版本表），不是条表副本
        "law_id": law_id,
    }


def fetch_articles(conn, article_nos: list[int],
                   law_id: str = DEFAULT_LAW_ID) -> list[dict]:
    """按条号取条，**保持输入顺序**，取不到的条直接跳过。

    保持输入顺序是因为置顶顺序按问句里条号出现的先后排（设计文档 4.1）；
    IN 查询返回的顺序由 MySQL 决定，不能指望。
    """
    if not article_nos:
        return []
    placeholders = ", ".join(["%s"] * len(article_nos))
    sql = ARTICLE_SQL.format(placeholders=placeholders)
    # article_no 在库里是字符串列，入参统一转字符串避免隐式转换导致索引失效
    params = [law_id] + [str(n) for n in article_nos]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        by_no = {int(row[0]): _row_to_block(row) for row in cur.fetchall()}
    return [by_no[n] for n in article_nos if n in by_no]
