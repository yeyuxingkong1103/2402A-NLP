"""把 law_articles.jsonl 灌进 MySQL 三表。

存在的理由：技术方案 4.2 定义了 13 个元数据字段，而 law_chunks.jsonl 只有 6 个；
其余（law_id / law_version / status / effective_date / article_no_cn / source_hash）
按 7.1 归属 MySQL 主表，是后续检索过滤（FR-3.3 效力过滤）与增量更新（FR-1.4）的依据。

本期只写条级 1260 行：article 表的 paragraph_no / item_no 恒为 NULL，
款/项级编号留在 Milvus 的块级 payload 里。
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from app.db.mysql import SCHEMA_PATH, apply_schema, connect, insert_many

# 民法典的单行取值。这些是事实性常量，2026-09-22 经用户确认；
# 换成多部法律时本模块要改成从配置读，而不是在这里加分支。
LAW_ID = "minfadian"
LAW_NAME = "中华人民共和国民法典"
LAW_VERSION = "v1"
EFFECTIVE_DATE = "2021-01-01"
STATUS = "现行有效"

# 列清单独立成常量：build_rows 的元组顺序与它一一对应，
# 两者放在一起才能一眼看出有没有错位
ARTICLE_COLUMNS = [
    "law_id", "law_version", "article_no", "article_no_cn", "paragraph_no", "item_no",
    "path", "text", "effective_date", "status", "replaced_by", "replaces",
    "source_hash", "team_id",
]

# 必填字段：Global Constraint 要求字段为空时中断而非静默跳过。
# 这里不依赖 verify_ingest.check_fields，是因为那个函数吃的是 Article 对象，
# 而本模块手上是 json.loads 出来的 dict
REQUIRED_FIELDS = ("number", "number_cn", "text", "path")


def source_hash(text: str) -> str:
    """条级文本指纹，md5 十六进制（32 位，与附录 C 的 CHAR(32) 对齐）。

    取条级而非块级，是因为技术方案 4.5 要求"按 source_hash 重算该条子块向量"——
    粒度必须是条，否则一条法条改一个字会让它所有子块都算成不同来源。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_rows(articles: list[dict]) -> dict[str, list[tuple]]:
    """把条级记录转成三表各自的行元组。不连库，纯函数，便于单测。"""
    law_rows = [(LAW_ID, LAW_NAME)]
    version_rows = [(LAW_ID, LAW_VERSION, EFFECTIVE_DATE, STATUS)]
    article_rows = []
    for art in articles:
        article_rows.append((
            LAW_ID, LAW_VERSION,
            str(art["number"]),          # 附录 C 定的是 VARCHAR(16)，存阿拉伯数字字符串
            art["number_cn"],
            None,                        # paragraph_no：条级行恒为 NULL，见模块文档串
            None,                        # item_no：同上
            art["path"],
            art["text"],
            EFFECTIVE_DATE, STATUS,
            None, None,                  # replaced_by / replaces：民法典无替代关系
            source_hash(art["text"]),
            None,                        # team_id：公开法条为 NULL，类案才带团队
        ))
    return {"law": law_rows, "law_version": version_rows, "article": article_rows}


def check_required(articles: list[dict]) -> None:
    """校验必填字段非空，任一为空即抛错。

    必须在 DELETE 之前调用：本模块是先清表再插入，源文件有问题时
    如果不先拦住，结果就是"表被清空 + 程序正常退出"，而不是报错。
    """
    for art in articles:
        for name in REQUIRED_FIELDS:
            # 先判 is None 再判空串：JSON 里的 null 经 str() 会变成 "None"，
            # 非空且 truthy，只做空串判断的话它会一路过关，最后以字面量
            # "None" 写进 article_no 这种权威字段——比报错难查得多
            value = art.get(name)
            if value is None or not str(value).strip():
                raise ValueError(f"第 {art.get('number', '?')} 条字段 {name} 为空")


def _upsert(conn, table: str, columns: list[str], rows: list[tuple],
            key_columns: list[str]) -> int:
    """按主键 upsert，返回处理行数。

    单独写一个而不给 insert_many 加开关：两者语义不同——insert_many 只做插入，
    本函数保证重跑安全。混在一起会让调用方看不清自己拿到的是哪一种。
    """
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(f"{c} = VALUES({c})" for c in columns if c not in key_columns)
    sql = (f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
           f" ON DUPLICATE KEY UPDATE {updates}")
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def load_to_mysql(articles_path: pathlib.Path, conn=None) -> dict[str, int]:
    """建表并灌数据，返回各表写入行数。

    article 那一项取自插入后的**库侧回读**而非 len(rows)：本函数只按 law_id 清表，
    别处（或上一次跑崩）留下的行不在它的视野里，不读库就会把残留当成成功上报。

    article 表在附录 C 里只有普通索引 idx_law_art、没有唯一键，所以
    ON DUPLICATE KEY UPDATE 无从触发；改为"先按 law_id 清表再插"，
    这样重跑得到的是 1260 行而不是 2520 行。

    清表与插入分属**两个事务**（DELETE 后即 commit，插入由 insert_many / _upsert 各自 commit），
    故不是原子的：中断会留下一张空表而非重复行，重跑即可恢复。要合并成一个事务，
    得给 insert_many 与 _upsert 都加一个"不自动提交"的开关，为这一处不划算。
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()
    try:
        # 先读源文件再建表清表：读失败或语料为空时，数据库还一动没动
        with open(articles_path, encoding="utf-8") as f:
            articles = [json.loads(line) for line in f if line.strip()]
        if not articles:
            # 空输入不是"没什么可做"：按本模块先删后插的顺序，它意味着
            # 把整张表清空并正常退出——宁可在这里炸掉
            raise ValueError(f"语料为空，拒绝清表：{articles_path}")
        check_required(articles)
        apply_schema(conn, SCHEMA_PATH)
        rows = build_rows(articles)

        with conn.cursor() as cur:
            cur.execute("DELETE FROM article WHERE law_id = %s", (LAW_ID,))
        conn.commit()

        # law / law_version 有主键，走 upsert 才经得起重跑；
        # article 无唯一键（附录 C 只给了普通索引），靠上面那句 DELETE 保证幂等
        counts = {}
        counts["law"] = _upsert(
            conn, "law", ["law_id", "law_name"], rows["law"], ["law_id"])
        counts["law_version"] = _upsert(
            conn, "law_version",
            ["law_id", "law_version", "effective_date", "status"],
            rows["law_version"], ["law_id", "law_version"])
        counts["article"] = insert_many(conn, "article", ARTICLE_COLUMNS, rows["article"])
        # 回读库侧真实行数：counts["article"] 是 insert_many 自己算的 len(rows)，
        # 残留行对它不可见，那正是 2520 行事件的机制。与 Milvus 侧 insert_chunks
        # 返回服务端 upsert_count 的口径对齐
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article WHERE law_id = %s", (LAW_ID,))
            actual = cur.fetchone()[0]
        if actual != len(rows["article"]):
            raise RuntimeError(
                f"article 表实际行数 {actual} 与提交行数 {len(rows['article'])} 不符")
        counts["article"] = actual
        return counts
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[3]
    result = load_to_mysql(root / "data" / "parsed" / "law_articles.jsonl")
    print(f"MySQL 入库完成：{result}")
