# 精确通路必须真查 MySQL：它的价值就在"库里有这条能取出来"，用 mock 测等于没测。
# 但本文件同时有两类用例：查库的（逐个挂 requires_mysql）与纯断言的（常量、SQL 文本）。
# 不再用模块级 pytestmark：那会把纯断言用例一起吞掉，MySQL 离线时跑闸门得到"全绿"，
# 而 DEFAULT_LAW_ID 与 SQL 列绑定其实无人验证——对"测试是唯一防线"的项目是静默缺口。
from contextlib import closing

import pytest

from app.db.mysql import connect
from app.retrieval.article_lookup import ARTICLE_SQL, DEFAULT_LAW_ID, fetch_articles


def _mysql_available() -> bool:
    try:
        connect().close()
        return True
    except Exception:
        return False


# 只挂在真正查库的用例上；纯断言用例不挂，MySQL 离线时必须照跑
requires_mysql = pytest.mark.skipif(not _mysql_available(), reason="MySQL 不在线")


@requires_mysql
def test_fetches_existing_article_as_father_block():
    with closing(connect()) as conn:
        rows = fetch_articles(conn, [584])
    assert len(rows) == 1
    block = rows[0]
    # 块形状必须与 Milvus 召回后的父块同构，否则下游要写两套分支
    assert block["article_no"] == 584
    assert block["chunk_type"] == "father"
    assert block["source"] == "exact"
    assert block["parent_id"] is None
    # 效力必须随块带出且键名不漂移：cite_verify 第②关按块里的 status 判「现行有效」，
    # 被置 None、键被删或改名都会让校验关拿不到效力
    assert block["status"] == "现行有效"
    # 条原文里必然含「第五百八十四条」，这是条文本自身的开头
    assert "第五百八十四条" in block["text"]


@requires_mysql
def test_returns_empty_for_nonexistent_article():
    # 第9999条不存在——精确通路返回空，下游就不会置顶（设计文档 4.1 的边界）
    with closing(connect()) as conn:
        assert fetch_articles(conn, [9999]) == []


@requires_mysql
def test_preserves_input_order_and_skips_missing():
    with closing(connect()) as conn:
        rows = fetch_articles(conn, [577, 9999, 584])
    assert [r["article_no"] for r in rows] == [577, 584]


@requires_mysql
def test_preserves_descending_input_order():
    # 上面那条样本本身是升序，退化实现（按条号升序、或直接吐字典的值）照样能过——
    # 而 MySQL 走 idx_law_art 索引扫描恰好就是按条号升序返回的，这种退化在真库上
    # 永远不会自己暴露。降序样本才能把"保持输入顺序"这条约定真正钉住：
    # 用户问「第584条和第577条」，置顶时必须先 584 再 577（设计文档 4.1）
    with closing(connect()) as conn:
        rows = fetch_articles(conn, [584, 9999, 577])
    assert [r["article_no"] for r in rows] == [584, 577]


@requires_mysql
def test_empty_input_short_circuits():
    # 空输入不该发出一条 WHERE article_no IN () 的非法 SQL
    with closing(connect()) as conn:
        assert fetch_articles(conn, []) == []


@requires_mysql
def test_block_carries_path_and_cn_number():
    with closing(connect()) as conn:
        block = fetch_articles(conn, [1])[0]
    assert block["article_no_cn"] == "一"
    # status 与上面那条测的是同一约定，这里再钉一次：两条取块路径都退化才该放过
    assert block["status"] == "现行有效", "status 供 cite_verify 校验关判效力，取错列/丢键都要红"
    assert block["path"], "路径用于导航与引用展示，空路径说明 JOIN 取错了列"
    assert block["chunk_id"] is None, "精确块无块 id，置顶时按 article_no 去重"


def test_article_sql_reads_status_from_law_version():
    # 这条纯文本断言故意不挂 requires_mysql：不查库，离线也要跑（见文件头）。
    # 为什么不得不测实现文本而不是行为——设计文档 4.3 第②关的判据读
    # law_version.status，但现库数据让行为测试没有判别力：article.status 与
    # law_version.status 1260 行完全同值（实测不一致 0 行），块里 status 的值
    # 无论来自哪张表都一样，把 v.status 手改回 a.status 则全部查库用例照样绿。
    # 能拦「改错列」的只剩 SQL 文本本身，所以在这里钉死列绑定；将来两列分叉
    # （法规更新）补库上夹具时这条也不可删——它守的是绑定，不是取值
    assert "v.status" in ARTICLE_SQL
    assert "JOIN law_version" in ARTICLE_SQL


def test_law_id_defaults_to_minfadian():
    # 纯常量断言，同样不挂标记：离线时它就是 DEFAULT_LAW_ID 的唯一防线
    assert DEFAULT_LAW_ID == "minfadian"
