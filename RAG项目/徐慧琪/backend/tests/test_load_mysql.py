# ETL 单测。用真实语料做夹具——条号、路径、正文长度都取实际值，
# 造数据测不出"民法典里有没有 255 字以上的路径"这类真问题。
import json
import pathlib

import pytest

from app.ingest.load_mysql import (
    ARTICLE_COLUMNS, EFFECTIVE_DATE, LAW_ID, LAW_NAME, LAW_VERSION, STATUS,
    build_rows, check_required, load_to_mysql, source_hash,
)

ARTICLES = pathlib.Path(__file__).resolve().parents[2] / "data" / "parsed" / "law_articles.jsonl"


@pytest.fixture
def sample_articles():
    # 取真实语料的前 3 条，而不是手写 dict
    with open(ARTICLES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:3]


def test_constants_match_confirmed_values():
    # 这几个取值是 2026-09-22 与用户确认过的，改动必须有人拍板
    assert LAW_ID == "minfadian"
    assert LAW_NAME == "中华人民共和国民法典"
    assert LAW_VERSION == "v1"
    assert EFFECTIVE_DATE == "2021-01-01"
    assert STATUS == "现行有效"


def test_source_hash_is_article_level_md5():
    # 技术方案 4.5 要求"按 source_hash 重算该条子块向量"，
    # 故指纹取条级：同条所有块共享同一个值，长度是 md5 的 32 位十六进制
    h = source_hash("第一条 测试")
    assert len(h) == 32
    assert h == source_hash("第一条 测试")
    assert h != source_hash("第二条 测试")


def test_check_required_rejects_empty_field(sample_articles):
    # 字段为空要中断而非静默跳过（Global Constraint）。取真实首条改一个字段，
    # 而不是手搓 dict——手搓的少了真实字段，恰好会绕过"字段名写错"这类错误
    bad = {**sample_articles[0], "text": ""}
    with pytest.raises(ValueError, match="字段 text 为空"):
        check_required([bad])


def test_check_required_rejects_json_null(sample_articles):
    # JSON 里的 null 经 str() 会变成 "None"：非空且 truthy，只判空串的写法会放它过关，
    # 随后 build_rows 把 str(None) 当条号写进 article_no 这个权威字段。
    # 这条测试与上一条的差别就在原值本身是不是 None，两者必须都能拦住
    bad = {**sample_articles[0], "number": None}
    with pytest.raises(ValueError, match="字段 number 为空"):
        check_required([bad])


def test_build_rows_shapes(sample_articles):
    rows = build_rows(sample_articles)
    assert set(rows) == {"law", "law_version", "article"}
    assert len(rows["law"]) == 1
    assert len(rows["law_version"]) == 1
    assert len(rows["article"]) == 3


def test_build_rows_article_columns_in_ddl_order(sample_articles):
    # 列顺序必须与 schema.sql 的列一致，错位会静默写进错误的列
    law_id, law_version, art_no, art_no_cn, para_no, item_no, path, text, \
        eff, status, replaced_by, replaces, sha, team_id = build_rows(sample_articles)["article"][0]
    assert law_id == LAW_ID
    assert law_version == LAW_VERSION
    assert art_no == "1"
    assert art_no_cn == "一"
    assert para_no is None
    assert item_no is None
    assert path.startswith("第一编")
    assert text.startswith("第一条")
    assert str(eff) == EFFECTIVE_DATE
    assert status == STATUS
    assert replaced_by is None and replaces is None
    assert len(sha) == 32
    assert team_id is None


def test_article_rows_cover_full_corpus_without_gaps():
    # 1260 条零缺号是入库链路的产出保证，ETL 不能把它弄丢
    with open(ARTICLES, encoding="utf-8") as f:
        arts = [json.loads(line) for line in f if line.strip()]
    rows = build_rows(arts)
    numbers = [int(r[2]) for r in rows["article"]]
    assert len(numbers) == 1260
    assert sorted(numbers) == list(range(1, 1261))


@pytest.mark.skipif(not pathlib.Path(ARTICLES).exists(), reason="语料不存在")
def test_load_to_mysql_is_idempotent():
    # 重跑必须仍是 1260 行而不是 2520 行——article 表没有唯一键，
    # 靠"先按 law_id 清表再插"保证幂等，这个测试就是它的守卫
    from app.db.mysql import SCHEMA_PATH, apply_schema, connect

    try:
        conn = connect()
    except Exception:
        pytest.skip("MySQL 未在线")
    try:
        apply_schema(conn, SCHEMA_PATH)
        first = load_to_mysql(ARTICLES, conn)
        second = load_to_mysql(ARTICLES, conn)
        assert first["article"] == 1260
        assert second["article"] == 1260
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article WHERE law_id = %s", (LAW_ID,))
            assert cur.fetchone()[0] == 1260
    finally:
        conn.close()


@pytest.mark.skipif(not pathlib.Path(ARTICLES).exists(), reason="语料不存在")
def test_empty_corpus_is_rejected_before_the_table_is_touched(tmp_path):
    # 空语料不是"没什么可做"：本模块先按 law_id 清表再插，放过去就是"清空权威表 +
    # 正常退出"。只断言抛 ValueError 不够——"先删完再抛"同样能让它绿，
    # 所以必须比对**库侧行数没动**，那才证明拦截发生在 DELETE 之前
    from app.db.mysql import connect

    try:
        conn = connect()
    except Exception:
        pytest.skip("MySQL 未在线")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article WHERE law_id = %s", (LAW_ID,))
            before = cur.fetchone()[0]
        # 基线先钉住：库侧本来就不是 1260 时，"行数没变"会是个没有意义的断言
        assert before == 1260, f"跑测试前 article 已是 {before} 行，基线不成立"
        with pytest.raises(ValueError, match="语料为空"):
            load_to_mysql(empty, conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article WHERE law_id = %s", (LAW_ID,))
            assert cur.fetchone()[0] == before
    finally:
        conn.close()


def test_article_columns_match_ddl_order():
    # ARTICLE_COLUMNS 才是真正传给 insert_many 的列清单，它自己必须被钉住：
    # 单独调换这个列表的顺序，上面那条元组解包测试照样绿，但每一行都会写错列
    assert ARTICLE_COLUMNS == [
        "law_id", "law_version", "article_no", "article_no_cn", "paragraph_no", "item_no",
        "path", "text", "effective_date", "status", "replaced_by", "replaces",
        "source_hash", "team_id",
    ]
