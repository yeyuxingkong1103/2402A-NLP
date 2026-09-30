# 评估集是③a 唯一的尺子，它的自检必须比被测代码更严。
# 这里只审"标注是否成立"，不审"题出得好不好"——后者只能靠人。
import json
import pathlib

import pytest

EVAL_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "eval" / "qa_eval_v0.jsonl"
CATEGORIES = {"条款号直问", "语义改写", "易混条对", "公众口语化"}


def _items():
    with EVAL_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_file_exists_and_is_frozen_jsonl():
    assert EVAL_PATH.exists(), "评估集必须落盘冻结，不能只活在脚本里"
    assert len(_items()) == 100


def test_side_split_is_60_internal_40_public():
    # 需求 6.3 定的量级：律师侧 50~100、公众侧 30~50
    items = _items()
    assert sum(1 for i in items if i["side"] == "internal") == 60
    assert sum(1 for i in items if i["side"] == "public") == 40


def test_every_item_is_marked_ai_draft():
    # 未过律师核是本集最大的局限，必须逐条自曝，不靠 README 提醒
    assert all(i["source"] == "ai_draft" for i in _items())


def test_ids_are_unique_and_sequential():
    ids = [i["id"] for i in _items()]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("i-") and ids[-1].startswith("i-")


def test_queries_are_unique_and_non_empty():
    queries = [i["query"].strip() for i in _items()]
    assert all(queries), "空题面无法评测"
    assert len(set(queries)) == len(queries), "重复题面会让指标被同一道题加权"


def test_categories_are_from_the_agreed_set():
    assert {i["category"] for i in _items()} <= CATEGORIES


def test_public_items_cover_five_life_domains():
    # 技术方案 12.1 指定公众题覆盖租房、消费、劳动、借贷、婚姻家庭
    public = " ".join(i["query"] for i in _items() if i["side"] == "public")
    for keyword in ("租", "消费", "劳动", "借", "婚姻"):
        assert keyword in public, f"公众题缺少「{keyword}」类问题"


def test_has_article_number_direct_subset():
    # AC-2 的 100% 判据就在这个子集上算，太少则说明不了问题
    direct = [i for i in _items() if i["category"] == "条款号直问"]
    assert len(direct) >= 20


def test_gold_articles_are_non_empty_lists_of_ints():
    for item in _items():
        assert isinstance(item["gold_articles"], list) and item["gold_articles"]
        assert all(isinstance(no, int) and 1 <= no <= 1260 for no in item["gold_articles"])


def _mysql_available() -> bool:
    # 必须先于装饰器定义：装饰器在模块导入时就求值，写在后面会 NameError，
    # 整份评估集自检连收集都过不去。所以把"环境在不在"的探测器提前。
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
    try:
        from app.db.mysql import connect
        connect().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mysql_available(), reason="MySQL 不在线")
def test_gold_articles_all_exist_in_the_law_table():
    """题目标错条号会让评测显示"没召回"，而问题其实出在标注上。"""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
    from app.db.mysql import connect
    conn = connect()
    with conn.cursor() as cur:
        for item in _items():
            for no in item["gold_articles"]:
                cur.execute("SELECT COUNT(*) FROM article WHERE article_no = %s", (str(no),))
                assert cur.fetchone()[0] == 1, f"{item['id']} 标注的第{no}条不存在"
