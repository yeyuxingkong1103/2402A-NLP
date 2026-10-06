# 冒烟集的查询集与报告渲染是纯逻辑，可离线断言。
# 真正跑检索的部分由 tools/smoke_retrieval.py 手动执行。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from smoke_retrieval import QUERIES, render_report


def test_queries_cover_the_confusable_pairs():
    # 冒烟集的价值就在易混条款：语义相近、条款号不同，
    # 纯 dense 容易混，纯 sparse 抓不住改写，只有双路融合才稳
    topics = {q["topic"] for q in QUERIES}
    assert "善意取得" in topics or any("善意" in t for t in topics)
    assert len(topics) >= 8, "覆盖太窄，证明不了链路"


def test_queries_have_required_keys():
    for q in QUERIES:
        assert set(q) >= {"topic", "query", "expect_articles"}
        assert q["query"].strip(), "空查询没有意义"
        # 期望条号允许为空列表（只验链路不验答案的题），但必须是列表
        assert isinstance(q["expect_articles"], list)


def test_query_count_is_in_agreed_range():
    # 设计文档说的是 15~20 条；少于 15 条覆盖不足，多于 20 条超出本期范围
    assert 15 <= len(QUERIES) <= 20


def test_render_report_states_it_is_not_a_benchmark():
    # 这是报告里最容易被人误读的地方，必须有显式声明
    text = render_report([])
    assert "冒烟" in text
    assert "不代表" in text or "不能作为" in text


def test_render_report_lists_each_result():
    results = [{"topic": "善意取得", "query": "买了别人无权处分的车",
                "hits": [{"article_no": 311, "chunk_type": "paragraph",
                          "text": "无处分权人将不动产或者动产转让给受让人的。"}],
                "expect_articles": [311], "dense_only_top1": 400}]
    text = render_report(results)
    assert "善意取得" in text
    # 用只在命中列表里才出现的串：上面已有一行"期望条号：[311]"，
    # 只断言 '311' 的话，命中列表整段坏掉也照样通过
    assert "1. ✅ 第 311 条（paragraph）" in text
    # 两个 top1 都要出现，人才看得出融合有没有改变排序
    assert "400" in text


def test_render_report_tolerates_missing_dense_only_field():
    # 缺诊断字段时不该抛错——渲染函数要容忍调用方不提供它
    results = [{"topic": "t", "query": "q", "hits": [], "expect_articles": []}]
    assert "t" in render_report(results)
