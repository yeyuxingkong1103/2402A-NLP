# -*- coding: utf-8 -*-
"""多路召回的纯逻辑测试（不依赖 Neo4j / MySQL / 向量库服务）。

本模块在守护什么
================

多路召回引入了三个**会静默出错**的点，都已实测踩过：

1. **中文数字字符类漏「千」**
   `第一千一百四十七条` 解析不出条号 —— 而民法典共 1260 条，
   意味着**所有 1000 条以上的法条图谱路静默失效**。
   实测：修复前「指名法条」问题命中率 30%，修复后 100%。

2. **文档名靠正则「猜」**
   初版用 `[一-龥]{2,20}(?:法|规定)` 抽候选文档名，贪婪匹配把整句
   「劳动法对试用期是怎么规定」当成文档名，导致 MySQL 路永远查不中。
   改为与库内真实文件名匹配。

3. **跨路 `source` 口径不一致**
   向量路是**文件名**（`X.txt`）、图谱路是**法律名**（`X`），
   直接比对会让「指名文档加分」静默失效。

这三条都不是「跑一遍就能发现」的问题 —— 它们只会让结果**悄悄变差**。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import graph_store as gs  # noqa: E402
from app.services.retrieval import _bare_name  # noqa: E402


# ---------------------------------------------------------------- 中文数字
def test_cn_to_int_basic():
    assert gs.cn_to_int("第十八条") == 18
    assert gs.cn_to_int("二十五") == 25
    assert gs.cn_to_int("577") == 577


def test_cn_to_int_handles_thousand():
    """「千」必须能解析 —— 民法典 1260 条里大量是四位数条号。"""
    assert gs.cn_to_int("第一千二百六十条") == 1260
    assert gs.cn_to_int("一千一百四十七") == 1147


def test_art_no_regex_includes_thousand():
    """回归：正则字符类必须含「千」。

    漏掉时 `_ART_NO.findall('第一千一百四十七条')` 返回空列表，
    图谱路对该条完全失效 —— 且**不报任何错**。
    """
    cases = {
        "第十八条": 18,
        "第五百七十七条": 577,
        "第一千零五十九条": 1059,
        "第一千一百四十七条": 1147,
        "第一千二百二十三条": 1223,
        "第一千二百六十条": 1260,
    }
    for text, expected in cases.items():
        found = gs._ART_NO.findall(text)
        assert found, f"{text} 解析不出条号（字符类是否漏了「千」？）"
        assert gs.cn_to_int(found[0]) == expected


def test_art_no_in_full_question():
    """真实问句里也要能抽出来（前面还有书名号内容）。"""
    q = "《中华人民共和国民法典》第一千一百四十七条规定了什么？"
    assert [gs.cn_to_int(x) for x in gs._ART_NO.findall(q)] == [1147]


# ---------------------------------------------------------------- 引用抽取
def test_self_reference_is_stripped():
    """chunk 开头的自我标识必须剥掉，否则每条都「引用自己」，图会被自环污染。"""
    t = "《中华人民共和国粮食安全保障法》第十八条规定，国家推进种业振兴。"
    assert gs.extract_citations("中华人民共和国粮食安全保障法", t) == []


def test_cross_law_citation_found():
    t = ("《中华人民共和国数据安全法》第三十一条规定，关键信息基础设施的运营者…"
         "适用《中华人民共和国网络安全法》的规定；")
    assert gs.extract_citations("中华人民共和国数据安全法", t) == ["中华人民共和国网络安全法"]


def test_multiple_citations_order_preserved_and_deduped():
    t = "《X法》第一条 依照《A法》和《B法》执行，另见《A法》第三条。"
    assert gs.extract_citations("X法", t) == ["A法", "B法"]


def test_self_prefix_without_comma():
    """自我标识后面不一定带逗号，也要能剥掉。"""
    t = "《中华人民共和国宪法》第一条 中华人民共和国是工人阶级领导的。"
    assert gs.extract_citations("中华人民共和国宪法", t) == []


# ---------------------------------------------------------------- 跨路名对齐
def test_bare_name_strips_extension_and_prefix():
    """跨路 source 口径对齐：文件名 vs 法律名必须归一后可比。"""
    assert _bare_name("中华人民共和国民法典.txt") == _bare_name("中华人民共和国民法典")
    assert _bare_name("中华人民共和国民法典.txt") == "民法典"
    # 医疗语料不需要剥「中国」：它的 chunk 只有 `source`（文件名）、
    # `law_name` 为空，图谱路也不适用（PDF 指南没有条文号），
    # 因此不存在「文件名 vs 法律名」的对齐需求。刻意不剥，避免过度归一。
    assert _bare_name("中国高血压防治指南2024修订版.pdf") == "中国高血压防治指南2024修订版"
    assert _bare_name(None) == ""
    assert _bare_name("") == ""


def test_bare_name_makes_doc_and_graph_match():
    """具体场景：文档路给文件名、图谱路给法律名，归一后应判定为同一份。"""
    doc_source = "中华人民共和国民法典.txt"     # 向量路/文档路
    graph_source = "中华人民共和国民法典"          # 图谱路
    assert _bare_name(doc_source) == _bare_name(graph_source)


# ---------------------------------------------------------------- uid
def test_uid_is_stable_and_distinct():
    assert gs.uid_of("民法典", "第一条") == gs.uid_of("民法典", "第一条")
    assert gs.uid_of("民法典", "第一条") != gs.uid_of("民法典", "第二条")
    assert gs.uid_of("民法典", "第一条") != gs.uid_of("刑法", "第一条")
