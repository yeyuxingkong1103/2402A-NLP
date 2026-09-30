# 单册还原：条要带正确路径，款要按空行拆开，头部元信息不能污染路径
import pytest
from app.ingest.structure import restore_articles

SAMPLE = """# 中华人民共和国民法典（中册）

> 本册范围：第三编 合同、第四编 人格权
---

## 第三编 合同

第一分编 通 则

### 第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。

第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。

婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定。

第四百六十五条 依法成立的合同，受法律保护。
"""


@pytest.fixture
def sample_md(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_parses_all_articles(sample_md):
    arts = restore_articles(sample_md)
    assert [a.number for a in arts] == [463, 464, 465]


def test_path_tracks_hierarchy(sample_md):
    # 路径是 FR-1.1/FR-7.3 要的「编 > 分编 > 章」，必须逐级拼出来
    arts = restore_articles(sample_md)
    assert arts[0].path == "第三编 合同 > 第一分编 通 则 > 第一章 一般规定"


def test_article_text_includes_number(sample_md):
    # 首段必须保留整行含条号——逐字命中要拿 text 去 PDF 里比，PDF 原文是含条号的
    arts = restore_articles(sample_md)
    assert arts[0].text.startswith("第四百六十三条 ")
    assert arts[0].number_cn == "四百六十三"


def test_paragraphs_split_by_blank_line(sample_md):
    # 第 464 条有两款，必须拆成 2 个 paragraph，不能粘成一坨
    arts = restore_articles(sample_md)
    art464 = next(a for a in arts if a.number == 464)
    assert len(art464.paragraphs) == 2
    assert art464.paragraphs[1].startswith("婚姻、收养")


def test_blockquote_header_does_not_pollute_path(sample_md):
    # 头部元信息里的「第三编 合同」不是标题，不能变成任何条的路径来源
    arts = restore_articles(sample_md)
    assert all("本册范围" not in a.path for a in arts)


def test_resets_cursor_when_entering_new_bian(tmp_path):
    # 进入新编必须清掉上一编的分编/章/节，否则路径会串编
    md = tmp_path / "two_bian.md"
    md.write_text(
        "## 第三编 合同\n\n第一分编 通 则\n\n### 第一章 一般规定\n\n"
        "第四百六十三条 甲。\n\n## 第四编 人格权\n\n### 第一章 一般规定\n\n"
        "第九百八十九条 乙。\n",
        encoding="utf-8")
    arts = restore_articles(md)
    assert arts[0].path == "第三编 合同 > 第一分编 通 则 > 第一章 一般规定"
    # 第四编没有分编，路径里不应残留第三编的「第一分编 通 则」
    assert arts[1].path == "第四编 人格权 > 第一章 一般规定"


def test_skips_region_before_first_article(sample_md):
    # 第一条之前的说明性文字不应产生任何条
    arts = restore_articles(sample_md)
    assert min(a.number for a in arts) == 463
