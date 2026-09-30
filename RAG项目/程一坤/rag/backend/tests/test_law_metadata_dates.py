r"""批次 10 任务 1：法规生效日期抽取增强的回归测试。

背景（实测证据，2026-09-20 扫库）：
- 6 部法规 effective_date 为空，根因不是"页面没写"，而是两类缺陷：
  1. 页面文本数字与年月日之间有空格（"自 2004 年 1 月 1 日起施行"），
     旧正则 r"自\s*(\d{4})年" 不容空格 → 匹配失败；
  2. "自公布之日起施行"类条款（实施条例/特别规定）没有映射到公布日期；
  3. 新闻页只有"发布时间：YYYY-MM-DD"结构化标注，旧词表不含"发布时间"。
- 修订语境（规则③）：修订说明后紧跟施行日期的，应优先取修订后的施行日期。

边界：只从页面实际内容抽取，抽不到返回 None，绝不编造。
"""

from datetime import date

from app.ingest.law_date_rules import (
    extract_effective_date,
    extract_promulgation_date,
)
from app.ingest.law_metadata_extractor import extract_legal_metadata


# ---------------------------------------------------------------------------
# ① 日期各段之间容忍空白（真实页面："本条例自 2004 年 1 月 1 日起施行"）
# ---------------------------------------------------------------------------
def test_effective_date_tolerates_whitespace_around_digits() -> None:
    text = "第六十七条 本条例自 2004 年 1 月 1 日起施行。本条例施行前已受到事故伤害的职工"
    assert extract_effective_date(text) == date(2004, 1, 1)


def test_effective_date_tolerates_fullwidth_and_nbsp() -> None:
    text = "本法自2008\u00a0年1\u00a0月1\u00a0日起施行。"
    assert extract_effective_date(text) == date(2008, 1, 1)


def test_promulgation_date_tolerates_whitespace_in_parenthesis() -> None:
    # 真实页面："（ 2020 年 12 月 25 日最高人民法院审判委员会…会议通过）"（NFKC 后数字半角，字间有空格）
    text = "（2020 年 12 月 25 日由最高人民法院审判委员会第 1825 次会议通过，自 2021 年 1 月 1 日起施行）"
    assert extract_promulgation_date(text) == date(2020, 12, 25)


# ---------------------------------------------------------------------------
# ② "自公布之日起施行" → effective_date = promulgation_date
#    （页面必须真有这句话，且公布日期已知；两者缺一不猜）
# ---------------------------------------------------------------------------
def test_effective_date_from_promulgation_day_phrase() -> None:
    text = "2008年9月18日中华人民共和国国务院令第535号公布自公布之日起施行) 第一章 总则 第三十八条 本条例自公布之日起施行。"
    effective = extract_effective_date(text, promulgation_date=date(2008, 9, 18))
    assert effective == date(2008, 9, 18)


def test_effective_date_promulgation_day_phrase_without_promulgation_returns_none() -> None:
    # 页面有"自公布之日起施行"但抽不到公布日期 → 不能编造，返回 None
    text = "自公布之日起施行) 第一章 总则"
    assert extract_effective_date(text, promulgation_date=None) is None


def test_effective_date_explicit_clause_wins_over_promulgation_phrase() -> None:
    # 页面同时有"自X年X月X日起施行"和"自公布之日起施行"时，明示日期更具体，优先
    text = "自公布之日起施行) 附则 第一百条 本条例自 2010 年 1 月 1 日起施行。"
    effective = extract_effective_date(text, promulgation_date=date(2008, 9, 18))
    assert effective == date(2010, 1, 1)


def test_effective_date_promulgation_phrase_requires_actual_phrase() -> None:
    # 页面没有"自公布之日起施行"字样 → 不得把公布日期当生效日期
    text = "2008年9月18日中华人民共和国国务院令第535号公布) 第一章 总则"
    assert extract_effective_date(text, promulgation_date=date(2008, 9, 18)) is None


# ---------------------------------------------------------------------------
# ③ 修订语境：修订说明后紧跟的施行日期优先（修订版生效日期）
# ---------------------------------------------------------------------------
def test_effective_date_revision_context_takes_priority() -> None:
    text = (
        "（2003年4月27日中华人民共和国国务院令第375号公布 "
        "根据2010年12月20日《国务院关于修改〈工伤保险条例〉的决定》修订，"
        "自 2011 年 1 月 1 日起施行）第一条 为了保障因工作遭受事故伤害的职工"
    )
    assert extract_effective_date(text) == date(2011, 1, 1)


# ---------------------------------------------------------------------------
# 结构化标注扩展："发布时间：YYYY-MM-DD"（最高法新闻页的真实格式）
# ---------------------------------------------------------------------------
def test_promulgation_date_from_publish_time_label() -> None:
    text = "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）来源：最高人民法院 发布时间：2020-12-30 21:29:28"
    assert extract_promulgation_date(text) == date(2020, 12, 30)


# ---------------------------------------------------------------------------
# 端到端：合成 HTML 走 extract_legal_metadata 全流程
# ---------------------------------------------------------------------------
def test_extract_legal_metadata_promulgation_day_effective(tmp_path) -> None:
    html = (
        "<html><body>"
        "<h1>中华人民共和国劳动合同法实施条例</h1>"
        "<p>（2008年9月3日国务院第25次常务会议通过 "
        "2008年9月18日中华人民共和国国务院令第535号公布 "
        "自公布之日起施行）</p>"
        "<p>第三十八条 本条例自公布之日起施行。</p>"
        "</body></html>"
    )
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    metadata = extract_legal_metadata("中华人民共和国劳动合同法实施条例", "http://x", page)
    assert metadata.promulgation_date == date(2008, 9, 18)
    assert metadata.effective_date == date(2008, 9, 18)


def test_extract_legal_metadata_whitespace_effective_date(tmp_path) -> None:
    html = (
        "<html><body>"
        "<h1>工伤保险条例</h1>"
        "<p>（2003 年 4 月 27 日中华人民共和国国务院令第375号公布）</p>"
        "<p>第六十七条 本条例自 2004 年 1 月 1 日起施行。</p>"
        "</body></html>"
    )
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    metadata = extract_legal_metadata("工伤保险条例", "http://x", page)
    assert metadata.effective_date == date(2004, 1, 1)
