# 校验模块是 AC-1 的证据生成器，也是硬门槛的执行者。
# 缺号与条数不符必须有反向用例——只有正向用例等于没验。
import pytest
from app.ingest.structure import Article
from app.ingest.verify_ingest import (
    IngestFailed, check_continuity, check_total, check_fields, check_verbatim,
)


def test_continuity_returns_empty_when_complete():
    assert check_continuity(list(range(1, 463)), 1, 462) == []


def test_continuity_reports_missing_numbers():
    # 反向用例：缺 210 和 261 必须被列出，不能返回空
    nums = [n for n in range(1, 463) if n not in (210, 261)]
    assert check_continuity(nums, 1, 462) == [210, 261]


def test_check_total_raises_on_mismatch():
    # 1260 是硬门槛，差一条也要中断
    with pytest.raises(IngestFailed, match="1260"):
        check_total(list(range(1259)), expect=1260)


def test_check_total_passes_on_exact_count():
    check_total(list(range(1260)), expect=1260)


def test_check_total_counts_unique_numbers():
    # 重复条号不能虚增条数——去重后 1259 条必须被判失败
    nums = list(range(1259)) + [0]
    with pytest.raises(IngestFailed, match="1260"):
        check_total(nums, expect=1260)


def test_check_fields_reports_empty_required_field():
    art = Article(number=1, number_cn="一", text="第一条 …", path="第一编")
    art.text = ""                       # 制造字段非空失败
    problems = check_fields([art])
    assert any("text" in p for p in problems)


def test_check_fields_passes_on_complete_article():
    art = Article(number=1, number_cn="一", text="第一条 …", path="第一编")
    assert check_fields([art]) == []


def test_verbatim_matches_ignoring_whitespace():
    # MinerU 把 PDF 硬换行合并成整行，PDF 原文那些位置是换行符，
    # 直接子串匹配会全数失配——两侧都去空白后再比
    art = Article(number=1, number_cn="一",
                  text="第一条为了保护民事主体的合法权益，调整民事关系。",
                  path="第一编")
    art.paragraphs = [art.text]
    pdf_text = "第一条为了保护民事主体的\n合法权益，调整民事关系。"
    assert check_verbatim([art], pdf_text) == []


def test_verbatim_reports_unmatched_article():
    art = Article(number=2, number_cn="二", text="第二条不存在的条文。", path="第一编")
    art.paragraphs = [art.text]
    assert check_verbatim([art], "第一条其他内容。") == [2]


def test_verbatim_tolerates_page_numbers_in_pdf():
    # PDF 文本层会把页码混进正文。实测：第 28 条在 PDF 里是
    # 「…其他近亲属；12（四）其他愿意担任…」，第 38 条是「…真实意愿的15前提下…」。
    # MinerU 已正确去掉页码，若基线不清洗，这 36 条会被误报成"未命中"
    art = Article(number=28, number_cn="二十八",
                  text="第二十八条…（三）其他近亲属；（四）其他愿意担任监护人的个人或者组织。",
                  path="第一编")
    art.paragraphs = [art.text]
    pdf_text = "第二十八条…（三）其他近亲属；12（四）其他愿意担任监护人的个人或者组织。"
    assert check_verbatim([art], pdf_text) == []


def test_verbatim_keeps_legitimate_dates():
    # 第 1260 条（附则）含「2021年1月1日」，是法条正文的一部分，不能被当页码剔掉。
    # 全库只有这一条含半角数字，删错了会直接漏掉对它的校验
    art = Article(number=1260, number_cn="一千二百六十",
                  text="第一千二百六十条 本法自2021年1月1日起施行。", path="附则")
    art.paragraphs = [art.text]
    assert check_verbatim([art], "第一千二百六十条 本法自2021年1月1日起施行。") == []
