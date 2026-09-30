"""入库一致性与防错校验（技术方案 3.4，对应 AC-1）。

存在的理由：这是入库的硬门槛。校验不通过就不许入库——一旦错切入了库，
后面检索、生成、引用回查全部建立在错数据上，且向量库里已经落了数据，返工代价极高。
"""
from __future__ import annotations

import re

from app.ingest.structure import Article

# 技术方案 3.4 定的条数合计基准：上册 462 + 中册 577 + 下册 221
EXPECTED_TOTAL = 1260


class IngestFailed(Exception):
    """校验不通过。调用方不得捕获后继续——技术方案 3.4 要求中断入库。"""


# 逐字命中比对前要去掉的空白（含换行、全角空格）
_WS = re.compile(r"\s+", re.UNICODE)

# PDF 文本层里混入的页码：半角数字串。
# 判据来自实测：全库 1260 条里只有第 1260 条含半角数字（「2021年1月1日」），
# 其数字后必跟 年/月/日；其余半角数字串都是页码——如第 28 条的「12」混在
# 「其他近亲属；12（四）」中间、第 38 条的「15」混在「真实意愿的15前提下」中间。
PAGE_NUMBER = re.compile(r"(?<![0-9])[0-9]{1,3}(?![0-9年月日])")


def strip_page_numbers(text: str) -> str:
    """去掉 PDF 文本层里混入的页码。

    这是**比对基线**的清洗，不动入库正文。技术方案 3.2 ⑥ 的"去页眉页脚/页码"
    同样适用于校验基线：MinerU 已经把页码正确去掉了，基线若带着页码，
    会把它的正确行为误判成"未命中"（实测上册 36 条就是这么被误报的）。
    """
    return PAGE_NUMBER.sub("", text)


def _squash(text: str) -> str:
    """去掉全部空白字符。

    MinerU 已把 PDF 的硬换行合并成整行，而 PDF 原文那些位置是换行符，
    因此逐字命中必须两侧都归一化后再比，否则会全数失配。
    这不是放宽标准——它让"逐字"比较的是字符本身而非排版。
    """
    return _WS.sub("", text)


def check_continuity(numbers: list[int], lo: int, hi: int) -> list[int]:
    """返回 [lo, hi] 区间内缺失的条号。空列表表示连续。"""
    present = set(numbers)
    return sorted(set(range(lo, hi + 1)) - present)


def check_total(numbers: list[int], expect: int = EXPECTED_TOTAL) -> None:
    """条数合计必须等于 expect，否则中断入库。重复条号按一条计。"""
    actual = len(set(numbers))
    if actual != expect:
        raise IngestFailed(f"条数合计不符：期望 {expect}，实际 {actual}")


def check_fields(articles: list[Article]) -> list[str]:
    """检查必填字段非空，返回问题描述列表。空列表表示通过。"""
    problems: list[str] = []
    for art in articles:
        for name in ("number_cn", "text", "path"):
            if not str(getattr(art, name)).strip():
                problems.append(f"第 {art.number} 条字段 {name} 为空")
    return problems


def check_verbatim(articles: list[Article], pdf_text: str) -> list[int]:
    """逐字命中校验：返回未能在 PDF 原文中命中的条号列表。

    注意本期只对上册执行——中册/下册的 PDF 在回收站未恢复（见设计文档第九节）。
    """
    # 先去掉基线里的页码，再统一去空白——两步都是比对口径，不改被校验的条文
    haystack = _squash(strip_page_numbers(pdf_text))
    unmatched: list[int] = []
    for art in articles:
        if _squash(art.text) not in haystack:
            unmatched.append(art.number)
    return unmatched
