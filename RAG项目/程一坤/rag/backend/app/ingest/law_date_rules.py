"""法规日期识别规则与文本预处理（从 law_metadata_extractor.py 抽出的独立块）。

为什么独立：这一块是"页面文字 → 日期"的纯解析规则，与文书类型/发布机关的
成套判定无关；规则本身随真实页面排版演进频繁调整（批次 10 的空白容忍、
"发布时间："词表补充都是这里改），单独成文件后改动面清晰、可被单测直击。

铁律：只认页面实际文字，抽不到就返回 None（交由人工补录），绝不编造。
"""

import re
from datetime import date


def _to_plain_text(html: str) -> str:
    """把 HTML 压成单行纯文本，便于用正则找日期；保留原有顺序。"""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML 实体中最常见的空格与连接符，替换掉避免切断"自2008年1月1日起施行"这类短语
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text)


def _chinese_number_to_int(value: str) -> int | None:
    """把"二〇一二""十""三十一"这类中文数字转成整数；无法解析返回 None。"""
    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    # 逐位形式：二〇一二
    if all(character in digits for character in value):
        return int("".join(str(digits[character]) for character in value))
    # 十位形式：十 / 十二 / 二十 / 三十一
    if "十" in value:
        head, _, tail = value.partition("十")
        tens = digits.get(head, 1) if head else 1
        ones = digits.get(tail, 0) if tail else 0
        return tens * 10 + ones
    return None


def _parse_ymd_date(year: str, month: str, day: str) -> date | None:
    """把三段数字拼成日期；非法日期返回 None（例如 2 月 30 日）。"""
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def extract_effective_date(plain_text: str, promulgation_date: date | None = None) -> date | None:
    """抽取生效日期。

    依据（按优先级排序，全部只认页面实际文字，抽不到就返回 None）：
    1. 修订语境："…修订，自YYYY年M月D日起施行" —— 修订版施行日期最具体，优先；
    2. 正文最后一条通常写明"自YYYY年M月D日起施行"；
    3. "自公布之日起施行"（页面必须真有这句话，且公布日期已抽到）
       → effective_date = promulgation_date。

    兼容性（批次 10 实测）：真实页面普遍存在"本条例自 2004 年 1 月 1 日起施行"
    这种数字与年月日之间带空白/全角空格的排版，旧正则不容空格导致 6 部法规
    生效日期漏抽 —— 现在各段之间一律容忍空白。
    """
    # 各段之间容忍任意空白（含 &nbsp; 转出的空格、全角空格已被 _to_plain_text 保留）
    d = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    patterns = [
        # 1) 修订语境优先："…修订" 之后 40 字内紧跟的施行日期
        rf"修订[^。；]{{0,40}}?自\s*{d}\s*起(?:施行|实施|生效)",
        # 2) 通用施行条款
        rf"自\s*{d}\s*起(?:施行|实施|生效)",
        rf"{d}\s*起(?:施行|实施|生效)",
    ]
    for pattern in patterns:
        match = re.search(pattern, plain_text)
        if match:
            groups = match.groups()
            parsed = _parse_ymd_date(groups[-3], groups[-2], groups[-1])
            if parsed:
                return parsed

    # 3) "自公布之日起施行"：页面真有这句话且公布日期已知 → 生效日期 = 公布日期；
    #    公布日期抽不到就不猜（留空交人工补录），绝不把公布日期硬当生效日期
    if promulgation_date is not None and re.search(r"自\s*公\s*布\s*之\s*日\s*起\s*(?:施行|实施|生效)", plain_text):
        return promulgation_date
    return None


def extract_promulgation_date(plain_text: str) -> date | None:
    """抽取公布（或通过、修订）日期。

    依据（按可靠度排序）：
    1. 页面结构化标注："公布日期：2012年12月28日"（含"发布时间：2020-12-30"，
       最高法新闻页用的是"发布时间"，批次 10 补进词表）
    2. 会议通过措辞："（2007年12月29日第十届全国人民代表大会常务委员会第三十一次会议通过）"
    3. 正文里的"XXXX年X月X日……公布/修订/修正"
    中文数字日期（二〇一二年十二月二十八日）同样支持。
    各段之间容忍空白（真实页面存在"公布日期： 2012 年 12 月 28 日"式排版）。
    """
    # 各段之间容忍任意空白
    d = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"

    # 1) 结构化标注（含"发布时间：2020-12-30"横杠格式与"公布日期：2012年12月28日"）
    labelled = re.search(
        rf"(?:公布日期|通过日期|成文日期|发布日期|发布时间)\s*[:：]\s*(?:{d}|(\d{{4}})-(\d{{1,2}})-(\d{{1,2}}))",
        plain_text,
    )
    if labelled:
        groups = [g for g in labelled.groups() if g is not None]
        if len(groups) == 3:
            parsed = _parse_ymd_date(*groups)
            if parsed:
                return parsed

    # 2) 正文中的"X年X月X日……公布/修订/修正"（公布日语义优先于会议通过日：
    #    国务院令"通过"在前、"公布"在后，公布日期才是法规的公布时间）
    inline = re.search(rf"{d}[^。；]{{0,20}}?(?:公布|修订|修正|发布)", plain_text)
    if inline:
        parsed = _parse_ymd_date(*inline.groups())
        if parsed:
            return parsed

    # 3) 会议通过措辞（括号内可再带"自…施行"等说明，不允许跨过闭括号）
    passed = re.search(rf"[（(]\s*{d}[^）)]{{0,60}}?会议通过", plain_text)
    if passed:
        parsed = _parse_ymd_date(*passed.groups())
        if parsed:
            return parsed

    # 4) 中文数字日期
    chinese = re.search(r"([一二三四五六七八九〇零十]{4})年([一二三四五六七八九十]{1,3})月([一二三四五六七八九十]{1,3})日", plain_text)
    if chinese:
        year = _chinese_number_to_int(chinese.group(1))
        month = _chinese_number_to_int(chinese.group(2))
        day = _chinese_number_to_int(chinese.group(3))
        if year and month and day:
            return _parse_ymd_date(str(year), str(month), str(day))
    return None
