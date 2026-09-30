"""引用回查校验（技术方案 6.2 / 设计文档 4.3，本系统的安全阀）。

为什么必须独立：校验用的是**字符串比对与查库**，不是"再问一次模型"。
模型自己说自己对不算数——把校验交给模型自评，安全阀就等于没有。

四关（缺一不可）：
  ① 存在性  该条在法条主表里存在
  ② 效力    该条现行有效
  ③ 款/项   引用的款号、项号真实存在（查该条全量子块，不是本轮召回集）
  ④ 原文    quote 是逐字摘录，且有最小长度约束

返回的是**失败原因列表**而不是布尔值：重生成时要把全部原因反馈给模型，
只报"失败了"会让它反复撞同一堵墙。故四关逐一走完、不遇错即返，
唯一提前返回的情形是条号根本解析不出（后三关无从谈起）。
"""
from __future__ import annotations

import re

from app.generation.schema import Answer
from app.retrieval.query_parse import extract_article_nos

# quote 的最短长度。短片段（"当事人""可以"）能骗过子串匹配，
# 十个字是"至少是一个有信息量的半句"的经验下限
MIN_QUOTE_LEN = 10

# 空白字符（含全角空格、制表、换行）。归一化只剔空白、不动标点：
# 抹平标点会把"逗号写成句号"的改写一起放过
_WHITESPACE_RE = re.compile(r"\s|　")

# 项号的括号前缀：「（一）」或「(一)」。Milvus 里存的是光杆中文数字
_ITEM_PAREN_RE = re.compile(r"^[（(](.+?)[）)]$")

# 款/项索引查不到该条时的缺省值。空集合表示"没有款级/项级子块"，
# 与子块缺失时的真实形状一致；第③关的单款特例据此判定
_EMPTY_INDEX: dict = {"paragraphs": set(), "items": set()}


def normalize_quote(text: str) -> str:
    """归一化：剔除全部空白字符。用于 quote 与原文的比对基准。"""
    return _WHITESPACE_RE.sub("", text or "")


def _parse_article_no(raw: str) -> int | None:
    """把引用里的条号还原成整数；解析不出返回 None。

    先走问句抽号器（认「第X条」，中文数字与全角数字都归一成整数）；
    抽不出时兜底认光杆数字——模型常把条号写成「584」，因为写法不统一
    就判"无法解析"属于假拒绝（假拒绝比漏放更伤）。isdigit()/int() 原生
    认全角数字，与 query_parse 的口径一致，故不必再转一次全角。
    多个条号（「第584条和第577条」）不认：一条引用只该对应一条法条。
    """
    nos = extract_article_nos(raw or "")
    if len(nos) == 1:
        return nos[0]
    bare = (raw or "").strip()
    return int(bare) if bare.isdigit() and int(bare) > 0 else None


def _normalize_item(item: str | None) -> str | None:
    """把引用里的项号剥成光杆形式，与 Milvus 的存储口径对齐。"""
    if not item:
        return None
    match = _ITEM_PAREN_RE.match(item.strip())
    return (match.group(1) if match else item).strip()


def _paragraph_exists(paragraph: int, paragraphs: set[int]) -> bool:
    """款号是否真实。

    单款条的特例：该条在库里没有款级子块时，实务上整条即一款，
    故「第一款」合法、「第二款」及以后仍判失败。
    """
    return paragraph in paragraphs or (not paragraphs and paragraph == 1)


def _check_existence(citation, no: int, article, paragraphs: set[int]) -> list[str]:
    """①②关，外加③关的款号部分。条不存在时只报①关：②③关没有可
    比对的库内事实，硬报只会给重生成叠噪音。"""
    if article is None:
        return [f"引用失败：第{no}条在法条主表中不存在"]
    failures = []
    if article["status"] != "现行有效":
        failures.append(f"引用失败：第{no}条效力为「{article['status']}」")
    if citation.paragraph is not None and not _paragraph_exists(citation.paragraph,
                                                               paragraphs):
        failures.append(f"引用失败：第{no}条无第{citation.paragraph}款")
    return failures


def _check_item(citation, no: int, items: set[str]) -> list[str]:
    """③关的项号部分。条级引用（无项）视为合法，不查。"""
    item = _normalize_item(citation.item)
    if item is None or item in items:
        return []
    return [f"引用失败：第{no}条无第{item}项"]


def _check_quote(citation, no: int, article) -> list[str]:
    """④关：逐字摘录且有最小长度。

    长度检查不依赖库内原文，故条不存在时照样要报；子串比对则没有基准，
    跳过即可（①关那条原因已代表该问题，不重复报）。
    """
    quote = normalize_quote(citation.quote)
    if len(quote) < MIN_QUOTE_LEN:
        return [f"引用失败：quote 过短（{len(quote)} 字 < {MIN_QUOTE_LEN}）"]
    if article is not None and quote not in normalize_quote(article["text"]):
        return [f"引用失败：quote 不是第{no}条的原文摘录"]
    return []


def _verify_one(citation, *, article_of, para_index_of) -> list[str]:
    """校验单条引用，返回其全部失败原因。

    条不存在时不再查款/项（省一次 Milvus 往返），但④关仍走完——
    要把全部原因一次报齐，不能只报第一条。
    """
    no = _parse_article_no(citation.article)
    if no is None:
        return [f"引用失败：条号「{citation.article}」无法解析为唯一条号"]
    article = article_of(no)
    # 索引查询器可能覆盖不到该条号（假语料只塞部分条）：缺省成空索引，
    # 让款/项两关在「条不存在」之外仍各自独立判定，而不是直接崩
    index = (para_index_of(no) if article is not None else None) or _EMPTY_INDEX
    failures = _check_existence(citation, no, article, index["paragraphs"])
    failures += _check_item(citation, no, index["items"])
    failures += _check_quote(citation, no, article)
    return failures


def verify_answer(ans: Answer, *, article_of, para_index_of) -> list[str]:
    """校验整条答案，返回全部失败原因；空列表表示四关全过。

    article_of(no) -> {"text", "status", "law_id"} | None
    para_index_of(no) -> {"paragraphs": set[int], "items": set[str]}
    两个查询器都注入，测试可塞假语料，运行时传 LawCorpus 的两个方法。
    """
    if ans.status != "ok":
        # 追问补齐与非法律问题不要求引用，它们本来就不该给结论
        return []
    if not ans.citations:
        return ["引用失败：答案没有任何引用，结论无依据"]
    failures: list[str] = []
    for citation in ans.citations:
        failures += _verify_one(citation, article_of=article_of,
                                para_index_of=para_index_of)
    return failures
