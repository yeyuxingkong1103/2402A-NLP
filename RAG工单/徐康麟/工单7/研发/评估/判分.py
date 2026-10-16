# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 纯函数判分模块。

**本模块不依赖索引、模型、网络与文件系统**，只做字符串/数值/区间判定，因此可以单独
import 做单元测试（``python 研发/评估/判分.py`` 会跑内置自检）。

口径总览
--------
* :func:`normalize`            —— 去空白、去千分位、全角→半角、去标点（保留小数点）、小写化
* :func:`keyword_coverage`     —— 答案命中多少个期望关键词 ``(命中, 总数, 覆盖率, 命中表, 缺失表)``
* :func:`is_correct`           —— 覆盖率 ≥ 阈值 且 未拒答 ⇒ 判对
* :func:`partial_correct`      —— 「答案没错但表述/单位不同」的兜底判据（见函数 docstring）
* :func:`retrieval_recall_at_k`—— top-k 片段合并正文对关键词的召回率
* :func:`page_hit`             —— 命中页是否落在期望页区间（支持 ``a-b`` 与 ``a-b;c-d``）
* :func:`cross_corpus_leak`    —— top-3 命中是否出现**不属于期望语料**的文件名

设计要点：数字边界
------------------
子串匹配天然会把 ``11.65`` 里的 ``1.65`` 当成命中，把 ``18.25`` 里的 ``18.2`` 当成命中。
本模块对所有**以数字开头/结尾**的关键词施加"数字边界"约束：匹配位置的相邻字符不能是
数字或小数点，从而消除这类假阳性（实测年报数值题必须处理，否则得分虚高）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

# 与 配置.py 保持同源；此处给默认值，保证本模块可被独立 import（不反向依赖配置）
try:  # pragma: no cover - 仅用于独立运行时兜底
    from 配置 import KEYWORD_THRESHOLD, ATOMIC_THRESHOLD, LEAK_TOP_N  # type: ignore
except Exception:  # noqa: BLE001
    KEYWORD_THRESHOLD = 0.6
    ATOMIC_THRESHOLD = 0.6
    LEAK_TOP_N = 3

_DIGITS = frozenset("0123456789")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

#: 归一化时**保留**的标点：小数点（保住 1.65 / 18.2 / 11.05 的精度）
_KEEP_PUNCT = frozenset(".")


# ===========================================================================
# 1) 归一化
# ===========================================================================
def _to_halfwidth(ch: str) -> str:
    """全角 → 半角（含全角空格）。"""
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    return ch


def normalize(text: Any) -> str:
    """归一化文本（**严格口径**），用于宽松比对。

    处理顺序：``None``/非字符串安全转换 → 全角转半角 → 千分位逗号删除（仅当夹在数字之间：
    ``824,961`` → ``824961``）→ 其余标点/符号/控制符替换为**空格**（不是删除！）→
    空白折叠为单个空格并去首尾 → 小写化。

    ⚠️ 标点为什么替换成空格而不是删除：
        年报表格里相邻单元格的数字原本靠 ``|`` / ``,`` 分隔，若把这些分隔符直接删掉，
        ``| 29,428,896,087.24 | 24,277,670,240.59 |`` 会粘成
        ``29428896087.2424277670240.59``，数字边界判定就会把**真命中**误判成不命中
        （实测把 recall@10 从 1.0 压到 0.0）。留一个空格既保住数字边界，又不影响中文短语匹配。

    >>> normalize('1,379.58 亿元')
    '1379.58 亿元'
    >>> normalize('| 824,961 | 745,165 |')
    '824961 745165'
    >>> normalize(None)
    ''
    """
    if text is None:
        return ""
    s = "".join(_to_halfwidth(ch) for ch in str(text))
    buf: list[str] = []
    for idx, ch in enumerate(s):
        if ch.isspace():
            buf.append(" ")
            continue
        if ch in _KEEP_PUNCT:            # 小数点
            buf.append(ch)
            continue
        # P=标点 S=符号 Z=分隔 C=控制/未分配（乱码输入的兜底）
        if unicodedata.category(ch)[0] in ("P", "S", "Z", "C"):
            if ch == "," and buf and buf[-1].isdigit():
                nxt = s[idx + 1] if idx + 1 < len(s) else ""
                if nxt.isdigit():        # 千分位逗号 → 真删除
                    continue
            buf.append(" ")              # 其余标点 → 空格分隔，防止数字粘连
            continue
        buf.append(ch)
    return " ".join("".join(buf).split()).lower()


def _loose(norm_text: str) -> str:
    """把严格口径里的空格全部去掉（宽松口径），用于容忍"两个字之间多一个空格"的 PDF 抽取噪声。"""
    return norm_text.replace(" ", "")


def _contains(hay: str, needle: str) -> bool:
    """带数字边界的子串匹配（hay/needle 均已 normalize）。

    ``11.65`` **不**命中关键词 ``1.65``；``18.25`` **不**命中 ``18.2``。
    """
    if not hay or not needle:
        return False
    start = 0
    size = len(needle)
    while True:
        idx = hay.find(needle, start)
        if idx < 0:
            return False
        end = idx + size
        ok = True
        if needle[0] in _DIGITS and idx > 0 and (hay[idx - 1] in _DIGITS or hay[idx - 1] == "."):
            ok = False
        if ok and needle[-1] in _DIGITS and end < len(hay) and (hay[end] in _DIGITS or hay[end] == "."):
            ok = False
        if ok:
            return True
        start = idx + 1


def contains_keyword(hay_strict: str, keyword: Any) -> bool:
    """关键词是否命中**已归一化**的正文：严格口径（保留空格，护住数字边界）或宽松口径（去空格）。

    两个口径"或"起来：严格口径防假阳性，宽松口径容忍 PDF 抽取产生的多余空格。
    """
    needle = normalize(keyword)
    if not needle:
        return False
    if _contains(hay_strict, needle):
        return True
    return _contains(_loose(hay_strict), _loose(needle))


def text_matches(text: Any, keyword: Any) -> bool:
    """单个文本是否命中关键词（对外便捷函数，双口径）。"""
    return contains_keyword(normalize(text), keyword)


# ===========================================================================
# 2) 关键词覆盖
# ===========================================================================
def _clean_keywords(keywords: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for kw in keywords or []:
        if kw is None:
            continue
        text = str(kw).strip()
        if text:
            out.append(text)
    return out


def keyword_coverage(answer: Any, keywords: Iterable[Any] | None) -> tuple[int, int, float, list[str], list[str]]:
    """答案对期望关键词的覆盖率。

    :returns: ``(命中数, 总数, 覆盖率, 命中关键词列表, 缺失关键词列表)``；
              关键词为空时返回 ``(0, 0, 0.0, [], [])``（不抛异常）。
    """
    kws = _clean_keywords(keywords)
    if not kws:
        return (0, 0, 0.0, [], [])
    hay = normalize(answer)
    hit: list[str] = []
    miss: list[str] = []
    for kw in kws:
        (hit if contains_keyword(hay, kw) else miss).append(kw)
    total = len(kws)
    return (len(hit), total, round(len(hit) / total, 4), hit, miss)


def is_correct(coverage: Any, refused: Any, threshold: float = KEYWORD_THRESHOLD) -> bool:
    """严格判对：**未拒答** 且 覆盖率 ≥ 阈值。"""
    if refused:
        return False
    try:
        cov = float(coverage)
    except (TypeError, ValueError):
        return False
    return cov >= float(threshold)


# ===========================================================================
# 3) 数值原子（用于"答案正确但表述/单位不同"的兜底判据）
# ===========================================================================
def numeric_atoms(text: Any) -> list[str]:
    """抽出归一化文本里的所有数字原子（千分位已去掉）。``'1,379.58亿元'`` → ``['1379.58']``。"""
    return _NUM_RE.findall(normalize(text))


#: 中文/英文单位换算可能出现的数量级差（亿 ↔ 十亿 ↔ 百万 ↔ 万 ↔ 千 ↔ 百 ↔ 十）
_SCALES = (1.0, 1e1, 1e2, 1e3, 1e4, 1e6, 1e8, 1e12)


def _number_equivalent(a: str, b: str, rel: float = 0.005) -> bool:
    """两个数字原子是否等价（允许单位换算引起的 10^n 倍差，相对误差 rel）。"""
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if x == 0.0 or y == 0.0:
        return x == y
    for scale in _SCALES:
        if abs(x * scale - y) <= rel * abs(y) or abs(y * scale - x) <= rel * abs(x):
            return True
    return False


def atomic_coverage(answer: Any, keywords: Iterable[Any] | None) -> tuple[int, int, float, list[str], list[str]]:
    """**数值原子**覆盖率：只看关键词里的数字，允许单位换算（亿元 ↔ billion）。

    用途：英文作答 / 单位改写导致的"字面不匹配但事实正确"。仅作补充判据，不单独定对错。
    """
    atoms: list[str] = []
    for kw in _clean_keywords(keywords):
        atoms.extend(numeric_atoms(kw))
    atoms = list(dict.fromkeys(atoms))
    if not atoms:
        return (0, 0, 0.0, [], [])
    hay_atoms = numeric_atoms(answer)
    hit, miss = [], []
    for atom in atoms:
        (hit if any(_number_equivalent(atom, h) for h in hay_atoms) else miss).append(atom)
    total = len(atoms)
    return (len(hit), total, round(len(hit) / total, 4), hit, miss)


def partial_correct(coverage: Any, refused: Any, answer: Any = "",
                    keywords: Iterable[Any] | None = None,
                    threshold: float = KEYWORD_THRESHOLD,
                    partial_threshold: float = 0.3,
                    atomic_threshold: float = ATOMIC_THRESHOLD) -> bool:
    """「部分正确」判据（严格判对之外的补充档，**不算入严格准确率**）。

    满足任一条件即为部分正确（前提：未拒答 且 未达到严格判对）：
      1. 关键词覆盖率 ∈ ``[partial_threshold, threshold)`` —— 答对了一部分事实；
      2. 关键词的**数值原子**覆盖率 ≥ ``atomic_threshold`` —— 数字都对上了，
         只是表述/单位/语言不同（典型：英文作答 vs 中文关键词）。
    """
    if refused or is_correct(coverage, refused, threshold):
        return False
    try:
        cov = float(coverage)
    except (TypeError, ValueError):
        cov = 0.0
    if cov >= float(partial_threshold):
        return True
    _n, _t, acov, _h, _m = atomic_coverage(answer, keywords)
    return acov >= float(atomic_threshold)


# ===========================================================================
# 4) 检索召回
# ===========================================================================
def retrieval_recall_at_k(hits_text: Sequence[Any] | None, keywords: Iterable[Any] | None,
                          k: int) -> tuple[int, int, float, list[str], list[str]]:
    """top-k 片段**合并正文**对关键词的召回率。

    :param hits_text: 已按相关度排序的片段正文列表（第 0 条最相关）
    :param k: 取前 k 条（``k`` 非法/超出时按实际条数处理）
    :returns: ``(命中关键词数, 关键词总数, recall@k, 命中列表, 缺失列表)``
    """
    kws = _clean_keywords(keywords)
    texts = [str(t or "") for t in (hits_text or [])]
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = len(texts)
    if kk < 0:
        kk = 0
    merged = normalize("\n".join(texts[:kk]))
    if not kws:
        return (0, 0, 0.0, [], [])
    hit, miss = [], []
    for kw in kws:
        (hit if contains_keyword(merged, kw) else miss).append(kw)
    total = len(kws)
    return (len(hit), total, round(len(hit) / total, 4), hit, miss)


# ===========================================================================
# 5) 页命中
# ===========================================================================
_RANGE_RE = re.compile(r"^(\d+)\s*(?:[-~—－]|\s至\s|,)\s*(\d+)$")
_SINGLE_RE = re.compile(r"^(\d+)$")


def parse_page_ranges(hint: Any) -> list[tuple[int, int]]:
    """解析期望页区间：``'5-51'`` → ``[(5, 51)]``；``'5-37;7-20'`` → ``[(5, 37), (7, 20)]``。"""
    ranges: list[tuple[int, int]] = []
    for part in re.split(r"[;；]", str(hint or "")):
        part = part.strip()
        if not part:
            continue
        match = _RANGE_RE.match(part)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            ranges.append((min(a, b), max(a, b)))
            continue
        match = _SINGLE_RE.match(part)
        if match:
            page = int(match.group(1))
            ranges.append((page, page))
    return ranges


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _basename(name: Any) -> str:
    """取文件名（去目录、去首尾空白、大小写归一），用于跨语料比对。"""
    text = str(name or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].casefold()


def page_hit(hits: Sequence[Mapping[str, Any]] | None, expected_pages_hint: Any,
             source_filter: Sequence[str] | str | None = None) -> dict[str, Any]:
    """命中页是否落在期望页区间。

    * ``expected_pages_hint`` 支持 ``"a-b"``、单页 ``"a"``、以及分号并列 ``"a-b;c-d"``；
    * ``source_filter`` 给定期望文件名（**保持与区间一一对应的顺序**）时，只统计这些文件的
      命中，且当"文件数 == 区间数"时按**下标配对**（第 i 个文件 ↔ 第 i 个区间）——这正是
      跨文档题（Q04/Q09）的语义；
    * 不给 ``source_filter`` 时，任一命中页落在任一区间即算命中。

    :returns: ``{"hit": bool, "ranges": [...], "matched": [...], "scanned": int, "reason": str}``
    """
    result: dict[str, Any] = {"hit": False, "ranges": [], "matched": [], "scanned": 0, "reason": ""}
    ranges = parse_page_ranges(expected_pages_hint)
    result["ranges"] = ranges
    hit_list = list(hits or [])
    if not ranges:
        result["reason"] = "no_valid_range"
        return result
    if not hit_list:
        result["reason"] = "no_hits"
        return result

    files: list[str] = []
    if source_filter:
        files = [str(x) for x in ([source_filter] if isinstance(source_filter, str) else source_filter)]
    filter_set = {_basename(f) for f in files} if files else None
    paired = bool(files) and len(files) == len(ranges) and len(ranges) > 1
    owner = {_basename(f): i for i, f in enumerate(files)} if paired else {}

    for rank, hit in enumerate(hit_list):
        page = _to_int(hit.get("page"))
        if page is None:
            continue
        name = _basename(hit.get("file_name"))
        if filter_set is not None and name not in filter_set:
            continue
        result["scanned"] += 1
        candidates = [ranges[owner[name]]] if paired and name in owner else ranges
        for lo, hi in candidates:
            if lo <= page <= hi:
                result["hit"] = True
                result["matched"].append({
                    "rank": rank, "chunk_id": hit.get("chunk_id"), "page": page,
                    "file_name": hit.get("file_name"), "range": [lo, hi],
                    "final_score": hit.get("final_score"), "score": hit.get("score"),
                })
                break
    if not result["hit"] and not result["reason"]:
        result["reason"] = "out_of_range"
    return result


# ===========================================================================
# 6) 跨语料泄漏
# ===========================================================================
def cross_corpus_leak(hits: Sequence[Mapping[str, Any]] | None,
                      expected_file_names: Sequence[str] | str | None,
                      top_n: int = LEAK_TOP_N) -> dict[str, Any]:
    """top-N 命中里是否出现**不属于期望语料**的文件名（公司级串档）。

    :param expected_file_names: 该题允许出现的语料文件名（通常来自问题集 ``source_file``）
    :returns: ``{"leak": bool, "foreign": [...], "expected": [...], "checked": int}``
    """
    if isinstance(expected_file_names, str):
        expected_list = [x for x in re.split(r"[;；]", expected_file_names) if x.strip()]
    else:
        expected_list = [str(x) for x in (expected_file_names or []) if str(x).strip()]
    expected = {_basename(x) for x in expected_list}
    hit_list = list(hits or [])
    try:
        limit = max(0, int(top_n))
    except (TypeError, ValueError):
        limit = LEAK_TOP_N
    foreign: list[dict[str, Any]] = []
    if not expected:
        return {"leak": False, "foreign": [], "expected": [], "checked": min(limit, len(hit_list)),
                "reason": "no_expected_corpus"}
    for rank, hit in enumerate(hit_list[:limit], start=1):
        name = _basename(hit.get("file_name"))
        if name and name not in expected:
            foreign.append({
                "rank": rank, "chunk_id": hit.get("chunk_id"), "file_name": hit.get("file_name"),
                "page": hit.get("page"), "final_score": hit.get("final_score"), "score": hit.get("score"),
            })
    return {"leak": bool(foreign), "foreign": foreign,
            "expected": sorted(expected), "checked": min(limit, len(hit_list))}


# ===========================================================================
# 7) 元数据辅助（纯函数）
# ===========================================================================
def split_source_files(source_file: Any) -> list[str]:
    """``source_file`` 用 ``;`` 并列多个文件名 → 列表。"""
    if isinstance(source_file, (list, tuple)):
        return [str(x).strip() for x in source_file if str(x).strip()]
    return [x.strip() for x in re.split(r"[;；]", str(source_file or "")) if x.strip()]


def company_of(file_name: Any) -> str:
    """从语料文件名派生公司简称。

    文件名形如 ``2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf``，
    按 ``__`` 切分后第 4 段（下标 3）就是公司简称；字段不足时降级返回主名。
    """
    stem = str(file_name or "").rsplit(".", 1)[0].strip()
    parts = [p.strip() for p in stem.split("__") if p.strip()]
    if len(parts) >= 4:
        return parts[3]
    return parts[0] if parts else ""


def companies_of(source_file: Any) -> str:
    """多语料题返回 ``A;B`` 形式的公司简称串。"""
    names = [company_of(f) for f in split_source_files(source_file)]
    return ";".join([n for n in names if n])


def classify_question(question: Any, lang: Any, keywords: Iterable[Any] | None,
                      source_file: Any) -> str:
    """题型归类（用于汇总的"按题型分组"）。

    规则（按优先级）：
      1. ``lang == 'en'``                                → 英文题
      2. 多份来源文件 且 题干含对比/综合词                 → 跨文档对比题
      3. 数值型关键词占比 ≥ 0.5                          → 数值题
      4. 题干含"哪些/如何/是什么/说明/分析"               → 定性题
      5. 其它                                            → 其他
    """
    text = str(question or "")
    kws = _clean_keywords(keywords)
    if str(lang or "").lower().startswith("en"):
        return "英文题"
    files = split_source_files(source_file)
    if len(files) > 1 and any(w in text for w in ("对比", "比较", "共同", "综合", "分别")):
        return "跨文档对比题"
    if kws:
        numeric_ratio = sum(1 for k in kws if _NUM_RE.search(normalize(k))) / len(kws)
        if numeric_ratio >= 0.5:
            return "数值题"
    if any(w in text for w in ("哪些", "如何", "是什么", "说明", "分析", "为什么", "怎样")):
        return "定性题"
    return "其他"


def citation_leak(citations: Sequence[Mapping[str, Any]] | None,
                  expected_file_names: Sequence[str] | str | None) -> dict[str, Any]:
    """引用（citations）里是否出现不属于期望语料的文件 —— "引用页属于别的公司" 的直接判据。"""
    if isinstance(expected_file_names, str):
        expected_list = [x for x in re.split(r"[;；]", expected_file_names) if x.strip()]
    else:
        expected_list = [str(x) for x in (expected_file_names or []) if str(x).strip()]
    expected = {_basename(x) for x in expected_list}
    foreign: list[dict[str, Any]] = []
    for cite in citations or []:
        name = _basename(cite.get("corpus") or cite.get("file_name"))
        if expected and name and name not in expected:
            foreign.append({"corpus": cite.get("corpus") or cite.get("file_name"),
                            "page": cite.get("page"), "chunk_id": cite.get("chunk_id")})
    return {"leak": bool(foreign), "foreign": foreign}


# ===========================================================================
# 自检（纯函数，无需索引）
# ===========================================================================
def _selftest() -> int:
    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got={got!r} want={want!r}")

    # --- normalize ---
    check("normalize/千分位", normalize("1,379.58 亿元"), "1379.58 亿元")
    check("normalize/全角百分号", normalize("不良贷款率 1.65％"), "不良贷款率 1.65")
    check("normalize/空", normalize(None), "")
    check("normalize/乱码不崩", normalize("\x00乱码\ufffd"), "乱码")
    check("normalize/大小写", normalize("ROE 11.05%"), "roe 11.05")
    check("normalize/表格分隔符不被删除", normalize("| 824,961 | 745,165 |"), "824961 745165")

    # --- 回归：相邻表格数字被逗号/竖线分隔时，真命中必须命中（曾因粘连被判不命中）---
    check("glue/大额千分位", keyword_coverage(
        "| 营业收入 | 29,428,896,087.24 | 24,277,670,240.59 |", ["29,428,896,087.24"])[2], 1.0)
    check("glue/相邻单元格", keyword_coverage("| 824,961 | 745,165 | 10.7% |",
                                          ["824961", "745165", "10.7%"])[2], 1.0)
    check("glue/空格分隔数字", keyword_coverage("745,165 10.7% 643,101", ["10.7%"])[2], 1.0)
    check("glue/仍防假阳性", keyword_coverage("| 11,650 | 18,250 |", ["1.65", "18.2"])[2], 0.0)
    check("glue/recall", retrieval_recall_at_k(
        ["| 824,961 | 745,165 | 10.7% |"], ["824961", "10.7%"], 1)[2], 1.0)
    check("glue/宽松口径容忍空格", keyword_coverage("每 10 股派发现金红利人民币 4.00 元",
                                              ["每10股派发现金红利人民币4.00元"])[2], 1.0)

    # --- keyword_coverage 正常 ---
    n, t, cov, hit, miss = keyword_coverage("2019年营业收入1,379.58亿元，同比增长18.2%", ["1,379.58", "18.2%", "281.95"])
    check("coverage/命中数", n, 2)
    check("coverage/总数", t, 3)
    check("coverage/缺失", miss, ["281.95"])

    # --- 数字边界（关键：防假阳性） ---
    check("boundary/11.65不命中1.65", keyword_coverage("不良贷款率11.65%", ["1.65"])[2], 0.0)
    check("boundary/18.25不命中18.2", keyword_coverage("增长18.25%", ["18.2"])[2], 0.0)
    check("boundary/1.65命中", keyword_coverage("不良贷款率1.65%", ["1.65"])[2], 1.0)
    check("boundary/18.2命中", keyword_coverage("同比增长18.2%", ["18.2"])[2], 1.0)
    check("boundary/1165不命中165", keyword_coverage("金额1165元", ["165"])[2], 0.0)

    # --- 空 / 乱码输入 ---
    check("empty/answer", keyword_coverage("", ["a"])[2], 0.0)
    check("empty/keywords", keyword_coverage("abc", [])[:3], (0, 0, 0.0))
    check("none/keywords", keyword_coverage("abc", None)[:3], (0, 0, 0.0))
    # 纯标点（归一化后为空）的关键词视为**无法匹配** → 计入缺失（保守，不让空洞关键词白送分）
    check("garbage/keywords", keyword_coverage("!!!|||", ["!!!|||"])[2], 0.0)
    check("garbage/不崩", type(keyword_coverage("!!!|||", ["!!!|||"])).__name__, "tuple")

    # --- is_correct ---
    check("is_correct/达标", is_correct(0.6, False, 0.6), True)
    check("is_correct/不达标", is_correct(0.59, False, 0.6), False)
    check("is_correct/拒答", is_correct(1.0, True, 0.6), False)
    check("is_correct/坏输入", is_correct("x", False, 0.6), False)

    # --- atomic_coverage ---
    _n, _t, acov, _h, _m = atomic_coverage("Net profit was RMB 61.036 billion", ["610.36亿元"])
    check("atomic/单位换算", acov, 1.0)

    # --- partial_correct ---
    check("partial/中档", partial_correct(0.4, False, "x", ["a"]), True)
    check("partial/已判对", partial_correct(0.8, False, "x", ["a"]), False)
    check("partial/拒答", partial_correct(0.4, True, "x", ["a"]), False)

    # --- retrieval_recall_at_k ---
    hits = ["营业收入实现1,379.58亿元", "同比增长18.2%", "净利润281.95亿元"]
    check("recall@1", retrieval_recall_at_k(hits, ["1,379.58", "281.95"], 1)[2], 0.5)
    check("recall@3", retrieval_recall_at_k(hits, ["1,379.58", "281.95"], 3)[2], 1.0)
    check("recall@0", retrieval_recall_at_k(hits, ["1,379.58"], 0)[2], 0.0)
    check("recall/k非法", retrieval_recall_at_k(hits, ["1,379.58"], "x")[2], 1.0)

    # --- page_hit ---
    h1 = [{"page": 20, "file_name": "a.pdf", "chunk_id": "c1"}]
    check("page/区间内", page_hit(h1, "5-67")["hit"], True)
    check("page/区间外", page_hit(h1, "5-19")["hit"], False)
    check("page/单页", page_hit(h1, "20")["hit"], True)
    check("page/空hint", page_hit(h1, "")["hit"], False)
    check("page/空hits", page_hit([], "5-67")["hit"], False)
    # 跨文档题：第 i 个文件配第 i 个区间（h2: b.pdf p30 / a.pdf p10）
    h2 = [{"page": 30, "file_name": "b.pdf", "chunk_id": "c2"}, {"page": 10, "file_name": "a.pdf", "chunk_id": "c3"}]
    check("page/配对-按序", page_hit(h2, "5-20;25-40", ["a.pdf", "b.pdf"])["hit"], True)
    check("page/配对-错序", page_hit(h2, "5-20;25-40", ["b.pdf", "a.pdf"])["hit"], False)
    check("page/不配对时取并集", page_hit(h2, "5-20;25-40")["hit"], True)
    check("page/配对-全落空", page_hit([{"page": 99, "file_name": "a.pdf"}], "5-20;25-40",
                                   ["a.pdf", "b.pdf"])["hit"], False)

    # --- cross_corpus_leak ---
    hits3 = [{"file_name": "招商银行.pdf", "chunk_id": "x", "page": 53, "final_score": 2.1, "rank": 1},
             {"file_name": "平安银行.pdf", "chunk_id": "y", "page": 30, "final_score": 1.9}]
    leak = cross_corpus_leak(hits3, "平安银行.pdf")
    check("leak/有泄漏", leak["leak"], True)
    check("leak/泄漏文件名", leak["foreign"][0]["file_name"], "招商银行.pdf")
    check("leak/无泄漏", cross_corpus_leak([hits3[1]], "平安银行.pdf")["leak"], False)
    check("leak/只看top3", cross_corpus_leak(hits3 + hits3, "招商银行.pdf")["leak"], True)
    check("leak/前3无泄漏", cross_corpus_leak(hits3[:1], "招商银行.pdf")["leak"], False)
    check("leak/多文件", cross_corpus_leak(hits3, "平安银行.pdf;招商银行.pdf")["leak"], False)
    check("leak/空期望", cross_corpus_leak(hits3, "")["leak"], False)

    # --- citation_leak ---
    check("citeleak/有", citation_leak([{"corpus": "招商银行.pdf", "page": 5}], "平安银行.pdf")["leak"], True)
    check("citeleak/无", citation_leak([{"corpus": "平安银行.pdf", "page": 5}], "平安银行.pdf")["leak"], False)

    # --- 元数据 ---
    fname = "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf"
    check("company", company_of(fname), "平安银行")
    check("companies", companies_of(fname + ";" + fname), "平安银行;平安银行")
    check("split", len(split_source_files(fname + ";" + fname)), 2)

    # --- 题型 ---
    check("题型/en", classify_question("x", "en", ["1"], "a.pdf"), "英文题")
    check("题型/数值", classify_question("营收多少", "zh", ["1,379.58", "18.2%"], "a.pdf"), "数值题")
    check("题型/跨文档", classify_question("对比两家", "zh", ["1"], "a.pdf;b.pdf"), "跨文档对比题")
    check("题型/定性", classify_question("有哪些创新", "zh", ["蛋壳平台"], "a.pdf"), "定性题")

    if failures:
        print("[判分自检] 失败 %d 项：" % len(failures))
        for item in failures:
            print("  -", item)
        return 1
    print("[判分自检] 全部通过（含数字边界、空/乱码输入、跨页区间、跨语料泄漏等边界用例）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
