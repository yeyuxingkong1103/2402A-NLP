# -*- coding: utf-8 -*-
"""法规名 / 条号匹配（评测的"检索侧判定"原语，批次 24 自 run_eval.py 拆出）。

为什么单独成文件：`run_eval.py` 拆分前已 628 行，超出项目"单文件 ≤300 行"硬规则
（见 `docs/目录与命名约定.md` §3.4）；本模块只装"把 golden 与检索结果对上"这件事，
**零逻辑改动**（函数体逐字搬移，对外函数名不变，调用方只改 import 路径）。

与 `answer_judging.py` 的分工：
- 本模块：判定**检索结果**里有没有目标法源（法规名归一 + 条号中文/阿拉伯数字归一）；
- `answer_judging.py`：判定**回答文本**是否合格（拒答口径、引用越界）。
两边的判定对象不同，不要互相 import。

条号归一为什么必须做：库内 `article_number` 存的是阿拉伯数字（"47"），
而评测集 golden 写的是中文（"第四十七条"），不归一会全部判为未命中。
"""

from __future__ import annotations

import re
from typing import Any

# 法规简称 → 库内 documents.title
# 顺序敏感：长名优先，避免"劳动合同法"吃掉"劳动合同法实施条例"
LAW_ALIASES: tuple[tuple[str, str], ...] = (
    ("劳动合同法实施条例", "中华人民共和国劳动合同法实施条例"),
    ("劳动争议司法解释（一）", "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）"),
    ("劳动争议司法解释（二）", "最高法发布劳动争议司法解释（二）和典型案例"),
    ("劳动争议调解仲裁法", "中华人民共和国劳动争议调解仲裁法"),
    ("劳动合同法", "中华人民共和国劳动合同法"),
    ("劳动法", "中华人民共和国劳动法"),
    ("工资支付暂行规定", "工资支付暂行规定"),
    ("职工带薪年休假条例", "职工带薪年休假条例"),
    ("工伤保险条例", "工伤保险条例"),
    ("女职工劳动保护特别规定", "女职工劳动保护特别规定"),
)


def normalize_article(value: str | None) -> str:
    """把条号统一成阿拉伯数字字符串，便于跨形态比较（"第十九条" / "19" → "19"）。"""
    if not value:
        return ""
    text = str(value).strip().replace(" ", "")
    text = text[1:-1] if text.startswith("第") and text.endswith("条") else text
    text = text.removesuffix("条")
    try:
        from app.ingest.chinese_number import to_arabic_number

        if re.fullmatch(r"[一二三四五六七八九十百千万零〇两\d]+", text):
            if text.isdigit():
                return str(int(text))
            return str(int(to_arabic_number(text)))
    except Exception:
        pass
    return text


def canonical_title(law_alias: str) -> str:
    for alias, title in LAW_ALIASES:
        if law_alias == alias or law_alias in alias or alias in law_alias:
            return title
    return law_alias


def title_matches(canonical: str, document_title: str | None) -> bool:
    if not document_title:
        return False
    return document_title == canonical or canonical in document_title or document_title in canonical


def article_matches(article_gold: str, article_actual: str | None) -> bool:
    left, right = normalize_article(article_gold), normalize_article(article_actual)
    return bool(left) and left == right


def first_golden_rank(articles: list[Any], golden: list[dict[str, str]]) -> int | None:
    """返回 golden 中任一 (法规, 条号) 在结果里的最早排名（1 起）；未命中返回 None。"""
    for rank, article in enumerate(articles, start=1):
        for gold in golden:
            canonical = canonical_title(gold["law"])
            if title_matches(canonical, article.document_title) and article_matches(
                gold["article"], article.article_number
            ):
                return rank
    return None


def absent_violation(articles: list[Any], expect_absent: list[dict[str, str]]) -> str | None:
    for article in articles:
        for gold in expect_absent or []:
            canonical = canonical_title(gold["law"])
            if title_matches(canonical, article.document_title) and article_matches(
                gold["article"], article.article_number
            ):
                return f"{gold['law']}{gold['article']}"
    return None
