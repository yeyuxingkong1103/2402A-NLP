"""条文（条）边界切分（自 chunker.py 拆出）。

只做一件事：把法规全文按"第 X 条"切成独立条文的列表。
切块主流程（chunk_document）与款/项识别（paragraph_items）不在这里。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _split_articles(content: str) -> list[tuple[str, str]]:
    """按"第 X 条"识别法律条文边界，返回 [(条文号, 条文内容)]。"""
    # 按法律条文编号切出独立父块的边界正则
    # 「条」后的合法续接分三类：
    #   1. 分隔符（空白/全角空格/冒号顿号逗号句号）或行尾 —— 标准排版；
    #   2. 正文汉字直接紧贴（"第一条劳动者…"，法院网页常见，去掉分隔符排版）；
    #   3. 排除"之/第"开头 —— "第X条第Y款""第X条之N"是引用续接或变体，
    #      不作为新条边界（之N 变体由组内 (?:之…) 处理）。
    article_pattern = re.compile(
        r"(?m)^\s*(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?)"
        r"(?=[\s　:：、。，]|$|(?![之第])\S)"
    )

    # 宽松提示仅用于识别疑似条文号，不直接作为切分边界
    hint_pattern = re.compile(
        r"(?m)^\s*第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?"
    )
    matches = list(article_pattern.finditer(content))

    # 提示存在但可靠识别不足时记录元数据，不记录正文
    hint_matches = list(hint_pattern.finditer(content))
    if hint_matches and len(matches) <= 1:
        logger.warning(
            "Article boundary recognition is unreliable",
            extra={"matched": len(hint_matches), "text_length": len(content)},
        )

    # 没有匹配到条文号时交给文档级处理
    if not matches:
        return []

    # 保存切分后的条文
    articles: list[tuple[str, str]] = []

    # 按相邻条文起点截取完整条文
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        article_content = content[start:end].strip()
        article_no = match.group(1)

        # 忽略没有正文的条文标题
        if article_content:
            articles.append((article_no, article_content))

    # 返回条文号和条文内容
    return articles
