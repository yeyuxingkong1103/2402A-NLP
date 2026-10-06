"""款/项识别（自 chunker.py 拆出）。

三级语义（docs/CONTEXT.md）中的后两级：
- 款：条内无编号自然段（_split_into_paragraphs），paragraph_no 从 1 依次编号；
- 项：款内的「（一）（二）」分项（_extract_items），item_no 按款内顺序编号。
条（一级）的边界切分在 app/ingest/article_splitter.py。

为什么款/项要单独成文件：条文最终会按"款/项"做成父子块（父块给 LLM 完整语境、
子块拿去算向量），所以"款怎么切、项怎么切"是检索质量的地基；把它从 chunker.py
里独立出来，既守住单文件 ≤300 行，也让这段规则能被单独测。
"""
from __future__ import annotations

import logging
import re

from app.ingest.chunk_text import _normalize_text

logger = logging.getLogger(__name__)

# 项号行首模式：行以「（一）」或「(一)」开头（半角/全角括号 + 中文数字）
_ITEM_LINE_PATTERN = re.compile(r"^\s*[（(]([一二三四五六七八九十百千万零〇]+)[）)]")

# 项号匹配模式：行首（或换行后）的项号标记，用于在款内切分各项
# 与 _ITEM_LINE_PATTERN 的区别：这里的 (?:^|\n) 允许项号出现在款文本中间
# （前一项的正文之后紧跟换行再起一项），而前者只判"这一行是不是项号行"。
_ITEM_BOUNDARY_PATTERN = re.compile(
    r"(?:^|\n)\s*[（(]([一二三四五六七八九十百千万零〇]+)[）)]"
)


def _split_into_paragraphs(article_content: str) -> list[str]:
    """把条文切成自然段（款）；以项号开头的行归属当前款而不是另起一段。

    参数：article_content —— 单条法条的正文（已含换行，可能带空行）。
    返回：款文本列表，按出现顺序；空串不保留。

    为什么用 `\\n\\s*\\n|\\n` 双分支切：源数据里段的切法不统一 —— HTML 抓取常留
    空行，手工整理的文本可能只有单换行。两种都当分隔符，避免"同一部法规
    切出的款数不一致"。
    """
    raw_parts = [
        part.strip()
        for part in re.split(r"\n\s*\n|\n", article_content)
        if part.strip()
    ]

    paragraphs: list[str] = []
    for part in raw_parts:
        # 项号行并入当前款（当前款即引导段所在的款）。
        # 若不合并，「（一）…」会各自成为一款，导致款数虚高 + 引导句与分项脱钩。
        if paragraphs and _ITEM_LINE_PATTERN.match(part):
            paragraphs[-1] = f"{paragraphs[-1]}\n{part}"
        else:
            paragraphs.append(part)

    return paragraphs


def _extract_items(paragraph_content: str) -> list[tuple[str, str]] | None:
    """款内识别项（（一）（二）…），返回 (项号阿拉伯数字, 项文本)；无项返回 None。

    参数：paragraph_content —— 单款文本（可能由多行项号行组成）。
    返回：`[(项号, 项文本), …]`；**无项时返回 None**（而非空列表），
          让调用方能区分"这一款本来就没有项"与"识别失败"。

    为什么延迟导入 to_arabic_number：chinese_number 也会被 article_number_rules
    等模块引用，放模块顶层会把"中文数字转换"链一起拉进来，延迟导入可避免
    潜在循环依赖。
    """
    from app.ingest.chinese_number import to_arabic_number

    matches = list(
        _ITEM_BOUNDARY_PATTERN.finditer(paragraph_content)
    )

    # 款内没有项号标记
    if not matches:
        return None

    items: list[tuple[str, str]] = []

    # 按相邻项号起点截取各项文本
    # 用"下一个项号的起点"当本项的终点，最后一个项到款尾 —— 这样项文本里
    # 即使再出现「（一）」这类字样，也不会被误当成新项（不重复扫描）。
    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(paragraph_content)
        )
        item_content = _normalize_text(paragraph_content[start:end])

        if not item_content:
            continue

        # 中文项号转阿拉伯数字；转换失败跳过该项并记录警告
        # 只跳单项、不整款失败：个别生僻写法（如"（廿一）"）不该牵连其余项。
        try:
            arabic_no = to_arabic_number(match.group(1))
        except ValueError:
            logger.warning(
                "Failed to convert item number: %s",
                match.group(1),
                extra={"chinese_number": match.group(1)},
            )
            continue

        items.append((arabic_no, item_content))

    # 全部转换失败时视为无项，整款按普通款处理
    if not items:
        return None

    return items
