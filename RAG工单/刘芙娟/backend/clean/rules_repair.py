"""修复类规则（FR-027 ~ FR-033，对应裁决 D2）。

**只执行一条修复**：跨页断句合并。
标题粘连与标题层级失真**只报告、不修复**——剥离"5.4"这类残留属于改写正文；
按编号重算 text_level 等于自造 MinerU 的契约，两件都不能做。
"""

from __future__ import annotations

import re

# 句末终止标点（含闭合引号/括号）
TERMINAL = "。！？；：…!?;:"
CLOSERS = "”’」』】）》）)\"'"

# 标题编号，如 "5.4.6.1" / "1.1"
_HEADING_NUM = re.compile(r"^\d+(?:\.\d+)+")
_HEADING_NUM_LOOSE = re.compile(r"^\d+\s+\S")

# 列表项开头，不应被当作断句续行
_LIST_MARKER = re.compile(r"^[（(]\d+[）)]|^[①-⑳]|^[-*•]")

# 粘连标题：句末标点之后又接了一段章节编号 + 标题文字，如
#   "…避免依据单次血压测量值频繁调整药物。5.4.6 血压≥ 180/110 mmHg 的紧急处理"
# 判据：句末标点 + 编号 + 空格 + 一段**不以标点收尾**的短文字（≤40 字）。
_GLUED_TAIL = re.compile(r"[。；]\s*(\d+(?:\.\d+)+)\s+([^\s].{0,40})$")

_CN_OR_WORD = re.compile(r"[㐀-䶿一-鿿A-Za-z0-9]$")
_START_OK = re.compile(r"^[㐀-䶿一-鿿A-Za-z]")

# 断句续行的强判据：**前块以汉字结尾、后块以汉字开头**（中间无空格）。
# 实测教训：只判「结尾无标点」会大量误合并——页脚注释结尾是 DOI 数字、
# 图题结尾是「图」、英文机构名结尾是字母，它们都不以标点收尾，
# 但都不是被截断的句子。放宽到「汉字接汉字」后，实测 6 处合并只剩 1 处正确项。
_CJK_END = re.compile(r"[㐀-䶿一-鿿]$")
_CJK_START = re.compile(r"^[㐀-䶿一-鿿]")

# 只允许这些类型的块参与合并：表格/图片/页脚注释/方程/列表都是自洽单元，
# 与相邻块合并没有意义（实测：页脚注释被粘进了正文段落）。
MERGEABLE_TYPES = frozenset({"text"})


def _ends_open(text: str) -> bool:
    """结尾是汉字、且没有终止标点 -> 句子可能是被截断的。"""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in TERMINAL:
        return False
    return bool(_CJK_END.search(stripped))


def _starts_continuation(text: str) -> bool:
    """开头是汉字，且不像新起一段（不是标题编号、不是列表项、不是空）。"""
    stripped = text.lstrip()
    if not stripped:
        return False
    if _HEADING_NUM.match(stripped) or _HEADING_NUM_LOOSE.match(stripped):
        return False
    if _LIST_MARKER.match(stripped):
        return False
    return bool(_CJK_START.match(stripped))


def should_merge(prev: dict, cur: dict, gap_rules: list[str], opts) -> bool:
    """判断相邻两个保留块是否应合并（FR-030 的全部条件）。"""
    if not getattr(opts, "repair", True):
        return False
    if prev.get("is_heading") or cur.get("is_heading"):
        return False
    # 只有正文块参与合并；表格/图片/页脚注释/方程/列表是自洽单元
    if prev.get("block_type") not in MERGEABLE_TYPES:
        return False
    if cur.get("block_type") not in MERGEABLE_TYPES:
        return False
    # 题注降级块、内容不完整的降级块，都是自洽单元
    if prev.get("caption_demoted") or cur.get("caption_demoted"):
        return False
    if prev.get("degraded") or cur.get("degraded"):
        return False

    page_diff = cur["page_idx"] - prev["page_idx"]
    if page_diff not in (0, 1):
        return False

    # 同页相邻：只允许中间夹着「空块」——这是实测的列切分/空行场景
    # （p0 的 "…然而，高" + 空块 + "管疾病流行的核心策略之一。"）。
    # 跨页相邻：中间夹的页眉/页码属正常噪声，不再额外约束。
    if page_diff == 0 and any(rule != "empty:text" for rule in gap_rules):
        return False

    if not _ends_open(prev["text"]):
        return False
    if not _starts_continuation(cur["text"]):
        return False
    return True


def merge_blocks(prev: dict, cur: dict) -> dict:
    """合并两块：文本相接，页码取起始页（块内引用以起始页为准）。"""
    merged = dict(prev)
    merged["text"] = prev["text"].rstrip() + cur["text"].lstrip()
    merged["merged_from"] = list(prev.get("merged_from", [prev["raw_idx"]])) + list(
        cur.get("merged_from", [cur["raw_idx"]])
    )
    merged["raw_text"] = merged["text"]
    return merged


# --------------------------------------------------------------------------
# 只报告、不修复的两项（D2）
# --------------------------------------------------------------------------

_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)+)")


def _number_depth(text: str) -> int:
    """"4.1.1.1 测量仪器" -> 4；"1 基层高血压管理基本要求" -> 1。"""
    match = _NUMBER_PREFIX.match(text.lstrip())
    return match.group(1).count(".") + 1 if match else 1


# --------------------------------------------------------------------------
# 裁决 2-B：标题粘连正文尾 -> 切出并还原为独立标题块
# --------------------------------------------------------------------------

def split_glued_heading(block: dict) -> tuple[str, str] | None:
    """返回 (段落文本, 切出的标题文本)；无可切内容返回 None。

    实测输入：「…避免依据单次血压测量值频繁调整药物。5.4.6 血压≥ 180/110 mmHg 的紧急处理」
    切出后：段落「…频繁调整药物。」 + 标题「5.4.6 血压≥ 180/110 mmHg 的紧急处理」
    """
    text = block.get("text") or ""
    match = _GLUED_TAIL.search(text)
    if not match:
        return None
    paragraph = text[: match.start() + 1]      # 保留句末标点
    heading = "%s %s" % (match.group(1), match.group(2))
    if not paragraph.strip() or not heading.strip():
        return None
    return paragraph, heading


# --------------------------------------------------------------------------
# 裁决 3-D：按章节编号重算层级 + 补标 MinerU 漏标的标题
# --------------------------------------------------------------------------

# 编号可解析时用编号深度，否则回退 MinerU 的 text_level
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\s")
_NUMBERED_TOP = re.compile(r"^(\d+)\s")

# 补标为标题的额外护栏：够短、且不以句末标点收尾（标题不会以「。」结尾）
_MAX_HEADING_CHARS = 60


def number_level(text: str) -> int | None:
    """"4.1.1.1 测量仪器" -> 4；"1 基层高血压管理基本要求" -> 1；无编号 -> None。"""
    stripped = text.lstrip()
    match = _NUMBERED.match(stripped) or _NUMBERED_TOP.match(stripped)
    if not match:
        return None
    return match.group(1).count(".") + 1


def looks_like_untagged_heading(text: str) -> bool:
    """形如章节标题、但 MinerU 没给 text_level 的块。

    实测：5.4.6.1 / 5.4.6.2 两块以章节号开头却没被标为标题，
    后果是 heading_path 缺一层，引用卡片会指错章节。
    """
    stripped = (text or "").lstrip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return False
    if number_level(stripped) is None:
        return False
    return stripped[-1] not in TERMINAL


def heading_level(block: dict) -> int:
    """层级：编号可解析时用编号深度，否则回退 text_level（缺省 1）。"""
    from_number = number_level(block.get("text") or "")
    if from_number is not None:
        return from_number
    level = block.get("text_level")
    return level if isinstance(level, int) and level > 0 else 1


# 题注误标为标题：MinerU 把图/表题注也给了 text_level，若不降级，
# 它会在 heading_path 里冒充一章，把 4 级章节树打断（实测 idx 53 / 84 / 141）。
_CAPTION_PREFIX = re.compile(r"^(?:图|表|续表|附图|附表|Figure|Table)\s*\d+")
_MAX_CAPTION_CHARS = 40
_FLOAT_TYPES = frozenset({"table", "image", "chart"})


def caption_like(block: dict, next_raw_block: dict | None) -> str | None:
    """返回命中的降级理由，否则 None。"""
    text = (block.get("text") or "").strip()
    if not text:
        return None
    if _CAPTION_PREFIX.match(text):
        return "caption:prefix"
    # 有章节编号的是真标题，哪怕它正下方就是图/表
    # （实测反例：'5.4.3.1 <80 岁、无合并症的高血压药物治疗方案（图 2）'）
    if number_level(text) is not None:
        return None
    if (
        next_raw_block is not None
        and next_raw_block.get("type") in _FLOAT_TYPES
        and len(text) <= _MAX_CAPTION_CHARS
    ):
        return "caption:before-float"
    return None


def detect_level_distortion(raw_blocks: list[dict]) -> str | None:
    """层级失真只作为**提示**输出（已按 3-D 重算，此处仅告知成因）。"""
    tagged = [
        b for b in raw_blocks
        if b.get("type") == "text" and "text_level" in b
    ]
    if len(tagged) < 5:
        return None
    max_level = max(b["text_level"] for b in tagged)
    deepest = max(_number_depth(b.get("text") or "") for b in tagged)
    if deepest <= max_level:
        return None
    return (
        "MinerU 给的 text_level 最大只有 %d，而章节编号最深到 %d 级（如 4.1.1.1）；"
        "已按 3-D 用编号深度重算层级（%d 个标题块受影响）。"
        % (max_level, deepest, len(tagged))
    )
