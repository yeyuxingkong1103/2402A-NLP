"""读取并校验 content_list.json（本步骤的唯一输入，FR-003）。"""

from __future__ import annotations

import json
import os

from .errors import CleanError, EXIT_BAD_INPUT, EXIT_UNKNOWN_TYPE

# docs/04 §5 记录的 MinerU 3.x pipeline 后端契约。
KNOWN_TYPES = frozenset({
    "text", "image", "table", "chart", "equation", "code", "list",
    "header", "footer", "page_number", "aside_text", "page_footnote",
})

# list 的 sub_type 已知取值（docs/04 §6）。
KNOWN_LIST_SUBTYPES = frozenset({"ref_text", "text"})


def load_content_list(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise CleanError(EXIT_BAD_INPUT, "content_list.json 不存在：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise CleanError(
            EXIT_BAD_INPUT, "JSON 解析失败：%s\n  %s" % (path, exc)
        ) from exc
    except OSError as exc:
        raise CleanError(EXIT_BAD_INPUT, "无法读取：%s\n  %s" % (path, exc)) from exc

    if not isinstance(data, list):
        raise CleanError(EXIT_BAD_INPUT, "content_list.json 顶层不是数组：%s" % path)
    if not data:
        raise CleanError(EXIT_BAD_INPUT, "content_list.json 为空数组：%s" % path)

    verify_contract(data)
    verify_page_idx(data)
    return data


def verify_contract(blocks: list[dict]) -> None:
    """块类型 / list 子类型必须落在已知集合内，未知即停止（FR-037）。"""
    unknown_types = sorted({b.get("type") for b in blocks} - KNOWN_TYPES)
    if unknown_types:
        raise CleanError(
            EXIT_UNKNOWN_TYPE,
            "出现 docs/04 §5 未列出的块类型，必须人工决策后回写文档与代码：\n  %s"
            % "、".join(repr(t) for t in unknown_types),
        )

    unknown_subtypes = sorted({
        b.get("sub_type")
        for b in blocks
        if b.get("type") == "list"
    } - KNOWN_LIST_SUBTYPES)
    if unknown_subtypes:
        raise CleanError(
            EXIT_UNKNOWN_TYPE,
            "出现未知的 list.sub_type，必须人工决策：\n  %s"
            % "、".join(repr(t) for t in unknown_subtypes),
        )


def verify_page_idx(blocks: list[dict]) -> None:
    """页码缺失不得以 0 填充后继续（specs/001 FR-005 的防御性延续）。"""
    for i, b in enumerate(blocks):
        if "page_idx" not in b:
            raise CleanError(
                EXIT_BAD_INPUT,
                "第 %d 个块缺 page_idx 字段；不得以 0 填充后继续。" % i,
            )
        if not isinstance(b["page_idx"], int) or b["page_idx"] < 0:
            raise CleanError(
                EXIT_BAD_INPUT,
                "第 %d 个块的 page_idx 非法：%r" % (i, b["page_idx"]),
            )
