"""转换类规则（FR-020 ~ FR-026，对应裁决 D3）。

只做三类**无损**转换，多一样都不做：
  ① 反转义 Markdown 残留反斜杠（\\~ -> ~、\\* -> *）
  ② 解码 HTML 实体（&lt; -> <）
  ③ 表格 HTML -> Markdown（rowspan/colspan 按行展开，文字零丢失）

方程块（equation，text_format 恒为 latex）**跳过全部字符级转换**——
LaTeX 里的反斜杠是语法，反转义会直接毁掉公式。
"""

from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser

# LaTeX 与 Markdown 共用的转义：\X -> X
_BACKSLASH_ESCAPE = re.compile(r"\\(.)", re.DOTALL)

# 跳过字符级转换的块类型
_NO_CHAR_CONVERSION = frozenset({"equation"})


def unescape_backslash(text: str) -> str:
    return _BACKSLASH_ESCAPE.sub(r"\1", text)


def unescape_html(text: str) -> str:
    return _html.unescape(text)


def convert_text(text: str, block_type: str, opts, stats: dict | None = None) -> str:
    """对单个块的正文字符串施加转换类规则。

    stats 用于逐条记录「哪条转换真的改动了字符」，供四类报告计数。
    """
    if not text or block_type in _NO_CHAR_CONVERSION:
        return text
    if not getattr(opts, "convert", True):
        return text

    step1 = unescape_backslash(text)
    if stats is not None and step1 != text:
        stats["backslash_unescape"] = stats.get("backslash_unescape", 0) + 1

    step2 = unescape_html(step1)
    if stats is not None and step2 != step1:
        stats["html_entity"] = stats.get("html_entity", 0) + 1

    return step2


# --------------------------------------------------------------------------
# 表格 HTML -> Markdown
# --------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """把 <table> 解析成 rows；保留 <sup>/<sub>/<br> 的原义。

    convert_charrefs=True：单元格里的 &lt; / &gt; / &amp; 在这里就解码成
    字面量（表格里 `收缩压 &lt; 120 mmHg` 的「小于号」属医学语义，
    不解码会污染检索，也违背 FR-026 ②）。
    """

    _KEEP_LITERAL = {"sup", "sub", "br"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self._row: list[dict] | None = None
        self._cell: dict | None = None
        self._buf: list[str] = []
        self._depth_keep = 0

    # -- 标签 --
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._cell = {
                "rowspan": _int_attr(attrs, "rowspan"),
                "colspan": _int_attr(attrs, "colspan"),
                "header": tag == "th",
                "text": "",
            }
            self._buf = []
        elif tag in self._KEEP_LITERAL and self._cell is not None:
            if tag == "br":
                self._buf.append("<br>")
            else:
                self._depth_keep += 1
                self._buf.append("<%s>" % tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None:
            self._cell["text"] = "".join(self._buf).strip()
            assert self._row is not None
            self._row.append(self._cell)
            self._cell = None
            self._buf = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in self._KEEP_LITERAL and tag != "br" and self._depth_keep > 0:
            self._depth_keep -= 1
            self._buf.append("</%s>" % tag)

    # -- 文本 --
    def handle_data(self, data):
        if self._cell is not None:
            self._buf.append(data)

    # 实体由 convert_charrefs=True 直接送进 handle_data，无需单独处理


def _int_attr(attrs, name: str) -> int:
    for key, value in attrs:
        if key.lower() == name:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return 1
    return 1


def expand_grid(rows):
    """把 rowspan/colspan 展开成矩形网格；合并单元格文字填充到它覆盖的每一格。"""
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()
    max_col = 0

    for r, row in enumerate(rows):
        col = 0
        for cell in row:
            while (r, col) in occupied:
                col += 1
            for rr in range(r, r + cell["rowspan"]):
                for cc in range(col, col + cell["colspan"]):
                    grid[(rr, cc)] = cell["text"]
                    occupied.add((rr, cc))
            col += cell["colspan"]
            max_col = max(max_col, col)

    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    return [[grid.get((r, c), "") for c in range(max_col)] for r in range(n_rows)]


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def html_table_to_markdown(table_html: str) -> str | None:
    """返回 Markdown 表格；无法解析出任何行时返回 None（调用方须显式降级）。"""
    if not table_html or "<" not in table_html:
        return None
    parser = _TableParser()
    parser.feed(table_html)
    parser.close()

    table_rows = expand_grid(parser.rows)
    if not table_rows:
        return None

    n_cols = max(len(r) for r in table_rows)
    table_rows = [r + [""] * (n_cols - len(r)) for r in table_rows]

    header, body = table_rows[0], table_rows[1:]
    lines = [
        "| " + " | ".join(_md_cell(c) for c in header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 各块类型的文本渲染
# --------------------------------------------------------------------------

def _join_lines(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return "\n".join(str(v) for v in value if str(v).strip())


def render_block_text(block: dict, opts, stats: dict | None = None) -> tuple[str, bool]:
    """返回 (文本, 是否降级)。降级 = 该块有内容但不完整（如表格丢了表体）。"""
    btype = block.get("type", "")

    if btype == "table":
        caption = convert_text(_join_lines(block.get("table_caption")), btype, opts, stats)
        footnote = convert_text(_join_lines(block.get("table_footnote")), btype, opts, stats)
        body_html = block.get("table_body") or ""
        body_md = html_table_to_markdown(body_html) if opts.convert else None

        if body_md is None:
            # 表体缺失/不可解析：不得静默丢弃，保留可得信息并标记降级
            img = block.get("img_path") or "(无图)"
            degraded_md = "[表格降级：未取到可解析的表体]\n表格图像：%s" % img
            parts = [p for p in (caption, degraded_md, footnote) if p]
            return "\n".join(parts), True

        if stats is not None:
            stats["table_to_markdown"] = stats.get("table_to_markdown", 0) + 1
        parts = [p for p in (caption, body_md, footnote) if p]
        return "\n".join(parts), False

    if btype in ("image", "chart"):
        caption = _join_lines(block.get("image_caption") or block.get("img_caption"))
        footnote = _join_lines(block.get("image_footnote") or block.get("img_footnote"))
        text = "\n".join(p for p in (caption, footnote) if p)
        if not text.strip():
            img = block.get("img_path") or "(无图)"
            return "[图片降级：无图注与图注脚]\n图片：%s" % img, True
        return text, False

    if btype == "list":
        items = block.get("list_items") or []
        return "\n".join(
            convert_text(str(i), btype, opts, stats) for i in items
        ), False

    # text / header / footer / page_number / aside_text / page_footnote /
    # equation / code
    return convert_text(block.get("text") or "", btype, opts, stats), False


def is_heading(block: dict) -> bool:
    """标题判定：type == text **且 text_level 键存在**（docs/04 §5）。

    两个常见错误在此被显式避开：
      - 不存在 type == "title" 这种类型；
      - 不能用 get("text_level", 0) >= 1，正文里该键是**缺失**而不是 0。
    """
    return block.get("type") == "text" and "text_level" in block


def to_page(page_idx: int) -> int:
    """0-based page_idx -> 1-based 物理页码（FR-020）。"""
    return page_idx + 1
