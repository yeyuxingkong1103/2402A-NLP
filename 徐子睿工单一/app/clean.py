# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：clean —— 文本清洗与规范化
# 说明：去页眉页脚、合并断行、全角转半角、数字/单位规范化。不改变语义，只提升检索与数值匹配质量。

import re
import html

# 页眉页脚：公司页眉行（可能带“招股意向书”与页码）、页码行（1-1-129 之类）
_HEADER_RE = re.compile(r"^\s*武汉兴图新科电子股份有限公司\s*(招股意向书)?\s*(1-1-\d+)?\s*$")
_FOOTER_RE = re.compile(r"^\s*招股意向书\s*(1-1-\d+)?\s*$")
_PAGENO_RE = re.compile(r"^\s*1-1-\d+\s*$")
# 行内出现的页眉片段（与正文糊在一起时）
_INLINE_HEADER_RE = re.compile(r"武汉兴图新科电子股份有限公司\s*招股意向书\s*1-1-\d+")

# 全角 -> 半角：仅数字、拉丁字母（保留中文标点与全角括号，避免伤可读性）
_FW_MAP = {}
for _i in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)):
    _FW_MAP[chr(_i)] = chr(_i - 0xFEE0)
_FW_MAP["\u3000"] = " "
_FW_RE = re.compile("|".join(re.escape(k) for k in _FW_MAP))


def to_halfwidth(s: str) -> str:
    return _FW_RE.sub(lambda m: _FW_MAP[m.group()], s)


def strip_html_table(s: str) -> str:
    """把 MinerU 的 HTML 表格压成“单元格用 | 分隔、行用换行”的纯文本，便于检索与数值抽取。"""
    if "<td" not in s and "<tr" not in s:
        return s
    txt = re.sub(r"<td[^>]*>", " | ", s)
    txt = re.sub(r"</tr>", "\n", txt)
    txt = re.sub(r"</td>|<tr[^>]*>|</?table[^>]*>", "", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]*\|[ \t]*", " | ", txt)
    txt = re.sub(r"(\s*\|\s*)+", " | ", txt)
    return txt.strip(" |\n ")


def clean_block(text: str, btype: str = "text") -> str:
    if btype == "table":
        text = strip_html_table(text)
    lines = []
    for ln in text.split("\n"):
        if _HEADER_RE.match(ln) or _FOOTER_RE.match(ln) or _PAGENO_RE.match(ln):
            continue
        lines.append(ln)
    s = "\n".join(lines)
    s = _INLINE_HEADER_RE.sub("", s)
    s = to_halfwidth(s)
    # 中文行内换行 -> 直接拼接（避免把词切断）；英文/数字间换行 -> 空格
    s = re.sub(r"(?<=[\u4e00-\u9fff，。；：、）】])\n(?=[\u4e00-\u9fff（【〔])", "", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    return s


def clean_blocks(blocks):
    out = []
    for b in blocks:
        t = clean_block(b.get("text", ""), b.get("type", "text"))
        if not t:
            continue
        nb = dict(b)
        nb["text"] = t
        out.append(nb)
    return out
