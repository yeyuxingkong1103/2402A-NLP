# -*- coding: utf-8 -*-
# 【表格清洗规范化模块 · table_parser.py】词坐标网格重建、多级表头还原、标题识别、Markdown化、表格行陈述句
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""表格解析核心层（对应设计文档第 3 章）。

pdfplumber 的 ``extract_tables`` 在招股书两类典型表上存在实测缺陷：
1. 无框线/弱框线的行表头列与末列被整体漏掉（如兴图 P185 的
   “国防领域/民用领域/合计”与最后一列“82.10%”等）；
2. 合并表头产生大量空列、单元格内硬换行；
3. 个别错版卡式表把“值/标签”切反。

本模块不直接信任单元格抽取结果，而是基于页内“词坐标 + 表格行带/列边”
重建规整字符矩阵，再做质量门控、多级表头还原、caption 识别、
Markdown 序列化与逐行陈述句生成。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CONFIG

# 表头词词典：命中即认为该行是表头行的组成部分
_HEADER_LEXICON = (
    "名称", "姓名", "项目", "序号", "类型", "金额", "占比", "比例", "持股",
    "关系", "单位", "年度", "期间", "日期", "编号", "类别", "内容", "指标",
    "计划总投资", "募集资金", "投资额", "出资额", "股东",
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*年(?:度)?(?:\d{1,2}\s*[-—~]\s*\d{1,2}\s*月)?")
_NUMERIC_RE = re.compile(r"^[\d,\.．，\s%％\[\]◆◆\-\-—~～万元亿]*\d[\d,\.．，\s%％\[\]◆\-—~～万元亿]*$")
_CAPTION_NOISE = re.compile(r"^(?:单位|资料来源|注)[：:]")


@dataclass
class ParsedTable:
    """清洗规范化后的表格结构。"""

    headers: List[str]                      # 还原后的单行表头
    rows: List[List[str]]                   # 数据行矩阵（与表头等长）
    markdown: str                           # Markdown 形式整表
    caption: str                            # 表标题（表格上方最近文本）
    page_no: int                            # 1 起始页码
    bbox: Tuple[float, float, float, float]
    claims: List[str] = field(default_factory=list)  # 逐行自然语言陈述句


# ---------------------------------------------------------------------------
# 基础清洗
# ---------------------------------------------------------------------------
def clean_cell(text: Optional[str]) -> str:
    """清洗单个单元格：去换行、归一化空白、剔除 CJK 间空格。

    :param text: 原始单元格文本
    :return: 规范化文本
    """
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\r", " ").replace("\u3000", " ")
    t = t.replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"\s+", " ", t).strip()
    # 英文/数字内部保留单空格；中文与中文、中文与数字/字母之间的空格多为折行产物
    t = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", t)
    t = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[0-9A-Za-z])", "", t)
    t = re.sub(r"(?<=[0-9A-Za-z%％])\s+(?=[\u4e00-\u9fa5])", "", t)
    t = re.sub(r"\s*([，。；：、%％])\s*", r"\1", t)
    return t.strip("-—|:： ")


def _join_words(words: List[str]) -> str:
    """把同格内的词拼接为完整文本。

    :param words: 词列表（已按坐标排序）
    :return: 拼接清洗后的文本
    """
    return clean_cell(" ".join(words))


# ---------------------------------------------------------------------------
# 词坐标网格重建
# ---------------------------------------------------------------------------
def _column_slots(table) -> List[Tuple[float, float]]:
    """根据全部单元格竖边构造列槽位，并在表格左右各补一个“越界词槽位”。

    :param table: pdfplumber Table 对象
    :return: 列槽位 (x_left, x_right) 列表（有序）
    """
    x0, _top, x1, _bot = table.bbox
    edges = {round(x0, 2), round(x1, 2)}
    for row in table.rows:
        for cell in row.cells:
            if cell is None:
                continue
            edges.add(round(cell[0], 2))
            edges.add(round(cell[2], 2))
    edges = sorted(edges)
    # 合并间距小于 2pt 的抖动边
    merged: List[float] = []
    for edge in edges:
        if not merged or abs(edge - merged[-1]) > 2.0:
            merged.append(edge)
        else:
            merged[-1] = (merged[-1] + edge) / 2.0
    slots = [(merged[i], merged[i + 1]) for i in range(len(merged) - 1)]
    # 左右外侧槽位：回收边框外的行表头与末列数据
    slots.insert(0, (x0 - CONFIG.table_side_margin_pt, x0 - 1.5))
    slots.append((x1 + 1.5, x1 + CONFIG.table_side_margin_pt))
    return slots


def _assign_slot(x_center: float, slots: List[Tuple[float, float]]) -> int:
    """按词的中心 x 坐标分配列槽位，找不到返回 -1。

    :param x_center: 词中心横坐标
    :param slots: 列槽位列表
    :return: 列下标
    """
    for idx, (left, right) in enumerate(slots):
        if left <= x_center <= right:
            return idx
    return -1


def rebuild_grid(table, words: List[dict]) -> List[List[str]]:
    """用页内词坐标把表格重建为规整 R×C 字符矩阵。

    行带采用 pdfplumber 检出的表格横线区间；列槽位由单元格竖边加左右越界槽位
    构成。每个词按中心坐标归入唯一 (行, 列)，从而修复漏掉的行表头列与末列。

    :param table: pdfplumber Table 对象
    :param words: pdfplumber ``extract_words`` 词列表
    :return: 重建后的二维文本矩阵
    """
    x0, top, x1, bottom = table.bbox
    slots = _column_slots(table)
    row_bands = [(r.bbox[1], r.bbox[3]) for r in table.rows]
    grid: List[Dict[int, List[Tuple[float, str]]]] = [
        dict() for _ in row_bands]

    for w in words:
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        # 仅取表格本体及左右越界槽位纵向范围内的词
        if not (top - 2.0 <= cy <= bottom + 2.0):
            continue
        if not (x0 - CONFIG.table_side_margin_pt <= cx <=
                x1 + CONFIG.table_side_margin_pt):
            continue
        col = _assign_slot(cx, slots)
        if col < 0:
            continue
        row_idx = -1
        for r_idx, (r_top, r_bottom) in enumerate(row_bands):
            if r_top - 2.0 <= cy <= r_bottom + 2.0:
                row_idx = r_idx
                break
        if row_idx < 0:
            continue
        grid[row_idx].setdefault(col, []).append((w["x0"], w["text"]))

    matrix: List[List[str]] = []
    for row in grid:
        cells = []
        for col in range(len(slots)):
            tokens = sorted(row.get(col, []), key=lambda item: item[0])
            cells.append(_join_words([t[1] for t in tokens]))
        matrix.append(cells)

    # 删除全空列与全空行
    matrix = _drop_empty(matrix)
    return matrix


def _drop_empty(matrix: List[List[str]]) -> List[List[str]]:
    """删除全空列与全空行，规整行宽。

    :param matrix: 原始矩阵
    :return: 压缩后的矩阵
    """
    if not matrix:
        return matrix
    n_cols = max(len(r) for r in matrix)
    for r in matrix:
        r += [""] * (n_cols - len(r))
    keep_cols = [c for c in range(n_cols)
                 if any(matrix[r][c] for r in range(len(matrix)))]
    trimmed = [[r[c] for c in keep_cols] for r in matrix]
    trimmed = [r for r in trimmed if any(cell for cell in r)]
    # 再次去掉因错位产生的行尾空列
    while trimmed and all((len(r) == 1 or not r[-1]) for r in trimmed) and \
            max(len(r) for r in trimmed) > 1:
        for r in trimmed:
            if len(r) > 1:
                r.pop()
    return trimmed


# ---------------------------------------------------------------------------
# 质量门控
# ---------------------------------------------------------------------------
def quality_ok(matrix: List[List[str]]) -> bool:
    """判断重建矩阵是否为“值得保留”的规整表格。

    :param matrix: 重建后的矩阵
    :return: 通过质量门控返回 True
    """
    rows = len(matrix)
    if rows < CONFIG.table_min_rows:
        return False
    cols = max(len(r) for r in matrix)
    if cols < CONFIG.table_min_cols:
        return False
    total = rows * cols
    filled = sum(1 for r in matrix for c in r if c)
    if filled / max(total, 1) < CONFIG.table_min_fill_rate:
        return False
    good_rows = sum(1 for r in matrix if sum(1 for c in r if c) >= 2)
    if good_rows / rows < CONFIG.table_min_row_ratio:
        return False
    avg_len = sum(len(c) for r in matrix for c in r if c) / max(filled, 1)
    if avg_len > CONFIG.table_max_cell_len:
        return False
    if not _header_sane(matrix):
        return False
    if _is_card_table(matrix):
        return False
    return True


def _is_card_table(matrix: List[List[str]]) -> bool:
    """识别“基本情况卡”式错版表（≥3 列标签-值-标签-值交错，行列归属不可靠）。

    招股书发行人概况卡被 pdfplumber 线框策略切表后，行内字段配对经常错位
    （如把“注册资本”的标签配到“法定代表人”的值），这类表整体交给 PyMuPDF
    正文承载更可靠，故在表格层拒绝。

    :param matrix: 重建后的矩阵
    :return: 判定为错版卡式表返回 True
    """
    cols = max(len(r) for r in matrix)
    # 仅拦截 ≥4 列（标签-值-标签-值交错）卡式表；3 列名表（如名称/持股/关系）放行
    if cols < 4 or len(matrix) < 2:
        return False

    def label_like(cell: str) -> bool:
        return bool(cell) and len(cell) <= 10 and not _is_numeric_cell(cell)

    label_hits, value_hits, pairs = 0, 0, 0
    for r in matrix:
        for ci in range(0, cols - 1, 2):
            lab = r[ci] if ci < len(r) else ""
            val = r[ci + 1] if ci + 1 < len(r) else ""
            if lab:
                pairs += 1
                if label_like(lab):
                    label_hits += 1
                if val:
                    value_hits += 1
    if pairs == 0:
        return False
    return label_hits / pairs >= 0.75 and value_hits / pairs >= 0.75


def _header_sane(matrix: List[List[str]]) -> bool:
    """表头合理性检查：拦截被 pdfplumber 错切的“卡式表”（值/标签颠倒、
    超长公司名/纯数字混入表头）。

    判定规则：
    - 表头任一非空格长度 ≤ 22（表头通常是短标签）；
    - 表头纯数值格占比 < 50%（允许“2010年1-6月”这类年度标签）；
    - 表头至少有一格命中表头词典或为年度标签。

    :param matrix: 重建后的矩阵
    :return: 表头可信返回 True
    """
    header = [c for c in matrix[0] if c]
    if not header:
        return False
    if max(len(c) for c in header) > 22:
        return False
    # 表头中若出现“1,840万股”这类非年度数值格，说明表体被错切（值跑到表头）
    for c in header:
        is_year = bool(_YEAR_RE.search(c)) or bool(
            re.match(r"^(?:19|20)\d{2}$", c)) or c in ("年度", "年")
        if not is_year and (_is_numeric_cell(c)
                            or re.search(r"\d[\d,\.]+\s*万?股", c)):
            return False
    numeric_ratio = sum(1 for c in header if _is_numeric_cell(c)) / len(header)
    if numeric_ratio >= 0.5:
        return False
    if any(any(w in c for w in _HEADER_LEXICON) for c in header):
        return True
    if any(_YEAR_RE.search(c) for c in header):
        return True
    # 两列键值表（如“股票种类/人民币普通股”）允许短标签表头
    return len(header) == 2 and all(len(c) <= 12 for c in header)


# ---------------------------------------------------------------------------
# 多级表头还原
# ---------------------------------------------------------------------------
def _is_numeric_cell(cell: str) -> bool:
    """判断单元格是否为纯数值/百分比/占位符。

    :param cell: 单元格文本
    :return: 数值型返回 True
    """
    return bool(cell) and bool(_NUMERIC_RE.match(cell))


def _header_like(row: List[str]) -> bool:
    """判断一行是否像表头（含表头词或年度分组，且数值格很少）。

    :param row: 矩阵行
    :return: 像表头返回 True
    """
    text_cells = [c for c in row if c]
    if not text_cells:
        return False
    numeric = sum(1 for c in text_cells if _is_numeric_cell(c))
    if numeric / len(text_cells) > 0.5:
        return False
    return any(any(w in c for w in _HEADER_LEXICON) for c in text_cells) or \
        bool(_YEAR_RE.search("".join(text_cells)))


def restore_headers(matrix: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """识别表头行并还原单行表头（支持两级表头与跨列年度分组）。

    :param matrix: 规整矩阵
    :return: (表头列表, 数据行列表)
    """
    if not matrix:
        return [], []
    n_cols = max(len(r) for r in matrix)
    for r in matrix:
        r += [""] * (n_cols - len(r))

    # 表头行数：最多 2 行。两级表头只出现在“年度/期间跨列分组 + 金额/占比
    # 子表头”的财务表中（第 0 行必须含年度模式），避免把“股东”等普通数据
    # 行误判成第二行表头。
    header_levels = 1
    if (len(matrix) >= 3 and _YEAR_RE.search("".join(matrix[0]))
            and _header_like(matrix[1])
            and any(not _is_numeric_cell(c) for c in matrix[1] if c)):
        # 数据区必须真实存在（第 2 行起至少一行不像表头）
        if any(not _header_like(matrix[i]) for i in range(2, len(matrix))):
            header_levels = 2

    if header_levels == 1:
        headers = [clean_cell(c) or f"列{i + 1}" for i, c in enumerate(matrix[0])]
        data = matrix[1:]
        return headers, data

    # 两级表头：上级年度分组前向填充后与下级（金额/占比等）组合
    upper = [clean_cell(c) for c in matrix[0]]
    lower = [clean_cell(c) for c in matrix[1]]
    # 年度分组在 PDF 中常被空格/列槽切成碎片（“2019年”+“1-6月”、
    # “2018”+“年度”），循环合并相邻碎片直到不能再拼出年度标签
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(1, len(upper)):
            a, b = upper[i - 1], upper[i]
            if not a or not b or len(a) > 10 or len(b) > 8:
                continue
            if _YEAR_RE.match(a + b) and (
                    "年" in b or "月" in b or re.match(
                        r"^(?:19|20)\d{2}$", a)):
                upper[i - 1] = a + b
                upper[i] = ""
                merged_any = True
    filled: List[str] = []
    last = ""
    for c in upper:
        if c:
            last = c
        filled.append(last)
    headers: List[str] = []
    for i in range(n_cols):
        up, low = filled[i], lower[i]
        if up and low and up != low:
            headers.append(f"{up}-{low}")
        elif low:
            headers.append(low)
        elif up:
            headers.append(up)
        else:
            headers.append(f"列{i + 1}")
    # 第 0 列若是行标签列，优先用下级标签名（类型/项目/名称）
    if lower[0]:
        headers[0] = lower[0]
    elif not upper[0] and headers[0].startswith("列"):
        headers[0] = "项目"
    return headers, matrix[2:]


# ---------------------------------------------------------------------------
# 表标题（caption）识别
# ---------------------------------------------------------------------------
def detect_caption(page, bbox: Tuple[float, float, float, float],
                   words: List[dict]) -> str:
    """识别表格正上方最近的 1~3 行文本作为表标题。

    :param page: pdfplumber 页面对象
    :param bbox: 表格 bbox
    :param words: 页内词列表
    :return: 拼接后的表标题（无则空串）
    """
    x0, top, _x1, _bot = bbox
    line_buckets: Dict[int, List[Tuple[float, str]]] = {}
    for w in words:
        cy = (w["top"] + w["bottom"]) / 2.0
        if not (top - CONFIG.table_caption_above_pt <= cy <= top - 2.0):
            continue
        cx = (w["x0"] + w["x1"]) / 2.0
        if not (x0 - 60 <= cx <= bbox[2] + 60):
            continue
        key = int(cy // 3.0)
        line_buckets.setdefault(key, []).append((w["x0"], w["text"]))
    lines = []
    for key in sorted(line_buckets):
        tokens = sorted(line_buckets[key], key=lambda item: item[0])
        line = clean_cell(" ".join(t[1] for t in tokens))
        if not line:
            continue
        # 过滤页眉页码等噪声
        if re.search(r"招股意向书|^\d{1,3}$|[\-—]\d{1,3}$", line):
            continue
        if _looks_like_data_line(line):
            continue
        lines.append(line)
    # 只保留最贴近表格的三行；表标题通常是编号小节或“……如下”引导句
    caption_lines = lines[-3:]
    caption_lines = [ln for ln in caption_lines
                     if len(ln) >= 3 or _CAPTION_NOISE.search(ln)]
    return " / ".join(caption_lines)


def _looks_like_data_line(line: str) -> bool:
    """判断表格上方文本行是否其实是上一张表/本区的数据碎片（不应当作标题）。

    判据：含 2 个以上数字簇、或 2 个以上百分号、或为超过 45 字且无标题引导的长句。

    :param line: 文本行
    :return: 像数据碎片返回 True
    """
    if len(re.findall(r"\d[\d,\.]{2,}", line)) >= 2:
        return True
    if line.count("%") + line.count("％") >= 2:
        return True
    # 短行内含“数值+百分号”（如“赵马克42.35%公司控股股东”）属上表数据行
    if re.search(r"\d[\d,\.]*\s*[%％]", line) and len(line) <= 30:
        return True
    # 无标点短行却串联 ≥2 个表头词（如“关联方名称持股比例与本公司关系”）；
    # 但“拟投资以下项目”等引导句不是数据碎片
    if (len(line) <= 30
            and not re.search(r"[，。；：、（）()【】]", line)
            and "如下" not in line and "以下" not in line
            and sum(1 for w in _HEADER_LEXICON if w in line) >= 2):
        return True
    heading_like = bool(
        re.match(r"^(?:第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、"
                 r"|（[一二三四五六七八九十]+）|\d+[\.、])", line))
    if len(line) > 45 and not heading_like and "如下" not in line:
        return True
    return False


# ---------------------------------------------------------------------------
# Markdown 与行陈述句
# ---------------------------------------------------------------------------
def _escape_md(cell: str) -> str:
    """转义 Markdown 表格中的竖线与换行。

    :param cell: 单元格文本
    :return: 安全文本
    """
    return cell.replace("|", "丨").replace("\n", " ").strip()


def to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """把表头与数据行序列化为 Markdown 表格。

    :param headers: 表头列表
    :param rows: 数据行矩阵
    :return: Markdown 字符串
    """
    lines = ["| " + " | ".join(_escape_md(h) for h in headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [_escape_md(c) for c in row]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return "\n".join(lines)


def _short_caption(caption: str) -> str:
    """取表标题中最像“小标题”的片段用于行陈述句前缀。

    :param caption: 完整 caption
    :return: 简短小标题
    """
    if not caption:
        return ""
    # 取“最贴近表格（最后出现）”的带编号小节标题，避免同页上下两张表
    # 时把上一张表的编号标题误挂到下一张表
    parts = re.split(r"\s*/\s*|[\n。；;]", caption)
    parts = [p.strip(" ：:，,") for p in parts]
    for part in reversed(parts):
        if re.search(r"[一二三四五六七八九十\d]+[、\.]", part) and len(part) <= 30:
            return part
    for part in reversed(parts):
        if part and len(part) <= 30:
            return part
    return caption[:30]


def build_row_claims(caption: str, headers: List[str],
                     rows: List[List[str]]) -> List[str]:
    """把每一数据行改写为自然语言“行陈述句”，供表格行块检索与单元格作答。

    例：表头 [关联方名称, 持股比例, 与本公司关系]，行 [赵马克, 42.35%, 公司控股股东]
    → “存在控制关系的关联方：赵马克，持股比例42.35%，与本公司关系：公司控股股东”

    :param caption: 表标题
    :param headers: 表头
    :param rows: 数据行
    :return: 行陈述句列表（与数据行对齐）
    """
    title = _short_caption(caption)
    claims: List[str] = []
    for row in rows:
        label = row[0] if row else ""
        # 双列键值表直接输出“键：值”，避免把表头名当成句子成分
        if len(headers) == 2:
            value = row[1] if len(row) > 1 else ""
            body = f"{label}：{value}" if label and value else (label or value)
            if title and title not in body:
                body = f"{title}：{body}"
            if body.strip():
                claims.append(body)
            continue
        pieces: List[str] = []
        for idx in range(1, len(headers)):
            val = row[idx] if idx < len(row) else ""
            if not val:
                continue
            head = headers[idx]
            # 年度-金额/占比 组合表头直接作为值前缀，保留财务语义
            pieces.append(f"{head}{val}" if re.search(r"金额|占比|比例", head)
                          else f"{head}：{val}")
        body = label
        if pieces:
            body = (label + "，" if label else "") + "，".join(pieces)
        if title and title not in body:
            body = f"{title}：{body}" if body else title
        # 纯数值合计行不生成陈述句（噪声大），但带标签的合计行保留
        if body.strip():
            claims.append(body)
    return claims


# ---------------------------------------------------------------------------
# 页面级入口
# ---------------------------------------------------------------------------
def parse_page_tables(plumber_page, page_no: int) -> List[ParsedTable]:
    """解析单页全部表格：识别→重建→门控→表头还原→序列化。

    :param plumber_page: pdfplumber 页面对象
    :param page_no: 1 起始页码
    :return: 清洗后的 ParsedTable 列表
    """
    result: List[ParsedTable] = []
    try:
        found = plumber_page.find_tables()
    except Exception:
        return result
    if not found:
        return result
    try:
        words = plumber_page.extract_words(
            x_tolerance=1.5, y_tolerance=3, extra_attrs=[])
    except Exception:
        words = []

    accepted_bboxes: List[Tuple[float, float, float, float]] = []
    for table in found:
        try:
            matrix = rebuild_grid(table, words)
            if not quality_ok(matrix):
                continue
            headers, data = restore_headers(matrix)
            if not data or not headers:
                continue
            # 数据行至少要有一行含 2 个非空格
            if sum(1 for r in data if sum(1 for c in r if c) >= 2) == 0:
                continue
            bbox = tuple(table.bbox)
            # 嵌套重叠表：若被已接受表大面积覆盖则跳过
            if any(_bbox_covered(bbox, old) for old in accepted_bboxes):
                continue
            caption = detect_caption(plumber_page, table.bbox, words)
            md = to_markdown(headers, data)
            claims = build_row_claims(caption, headers, data)
            result.append(ParsedTable(
                headers=headers, rows=data, markdown=md, caption=caption,
                page_no=page_no, bbox=bbox, claims=claims))
            accepted_bboxes.append(bbox)
        except Exception:
            # 单张表清洗失败不影响整页解析，直接跳过
            continue
    return result


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """计算 bbox 面积。"""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_covered(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> bool:
    """判断 bbox a 是否被 b 覆盖 80% 以上（用于嵌套表去重）。

    :param a: 待判断 bbox
    :param b: 已接受 bbox
    :return: 被覆盖返回 True
    """
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = _bbox_area(a) or 1.0
    return inter / area_a >= 0.8
