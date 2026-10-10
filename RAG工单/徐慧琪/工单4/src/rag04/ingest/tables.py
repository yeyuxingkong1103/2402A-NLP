# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""表格抽取：pdfplumber → Markdown，保留结构化语义。"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import pdfplumber

from rag04.schema import TableBlock

logger = logging.getLogger("rag04.tables")

_WS = re.compile(r"\s+")
_CJK = re.compile(r"[一-鿿]")
_ASCII_DIGIT = re.compile(r"[0-9]")
# 报告期/日期列头：`2019年1-6月`、`2018年12月31日/2018年度`、`2019/6/30`
_PERIOD = re.compile(r"^\d{4}(?:年|[-/]\d{1,2}[-/]\d{1,2})")

# RC4：词级行标签对照参数
_KV_COL_GAP = 10.0        # 同一视觉行内，词间距 > 该值即视为不同栏位（pt）
_KV_MAX_XGAP = 60.0       # 标签与值相邻栏位的最大间距：更远说明跨了列/页宽，不配对
_KV_LINE_TOL = 3.0        # y 中心差 <= 该值视为同一视觉行（pt）
_KV_BAND_PAD = 6.0        # 表格 bbox 的 y 容差（pt）
_KV_LABEL_MIN, _KV_LABEL_MAX = 2, 20
_KV_VALUE_MIN = 2         # 单字值（释义表的连接列 `指` 等）一律不要
_KV_VALUE_MAX = 60
_KV_BOX_TOL = 2.0         # 网格 bbox 与词 x 范围的容差（判定网格是否覆盖整行）
_KV_MAX_PAIRS = 30


def _clean(cell) -> str:
    if cell is None:
        return ""
    return _WS.sub(" ", str(cell).replace("\n", " ")).strip()


def _escape(cell: str) -> str:
    """转义 Markdown 表格里的反斜杠、竖线与换行，避免破坏表结构。"""
    # 必须先转义反斜杠：否则既有 `\|` 会被写成 `\\|`，
    # GFM 把 `\\` 读作字面反斜杠，后面的 `|` 又变回活分隔符。
    return cell.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def rows_to_markdown(rows: list[list[str]]) -> str:
    """二维表 → Markdown 表格。首行作表头；空表或零列返回空串。"""
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return ""
    norm = [_clean_cells_row(r, n_cols) for r in rows]
    head, body = norm[0], norm[1:]
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * n_cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _clean_cells_row(row: list[str], n_cols: int) -> list[str]:
    cells = [_escape(_clean(c)) for c in row]
    if len(cells) < n_cols:
        cells += [""] * (n_cols - len(cells))
    return cells[:n_cols]


def _visual_lines(words: list[dict]) -> list[list[dict]]:
    """按 y 中心把词聚成视觉行（每行按 x 排序）。"""
    buckets: list[tuple[float, list[dict]]] = []
    for w in words:
        yc = (float(w.get("top", 0.0)) + float(w.get("bottom", 0.0))) / 2
        for i, (by, ws) in enumerate(buckets):
            if abs(by - yc) <= _KV_LINE_TOL:
                buckets[i] = ((by * len(ws) + yc) / (len(ws) + 1), ws + [w])
                break
        else:
            buckets.append((yc, [w]))
    return [sorted(ws, key=lambda w: float(w["x0"]))
            for _, ws in sorted(buckets, key=lambda b: b[0])]


def _line_segments(line: list[dict],
                   gap: float = _KV_COL_GAP) -> list[tuple[str, float, float]]:
    """把视觉行按栏位间距切成若干连续文本段（左→右），带回 x 起止。"""
    segs: list[list] = []          # [texts, x_start, x_end]
    for w in line:
        x0, x1 = float(w["x0"]), float(w["x1"])
        if segs and x0 - segs[-1][2] <= gap:
            segs[-1][0].append(str(w.get("text", "")))
            segs[-1][2] = max(segs[-1][2], x1)
        else:
            segs.append([[str(w.get("text", ""))], x0, x1])
    return [("".join(t), x0, x1) for t, x0, x1 in segs]


def _looks_like_label(seg: str) -> bool:
    """行标签形态：短、含中文、不带数字（数值/序号对不是标签）。"""
    s = seg.strip()
    return (_KV_LABEL_MIN <= len(s) <= _KV_LABEL_MAX
            and bool(_CJK.search(s)) and not _ASCII_DIGIT.search(s)
            and s[-1] not in "。！？；，、：:;") if s else False


def row_bands(table) -> list[tuple[float, float]]:
    """pdfplumber 表网格的行带 (top, bottom) 列表；取不到时返回空（宁缺勿错）。"""
    try:
        return [(float(r.bbox[1]), float(r.bbox[3])) for r in table.rows]
    except Exception as e:
        logger.warning("表网格行带读取失败：%s", e)
        return []


def _band_index(yc: float, bands: list[tuple[float, float]]) -> int | None:
    """y 中心落在哪个行带内（带 _KV_BAND_PAD 容差）。"""
    for i, (top, bottom) in enumerate(bands):
        if top - _KV_BAND_PAD <= yc <= bottom + _KV_BAND_PAD:
            return i
    return None


def _same_column(segs: list[tuple], seg: tuple) -> list[tuple]:
    """与 seg 横向显著重叠的段（同列）。``segs`` 元素为 (y, text, x0, x1)。"""
    x0, x1 = seg[1], seg[2]
    out = []
    for s in segs:
        if s[3] - s[2] <= 0:
            continue
        ov = min(x1, s[3]) - max(x0, s[2])
        if ov > 0.5 * min(x1 - x0, s[3] - s[2]):
            out.append(s)
    return out


def _has_band_neighbor(band_segs: list[tuple], seg: tuple, yc: float) -> bool:
    """同一行带内、与 seg 横向显著重叠的**另一视觉行**文本 → seg 只是被折行切碎的片段。

    真实 p22 实测：`主要生产经营地` 的值 `湖北省武汉市东湖新技 / 术开发区关山大道1号 /
    软件产业三期A3栋8 / 层` 折成 4 行，网格只把中间一行当整格；
    `注册地址` 的值同样折成 3 行。凡同一列在行带内还有上/下续行，本段即片段。
    """
    x0, x1 = seg[1], seg[2]
    w = x1 - x0
    if w <= 0:
        return True
    for y, _text, bx0, bx1 in band_segs:
        if abs(y - yc) <= 0.5:
            continue                       # 同一视觉行：本行自己的栏位
        ov = min(x1, bx1) - max(x0, bx0)
        if ov > 0.5 * min(w, bx1 - bx0):
            return True
    return False


def label_value_hints(page, bbox, row_bands=None) -> list[tuple[str, str]]:
    """RC4：从词级版面提取「行标签 = 数值」对照，修复表格网格错行。

    根因（id543，实测 p22）：pdfplumber 只把表格中部两栏识别为网格，
    且列序为 [值, 下一行标签]，于是 `5,520.00万元` 被配到「法定代表人」，
    「注册资本」整行丢失——证据在上下文里但行列对应关系已不存在。
    词级版面（同行、左标签右数值、栏位间距 > 10pt）才是真值来源：
    逐视觉行按栏位切段，相邻两段配对为 (标签, 值)，标签必须是短中文且不含
    数字（`1,840万股`、`占发行后总股本比例` 这类数值对自动跳过）。

    **复核收紧（review round 1）**：本对照块会进入上下文且表头声称权威，
    因此配对必须无歧义，宁可少而准。除上述几何条件外，再加以下闸门：

    1. 行带完整性：标签段与数值段所在的**表网格行带**（``row_bands``，
       来自 ``find_tables().rows``）内，同列不得还有别的折行文本。真实 p22
       的 `主要生产经营地` / `注册地址` 单元格跨 3~4 个视觉行，旧实现只取
       中间一行当整格，产出 `术开发区关山大道1号`、`关山大道1号软件产业三期A3栋`
       这类片段值 —— 现在直接不产出。
    2. 数值行判据：本视觉行必须至少含一个带阿拉伯数字的栏位，否则视为
       表头/纯文字行（`序号=股东名称`、`持股数量（万股）=持股比例（%）` 不产出）。
    3. 报告期列头行：整行出现 ≥2 个 `2019年1-6月` / `2019/6/30` 一律跳过
       （`项目=2019年1-6月` 这类整行年度列头不是行标签）。
    4. 完整网格的首行带是表头，直接跳过；网格只识别出中部若干栏（行带里的词
       越出网格 x 范围，如真实 p22）时该判定不生效，避免误杀 `中文名称=…`。
    5. 纯文字值：所在列必须在别的视觉行上出现过数值，否则该列是分类/名称列
       （`项目=存货类别`、`合计=设备销售` 不产出；真实 p22 的
       `法定代表人=程家明` 因同列还有成立日期而保留）。
    6. 短值（单字，如释义表的连接列 `指`）与左右同文的自配对不产出。
    7. 取不到行带时返回空：无法验证完整性就不声称权威。
    """
    try:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    except Exception as e:                      # 词级提取失败只是少了对照，不阻断
        logger.warning("词级版面提取失败（第 %s 页）：%s", getattr(page, "page_number", "?"), e)
        return []

    bands = [(float(t), float(b)) for t, b in (row_bands or [])]
    if not bands:
        return []                     # 行带不可得：不产出（表头声称权威，宁缺勿错）

    lines = _visual_lines(words)
    band_segs: dict[int, list[tuple]] = {}      # 行带 → [(y, text, x0, x1)]
    for line in lines:
        yc = (float(line[0]["top"]) + float(line[0]["bottom"])) / 2
        bi = _band_index(yc, bands)
        if bi is None:
            continue
        for text, x0, x1 in _line_segments(line):
            band_segs.setdefault(bi, []).append((yc, text, x0, x1))

    all_segs = [s for segs_ in band_segs.values() for s in segs_]

    # 网格是否覆盖整张表的横向范围。p22 这类「只识别出中部两栏」的残缺网格
    # （列 x 范围外还有词）第一行带是数据行；完整网格的第一行带才是表头。
    bx0, bx1 = float(bbox[0]), float(bbox[2])
    covered = all(bx0 - _KV_BOX_TOL <= x0 and x1 <= bx1 + _KV_BOX_TOL
                  for segs_ in band_segs.values()
                  for _y, _t, x0, x1 in segs_)

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        yc = (float(line[0]["top"]) + float(line[0]["bottom"])) / 2
        bi = _band_index(yc, bands)
        if bi is None:
            continue
        if covered and bi == 0:
            continue      # 完整网格的首行带是表头：`项目=2019年1-6月` 一类不产出
        segs_in_band = band_segs[bi]
        segs = _line_segments(line)
        if not any(_ASCII_DIGIT.search(t) for t, _x0, _x1 in segs):
            continue      # 本视觉行无任何数值：表头/纯文字行，不做标签对照
        if sum(1 for t, _x0, _x1 in segs if _PERIOD.search(t)) >= 2:
            continue      # 整行都是报告期/年度列头（项目=2019年1-6月 …）：不是行标签
        for i in range(0, len(segs) - 1, 2):
            label, value = segs[i][0].strip(), segs[i + 1][0].strip()
            if not _looks_like_label(label) or len(value) < _KV_VALUE_MIN \
                    or len(value) > _KV_VALUE_MAX:
                continue
            if label == value:
                continue      # 左右同文：相邻单元格，不是键值对
            if segs[i + 1][1] - segs[i][2] > _KV_MAX_XGAP:
                continue      # 相邻栏位隔得太远：跨列/跨页宽，不是同一键值对
            if _has_band_neighbor(segs_in_band, segs[i], yc) \
                    or _has_band_neighbor(segs_in_band, segs[i + 1], yc):
                continue      # 标签或值只是折行单元格的一个片段
            if not _ASCII_DIGIT.search(value):
                # 纯文字值：所在列必须在别的行上出现过数值，否则该列是分类/名称列
                # （`项目=存货类别`、`合计=设备销售` 这类列头互配一律不产出）；
                # 真实 p22 的 `法定代表人=程家明` 因同列还有成立日期而保留。
                col = _same_column(all_segs, segs[i + 1])
                if not any(_ASCII_DIGIT.search(t) for y2, t, _a, _b in col
                           if abs(y2 - yc) > 0.5):
                    continue
            if (label, value) in seen:
                continue
            seen.add((label, value))
            out.append((label, value))
            if len(out) >= _KV_MAX_PAIRS:
                return out
    return out


def _kv_markdown(pairs: list[tuple[str, str]]) -> str:
    head = ("[行标签对照（按版面几何校验整格无歧义后提取，修复表格网格错行；"
            "仅收录完整单元格的标签↔数值，不完整即不列）]")
    lines = [head, "| 行标签 | 值 |", "| --- | --- |"]
    lines += [f"| {_escape(a)} | {_escape(b)} |" for a, b in pairs]
    return "\n".join(lines)


def _tables_from_page(page, doc_id: str, page_no: int) -> list[TableBlock]:
    """从已打开的 pdfplumber Page 抽表；调用方负责异常捕获与释放页面。"""
    out: list[TableBlock] = []
    tables = page.extract_tables() or []
    try:
        found = list(page.find_tables()) if tables else []
    except Exception:
        found = []
    if len(found) != len(tables):
        found = []                                # 网格与抽取不一致：不敢做对照

    for idx, tbl in enumerate(tables):
        if not tbl or len(tbl) < 2:
            continue
        if not any(_clean(c) for row in tbl for c in row):
            continue  # 全空单元格：抽取噪声，不产出空内容块
        md = rows_to_markdown(tbl)
        if not md:
            continue
        if found:                                 # RC4：补行标签↔数值对照
            hints = label_value_hints(page, found[idx].bbox,
                                      row_bands=row_bands(found[idx]))
            if hints:
                md = f"{md}\n\n{_kv_markdown(hints)}"
        n_cols = max(len(r) for r in tbl)
        out.append(TableBlock(
            doc_id=doc_id, page=page_no,
            bbox=tuple(page.bbox), markdown=md,
            n_rows=len(tbl), n_cols=n_cols,
        ))
    return out


def extract_tables(pdf_path: Path, doc_id: str, page_no: int) -> list[TableBlock]:
    """抽取指定页（1-based）的所有表格。失败记录日志并返回空列表。"""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                return []
            return _tables_from_page(pdf.pages[page_no - 1], doc_id, page_no)
    except Exception as e:
        logger.warning("第 %d 页表格抽取失败：%s: %s", page_no, type(e).__name__, e)
    return []


def iter_all_tables(pdf_path: Path, doc_id: str) -> Iterator[TableBlock]:
    """逐页产出表格。打开失败记录后结束；单页失败跳过，不中断。"""
    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as e:
        logger.error("打开 PDF 失败，无法抽表：%s", e)
        return

    try:
        try:
            pages = pdf.pages
        except Exception as e:
            logger.error("读取 PDF 页面失败，无法抽表：%s", e)
            return
        for pno, page in enumerate(pages, start=1):
            try:
                yield from _tables_from_page(page, doc_id, pno)
            except Exception as e:
                logger.warning("第 %d 页表格抽取失败：%s: %s",
                               pno, type(e).__name__, e)
                continue
            finally:
                # 释放本页解析缓存（chars/rects/layout 等），控制长文档 RSS；
                # Page.close() 只清缓存，不影响文档与其它页。
                page.close()
    finally:
        pdf.close()
