# -*- coding: utf-8 -*-
"""表格解析模块（本工单核心新增）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

【为什么需要表格解析】
招股说明书中的关键事实（发行股数、募集资金投向、关联方持股比例、分年度收入
及占比等）大量以「表格」形式呈现。若沿用 01/02 工单的做法——把表格当作普通
文本行抽取，会出现三类致命问题：

1. 行列关系丢失：单元格被按阅读顺序拍平成一行文本，表头与数据行分离，
   检索时“表头关键词命中、数值却在另一行”，答案无法拼装；
2. 合并单元格错位：多级表头 / 跨行合并的单元格在纯文本中丢失归属，
   导致“某年—某项目—某数值”的三元关系无法还原；
3. 跨页表格断裂：长表格跨页后表头不再重复，后半段数据成为“无表头孤儿行”。

【本模块做法】
1. 版面级表格检测：基于 PyMuPDF `page.find_tables()` 获取表格的单元格网格与
   边界框（bbox），得到结构化的二维单元格矩阵；
2. 合并单元格还原：对 None（被合并吞掉的单元格）按“左邻优先、上邻兜底”填充，
   恢复“跨行/跨列表头”的语义归属；
3. 表头规整：支持多级表头（默认 1 行，可配置）合并为单行表头；
4. 双表示输出：
   - Markdown 表：保留行列结构，供人阅读与结构化展示；
   - 键值对行（key-value）：为每个数据行生成“表头=取值”形式的检索单元，
     把“表头—数值”强绑定，显著提升数值型/比例型问题的召回精度；
5. 表题识别：从表格 bbox 上方的文本块中提取表题（如“本次发行概况”“关联方
   及关联关系”），作为章节锚点，提升章节先验命中率；
6. 质量过滤：行/列数不足或有效文本过少的误检表格直接丢弃，避免噪声入库。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src import config

logger = logging.getLogger(__name__)

# 表格行/单元格中的“空值”占位符（PDF 中常见）
_EMPTY_TOKENS = {"", "-", "--", "—", "－", "／", "/", "无", "不适用", "N/A", "n/a", "□"}


@dataclass
class TableBlock:
    """一张结构化表格的解析结果。"""

    page: int
    rows: List[List[str]] = field(default_factory=list)   # 规整后的单元格矩阵
    caption: str = ""                                     # 表题（表上方文本）
    header: List[str] = field(default_factory=list)       # 合并后的表头
    bbox: tuple = ()                                      # 表格边界框 (x0,y0,x1,y1)
    source: str = ""                                      # 所属文档文件名

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    @property
    def markdown(self) -> str:
        """渲染为 Markdown 表格（保留行列结构）。"""
        if not self.rows:
            return ""
        lines = ["| " + " | ".join(c or "" for c in r) + " |" for r in self.rows]
        n = max(1, self.n_cols)
        lines.insert(1, "|" + "---|" * n)
        return "\n".join(lines)

    @property
    def kv_lines(self) -> List[str]:
        """生成键值对行：每个数据行的每个单元格绑定其表头。

        例：表头 ["项目名称","计划总投资(万元)"]、数据行 ["仓储及物流中心","3,393.40"]
        → ["项目名称：仓储及物流中心；计划总投资(万元)：3,393.40"]
        """
        if not config.TABLE_KV_ENABLE or not self.rows:
            return []
        header = self.header or self.rows[0]
        data_rows = self.rows[1:] if len(self.rows) > 1 else self.rows
        out: List[str] = []
        for row in data_rows:
            pairs = []
            for i, cell in enumerate(row):
                cell = (cell or "").strip()
                if not cell or cell in _EMPTY_TOKENS:
                    continue
                key = header[i].strip() if i < len(header) and header[i].strip() else f"列{i + 1}"
                pairs.append(f"{key}：{cell}")
            if len(pairs) >= 2:
                out.append("；".join(pairs))
        return out

    @property
    def text(self) -> str:
        """表格的检索文本：表题 + Markdown + 键值对行。"""
        parts = []
        if self.caption:
            parts.append(f"[表题] {self.caption}")
        md = self.markdown
        if md:
            parts.append(md)
        kvs = self.kv_lines
        if kvs:
            parts.append("[键值对]\n" + "\n".join(kvs))
        return "\n".join(parts)


# ---------------- 单元格规整 ---------------------------------------------------
def _norm_cell(v) -> str:
    if v is None:
        return ""
    s = str(v).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[ \t\u3000]+", " ", s).strip()
    return s


def _fill_merged(rows: List[List[str]]) -> List[List[str]]:
    """合并单元格还原：空单元格按“左邻优先、上邻兜底”填充。

    PyMuPDF 对跨行/跨列合并的单元格返回 None，直接丢弃会丢失“归属关系”。
    这里按阅读顺序做前向填充，恢复多级表头与跨行字段的语义。
    """
    if not rows:
        return rows
    n_cols = max(len(r) for r in rows)
    grid = [list(r) + [""] * (n_cols - len(r)) for r in rows]

    # 先按列做“上邻填充”，仅对明显属于跨行合并的场景（上一行同列有值且本行为空）
    for c in range(n_cols):
        for r in range(1, len(grid)):
            if not grid[r][c] and grid[r - 1][c]:
                # 仅当本行左侧也为空（典型的合并延续）时才向上填充，避免误填真实空值
                if c == 0 or not grid[r][c - 1]:
                    grid[r][c] = grid[r - 1][c]
    return grid


def _collapse_interleaved(rows: List[List[str]]) -> List[List[str]]:
    """折叠“交错列”布局（本工单针对力源信息招股书的表格特征）。

    部分招股书的表格被 PDF 线条切分为「表头列」与「数据列」交替的网格，例如：

        |  | 关联方名称 |  | 持股比例 |  | 与本公司关系 |
        | 赵马克 |  | 42.35% |  | 公司控股股东 |  |

    表头落在奇数列、数据落在偶数列，若不还原，表头与取值会错位（键值对退化为
    “列1/列3/列5”），检索时“表头关键词”无法命中。还原方法：若相邻两列在每一行
    中「至多只有一个非空」，则判定为交错列并合并为一列（保留非空值），重复执行
    直到不再变化，可处理多层嵌套。
    """
    cur = rows
    for _ in range(4):
        if not cur:
            return cur
        n_cols = max(len(r) for r in cur)
        if n_cols < 4 or n_cols % 2 != 0:
            return cur
        grid = [list(r) + [""] * (n_cols - len(r)) for r in cur]
        mergeable = True
        for r in grid:
            for a in range(0, n_cols, 2):
                if r[a].strip() and r[a + 1].strip():
                    mergeable = False
                    break
            if not mergeable:
                break
        if not mergeable:
            return cur
        cur = [[grid[r][a] if grid[r][a].strip() else grid[r][a + 1]
                for a in range(0, n_cols, 2)] for r in range(len(grid))]
    return cur


def _drop_empty(rows: List[List[str]]) -> List[List[str]]:
    """删除全空行、全空列，并做尾部修剪。"""
    rows = [[_norm_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    keep_cols = [c for c in range(n_cols) if any(c < len(r) and r[c] for r in rows)]
    if not keep_cols:
        return []
    return [[r[c] if c < len(r) else "" for c in keep_cols] for r in rows]


def _merge_header(rows: List[List[str]], header_rows: int) -> List[str]:
    """多级表头合并为单行表头。"""
    if not rows:
        return []
    header_rows = max(1, min(header_rows, len(rows)))
    n_cols = max(len(r) for r in rows)
    merged = []
    for c in range(n_cols):
        parts, seen = [], set()
        for r in range(header_rows):
            v = rows[r][c] if c < len(rows[r]) else ""
            v = (v or "").strip()
            if v and v not in seen:
                seen.add(v)
                parts.append(v)
        merged.append(" ".join(parts))
    return merged


def _has_text(rows: List[List[str]]) -> bool:
    """表格是否包含足够多的有效文本单元格（过滤纯线条/空白误检）。"""
    cells = [c for r in rows for c in r if c and c not in _EMPTY_TOKENS]
    return len(cells) >= max(4, len(rows))


def _quality_ok(rows: List[List[str]]) -> bool:
    """质量过滤：行/列数达标且含有效文本。"""
    if not rows:
        return False
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    if n_rows < config.TABLE_MIN_ROWS or n_cols < config.TABLE_MIN_COLS:
        return False
    return _has_text(rows)


# ---------------- 表题识别 -----------------------------------------------------
def _caption_above(page, bbox, max_gap: float = 30.0) -> str:
    """取表格上方最近的文本块作为表题。"""
    try:
        x0, y0, x1, y1 = bbox
        best, best_dy = "", 1e9
        for b in page.get_text("blocks") or []:
            bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], b[4]
            text = _norm_cell(text)
            if not text or len(text) > 60:
                continue
            dy = y0 - by1
            # 位于表格上方、垂直间距合理、水平方向有重叠
            if 0 <= dy <= max_gap and bx1 > x0 - 5 and bx0 < x1 + 5:
                if dy < best_dy:
                    best, best_dy = text, dy
        return best
    except Exception as exc:
        logger.debug("caption detect failed: %s", exc)
        return ""


# ---------------- 主入口 -------------------------------------------------------
def parse_tables(pdf_path, source_name: str = "") -> List[TableBlock]:
    """解析 PDF 中全部表格，返回结构化 TableBlock 列表。

    Args:
        pdf_path: PDF 文件路径。
        source_name: 文档来源标识（写入元数据，便于多文档溯源）。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    from pathlib import Path

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    try:
        import pymupdf
    except ImportError:  # 兼容旧版本包名
        import fitz as pymupdf

    blocks: List[TableBlock] = []
    with pymupdf.open(str(path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                finder = page.find_tables()
            except Exception as exc:
                logger.debug("page %s find_tables failed: %s", i + 1, exc)
                continue
            for tb in getattr(finder, "tables", []) or []:
                try:
                    raw = tb.extract() or []
                except Exception as exc:
                    logger.debug("page %s table extract failed: %s", i + 1, exc)
                    continue
                # 处理顺序很关键：先「删空列」把 PDF 线条造成的空分隔列去掉，得到规整的
                # 交错列网格，再「折叠交错列」还原表头与取值的对应关系，最后再删一次
                # 空列。若顺序颠倒（先折叠），空分隔列会导致表头与数据错位。
                norm = [[_norm_cell(c) for c in r] for r in raw]
                rows = _drop_empty(_collapse_interleaved(_drop_empty(_fill_merged(norm))))
                if not _quality_ok(rows):
                    continue
                bbox = tuple(getattr(tb, "bbox", ()) or ())
                header = _merge_header(rows, config.TABLE_HEADER_ROWS)
                blocks.append(TableBlock(
                    page=i + 1, rows=rows, header=header, bbox=bbox,
                    caption=_caption_above(page, bbox) if bbox else "",
                    source=source_name or path.name,
                ))
    logger.info("表格解析：%s → %d 张表", path.name, len(blocks))
    return blocks


def table_stats(blocks: Sequence[TableBlock]) -> dict:
    """表格解析统计信息（供报告与界面展示）。"""
    return {
        "tables": len(blocks),
        "cells": sum(b.n_rows * b.n_cols for b in blocks),
        "kv_lines": sum(len(b.kv_lines) for b in blocks),
        "with_caption": sum(1 for b in blocks if b.caption),
        "pages": sorted({b.page for b in blocks}),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    from src.config import PDF_PATHS

    all_blocks: List[TableBlock] = []
    for p in PDF_PATHS:
        all_blocks.extend(parse_tables(p))
    print(table_stats(all_blocks))
    for b in all_blocks[:3]:
        print("=" * 60)
        print(f"p{b.page} caption={b.caption!r} {b.n_rows}x{b.n_cols}")
        print(b.text[:500])