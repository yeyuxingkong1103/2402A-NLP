# -*- coding: utf-8 -*-
"""PDF 表格分析调用模块（pdfplumber）。

基于 pdfplumber 抽取 PDF 中的表格，并做轻量分析：识别表头、清洗空行/空列、
转成「记录字典」或 CSV/JSON 落盘，方便在 RAG 流程中对表格型 PDF 结构化取数。

实现要点：

1. **懒加载**：顶层不 ``import pdfplumber``，首次真正抽取时才加载，导入零开销；
2. **抽表**：对每页调用 ``page.find_tables(...)`` 拿到带坐标的表格，再 ``.extract()`` 取行；
   默认用 ``lines`` 策略（框线），找不到时自动回退 ``text`` 策略（空白对齐）；
3. **分析**：识别表头（首个非空行）、去掉全空行/全空列、统计行列数；
4. **落盘**：每张表可存为 CSV / JSON，多表可一次性导出，或（可选）转 pandas DataFrame。

运行环境：

    使用已安装 pdfplumber 的 rags_ 环境（或 base）运行，例如：

        D:/an/envs/rags_/python.exe pdf_table.py <xx.pdf>
        D:/an/envs/rags_/python.exe -c "from pdf_table import extract_tables; ts = extract_tables('a.pdf')"

用法示例（详见同目录 README.md）：

    from pdf_table import extract_tables, analyze_pdf

    tables = extract_tables("报表.pdf")              # 抽取所有表格
    for t in tables:
        print(t.page_number, t.n_rows, t.n_cols)     # 页码 / 行列数
        print(t.header)                              # 表头
        print(t.to_dicts())                          # [{列名: 值, ...}, ...]

    info = analyze_pdf("报表.pdf")                    # 汇总：每页表格数、表头、行列数
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag2.pdf_table")

# 表格抽取策略 → pdfplumber 的 vertical/horizontal 参数组合
_STRATEGIES: dict[str, dict[str, str]] = {
    "lines": {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    "lines_strict": {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
    "text": {"vertical_strategy": "text", "horizontal_strategy": "text"},
    "explicit": {"vertical_strategy": "explicit", "horizontal_strategy": "explicit"},
}

# 默认表格设置（lines 策略 + 容差）
DEFAULT_TABLE_SETTINGS: dict[str, Any] = {
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "edge_min_length": 3,
}


def _parse_pages(pages: int | list[int] | str | None, total: int) -> list[int]:
    """把页码参数规整成 1 起始的页码列表。

    参数：
        pages: None/\"all\" → 全部；int → 单页；list[int] → 指定页；\"1-10,12\" → 范围+散页。
        total: PDF 总页数。
    """
    if pages is None or pages == "all":
        return list(range(1, total + 1))
    if isinstance(pages, int):
        return [pages]
    if isinstance(pages, str):
        result: list[int] = []
        for part in re.split(r"[,，\s]+", pages.strip()):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                result.extend(range(int(a), int(b) + 1))
            else:
                result.append(int(part))
        return result
    return list(pages)


def _clean_rows(rows: list[list[str | None]]) -> list[list[str | None]]:
    """去掉全空行与全空列（值统一为 None 或空白字符串判定）。"""
    if not rows:
        return rows

    def blank(v: str | None) -> bool:
        return v is None or str(v).strip() == ""

    # 去全空行
    rows = [r for r in rows if any(not blank(c) for c in r)]
    if not rows:
        return rows

    n_cols = max(len(r) for r in rows)
    # 补齐列宽，再去全空列
    padded = [list(r) + [None] * (n_cols - len(r)) for r in rows]
    keep = [i for i in range(n_cols) if any(not blank(r[i]) for r in padded)]
    return [[r[i] for i in keep] for r in padded]


def _stringify(v: str | None) -> str:
    """单元格取值 → 字符串（None/空 → 空串，去首尾空白）。"""
    return "" if v is None else str(v).strip()


@dataclass
class TableResult:
    """单张 PDF 表格的分析结果。"""

    page_number: int
    rows: list[list[str | None]]
    bbox: tuple | None = None          # (x0, top, x1, bottom)，页面坐标
    index: int = 0                     # 该页内第几张表（0 起）
    _header_row: int = 0               # 表头所在行（默认首行）

    # ------------------------------------------------------------------ 基本属性
    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        """表头（默认首个非空行）。"""
        for i, row in enumerate(self.rows):
            if any(_stringify(c) for c in row):
                self._header_row = i
                break
        if self.n_rows > self._header_row:
            return [_stringify(c) for c in self.rows[self._header_row]]
        return []

    @property
    def data_rows(self) -> list[list[str | None]]:
        """去掉表头后的数据行。"""
        return self.rows[self._header_row + 1:]

    def clean(self) -> "TableResult":
        """返回清洗后的副本（去全空行/全空列）。"""
        return TableResult(self.page_number, _clean_rows(self.rows), self.bbox, self.index)

    # ------------------------------------------------------------------ 导出
    def to_dicts(self, header: list[str] | None = None) -> list[dict]:
        """把数据行转成记录字典列表（列名来自表头）。

        参数：
            header: 自定义列名；缺省用表头行。列名重复时自动加 ``_2``、``_3`` 去重。
        """
        names = list(header) if header else self.header
        if not names:
            return [{"col%d" % (i + 1): _stringify(c) for i, c in enumerate(r)} for r in self.data_rows]
        seen: dict[str, int] = {}
        unique: list[str] = []
        for n in names:
            key = n if n else "col"
            if key in seen:
                seen[key] += 1
                key = f"{key}_{seen[key]}"
            else:
                seen[key] = 1
            unique.append(key)
        result = []
        for row in self.data_rows:
            record = {}
            for i, name in enumerate(unique):
                record[name] = _stringify(row[i]) if i < len(row) else ""
            result.append(record)
        return result

    def to_json(self, ensure_ascii: bool = False) -> str:
        """序列化为 JSON 字符串（含页码与记录列表）。"""
        payload = {
            "page_number": self.page_number,
            "index": self.index,
            "bbox": list(self.bbox) if self.bbox else None,
            "header": self.header,
            "rows": self.to_dicts(),
        }
        return json.dumps(payload, ensure_ascii=ensure_ascii, indent=2)

    def to_csv(self) -> str:
        """序列化为 CSV 字符串（含表头行）。"""
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in self.rows:
            writer.writerow([_stringify(c) for c in row])
        return buf.getvalue()

    def to_dataframe(self):  # 可选：依赖 pandas
        """转成 pandas.DataFrame（未安装 pandas 时抛出 ImportError）。"""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - 依赖缺失提示
            raise ImportError("未安装 pandas，无法转 DataFrame（本模块默认仅需 pdfplumber）。") from exc
        if self.header:
            return pd.DataFrame(self.data_rows, columns=self.header)
        return pd.DataFrame(self.rows)

    def __len__(self) -> int:
        return self.n_rows

    def __iter__(self):
        return iter(self.rows)

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        return f"TableResult(第{self.page_number}页#{self.index}, {self.n_rows}行x{self.n_cols}列)"


# ---------------------------------------------------------------------- 主类
class PDFTableExtractor:
    """pdfplumber 表格抽取器（懒加载，可复用）。"""

    def __init__(
        self,
        strategy: str = "lines",
        fallback_to_text: bool = True,
        table_settings: dict[str, Any] | None = None,
        clean: bool = True,
    ) -> None:
        """初始化。

        参数：
            strategy:          抽取策略 lines / lines_strict / text / explicit，默认 lines。
            fallback_to_text:  lines 策略抽不到表时，是否自动回退 text 策略再试一次。
            table_settings:    额外覆盖 pdfplumber 的 TableSettings（优先级最高）。
            clean:             是否自动去掉全空行/全空列。
        """
        if strategy not in _STRATEGIES:
            raise ValueError(f"未知策略：{strategy}，可选 {list(_STRATEGIES)}")
        self.strategy = strategy
        self.fallback_to_text = fallback_to_text
        self.table_settings = dict(table_settings or {})
        self.clean = clean

    # ------------------------------------------------------------------ 底层
    def _open(self, pdf_path: str | Path):
        """懒加载 pdfplumber 并打开 PDF。"""
        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - 依赖缺失提示
            raise ImportError(
                "未找到 pdfplumber，请使用 rags_ 环境运行：D:/an/envs/rags_/python.exe。"
            ) from exc
        return pdfplumber.open(Path(pdf_path))

    def _settings(self, strategy: str) -> dict[str, Any]:
        settings = dict(DEFAULT_TABLE_SETTINGS)
        settings.update(_STRATEGIES[strategy])
        settings.update(self.table_settings)
        return settings

    # ------------------------------------------------------------------ 抽取
    def extract_page(self, page, page_number: int) -> list[TableResult]:
        """从单个 pdfplumber.Page 抽取表格。"""
        tables: list[TableResult] = []
        found = page.find_tables(self._settings(self.strategy))
        if not found and self.fallback_to_text and self.strategy != "text":
            logger.debug("第 %d 页 lines 策略未抽到表格，回退 text 策略", page_number)
            found = page.find_tables(self._settings("text"))
        for i, tb in enumerate(found):
            rows = tb.extract() or []
            rows = _clean_rows(rows) if self.clean else rows
            if rows:
                tables.append(
                    TableResult(page_number, rows, tuple(tb.bbox), i)
                )
        return tables

    def extract(
        self,
        pdf_path: str | Path,
        pages: int | list[int] | str | None = None,
    ) -> list[TableResult]:
        """抽取 PDF 中的全部（或指定页）表格。

        参数：
            pdf_path: PDF 路径。
            pages:    页码范围，None/"all" 全部；int 单页；list[int] 多页；"1-10,12" 范围+散页。

        返回：
            List[TableResult]，按页序排列。
        """
        pdf = Path(pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"PDF 不存在：{pdf}")
        result: list[TableResult] = []
        with self._open(pdf) as doc:
            page_numbers = _parse_pages(pages, len(doc.pages))
            for n in page_numbers:
                if not 1 <= n <= len(doc.pages):
                    logger.warning("页码越界，跳过：%d", n)
                    continue
                result.extend(self.extract_page(doc.pages[n - 1], n))
        logger.info("抽取完成：%s 共 %d 张表", pdf.name, len(result))
        return result


# ---------------------------------------------------------------------- 便捷函数
def extract_tables(
    pdf_path: str | Path,
    pages: int | list[int] | str | None = None,
    *,
    strategy: str = "lines",
    fallback_to_text: bool = True,
    table_settings: dict[str, Any] | None = None,
    clean: bool = True,
) -> list[TableResult]:
    """便捷函数：抽取 PDF 表格（等价于 ``PDFTableExtractor(...).extract(...)``）。"""
    return PDFTableExtractor(
        strategy=strategy,
        fallback_to_text=fallback_to_text,
        table_settings=table_settings,
        clean=clean,
    ).extract(pdf_path, pages=pages)


def analyze_pdf(
    pdf_path: str | Path,
    pages: int | list[int] | str | None = None,
    **kwargs: Any,
) -> dict:
    """汇总 PDF 的表格结构：每页表格数、各表表头与行列数。"""
    tables = extract_tables(pdf_path, pages=pages, **kwargs)
    per_page: dict[str, list] = {}
    for t in tables:
        per_page.setdefault(t.page_number, []).append(
            {"index": t.index, "header": t.header, "n_rows": t.n_rows, "n_cols": t.n_cols}
        )
    return {
        "file": str(Path(pdf_path).resolve()),
        "table_count": len(tables),
        "pages": per_page,
    }


def tables_to_csv(
    tables: list[TableResult],
    output_dir: str | Path,
    stem: str | None = None,
) -> list[str]:
    """把多张表保存为 CSV（每表一个文件），返回保存路径列表。

    文件名：``<stem>_p<页码>_t<序号>.csv``；未指定 stem 时用 "table"。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or "table"
    paths: list[str] = []
    for t in tables:
        name = f"{stem}_p{t.page_number}_t{t.index}.csv"
        p = out / name
        p.write_text(t.to_csv(), encoding="utf-8-sig")  # 带 BOM，Excel 打开不乱码
        paths.append(str(p))
    logger.info("已导出 %d 个 CSV 到 %s", len(tables), out)
    return paths


def tables_to_json(
    tables: list[TableResult],
    output_path: str | Path,
) -> str:
    """把所有表合并写入单个 JSON 文件，返回文件路径。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(t.to_json()) for t in tables]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已导出 %d 张表到 %s", len(tables), out)
    return str(out)


__all__ = [
    "TableResult",
    "PDFTableExtractor",
    "extract_tables",
    "analyze_pdf",
    "tables_to_csv",
    "tables_to_json",
]


if __name__ == "__main__":  # pragma: no cover - 需 rags_ 环境与真实 PDF
    import sys

    if len(sys.argv) < 2:
        print("用法：D:/an/envs/rags_/python.exe pdf_table.py <pdf路径> [页码]")
        print("示例：D:/an/envs/rags_/python.exe pdf_table.py 报表.pdf 1-10")
        sys.exit(0)

    pdf = sys.argv[1]
    pages = sys.argv[2] if len(sys.argv) > 2 else None
    tables = extract_tables(pdf, pages=pages)
    print(f"共抽取 {len(tables)} 张表：")
    for t in tables:
        print(" ", t)
        print("   表头:", t.header)
        print("   首行:", t.to_dicts()[:1])
