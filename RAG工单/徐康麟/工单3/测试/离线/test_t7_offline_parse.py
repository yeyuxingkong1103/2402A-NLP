# -*- coding: utf-8 -*-
"""离线级：表格解析与归一化锚点断言（确定性，不依赖 LLM）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

锚点来自 ``设计/需求分析.md`` §4.4/§4.5 与 ``设计/验收标准.md`` §3.1：
    * PDF1 物理 130（表头页）/131（续表页）：``logical_cols == 5``、表头无相邻 ``None``、
      单元格无换行、续表被识别并继承表头；
    * PDF1 物理 152 = 正文证据页（``tables == 0``）、153 = 示意表（辅助）；
    * PDF2 物理 22 募投 5 项 + 第 5 项占位符按「未披露」、物理 157 两张表（赵马克 / 7 家企业）；
    * **缺陷只认行内横向重复**（§3.1）；退化表不进索引是设计选择（§3.2）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from common import artifacts, assertions, paths, pdf_probe

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def _blocks_on(tables: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    """取 ``page_start`` 落在该物理页的表格块。"""
    return [b for b in tables if int(b.get("page_start", 0)) == int(page)]


def _md(block: dict[str, Any]) -> str:
    """块 markdown。"""
    return str(block.get("markdown") or "")


def test_page_counts_match_pdf(pages_pdf1: list[dict[str, Any]], pdf2_stem: str,
                              discovered_pdfs: list[Any]) -> None:
    """解析产物页数 = PDF 真实页数（PDF1 548 / PDF2 350）。"""
    counts = {p.name: pdf_probe.page_count(str(p)) for p in discovered_pdfs}
    assert len(pages_pdf1) == counts[paths.resolve_corpus("1").name]
    pdf2_pages = artifacts.load_pages(pdf2_stem)
    assert len(pdf2_pages) == counts[paths.resolve_corpus("2").name]
    assert sorted({int(r["page"]) for r in pages_pdf1}) == list(range(1, len(pages_pdf1) + 1))


def test_pdf1_page130_header_normalized(tables_pdf1: list[dict[str, Any]]) -> None:
    """物理 130：逻辑列 5（物理 10）、表头无 ``None``、单元格无换行。"""
    blocks = _blocks_on(tables_pdf1, 130)
    assert blocks, "物理 130 未提取到表格块"
    block = blocks[0]
    header = list(block.get("header") or [])
    assert None not in header, f"表头出现 None：{header}"
    assert int(block["logical_cols"]) == 5, f"逻辑列数应为 5，实测 {block['logical_cols']}"
    assert int(block["flat_cols"]) >= int(block["logical_cols"])
    assert int(block["header_row_count"]) >= 2, "物理 130 是两级表头，应被识别为 ≥2 行表头"

    bad = [cell for cell in header if "\n" in str(cell)]
    assert not bad, f"表头单元格含换行：{bad}"
    for row in block.get("rows") or []:
        bad_cells = [cell for cell in row if "\n" in str(cell)]
        assert not bad_cells, f"数据行单元格含换行：{bad_cells}"


def test_pdf1_page131_continuation_absorbed(tables_pdf1: list[dict[str, Any]]) -> None:
    """物理 131 续表：被识别为物理 130 同一逻辑块（或 ``continued_from`` 非空）。"""
    absorbed = [b for b in tables_pdf1
                if int(b.get("page_start", 0)) == 130 and 131 in [int(p) for p in b.get("absorbed_pages") or []]]
    separate = [b for b in _blocks_on(tables_pdf1, 131)
                if str(b.get("continued_from") or "").strip() and bool(b.get("header_inherited"))]
    assert absorbed or separate, "物理 131 既未被并入 130，也没有合法的续表标记（continued_from/header_inherited）"
    if absorbed:
        block = absorbed[0]
        assert int(block["page_end"]) >= 131
        assert "军种五" in _md(block), "续表首行「军种五」应出现在合并后的 markdown 中"


def test_pdf1_page152_is_text_and_153_is_table(tables_pdf1: list[dict[str, Any]],
                                              pages_pdf1: list[dict[str, Any]]) -> None:
    """物理 152 无表且含答案正文；物理 153 有示意表（引用优先 152）。"""
    assert not _blocks_on(tables_pdf1, 152), "物理 152 不应有表格块（答案在正文段）"
    page152 = next(r for r in pages_pdf1 if int(r["page"]) == 152)
    text = str(page152.get("text") or "")
    assert "电子元器件制造企业" in text and "金属壳体制造企业" in text
    assert int(page152.get("table_count", 0)) == 0
    blocks_153 = _blocks_on(tables_pdf1, 153)
    assert blocks_153, "物理 153 应有上下游示意表（辅助证据页）"
    assert _md(blocks_153[0]).strip(), "物理 153 的表 markdown 不应为空"


def test_pdf2_page22_investment_items(tables_pdf2: list[dict[str, Any]]) -> None:
    """PDF2 物理 22：募投 5 项齐全，第 5 项占位符按「未披露」（不是 0、不是 None）。"""
    blocks = _blocks_on(tables_pdf2, 22)
    target = [b for b in blocks if "仓储及物流中心" in _md(b)]
    assert target, "物理 22 未找到募投汇总表块"
    md = _md(target[0])
    for token in ("仓储及物流中心", "研发中心", "电子商务平台", "扩充产品种类和数量",
                  "其他与主营业务相关的营运资金", "3,393.40", "1,526.38", "2,492.78", "9,000.00"):
        assert token in md, f"募投表缺少要素 {token!r}"
    assert "未披露" in md, "第 5 项计划总投资原文是 [ ◆ ] → 必须按「未披露」表述"
    # 序号 5 那一行必须同时含项目名与「未披露」，且不得出现 0 值
    row5 = [line for line in md.splitlines() if "其他与主营业务相关的营运资金" in line]
    assert row5, "募投表第 5 行缺失（不得静默丢弃）"
    assert "未披露" in row5[0] and "0" not in row5[0], f"第 5 行占位符处理有误：{row5[0]!r}"
    assert int(target[0]["logical_cols"]) == 3, f"募投表逻辑列应为 3，实测 {target[0]['logical_cols']}"


def test_pdf2_page24_placeholder_not_zero(tables_pdf2: list[dict[str, Any]]) -> None:
    """PDF2 物理 24：发行费用概算表的 ``[ ◆ ]`` 归一化为「未披露」，不得变成 0。"""
    blocks = _blocks_on(tables_pdf2, 24)
    target = [b for b in blocks if "承销费" in _md(b)]
    assert target, "物理 24 未找到发行费用概算表"
    md = _md(target[0])
    assert "未披露" in md and "◆" not in md, f"占位符未按「未披露」归一化：{md[:120]!r}"
    assert "0.00" not in md, "占位符不得被当作 0"


def test_pdf2_page157_two_tables_present(tables_pdf2: list[dict[str, Any]]) -> None:
    """PDF2 物理 157：关联方两表内容齐全（赵马克 42.35% / 7 家企业）。"""
    blocks = _blocks_on(tables_pdf2, 157)
    assert len(blocks) >= 1, "物理 157 未提取到表格块"
    joined = "\n".join(_md(b) for b in blocks)
    for token in ("赵马克", "42.35%", "公司控股股东"):
        assert token in joined, f"物理 157 表格缺少 {token!r}"
    for company in ("融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"):
        assert company in joined, f"物理 157 表格缺少企业 {company!r}"


def test_no_horizontal_row_duplication_on_anchor_tables(tables_pdf1: list[dict[str, Any]],
                                                        tables_pdf2: list[dict[str, Any]]) -> None:
    """锚点表「行内横向重复数 = 0」（**只认行内**；纵向同值不设门槛，§3.1）。"""
    anchors: list[tuple[str, dict[str, Any]]] = []
    anchors += [("PDF1 p130", b) for b in _blocks_on(tables_pdf1, 130)]
    anchors += [("PDF1 p153", b) for b in _blocks_on(tables_pdf1, 153)]
    anchors += [("PDF2 p22", b) for b in _blocks_on(tables_pdf2, 22)]
    anchors += [("PDF2 p157", b) for b in _blocks_on(tables_pdf2, 157)]
    assert anchors, "锚点表集合为空（解析产物缺失）"
    problems: list[str] = []
    for label, block in anchors:
        for item in assertions.table_horizontal_defects(block):
            if item.defect:
                problems.append(f"{label} {block.get('table_id')} 行{item.row_index}列{item.col_index}={item.value!r}")
    assert not problems, "出现行内横向重复（合并单元格复制填充）：\n" + "\n".join(problems)


def test_degenerate_tables_absent_from_evidence_pages_and_index(
    tables_pdf1: list[dict[str, Any]], tables_pdf2: list[dict[str, Any]],
    chunks_all: list[dict[str, Any]], golden_items: list[Any],
) -> None:
    """§3.2：**captain 复核过的证据页上退化表 = 0**，且退化表 ``table_id`` 不进索引。

    口径说明（避免把设计选择判成缺陷）：
        * 门槛页 = ``验收标准.md`` §3.2 列出的那批页（captain 逐页比对过，结论为 0）；
        * golden 里额外出现的**延伸证据页**（如 PDF2 物理 259、PDF1 物理 332）若落到退化表页上，
          只**留痕记录**、不判缺陷 —— 题 3/4 的主锚点是 157、题 957 的答案正文在 154，
          退化表本就不进索引（§3.3），把它们当缺陷等于用设计选择判实现失败。
    """
    # captain 独立验证过的证据页清单（门槛）
    captain_pages = {
        1: {22, 27, 30, 129, 130, 131, 151, 152, 153, 154, 157, 160, 479, 490},
        2: {21, 22, 24, 30, 31, 144, 157, 158, 306, 340},
    }
    # golden 声明的全部证据页（含延伸页，仅作记录）
    golden_pages: dict[int, set[int]] = {1: set(), 2: set()}
    for item in golden_items:
        key = 2 if item.corpus == "pdf2" else 1
        golden_pages[key].update(int(p) for p in item.evidence_pages)

    gate_hits: list[str] = []
    extra_hits: list[str] = []
    degenerate_ids: set[str] = set()
    for corpus, tables in (("PDF1", tables_pdf1), ("PDF2", tables_pdf2)):
        key = 1 if corpus == "PDF1" else 2
        for block in tables:
            if not bool(block.get("degenerate")):
                continue
            degenerate_ids.add(str(block.get("table_id")))
            covered = {int(block.get("page_start", 0)), int(block.get("page_end", 0))}
            covered.update(int(p) for p in block.get("row_pages") or [])
            covered.update(int(p) for p in block.get("absorbed_pages") or [])
            label = f"{corpus} {block.get('table_id')} 覆盖 {sorted(covered)}"
            if covered & captain_pages[key]:
                gate_hits.append(label)
            elif covered & golden_pages[key]:
                extra_hits.append(label)

    assert not gate_hits, ("captain 复核的证据页上出现退化表（与 §3.2 不符）：\n"
                           + "\n".join(gate_hits))
    if extra_hits:
        paths.ensure_trace_dir()
        (paths.TRACE_DIR / "degenerate_on_extended_evidence_pages.json").write_text(
            json.dumps({
                "work_order": assertions.WORK_ORDER,
                "note": ("以下退化表落在 golden 声明的**延伸**证据页上（不在 captain 门槛清单内）："
                         "只作留痕，按 §3.3 不判缺陷；如需升级为门槛请由 captain 裁定"),
                "ragas": assertions.RAGAS_BANNER,
                "items": extra_hits,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("延伸证据页上的退化表（留痕，不判缺陷）：", extra_hits)

    in_index = {str(c.get("table_id")) for c in chunks_all if c.get("table_id")} & degenerate_ids
    assert not in_index, f"退化表进入了索引：{sorted(in_index)}"


def test_table_markdown_nonempty_rate_recorded(tables_pdf1: list[dict[str, Any]],
                                              tables_pdf2: list[dict[str, Any]]) -> None:
    """记录「表块 markdown 非空率」基线（§3.2 明示：只作留痕，不作通过门槛）。"""
    total = len(tables_pdf1) + len(tables_pdf2)
    nonempty = sum(1 for b in list(tables_pdf1) + list(tables_pdf2) if _md(b).strip())
    assert total > 0
    paths.ensure_trace_dir()
    target = paths.TRACE_DIR / "baseline_table_markdown.json"
    target.write_text(json.dumps({
        "work_order": assertions.WORK_ORDER,
        "metric": "table markdown 非空率（基线值，非门槛）",
        "total": total, "nonempty": nonempty,
        "rate": round(nonempty / total, 4),
        "source": "设计/验收标准.md §3.2（基线 615/668 = 92.1%）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert 0.0 < nonempty / total <= 1.0
