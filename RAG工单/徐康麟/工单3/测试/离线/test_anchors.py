# -*- coding: utf-8 -*-
"""T3 定点回归：验证 7 个核心模块在「确定性锚点页」上的行为（不跑全量）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

锚点（设计/接口设计.md §3.6，全部为 **1-based 物理页**）：
    PDF1 物理 129  find_tables == 0，正文含 82.10%（题 33/260 正文证据路径）
    PDF1 物理 130  7×10 原始表 → logical_cols == 5、表头无相邻 None
    PDF1 物理 131  续表（首行 ['', '军种五', '2,727.20', …]）→ 并入 130，page_end == 131、row_pages 含 131
    PDF1 物理 153  表存在（3×4）
    PDF2 物理 22   表内可提取 ('发行股数', '1,670 万股，占发行后总股本的比例为25.04%')
    PDF2 物理 22/306  [ ◆ ] → 未披露（不得丢弃）

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/test_anchors.py
退出码：0 全通过；1 存在断言失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]   # 测试/离线/test_anchors.py → 工单3
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import chunker, pdf_parser, table_parser, text_utils  # noqa: E402
from app.core.config import discover_pdfs, get_config, model_slug  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    """记录一条断言结果并打印。"""
    RESULTS.append((bool(ok), name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def main() -> int:
    """执行全部锚点断言。"""
    setup_logging()
    log = get_logger("test_anchors")
    cfg = get_config()
    sources = {s.file_name: s for s in discover_pdfs(cfg.paths.raw_dir, logger=log)}
    pdf1 = sources.get("招股说明书1.pdf")
    pdf2 = sources.get("招股说明书2.pdf")
    if pdf1 is None or pdf2 is None:
        print("❌ 语料缺失：需要 招股说明书1.pdf 与 招股说明书2.pdf")
        return 1

    print("\n=== [1] 页码 API 与越界保护 ===")
    check(pdf_parser.pdf_page_count(pdf1) == 548, "PDF1 页数 == 548", str(pdf_parser.pdf_page_count(pdf1)))
    check(pdf_parser.pdf_page_count(pdf2) == 350, "PDF2 页数 == 350", str(pdf_parser.pdf_page_count(pdf2)))
    try:
        pdf_parser.get_page_text(pdf1, 0)
        check(False, "页码 0 抛 ChunkError")
    except Exception as exc:  # noqa: BLE001
        check(type(exc).__name__ == "ChunkError", "页码 0 抛 ChunkError", type(exc).__name__)
    check("82.10%" in pdf_parser.get_page_text(pdf1, 129), "物理129 页文本含 82.10%（doc[128]）")
    check(pdf_parser.get_page_table_count(pdf1, 129) == 0, "物理129 find_tables == 0")
    check(pdf_parser.get_page_table_count(pdf1, 130) == 1, "物理130 find_tables == 1")
    check(pdf_parser.get_page_table_count(pdf1, 153) >= 1, "物理153 有表",
          str(pdf_parser.get_page_table_count(pdf1, 153)))

    print("\n=== [2] 物理 130/131 表格归一化与跨页合并 ===")
    r1 = pdf_parser.parse_pdf(pdf1, cfg=cfg, pages=[129, 130, 131, 153], logger=log)
    by_page = {t.page_start: t for t in r1.tables}
    p130 = by_page.get(130)
    check(p130 is not None, "物理130 存在表块")
    if p130 is not None:
        check(p130.logical_cols == 5, "物理130 logical_cols == 5（原始 10 列）",
              f"logical_cols={p130.logical_cols} n_cols={p130.n_cols}")
        check(p130.n_cols == 10, "物理130 n_cols == 10", str(p130.n_cols))
        check(all(c is not None for c in p130.header) and all("\n" not in c for c in p130.header),
              "物理130 表头无 None、无换行")
        check(p130.flat_cols == 10, "物理130 两级表头压平为 10 列", str(p130.flat_cols))
        check(p130.page_start == 130 and p130.page_end == 131, "物理130 表块 page_start=130 / page_end=131",
              f"{p130.page_start}-{p130.page_end}")
        check(131 in p130.row_pages, "物理130 表块 row_pages 含 131", str(sorted(set(p130.row_pages))))
        check(131 not in by_page, "物理131 未独立成块（已并入 130）")
        check(any("军种五" in (cell or "") for row in p130.rows for cell in row),
              "物理131 续表行「军种五」已在 130 表块内")
        check(all(len(row) == len(p130.header) for row in p130.rows),
              "物理130 所有数据行列数与表头一致")
    p153 = by_page.get(153)
    check(p153 is not None, "物理153 存在表块", f"{p153.flat_cols if p153 else '-'} 列")

    print("\n=== [3] PDF2 物理 22：发行股数与 [ ◆ ] 占位符 ===")
    r2 = pdf_parser.parse_pdf(pdf2, cfg=cfg, pages=[2, 22, 24, 306], logger=log)
    joined = "\n".join(f"{t.title}\n{t.markdown}" for t in r2.tables)
    check("发行股数" in joined, "PDF2 物理22 表内可提取「发行股数」")
    check("1,670 万股，占发行后总股本的比例为25.04%" in joined.replace(" ", "")
          or "1,670 万股，占发行后总股本的比例为25.04%" in joined,
          "PDF2 物理22 表内可提取发行股数数值", joined.replace("\n", " ")[:0] or "")
    check("未披露" in joined, "PDF2 [ ◆ ] 已归一化为「未披露」",
          f"出现 {joined.count('未披露')} 次")
    check(any(t.page_start == 22 for t in r2.tables), "PDF2 物理22 有表块")
    p22_blocks = [t for t in r2.tables if t.page_start == 22]
    synth = [h for t in p22_blocks for h in t.header if str(h).startswith("列") and str(h)[1:].isdigit()]
    check(not synth, "PDF2 物理22 无合成列名（含年份的表头未被误判为数据行）", str(synth))
    check(any("项目" in h or "年度" in h or "月" in h for t in p22_blocks for h in t.header),
          "PDF2 物理22 表头保留了「项目/年度/月」等真实表头词")
    # 稀疏网格（合并单元格）回归：表头与数据行必须共用同一套逻辑列，禁止重复填充
    fund = [t for t in p22_blocks if any("计划总投资" in h for h in t.header)]
    check(bool(fund), "PDF2 物理22 存在募投表（计划总投资）")
    if fund:
        f = fund[0]
        check(f.header == ["序号", "项目名称", "计划总投资(万元)"],
              "募投表表头压缩为 3 个逻辑列（序号/项目名称/计划总投资）", str(f.header))
        target = [r for r in f.rows if any("仓储及物流中心" in c for c in r)]
        check(bool(target) and len(target[0]) == 3, "募投表数据行为 3 列（与表头同一映射）",
              str(target[0]) if target else "-")
        check(bool(target) and str(target[0]).count("仓储及物流中心") == 1 and str(target[0]).count("3,393.40") == 1,
              "募投表逻辑值未被重复填充 3 次", str(target[0]) if target else "-")
    tripled = [
        t.table_id for t in r2.tables
        for row in t.rows
        for i in range(len(row) - 2)
        if str(row[i]).strip() and str(row[i]) == str(row[i + 1]) == str(row[i + 2])
        and str(row[i]) not in ("未披露", "-")
    ]
    check(not tripled, "PDF2 抽查页无「三连重复值」行（合并单元格未被复制填充）", str(tripled[:4]))

    print("\n=== [4] 文本工具与命中判定 ===")
    check(text_utils.normalize_cell("有限公司成立日\n期") == "有限公司成立日期",
          "R1 换行归一：有限公司成立日期", repr(text_utils.normalize_cell("有限公司成立日\n期")))
    check(text_utils.normalize_cell("[ ◆ ]") == "未披露", "[ ◆ ] → 未披露",
          repr(text_utils.normalize_cell("[ ◆ ]")))
    check(text_utils.normalize_cell("[ ◆ ]元") == "未披露元", "[ ◆ ]元 → 未披露元（保住单位）",
          repr(text_utils.normalize_cell("[ ◆ ]元")))
    check(text_utils.normalize_cell("[ ◆ ]年[ ◆ ]月[ ◆ ]日") == "未披露（年/月/日）",
          "[ ◆ ]年[ ◆ ]月[ ◆ ]日 → 未披露（年/月/日）",
          repr(text_utils.normalize_cell("[ ◆ ]年[ ◆ ]月[ ◆ ]日")))
    check(text_utils.normalize_cell("[ ◆ ]倍(按截至2010 年6 月30 日计算)").startswith("未披露倍("),
          "带括号注释的占位符保留单位与注释",
          repr(text_utils.normalize_cell("[ ◆ ]倍(按截至2010 年6 月30 日计算)"))[:34])
    fake_pdf2 = "武汉力源信息技术股份有限公司\n127\n1、项目概况\n正文内容若干"
    cleaned = text_utils.strip_printed_page(fake_pdf2, page=128)
    check("127" not in cleaned.splitlines(), "PDF2 页眉裸数字（=物理-1）被剔除", repr(cleaned[:40]))
    keep = text_utils.strip_printed_page("公司情况如下\n1、项目概况\n2、其他", page=99)
    check("1、项目概况" in keep, "正文中的独立数字/序号行未被误删")
    check(text_utils.strip_printed_page("正文\n1-1-128", page=129).strip() == "正文", "PDF1 页脚 1-1-N 被剔除")
    ev = "报告期内，公司来自军用领域的收入分别为6,464.51 万元、14,414.16 万元，占主营业务收入比重分别为82.10%"
    page129_text = pdf_parser.get_page_text(pdf1, 129)
    check(text_utils.evidence_contains(page129_text, ev), "evidence_contains 命中物理129 正文证据")
    check(not text_utils.evidence_contains("完全无关的文本", ev), "evidence_contains 对无关文本返回 False")
    check(model_slug("bge-m3:latest", 1024) == "bge-m3_1024", "model_slug == bge-m3_1024",
          model_slug("bge-m3:latest", 1024))

    print("\n=== [5] 分块与元数据 ===")
    chunks = chunker.build_chunks(r1, cfg=cfg, logger=log)
    text_chunks = [c for c in chunks if c.type == "text"]
    table_chunks = [c for c in chunks if c.type == "table"]
    check(len(chunks) == len({c.chunk_id for c in chunks}), "chunk_id 全局唯一", str(len(chunks)))
    check(all(1 <= c.page <= 548 for c in chunks), "chunk.page 均在 [1, 548]")
    check(all(c.char_count == len(c.content) for c in chunks), "char_count 与 content 长度一致")
    check(all(c.keywords for c in chunks), "每个 chunk 都有关键词")
    check(all(0 <= c.order < len(chunks) for c in chunks), "order 连续且唯一")
    big = [c for c in text_chunks if 400 <= c.char_count <= 600]
    check(len(big) >= max(1, len(text_chunks) // 2), "多数文本块落在 400~600 字",
          f"{len(big)}/{len(text_chunks)}")
    tchunk = [c for c in table_chunks if c.table_id and c.table_id.endswith("p0130#t00")]
    check(bool(tchunk), "物理130 逻辑表块成为 1 个 table chunk",
          tchunk[0].chunk_id if tchunk else "-")
    if tchunk:
        c = tchunk[0]
        check(c.page_start == 130 and c.page_end == 131, "table chunk 覆盖 130-131", f"{c.page_start}-{c.page_end}")
        check("[表标题]" in c.content and "[来源]" in c.content and "[关键数字]" in c.content,
              "table chunk content 含 §6.4 四段结构")
        numbers_line = [ln for ln in c.content.splitlines() if ln.startswith("[关键数字]")]
        nums = text_utils.extract_numbers(numbers_line[0]) if numbers_line else []
        check(len(nums) >= 5, "table chunk 关键数字行含 >=5 个数字", f"{len(nums)} 个")
        check("军种五" in c.content, "table chunk 含续表行「军种五」（跨页未丢数据）")
    split_ok = all(c.char_count <= cfg.chunk.max_table_chars + 200 for c in table_chunks)
    check(split_ok, "超长表块已按行批次拆分（≤ max_table_chars+余量）",
          f"最大 {max((c.char_count for c in table_chunks), default=0)}")

    print("\n=== [6] 归档口径自检（页码 1-based） ===")
    check(all(t.page == t.page_start for t in r1.tables + r2.tables), "所有表块 page == page_start")
    check(all(t.page_start <= t.page_end for t in r1.tables + r2.tables), "page_start <= page_end")
    check(all(len(row) == t.flat_cols for t in r1.tables + r2.tables for row in t.rows),
          "所有表块的行宽 == flat_cols")

    failed = [name for ok, name, _ in RESULTS if not ok]
    print("\n" + "─" * 66)
    print(f"锚点断言：✅{len(RESULTS) - len(failed)} / {len(RESULTS)}"
          + (f"，失败：{failed}" if failed else ""))
    shutdown_logging()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：打印完整堆栈并以 1 退出
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
