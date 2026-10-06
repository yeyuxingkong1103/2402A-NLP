"""入库链路编排。只负责串流程与产出物落地，不含任何业务规则
（规则在 structure/chunk/verify_ingest 里，都可单测）。

与技术方案 8.1 的差异：8.1 把编排放在 app/chains/ingest_chain.py，但本期不引入
LangChain（不向量化、无链式组件），用纯编排脚本更符合 2.2 的"链只做编排"原则。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict

import pypdf

from app.ingest.chunk import build_chunks
from app.ingest.ocr_paddle import low_quality_pages
from app.ingest.parse_mineru import locate_product_md, run_mineru
from app.ingest.structure import restore_articles
from app.ingest.to_docx import md_to_docx
from app.ingest.verify_ingest import (
    EXPECTED_TOTAL, IngestFailed, check_continuity, check_fields,
    check_total, check_verbatim,
)

# 目录约定（技术方案 7.4）：raw 只读永久留档、interim 可删可重建、parsed 是入库成品
ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PARSED_DIR = ROOT / "data" / "parsed"

# (源文件名, 区间下界, 区间上界)。区间来自官方结构，三册首尾相接共 1260 条
VOLUMES = [
    ("中华人民共和国民法典（上册）.pdf", 1, 462),
    ("中华人民共和国民法典（中册）.md", 463, 1039),
    ("中华人民共和国民法典（下册）.docx", 1040, 1260),
]


def assign_ranges(numbers: list[int], lo: int, hi: int) -> list[int]:
    """只保留落在本册区间内的条号。"""
    return sorted(n for n in numbers if lo <= n <= hi)


def drop_boundary(numbers: list[int], lo: int, hi: int) -> tuple[list[int], list[int]]:
    """切出本册区间内的条号，并显式返回被丢弃的。

    上册 PDF 会多出 463~466（按页抽取导致边界共用页带上第三编开头），必须丢弃
    以免与中册重叠；丢弃清单要进报告，不能静默（《数据来源说明》定的口径
    是"严格不重叠以 md/docx 为准"）。
    """
    kept = assign_ranges(numbers, lo, hi)
    dropped = sorted(set(numbers) - set(kept))
    return kept, dropped


def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    """抽取 PDF 文本层，供逐字命中校验用。"""
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse(src: pathlib.Path, work_dir: pathlib.Path, synthetic: pathlib.Path,
           report: list[str]) -> pathlib.Path:
    """按输入类型选解析路径，返回 MinerU 产物 markdown 路径。"""
    parse_src = src
    if src.suffix.lower() == ".md":
        # MinerU 的 -p 只认 pdf/image/docx/pptx/xlsx，md 必须先合成 docx
        parse_src = md_to_docx(src, synthetic / (src.stem + ".docx"))
        report.append(f"  - 输入是 md，MinerU 不支持该格式，已合成 {parse_src.name} 后解析")
    run_mineru(parse_src, work_dir)
    return locate_product_md(work_dir)


def main(outdir: pathlib.Path = PARSED_DIR,
         interim: pathlib.Path = INTERIM_DIR) -> int:
    """跑完整条入库链路。返回进程退出码。

    outdir 只落入库成品；MinerU 产物与合成的 docx 一律进 interim。两者分开，
    是为了让"删掉中间产物重跑"不会碰到成品（技术方案 7.4）。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    synthetic = interim / "_synthetic"
    report: list[str] = [f"## 分册明细\n"]
    all_articles = []
    all_numbers: list[int] = []
    all_chunks = []

    for filename, lo, hi in VOLUMES:
        src = DATA_DIR / filename
        report.append(f"\n### {filename}（区间 {lo}-{hi}）")
        product = _parse(src, interim / src.stem, synthetic, report)

        md_text = product.read_text(encoding="utf-8")
        flagged = low_quality_pages(md_text)
        report.append(f"  - 低质页：{flagged or '无'}（PaddleOCR-VL 触发式兜底，无则不跑）")

        articles = restore_articles(product)
        kept, dropped = drop_boundary([a.number for a in articles], lo, hi)
        if dropped:
            report.append(f"  - 丢弃区间外条号 {dropped}（边界共用页，已由邻册覆盖）")
        keep_set = set(kept)
        articles = [a for a in articles if a.number in keep_set]

        missing = check_continuity(kept, lo, hi)
        if missing:
            raise IngestFailed(f"{filename} 条号不连续，缺号：{missing}")
        field_problems = check_fields(articles)
        if field_problems:
            raise IngestFailed(f"{filename} 字段非空校验失败：{field_problems}")
        report.append(f"  - 条号 {len(kept)} 条，无缺号 ✅")

        # 逐字命中：只有上册有 PDF 可比对，中/下册 PDF 在回收站未恢复（见设计文档第九节）
        if src.suffix.lower() == ".pdf":
            unmatched = check_verbatim(articles, extract_pdf_text(src))
            report.append(f"  - 逐字命中：{len(articles) - len(unmatched)}/{len(articles)}"
                          + ("" if not unmatched else f"，未命中 {unmatched}"))

        all_articles.extend(articles)
        all_numbers.extend(kept)
        all_chunks.extend(build_chunks(articles))

    check_total(all_numbers, expect=EXPECTED_TOTAL)
    report.append(f"\n## 合计\n\n- 条数 {len(set(all_numbers))} / {EXPECTED_TOTAL} ✅")
    report.append(f"- 父子块 {len(all_chunks)} 块")

    (outdir / "law_articles.jsonl").write_text(
        "\n".join(json.dumps(asdict(a), ensure_ascii=False) for a in all_articles),
        encoding="utf-8")
    (outdir / "law_chunks.jsonl").write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in all_chunks),
        encoding="utf-8")
    (outdir / "verify_report.md").write_text("# 入库校验报告\n" + "\n".join(report) + "\n",
                                             encoding="utf-8")
    print(f"入库完成：{len(all_articles)} 条，{len(all_chunks)} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
