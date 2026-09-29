# -*- coding: utf-8 -*-
"""同一部法规「PDF 解析 vs 库内 HTML 解析」对照，验证 PDF 通道与既有 HTML 通道是否等价。

对照对象（默认）：中华人民共和国劳动合同法
- PDF 侧：真实 MinerU 解析 → cleaner 后的正文（本脚本自己现跑，或 `--cleaned` 指定既有产物）
- HTML 侧：MySQL `documents#N` 已审核版本的 `cleaned_content`

比较维度：字符数、块数、条号序列、逐条正文（规范化后）、款/项子块数、结构性差异（目录 / Markdown 残留）。

用法：
    python scripts/e2e/compare_pdf_vs_html.py                       # 现跑 MinerU 再对照
    python scripts/e2e/compare_pdf_vs_html.py --cleaned some.cleaned.txt   # 用已有产物，不联网
    python scripts/e2e/compare_pdf_vs_html.py --document-id 10 --pdf jm_labor_law.pdf \
        --title 中华人民共和国劳动法

退出码：0 = 对照完成且条号序列一致；1 = 条号序列不一致（结构性差异，需排查）；2 = 前置不满足。

已知差异（登记在案，非缺陷）：见 `reports/batch22_pdf_pipeline_acceptance.md` 第四节
——PDF 版面换行会被还原成 ``\\n\\n``，同一「款」可能多切出 1 个子块；条级正文完全一致。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

# 项目根 = scripts/e2e/ 向上两层
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_project_env, mysql_config  # noqa: E402

# 除 _env 剔除的 4 个外，all_proxy 同样会劫持外网调用
for _name in ("all_proxy", "ALL_PROXY"):
    os.environ.pop(_name, None)

load_project_env(PROJECT_ROOT)

import pymysql  # noqa: E402

from app.ingest.chunker import chunk_document  # noqa: E402
from app.ingest.parser import ParsedDocument, parse_document  # noqa: E402

PDF_DIR = PROJECT_ROOT / "data" / "pdf_samples"

DEFAULT_PDF = "jm_labor_contract_law.pdf"
DEFAULT_TITLE = "中华人民共和国劳动合同法"
DEFAULT_DOCUMENT_ID = 9

# 规范化：去掉空白与 Markdown 标记，只留文字与标点，供逐条比对
_NOISE = re.compile(r"[\s#`*_]+")


def normalize(text: str) -> str:
    return _NOISE.sub("", text)


def load_html_content(document_id: int) -> tuple[str, int, str]:
    """取该文档最新 approved 版本的 cleaned_content。"""
    connection = pymysql.connect(charset="utf8mb4", **mysql_config())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT v.id, v.content_hash, v.cleaned_content, d.source_url
                FROM documents d
                JOIN document_versions v ON v.document_id = d.id
                WHERE d.id = %s AND v.version_status = 'approved'
                ORDER BY v.id DESC LIMIT 1
                """,
                (document_id,),
            )
            row = cursor.fetchone()
    finally:
        connection.close()
    if not row:
        raise SystemExit(f"documents#{document_id} 没有已审核版本")
    return row[2], row[0], row[3]


def chunk_stats(chunks) -> dict:
    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]
    items = [c for c in children if c.item_no]
    return {
        "total": len(chunks),
        "parents": len(parents),
        "children": len(children),
        "items": len(items),
        "plain_paragraphs": len(children) - len(items),
        "article_numbers": [c.article_no for c in parents],
    }


def resolve_pdf(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    return PDF_DIR / name


def fresh_pdf_content(pdf_path: Path, out_dir: Path) -> str:
    """现跑一次完整 PDF 链路（真实 MinerU，消耗额度），返回 cleaner 后的正文。"""
    from app.models.mineru import build_mineru_client_from_settings
    from app.models.qwen_vl import build_qwen_vl_client_from_settings

    print(f"（PDF 侧产物来源：现跑 MinerU 解析 {pdf_path.name}）")
    parsed = parse_document(
        pdf_path,
        build_mineru_client_from_settings(),
        build_qwen_vl_client_from_settings(),
    )
    cleaned_path = out_dir / f"{pdf_path.stem}.cleaned.txt"
    cleaned_path.write_text(parsed.content, encoding="utf-8")
    print(f"（已落盘：{cleaned_path}）")
    return parsed.content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF 解析 vs 库内 HTML 解析 对照")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help=f"PDF 样本名或路径（默认 {DEFAULT_PDF}）")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="文档标题（切块用）")
    parser.add_argument(
        "--document-id",
        type=int,
        default=DEFAULT_DOCUMENT_ID,
        help=f"库内对照文档 ID（默认 {DEFAULT_DOCUMENT_ID}）",
    )
    parser.add_argument(
        "--cleaned",
        default=None,
        help="既有 PDF 清洗产物（给了就不联网重跑，直接用它做 PDF 侧）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="现跑解析时的产物目录（默认写系统临时目录）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cleaned:
        cleaned_path = Path(args.cleaned)
        if not cleaned_path.is_file():
            print(f"❌ 指定的清洗产物不存在：{cleaned_path}")
            return 2
        pdf_content = cleaned_path.read_text(encoding="utf-8")
        pdf_label = f"{cleaned_path.name}（既有产物，未重新解析）"
    else:
        pdf_path = resolve_pdf(args.pdf)
        if not pdf_path.is_file():
            print(f"❌ PDF 不存在：{pdf_path}")
            print(f"   可用样本：{sorted(p.name for p in PDF_DIR.glob('*.pdf'))}")
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="pdf_cmp_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_content = fresh_pdf_content(pdf_path, out_dir)
        pdf_label = f"{pdf_path.name}（本次 MinerU 解析 + cleaner）"

    html_content, version_id, source_url = load_html_content(args.document_id)

    print("=" * 78)
    print(f"PDF 解析 vs 库内 HTML 解析 —— {args.title}")
    print("=" * 78)
    print(f"  PDF 侧 ：{pdf_label}")
    print(f"  HTML 侧：documents#{args.document_id}  approved version#{version_id}")
    print(f"           来源 {source_url}")
    print()

    pdf_doc = ParsedDocument(
        source_path=PROJECT_ROOT / "data" / "x.pdf",
        content=pdf_content,
        content_format="pdf",
        parser_name="mineru",
    )
    html_doc = ParsedDocument(
        source_path=PROJECT_ROOT / "data" / "x.html",
        content=html_content,
        content_format="html",
        parser_name="html",
    )

    pdf_chunks = chunk_document(pdf_doc, document_title=args.title, document_id="cmp-pdf")
    html_chunks = chunk_document(html_doc, document_title=args.title, document_id="cmp-html")
    pdf_stats = chunk_stats(pdf_chunks)
    html_stats = chunk_stats(html_chunks)

    print("-" * 78)
    print("① 体量对比")
    print(f"  {'':16s}{'PDF':>14s}{'HTML':>14s}")
    print(f"  {'清洗后字符数':16s}{len(pdf_content):>14,}{len(html_content):>14,}")
    print(f"  {'总块数':16s}{pdf_stats['total']:>14}{html_stats['total']:>14}")
    print(f"  {'条（父块）':16s}{pdf_stats['parents']:>14}{html_stats['parents']:>14}")
    print(f"  {'款/项（子块）':16s}{pdf_stats['children']:>14}{html_stats['children']:>14}")
    print(f"  {'  其中项':16s}{pdf_stats['items']:>14}{html_stats['items']:>14}")
    print(f"  {'  其中款':16s}{pdf_stats['plain_paragraphs']:>14}{html_stats['plain_paragraphs']:>14}")

    print()
    print("-" * 78)
    print("② 条号序列对比")
    pdf_nos = pdf_stats["article_numbers"]
    html_nos = html_stats["article_numbers"]
    print(f"  PDF  首 6 = {pdf_nos[:6]}  末 4 = {pdf_nos[-4:]}")
    print(f"  HTML 首 6 = {html_nos[:6]}  末 4 = {html_nos[-4:]}")
    sequence_matches = pdf_nos == html_nos
    if sequence_matches:
        print("  ✅ 条号序列逐项一致")
    else:
        print("  ⚠ 条号序列不一致，首个差异：")
        for index in range(max(len(pdf_nos), len(html_nos))):
            left = pdf_nos[index] if index < len(pdf_nos) else "(缺)"
            right = html_nos[index] if index < len(html_nos) else "(缺)"
            if left != right:
                print(f"     第 {index + 1} 项：PDF={left}  HTML={right}")
                break
        print(f"     PDF 独有 = {set(pdf_nos) - set(html_nos)}")
        print(f"     HTML 独有 = {set(html_nos) - set(pdf_nos)}")

    print()
    print("-" * 78)
    print("③ 逐条正文对比（规范化后按条号配对）")
    pdf_map = {c.article_no: c.content for c in pdf_chunks if c.chunk_level == "parent"}
    html_map = {c.article_no: c.content for c in html_chunks if c.chunk_level == "parent"}
    common = [no for no in pdf_nos if no in html_map]
    same: list[str] = []
    diff: list[tuple[str, str, str]] = []
    for article_no in common:
        left = normalize(pdf_map[article_no])
        right = normalize(html_map[article_no])
        if left == right:
            same.append(article_no)
        else:
            diff.append((article_no, pdf_map[article_no], html_map[article_no]))

    print(f"  配对条数 = {len(common)}  完全一致 = {len(same)}  有差异 = {len(diff)}")
    print(f"  一致率 = {len(same) / len(common):.1%}" if common else "  无可配对条文")

    print()
    print("  差异样例（最多 4 条，逐字符比对后的首处不同）：")
    for article_no, left, right in diff[:4]:
        left_n = normalize(left)
        right_n = normalize(right)
        pos = 0
        while pos < min(len(left_n), len(right_n)) and left_n[pos] == right_n[pos]:
            pos += 1
        print(f"    【{article_no}】PDF {len(left_n)} 字 / HTML {len(right_n)} 字")
        print(f"      PDF  …{left_n[max(0, pos - 25):pos + 45]}…")
        print(f"      HTML …{right_n[max(0, pos - 25):pos + 45]}…")

    # 结构性差异：PDF 独有的 Markdown 标记 / 目录段
    print()
    print("-" * 78)
    print("④ 结构性差异")
    pdf_md_headings = [l for l in pdf_content.split("\n") if l.startswith("#")]
    html_md_headings = [l for l in html_content.split("\n") if l.startswith("#")]
    print(f"  PDF 侧残留 Markdown 标题行 = {len(pdf_md_headings)} 行")
    print(f"  HTML 侧残留 Markdown 标题行 = {len(html_md_headings)} 行")

    # 「目录」在 PDF 里常排成「目 录」，比对时先去掉空白
    flat_pdf = re.sub(r"\s+", "", pdf_content)
    flat_html = re.sub(r"\s+", "", html_content)
    print(f"  含「目录」段：PDF={('目录' in flat_pdf)}  HTML={('目录' in flat_html)}")

    # 款/项子块数逐条对齐，定位剩余的成块差异
    print()
    print("-" * 78)
    print("⑤ 逐条子块（款/项）数对比 —— 定位剩余差异")
    pdf_child_count: dict = {}
    for chunk in pdf_chunks:
        if chunk.chunk_level == "child":
            pdf_child_count[chunk.article_no] = pdf_child_count.get(chunk.article_no, 0) + 1
    html_child_count: dict = {}
    for chunk in html_chunks:
        if chunk.chunk_level == "child":
            html_child_count[chunk.article_no] = html_child_count.get(chunk.article_no, 0) + 1
    mismatches = [
        (no, pdf_child_count.get(no, 0), html_child_count.get(no, 0))
        for no in common
        if pdf_child_count.get(no, 0) != html_child_count.get(no, 0)
    ]
    print(f"  子块数不一致的条文 = {len(mismatches)} 条")
    for no, left, right in mismatches[:8]:
        print(f"    【{no}】PDF 子块 {left} 个 / HTML 子块 {right} 个")
        pdf_parents = [c for c in pdf_chunks if c.article_no == no and c.chunk_level == "parent"]
        if pdf_parents:
            preview = pdf_parents[0].content.replace("\n", "⏎")
            print(f"      PDF 父块原文：{preview[:120]}")
        html_parent = [c for c in html_chunks if c.article_no == no and c.chunk_level == "parent"]
        if html_parent:
            preview = html_parent[0].content.replace("\n", "⏎")
            print(f"      HTML 父块原文：{preview[:120]}")
    if mismatches:
        print("  ℹ️ 款级子块数不一致属**已登记差异**（PDF 版面换行被还原成 \\n\\n），")
        print("     条级正文一致即可接受；详见 reports/batch22_pdf_pipeline_acceptance.md 第四节。")

    # 章节标题存活
    print()
    print("  章节标题出现次数（PDF / HTML）：")
    for heading in ("第一章", "第二章", "第八章", "总则", "附则"):
        print(f"      {heading:<6} PDF={pdf_content.count(heading):>2}  HTML={html_content.count(heading):>2}")
    print("=" * 78)

    # 条号序列一致是"结构等价"的最低要求；不一致直接以非 0 退出，避免被当成通过
    if not sequence_matches:
        print("❌ 条号序列不一致：PDF 通道与 HTML 通道结构不等价，需排查")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
