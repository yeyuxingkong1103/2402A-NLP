# -*- coding: utf-8 -*-
"""
工单01 建索引脚本：解析《招股说明书1.pdf》→ 分块 → 向量索引 + BM25 索引
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

功能：
  1. 解析 PDF（PyMuPDF 提取文字层；可选 pdfplumber 解析表格层）
  2. 按流水线预设策略分块（fixed / recursive / semantic / structure）
  3. 写入向量库（Chroma 为主，缺失时自动降级为 NumPy 本地库）
     并同步构建 BM25 倒排索引（工单01 要求「向量检索」与全文检索双通道）
  4. 打印索引统计（页数 / 块数 / 耗时），并保存到 results/index_stats.json

用法：
    python "工单01-基于PDF文档的问答系统/src/build_index.py"
    python .../src/build_index.py --preset wo03_table     # 额外解析表格（验收项：表格数据）
    python .../src/build_index.py --preset wo06_hybrid    # 混合检索+级联重排（Web 服务默认）
    python .../src/build_index.py --no-rebuild            # 增量追加，不重建

说明：
  · 工单01 默认预设为 wo01_baseline（fixed 分块 + 纯向量检索），索引本身
    同时包含向量库与 BM25，供后续工单 02/06 直接复用。
  · 表格解析（pdfplumber 逐页）在数百页招股书上较慢，默认关闭，按需开启。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config                        # noqa: E402
from rag_core.pipeline import PRESETS, Pipeline    # noqa: E402

# 本工单的产出目录
CASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：解析《招股说明书1.pdf》并建立向量索引 + BM25 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python build_index.py\n"
               "  python build_index.py --preset wo03_table --with-tables\n",
    )
    p.add_argument("--pdf", default=str(config.PDF_PROSPECTUS_1),
                   help="待解析的 PDF 路径，默认《招股说明书1.pdf》")
    p.add_argument("--doc-name", default=None,
                   help="文档名（用于答案溯源），默认取 PDF 文件名")
    p.add_argument("--preset", default="wo01_baseline", choices=sorted(PRESETS),
                   help="流水线预设（决定分块策略/是否含表格等），默认 wo01_baseline")
    p.add_argument("--collection", default="prospectus",
                   help="向量库集合名，默认 prospectus（Web 服务同用此集合）")
    p.add_argument("--with-tables", dest="with_tables", action="store_true",
                   default=None, help="强制开启表格解析（覆盖预设）")
    p.add_argument("--no-tables", dest="with_tables", action="store_false",
                   help="强制关闭表格解析（覆盖预设）")
    p.add_argument("--no-rebuild", action="store_true",
                   help="不先清空集合，直接把新块追加进已有索引")
    p.add_argument("--out", default=str(RESULTS_DIR / "index_stats.json"),
                   help="索引统计输出路径，默认 results/index_stats.json")
    p.add_argument("--debug", action="store_true", help="异常时打印完整堆栈")
    return p


# ---------------------------------------------------------------------------
# 统计信息补充
# ---------------------------------------------------------------------------
def _chunk_length_stats() -> dict:
    """从 BM25 索引回读块长度分布（用于资源消耗与分块质量分析）。"""
    from rag_core.bm25 import BM25Retriever

    bm25_path = config.INDEX_DIR / "bm25.pkl"
    if not bm25_path.exists():
        return {}
    bm = BM25Retriever.load(bm25_path)
    lens = [len(d.get("text", "")) for d in bm.docs]
    if not lens:
        return {}
    lens_sorted = sorted(lens)
    return {
        "chunk_chars_total": sum(lens),
        "chunk_chars_avg": round(sum(lens) / len(lens), 1),
        "chunk_chars_min": lens_sorted[0],
        "chunk_chars_max": lens_sorted[-1],
        "chunk_chars_p50": lens_sorted[len(lens_sorted) // 2],
    }


def _print_report(stats: dict) -> None:
    """把索引统计打印成人读的报表。"""
    doc = (stats.get("docs") or [{}])[0]
    types = stats.get("chunk_types", {})
    type_desc = " / ".join(f"{k} {v}" for k, v in sorted(types.items())) or "-"

    print("\n" + "=" * 66)
    print("  索引构建完成")
    print("=" * 66)
    print(f"  文档            : {doc.get('name', '?')}（{stats.get('pdf', '')}）")
    print(f"  页数            : {doc.get('pages', 0)}")
    print(f"  块数            : {stats.get('n_chunks', 0)}（{type_desc}）")
    print(f"  向量库条数      : {stats.get('n_vectors', 0)}"
          f"（后端：{stats.get('vector_backend', '?')}）")
    print(f"  BM25 文档数     : {stats.get('n_bm25_docs', 0)}")
    print(f"  解析+建索引耗时 : {stats.get('build_seconds', 0)} s")
    if stats.get("chunk_chars_avg"):
        print(f"  块长度          : 平均 {stats['chunk_chars_avg']} 字符"
              f"（最小 {stats['chunk_chars_min']} / 最大 {stats['chunk_chars_max']}）")
    print(f"  嵌入模型        : {stats.get('embed_model', '?')}")
    print(f"  流水线预设      : {stats.get('preset', '?')}")
    print(f"  索引目录        : {stats.get('index_dir', '')}")
    print(f"  统计已保存至    : {stats.get('stats_file', '')}")
    print("-" * 66)
    print("  下一步（以下命令均在项目根目录 工单作业/ 下执行）：")
    print('    python "工单01-基于PDF文档的问答系统/src/qa_cli.py" '
          '-q "武汉兴图新科电子股份有限公司法定代表人是谁？"')
    print('    python "工单01-基于PDF文档的问答系统/src/compare_rag_vs_llm.py"')
    print('    python "工单01-基于PDF文档的问答系统/src/run_evaluation.py"')
    print('    python "工单01-基于PDF文档的问答系统/src/serve.py"')
    print("=" * 66)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[错误] 找不到 PDF 文件：{pdf_path}")
        print("       请确认附件《招股说明书1.pdf》存在，或用 --pdf 指定路径。")
        return 2

    doc_name = args.doc_name or pdf_path.stem
    cfg = PRESETS[args.preset]
    if args.with_tables is not None:
        cfg = replace(cfg, with_tables=args.with_tables)

    print("=" * 66)
    print("  工单01 建索引：基于PDF文档的问答系统")
    print(f"  工单编号：人工智能NLP-RAG-基于PDF文档的问答系统")
    print("=" * 66)
    print(f"  PDF      : {pdf_path}")
    print(f"  文档名   : {doc_name}")
    print(f"  预设     : {args.preset}  "
          f"(chunk={cfg.chunk_strategy}, strategy={cfg.strategy}, "
          f"tables={cfg.with_tables}, reranker={cfg.reranker})")
    print(f"  集合     : {args.collection}")
    if not cfg.with_tables:
        print("  [提示] 当前未解析表格。若需验收「表格数据解析」，请加 "
              "--with-tables 或 --preset wo03_table。")
    print("-" * 66)

    t0 = time.perf_counter()
    try:
        pipeline = Pipeline(cfg, collection=args.collection)
        stats = pipeline.build_index([pdf_path], doc_names=[doc_name],
                                     rebuild=not args.no_rebuild, verbose=True)
    except Exception as e:                       # 容错：解析/建库失败给出可操作提示
        print(f"\n[错误] 建索引失败：{e}")
        print("       常见原因：PDF 损坏或被加密、缺少依赖（PyMuPDF/pdfplumber/"
              "sentence-transformers）、磁盘空间不足。")
        if args.debug:
            traceback.print_exc()
        return 3

    # ---- 补充统计信息并落盘 ----
    from rag_core.vectorstore import backend as vector_backend

    stats.update({
        "pdf": str(pdf_path),
        "pdf_size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
        "doc_name": doc_name,
        "preset": args.preset,
        "preset_config": cfg.to_dict(),
        "embed_model": config.EMBED_MODEL_NAME,
        "vector_backend": vector_backend(),
        "index_dir": str(config.INDEX_DIR),
        "llm_model": config.LLM_MODEL,
        "python": platform.python_version(),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    stats.update(_chunk_length_stats())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats["stats_file"] = str(out_path)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    _print_report(stats)
    print(f"  总墙钟耗时：{time.perf_counter() - t0:.1f} s")
    if args.preset != "wo06_hybrid":
        print("  [提示] Web 服务（serve.py / rag_core.api）默认使用 wo06_hybrid 预设，")
        print("         如需与网页端完全一致，请执行："
              "python src/build_index.py --preset wo06_hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
