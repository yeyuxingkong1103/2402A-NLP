# -*- coding: utf-8 -*-
"""T9 表格解析增益实测（回答 优化/README.md 的必答问题 1）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

问题：「表格解析是否真的带来增益 —— 同一评测集下，『仅文本块』vs『文本块 + 表格块』的差异。」

本脚本**不重建索引**（不臆造成本），而是用已落盘的逐题检索结果做**反事实裁剪**：
    ① 读 `优化/评估结果/accuracy_report.json`（含逐题 top_chunks / support_chunks）；
    ② 到 `研发/data/index/rag.sqlite3` 的 `chunks` 表取每个块的 `type`；
    ③ 计算三种口径的命中题数：
        full   = top-k ∪ 支持块（= 报告主口径）
        text   = 去掉表格块（`type == "table"`）后剩余块
        table  = 只留表格块
    ④ 「表格增益」= full 命中题数 − text 命中题数（>0 即为表格块不可替代的题）；
       并列出「命中块类型分布」与「哪些题依赖表格块」。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/table_gain_ablation.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/table_gain_ablation.py --report 优化/评估结果/accuracy_report.run2.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))

from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retrieval_utils import is_evidence_hit  # noqa: E402

OUT_JSON = REPO_ROOT / "优化" / "评估结果" / "表格增益实测.json"
OUT_MD = REPO_ROOT / "优化" / "评估结果" / "表格增益实测.md"


class ChunkView:
    """最小块视图（供 ``is_evidence_hit`` 使用：只要有 ``content`` 即可）。"""

    __slots__ = ("chunk_id", "file_name", "page", "type", "content")

    def __init__(self, chunk_id: str, file_name: str, page: int, kind: str, content: str) -> None:
        self.chunk_id = chunk_id
        self.file_name = file_name
        self.page = page
        self.type = kind
        self.content = content


def load_chunks(sqlite_path: Path, chunk_ids: Sequence[str], *, logger: Any) -> dict[str, ChunkView]:
    """按 chunk_id 批量取块（含 content 与 type），缺失的 id 显式记 WARNING。"""
    with logger.enter("load_chunks", {"ids": len(chunk_ids), "db": str(sqlite_path)}) as span:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            views: dict[str, ChunkView] = {}
            missing: list[str] = []
            for chunk_id in chunk_ids:
                row = conn.execute("SELECT chunk_id,file_name,page,type,content FROM chunks WHERE chunk_id=?",
                                   (str(chunk_id),)).fetchone()
                if row is None:
                    missing.append(str(chunk_id))
                    continue
                views[row[0]] = ChunkView(row[0], row[1], int(row[2]), str(row[3]), str(row[4] or ""))
        finally:
            conn.close()
        if missing:
            logger.log_event("table_gain.chunk_missing", level="WARNING", count=len(missing),
                             sample=missing[:5], degrade="缺失块不参与命中判定，结果按现有块计算")
        span.set_output({"loaded": len(views), "missing": len(missing)})
        return views


def main(argv: Sequence[str] | None = None) -> int:
    """入口：反事实裁剪 → 表格增益 → 落盘 JSON/Markdown。"""
    parser = argparse.ArgumentParser(description="表格解析增益实测（反事实裁剪口径）")
    parser.add_argument("--report", default=str(REPO_ROOT / "优化" / "评估结果" / "accuracy_report.json"))
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("table_gain_ablation")
    started = time.perf_counter()
    with log.enter("main", {"report": args.report}) as span:
        report_path = Path(args.report)
        if not report_path.is_file():
            log.log_event("table_gain.missing_report", level="ERROR", path=str(report_path))
            print(f"❌ 缺少评估产物：{report_path}")
            return 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = payload["results"]
        all_ids: list[str] = []
        for row in rows:
            all_ids += list(row.get("top_chunks") or []) + list(row.get("support_chunks") or [])
        views = load_chunks(cfg.paths.index_dir / "rag.sqlite3", all_ids, logger=log)

        per_question: list[dict[str, Any]] = []
        hit_full = hit_text = hit_table = 0
        table_dependent: list[int] = []
        cited_types: dict[str, int] = {}
        for row in rows:
            ids = [cid for cid in (row.get("top_chunks") or []) if cid in views]
            support = [cid for cid in (row.get("support_chunks") or []) if cid in views]
            context = [views[cid] for cid in ids + support]
            text_only = [chunk for chunk in context if chunk.type != "table"]
            table_only = [chunk for chunk in context if chunk.type == "table"]
            evidence = str(row.get("evidence_verbatim") or "")
            full_ok = bool(is_evidence_hit(context, evidence))
            text_ok = bool(is_evidence_hit(text_only, evidence))
            table_ok = bool(is_evidence_hit(table_only, evidence))
            hit_full += int(full_ok)
            hit_text += int(text_ok)
            hit_table += int(table_ok)
            if full_ok and not text_ok:
                table_dependent.append(int(row["id"]))
            cited = []
            for cid in row.get("top_chunks") or []:
                if cid in views:
                    cited.append(views[cid].type)
            for kind in cited:
                cited_types[kind] = cited_types.get(kind, 0) + 1
            per_question.append({
                "id": row["id"], "context_chunks": len(context),
                "text_chunks": len(text_only), "table_chunks": len(table_only),
                "hit_full": full_ok, "hit_text_only": text_ok, "hit_table_only": table_ok,
                "table_dependent": bool(full_ok and not text_ok),
                "top_chunk_types": cited,
            })

        total = len(rows)
        result = {
            "work_order": WORK_ORDER,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_report": str(report_path),
            "method": "反事实裁剪：用已落盘 top-k ∪ 支持块，按 chunks.type 去掉表格块后重算 is_evidence_hit",
            "total": total,
            "hit_full": hit_full, "hit_text_only": hit_text, "hit_table_only": hit_table,
            "table_gain_questions": total - hit_text,
            "table_dependent_ids": table_dependent,
            "top_chunk_type_distribution": cited_types,
            "per_question": per_question,
            "note": ("『仅文本块』口径 = 从返回块里删掉所有表格块后仍能命中证据原文；差值即表格块不可替代的题数。"
                     "反事实裁剪不改索引、不改检索参数，因此不引入新的实验偏差；"
                     "但它度量的是「表格块在**已返回上下文**中的贡献」，不是「重建无表格索引后的召回率」，"
                     "后者需重建索引（≈185 s + 嵌入），本工单未做，如实标注。"),
        }
        OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# 表格解析增益实测（反事实裁剪口径）",
            "",
            f"> 工单：{WORK_ORDER}　生成时间：{result['generated_at']}",
            f"> 数据源：`{report_path.relative_to(REPO_ROOT)}`（{total} 题）",
            "",
            "## 1. 结论",
            "",
            f"- **文本块 + 表格块**：命中 **{hit_full}/{total}**（报告主口径）",
            f"- **仅文本块**（删掉所有 `type=table` 的返回块）：命中 **{hit_text}/{total}**",
            f"- **仅表格块**：命中 **{hit_table}/{total}**",
            f"- → **表格块的不可替代增益 = {total - hit_text} 题**"
            f"（依赖表格块的题：{table_dependent or '无'}）",
            f"- 返回块类型分布（top-k 内）：{cited_types}",
            "",
            "## 2. 逐题明细",
            "",
            "| id | 上下文块数（文本/表格） | 全量命中 | 仅文本块 | 仅表格块 | 依赖表格块 | top-k 块类型 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in per_question:
            lines.append(f"| {row['id']} | {row['context_chunks']}"
                         f"（{row['text_chunks']}/{row['table_chunks']}） | "
                         f"{'✅' if row['hit_full'] else '❌'} | {'✅' if row['hit_text_only'] else '❌'} | "
                         f"{'✅' if row['hit_table_only'] else '❌'} | "
                         f"{'**是**' if row['table_dependent'] else '否'} | {row['top_chunk_types']} |")
        lines += ["", "## 3. 口径与局限（诚实声明）", "",
                  result["note"], "",
                  "- 「表格块」判定依据是索引里 `chunks.type` 字段（由 `研发/app/core/table_parser.py` 产出），"
                  "不是人工分类。", ""]
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"文本块+表格块 命中 {hit_full}/{total}；仅文本块 {hit_text}/{total}；仅表格块 {hit_table}/{total}")
        print(f"表格块不可替代增益 = {total - hit_text} 题（{table_dependent or '无'}）")
        print(f"产物：{OUT_JSON.relative_to(REPO_ROOT)} / {OUT_MD.relative_to(REPO_ROOT)}")
        print(f"耗时 {round((time.perf_counter() - started) * 1000, 2)} ms")
        span.set_output({"hit_full": hit_full, "hit_text_only": hit_text, "table_gain": total - hit_text})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底
        try:
            get_logger("table_gain_ablation").log_event("table_gain.failed", level="ERROR",
                                                        error_type=type(exc).__name__, message=str(exc),
                                                        stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
