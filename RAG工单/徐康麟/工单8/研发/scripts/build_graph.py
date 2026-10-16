# -*- coding: utf-8 -*-
"""工单8 图谱构建脚本：抽取实体与关系 → 落盘 entities/relations/graph.json/GraphML。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

输入（**只读**）：
    * 分块：``E:\\gao6gongdan\\工单7\\研发\\data\\index\\bge-m3_1024\\chunks_indexed.jsonl``
      （8847 块 = 文本 7325 + 表格 1417 + 图像 105；工单7 已解析完毕，本脚本**不重新解析 PDF**）
输出（写入本工单）：
    * ``研发/data/graph/entities.jsonl`` / ``relations.jsonl`` / ``graph.json`` / ``graph.graphml``
    * ``研发/data/graph/图谱构建统计.json``（构建实测数字，供报告与截图引用）
    * ``研发/data/graph/llm_关系补抽.jsonl``（启用 ``--llm-sample`` 时）

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_graph.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_graph.py --llm-sample 500 --limit-chunks 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

from app.graphrag.config import GRAPH_FILES, WORK_ORDER, ensure_dirs, get_config  # noqa: E402
from app.graphrag.extractor import GraphExtractor, iter_chunks  # noqa: E402
from app.graphrag.llm import GraphLLM  # noqa: E402
from app.graphrag.store import build_store, save_graph  # noqa: E402

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答


def _safe_logger(path: Path) -> Any:
    """轻量日志：同时写文件与 stdout 摘要（长日志落文件，不刷终端）。

    健壮性：日志文件被别的进程占用（Windows 上表现为 ``PermissionError``）时，
    自动改用带时间戳的备用文件名，**绝不因为日志打不开而中断图谱构建**。
    """
    class _Log:
        def __init__(self, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.handle = None
            self.path: Path | None = None
            for candidate in (target,
                              target.with_name(f"{target.stem}_{time.strftime('%H%M%S')}{target.suffix}")):
                try:
                    self.handle = candidate.open("a", encoding="utf-8")
                    self.path = candidate
                    break
                except OSError as exc:  # noqa: PERF203 —— 逐个候选尝试，失败就换下一个
                    print(f"[WARN] 日志文件不可写（{type(exc).__name__}: {exc}），尝试备用名")
            if self.handle is None:
                print("[WARN] 所有日志候选都不可写 ⇒ 本次不写日志文件（主流程继续）")

        def log_event(self, event: str, level: str = "INFO", **fields: Any) -> None:
            if self.handle is None:
                return
            try:
                line = json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "level": level,
                                   "event": event, **fields}, ensure_ascii=False, default=str)
                self.handle.write(line + "\n")
                self.handle.flush()
            except OSError:  # 运行中写失败（句柄/磁盘）⇒ 停止写日志，不影响主流程
                self.handle = None
                return
            if level in {"WARNING", "ERROR"}:
                print(f"[{level}] {event}: {line[:200]}")

        def close(self) -> None:
            if self.handle is not None:
                self.handle.close()

    return _Log(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 图谱构建：{WORK_ORDER}")
    parser.add_argument("--chunks", default=None, help="分块 jsonl 路径（默认取工单7 索引内的 chunks_indexed.jsonl）")
    parser.add_argument("--limit-chunks", type=int, default=0, help="只处理前 N 块（0 = 全量，调试用）")
    parser.add_argument("--include-tables", action="store_true", help="同时抽表格块（默认只抽文本块）")
    parser.add_argument("--llm-sample", type=int, default=None,
                        help="用 Ollama qwen2.5:3b 抽样补抽关系的块数（默认取配置值；0 = 关闭）")
    parser.add_argument("--llm-model", default=None, help="覆盖补抽模型名（默认 qwen2.5:3b）")
    parser.add_argument("--out-dir", default=None, help="覆盖图谱输出目录")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    log = _safe_logger(Path(cfg.log_dir) / "build_graph.log")
    started = time.perf_counter()
    chunks_path = Path(args.chunks) if args.chunks else cfg.chunks_source()
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.graph_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] 分块来源：{chunks_path}")
    log.log_event("build.start", work_order=WORK_ORDER, chunks=str(chunks_path), out_dir=str(out_dir),
                  limit_chunks=args.limit_chunks, include_tables=bool(args.include_tables),
                  config=cfg.fingerprint())

    extractor = GraphExtractor(cfg, logger=log)
    scanned = 0
    t_scan = time.perf_counter()
    try:
        for row in iter_chunks(chunks_path, include_table=bool(args.include_tables), logger=log):
            extractor.extract_chunk(row)
            scanned += 1
            if args.limit_chunks and scanned >= int(args.limit_chunks):
                break
            if scanned % 1000 == 0:
                print(f"      已扫描 {scanned} 块，实体 {len(extractor.entities())}，"
                      f"关系 {len(extractor.relations())}")
    except Exception as exc:  # noqa: BLE001 —— 抽取阶段任何异常都要留下可诊断信息
        log.log_event("build.extract_failed", level="ERROR", error=f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] 抽取失败：{type(exc).__name__}: {exc}")
        return 2
    scan_ms = round((time.perf_counter() - t_scan) * 1000, 2)
    stats = extractor.stats()
    print(f"[2/5] 规则抽取完成：扫描 {scanned} 块，实体 {stats['entities']}，关系 {stats['relations']}，"
          f"耗时 {scan_ms / 1000:.1f}s")
    # ---- 高频噪声剪枝（数据自证：出现块数 > max_doc_ratio×扫描块数 的实体剪除）----
    prune = extractor.prune_high_frequency(chunks_scanned=scanned, logger=log)
    # ---- 最低证据量剪枝（只出现一次的非主体实体 = 正则跨词残片）----
    prune_low = extractor.prune_low_support(logger=log)
    if prune["pruned"] or prune_low["pruned"]:
        stats = extractor.stats()
        print(f"      剪枝：高频 {prune['pruned']} 个（阈值 {prune['limit_chunks']} 块）"
              f"+ 低频 {prune_low['pruned']} 个（仅出现 {prune_low['min_mentions']} 次）"
              f"⇒ 实体 {stats['entities']}，关系 {stats['relations']}")

    # ---- 可选：Ollama 抽样补抽关系（默认 500 块；固定种子 ⇒ 可复现）----
    llm_rows: list[dict[str, Any]] = []
    llm_meta: dict[str, Any] = {"enabled": False}
    sample_n = int(args.llm_sample if args.llm_sample is not None
                   else (cfg.llm.get("sample_chunks") or 0))
    if sample_n > 0 and bool(cfg.llm.get("enabled", True)):
        llm_cfg = dict(cfg.llm)
        if args.llm_model:
            llm_cfg["model"] = args.llm_model
        llm = GraphLLM(type("_C", (), {"llm": llm_cfg})(), logger=log)  # 只借 llm 段
        if not llm.available():
            llm_meta = {"enabled": False, "reason": "Ollama 不可用或模型缺失",
                        "model": llm.model, "base_url": llm.base_url}
            print(f"[3/5] 跳过 LLM 补抽：{llm_meta['reason']}")
        else:
            print(f"[3/5] Ollama 抽样补抽：模型={llm.model}，计划抽样 {sample_n} 块（固定种子，可复现）")
            rows = [r for r in iter_chunks(chunks_path, include_table=False, logger=log)
                    if len(str(r.get("content") or "")) >= 120]
            rng = random.Random(int(cfg.llm.get("sample_seed") or 20260108))
            picked = rng.sample(rows, min(sample_n, len(rows)))
            t_llm = time.perf_counter()
            ok_chunks = 0
            for idx, row in enumerate(picked, 1):
                text = str(row.get("content") or "")[: int(cfg.llm.get("max_chars_per_chunk") or 320)]
                try:
                    found = llm.extract_relations(text, chunk_id=str(row.get("chunk_id") or ""),
                                                  limit=int(cfg.llm.get("max_relations_per_chunk") or 3))
                except Exception as exc:  # noqa: BLE001 —— 单块失败不中断抽样
                    log.log_event("build.llm_chunk_failed", level="WARNING", idx=idx,
                                  error=f"{type(exc).__name__}: {exc}")
                    continue
                if found:
                    ok_chunks += 1
                for item in found:
                    llm_rows.append({**item, "page": int(row.get("page") or 0),
                                     "file_name": str(row.get("file_name") or "")})
                if idx % 50 == 0:
                    print(f"      补抽进度 {idx}/{len(picked)}，累计关系 {len(llm_rows)}")
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            merge = extractor.add_llm_relations(llm_rows)
            llm_meta = {"enabled": True, "model": llm.model, "sampled_chunks": len(picked),
                        "chunks_with_relations": ok_chunks, "relations_raw": len(llm_rows),
                        "elapsed_ms": llm_ms, "avg_ms_per_chunk": round(llm_ms / max(len(picked), 1), 2),
                        **merge}
            (out_dir / GRAPH_FILES["llm_supplement"]).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in llm_rows) + ("\n" if llm_rows else ""),
                encoding="utf-8")
            print(f"      补抽完成：{llm_meta}")
            stats = extractor.stats()
    else:
        print("[3/5] 未启用 LLM 补抽（--llm-sample 0 或配置关闭）")

    # ---- 落盘 ----
    print("[4/5] 写图谱产物 …")
    entities = [e.to_dict() for e in extractor.entities()]
    relations = [r.to_dict() for r in extractor.relations()]
    store = build_store(entities, relations, stats={**stats, "chunks_scanned": scanned,
                                                    "scan_ms": scan_ms, "llm": llm_meta,
                                                    "prune": prune, "prune_low_support": prune_low,
                                                    "config": cfg.to_dict()})
    written = save_graph(cfg, store, entities=entities, relations=relations, logger=log)
    # graph.json 落到 --out-dir（save_graph 用的是 cfg.graph_dir；此处保证两者一致）
    if out_dir != Path(cfg.graph_dir):
        for key, path in list(written.items()):
            src = Path(path)
            dst = out_dir / src.name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            written[key] = str(dst)
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    summary = {
        "work_order": WORK_ORDER,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_source": str(chunks_path),
        "chunks_scanned": scanned,
        "entities": stats["entities"],
        "relations": stats["relations"],
        "mentions": stats["mentions"],
        "entities_by_type": stats["entities_by_type"],
        "relations_by_type": stats["relations_by_type"],
        "relations_by_extractor": stats["relations_by_extractor"],
        "counters": stats["counters"],
        "prune": prune,
        "prune_low_support": prune_low,
        "llm_supplement": llm_meta,
        "elapsed_ms": total_ms,
        "outputs": written,
        "config_fingerprint": cfg.fingerprint(),
    }
    (out_dir / GRAPH_FILES["build_stats"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log_event("build.done", **{k: v for k, v in summary.items() if k != "outputs"})
    log.close()
    print(f"[5/5] 完成：实体 {stats['entities']}，关系 {stats['relations']}，总耗时 {total_ms / 1000:.1f}s")
    print(f"      产物目录：{out_dir}")
    for key in ("entities", "relations", "graph", "graphml"):
        print(f"      - {key}: {written.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
