# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""20_build_graph.py —— LightRAG 知识图谱构建（索引构建流程第 ⑤~⑦ 步）。

流程：抽样 → LLM 逐块抽取（串行）→ 实体归并 / 关系去重 → 导出图谱 → 可视化。

用法（工作目录 = E:\\gao6gongdan\\工单12）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/20_build_graph.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/20_build_graph.py --pilot 3      # 计时试点
    pwsh -NoProfile -File run_py.ps1 研发/scripts/20_build_graph.py --resume        # 断点续跑（默认开）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/20_build_graph.py --sample-size 200 --time-budget 900

重活必须经 安全跑.ps1 包一层（监察内存并落盘日志）：
    pwsh -NoProfile -File E:\\gao6gongdan\\_共享工具\\安全跑.ps1 `
         -WorkDir "E:\\gao6gongdan\\工单12" `
         -Command "pwsh -NoProfile -File run_py.ps1 研发/scripts/20_build_graph.py" `
         -LogName 20_build_graph

纪律：只跑小样本（≤400 块，绝不跑全量 3013 块）；单块 1 次 LLM 调用、全程串行；
      失败块入 failed_chunks.jsonl 后继续；逐块 append，支持 --resume；阈值全部来自配置。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from lightrag.config import dump_effective_config, load_config                      # noqa: E402
from lightrag.corpus import load_chunks                                             # noqa: E402
from lightrag.graph_extract import (load_question_pages, read_done_ids,             # noqa: E402
                                    run as run_extract, select_sample, unload_models)
from lightrag.graph_store import (build_graph, load_records, sampling_caliber_text,  # noqa: E402
                                 save_graph)
from lightrag.graph_viz import draw_graph                                           # noqa: E402
from lightrag.logging_util import get_logger, write_json                            # noqa: E402
from lightrag.ollama_client import OllamaClient                                     # noqa: E402

GRAPH_PNG_NAME = "知识图谱可视化.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LightRAG 知识图谱构建（小样本抽样 + LLM 抽取）")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 部署/配置/lightrag.json）")
    ap.add_argument("--sample-size", type=int, default=None, help="抽样块数上限（默认取配置 graph.sample_size）")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="本次跑数的时间预算（秒），到点先落盘退出，默认取配置 graph.extract_time_budget_s")
    ap.add_argument("--pilot", type=int, default=0, help="计时试点：只跑样本中均匀抽取的 N 块")
    ap.add_argument("--target-total", type=int, default=None,
                    help="本次跑数后 raw_extractions.jsonl 的期望总记录数（配合 --resume 用来把样本从 200 砍到 80）；到量即停")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有 raw_extractions.jsonl，重跑全部样本")
    ap.add_argument("--skip-viz", action="store_true", help="跳过可视化出图")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("20_build_graph", cfg, console=True)
    t_all = time.perf_counter()
    summary: dict[str, object] = {"work_order": "人工智能NLP-RAG 项目-LightRAG 优化", "step": "20_build_graph"}

    # 生效配置留痕（审计「本次跑数用的是哪套阈值」）
    try:
        write_json(Path(cfg.path_str("log_dir")) / "effective_config.json", cfg.to_dict())
    except Exception as exc:  # noqa: BLE001
        log.warning("生效配置落盘失败：%s: %s", type(exc).__name__, exc)

    # ---------------------------------------------------------------- 载入语料
    chunks_path = Path(cfg.path_str("processed_dir")) / "chunks.jsonl"
    if not chunks_path.is_file():
        log.error("找不到块表：%s", chunks_path)
        return 2
    chunks = [c.to_dict() for c in load_chunks(chunks_path)]
    log.info("载入块表 %s：%d 块", chunks_path.name, len(chunks))

    # ---------------------------------------------------------------- 抽样
    q_path = Path(str(cfg.eval.question_file))
    q_pages = load_question_pages(q_path, tolerance=int(cfg.eval.page_tolerance))
    log.info("16 题命中页（含 ±%d 页容差）：%s", int(cfg.eval.page_tolerance),
             {k: len(v) for k, v in q_pages.items()})
    sample, sample_meta = select_sample(
        chunks, cfg, question_pages=q_pages, limit=args.sample_size, logger=log)

    if args.pilot and args.pilot > 0 and len(sample) > args.pilot:
        step = (len(sample) - 1) / max(args.pilot - 1, 1)
        sample = [sample[min(int(round(i * step)), len(sample) - 1)] for i in range(args.pilot)]
        log.info("试点模式：从样本中均匀抽取 %d 块", len(sample))

    if not sample:
        log.error("抽样结果为空，终止")
        summary["error"] = "抽样为空"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 3

    graph_dir = cfg.path("graph_dir")
    raw_path = Path(graph_dir) / "raw_extractions.jsonl"
    failed_path = Path(graph_dir) / "failed_chunks.jsonl"
    resume = not args.no_resume
    done_before = read_done_ids(raw_path) if resume else set()
    log.info("抽样口径：%d/%d 块（种子 %s）；已存在抽取记录 %d 条；resume=%s",
             sample_meta["selected"], sample_meta["chunk_total"], sample_meta["sample_seed"],
             len(done_before), resume)

    # ---------------------------------------------------------------- 探活
    client = OllamaClient(cfg, logger=log)
    health = client.health()
    if not health.get("ok"):
        log.error("Ollama 不可用或缺少模型：missing=%s error=%s", health.get("missing"), health.get("error"))
        summary["error"] = f"Ollama 探活失败：{health}"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 4
    log.info("Ollama 探活通过：生成模型=%s（嵌入模型本步不用，避免多模型同时驻留）", cfg.llm.extract_model)

    # ---------------------------------------------------------------- 抽取
    budget = args.time_budget if args.time_budget is not None else float(cfg.graph.get("extract_time_budget_s") or 0)
    budget = float(budget) if budget and budget > 0 else None
    max_new = None
    if args.target_total is not None:
        max_new = max(int(args.target_total) - len(done_before), 0)
        log.info("目标总量 %d，已存在 %d 条，本次最多再抽 %d 块",
                 int(args.target_total), len(done_before), max_new)
    try:
        estats = run_extract(sample, client, cfg, raw_path=raw_path, failed_path=failed_path,
                             resume=resume, time_budget_s=budget, max_new=max_new, logger=log)
        log.info("抽取完成：尝试 %d 块，成功 %d，失败 %d，调用耗时 %.1fs，墙钟 %.1fs",
                 estats["attempted"], estats["succeeded"], estats["failed"],
                 estats["call_seconds"], estats["wall_seconds"])
    finally:
        unload_models(cfg, logger=log)          # 收尾显式卸载，避免模型常驻

    # ---------------------------------------------------------------- 建图
    records = load_records(raw_path)
    log.info("读回抽取记录 %d 条（含历史续跑）", len(records))
    if not records:
        summary["error"] = "无任何抽取记录，无法建图"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 5
    nodes, edges, st = build_graph(records, cfg, logger=log)

    sample_meta["extracted_records"] = len(records)   # 实际建成图谱的记录数（可能 < selected）
    caliber = sampling_caliber_text(sample_meta)
    meta: dict[str, object] = {
        "work_order": "人工智能NLP-RAG 项目-LightRAG 优化",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "抽样口径": caliber,
        "sampling": sample_meta,
        "extract_stats": estats,
        "graph_stats": st,
        "llm": {"extract_model": str(cfg.llm.extract_model), "keep_alive": str(cfg.llm.keep_alive),
                "temperature": float(cfg.llm.extract_temperature),
                "batch_chunks": int(cfg.graph.batch_chunks)},
        "paths": {"raw_extractions": str(raw_path), "failed_chunks": str(failed_path)},
        "wall_seconds": round(time.perf_counter() - t_all, 3),
    }

    # ---------------------------------------------------------------- 可视化
    if not args.skip_viz:
        try:
            png = Path(cfg.path_str("screenshot_dir")) / GRAPH_PNG_NAME
            footnote = (
                f"小样本建图：实际建图 {sample_meta.get('extracted_records') or sample_meta['selected']} 块"
                f" / 全量 {sample_meta['chunk_total']} 块"
                f"｜节点 {len(nodes)} 个、关系 {len(edges)} 条"
                f"（约束相容 {st['edges_constraint_ok']} / 已标记 {st['edges_constraint_flagged']}）"
                f"｜实体类型 {st['entity_types_used']}/{st['entity_types_declared']} 类"
                f"｜关系类型 {st['relation_types_used']}/{st['relation_types_declared']} 类"
                f"｜抽取用模型 {cfg.llm.extract_model}"
            )
            out = draw_graph(nodes, edges, png, cfg=cfg,
                             title="招股说明书知识图谱（LightRAG 实体关系图）", footnote=footnote)
            meta["viz"] = {"知识图谱可视化": str(out)}
            log.info("图谱可视化已出图：%s", out)
        except Exception as exc:  # noqa: BLE001 - 出图失败不影响图谱产物
            log.warning("图谱可视化失败：%s: %s", type(exc).__name__, exc)
            meta["viz_error"] = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- 落盘
    paths = save_graph(nodes, edges, cfg=cfg, meta=meta)
    write_json(Path(graph_dir) / "graph_stats.json", st)
    log.info("图谱产物：%s", json.dumps(paths, ensure_ascii=False))

    summary.update({
        "nodes": len(nodes), "edges": len(edges),
        "entity_type_distribution": st["entity_type_distribution"],
        "relation_type_distribution": st["relation_type_distribution"],
        "sample": {"selected": sample_meta["selected"], "chunk_total": sample_meta["chunk_total"],
                   "seed": sample_meta["sample_seed"],
                   "layer_counts": sample_meta["layer_counts"],
                   "question_page_picked": sample_meta["question_page_picked"]},
        "extract": {"attempted": estats["attempted"], "succeeded": estats["succeeded"],
                    "failed": estats["failed"], "wall_seconds": estats["wall_seconds"],
                    "stopped_by_budget": estats["stopped_by_budget"], "remaining": estats["remaining"]},
        "caliber": caliber,
        "paths": paths,
        "wall_seconds_total": round(time.perf_counter() - t_all, 3),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
