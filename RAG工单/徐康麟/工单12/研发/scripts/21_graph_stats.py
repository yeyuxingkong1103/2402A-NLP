# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""21_graph_stats.py —— 图谱统计与「类型分布」截图。

读 研发/data/graph/{nodes,edges}.jsonl，重算统计并落盘 graph_stats.json，
再画一张真实计数的类型分布图（中文命名，落 测试/测试截图/）。

用法（工作目录 = E:\\gao6gongdan\\工单12）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/21_graph_stats.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/21_graph_stats.py --no-viz

本脚本**不调 LLM**、不建索引，只做统计与出图。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from lightrag.config import load_config                                   # noqa: E402
from lightrag.graph_store import graph_stats, sampling_caliber_text       # noqa: E402
from lightrag.graph_viz import draw_type_distribution                     # noqa: E402
from lightrag.logging_util import get_logger, write_json                  # noqa: E402

DIST_PNG_NAME = "知识图谱类型分布.png"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LightRAG 图谱统计与类型分布截图")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-viz", action="store_true", help="只出统计，不出图")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("21_graph_stats", cfg, console=True)

    graph_dir = Path(cfg.path_str("graph_dir"))
    nodes = _read_jsonl(graph_dir / "nodes.jsonl")
    edges = _read_jsonl(graph_dir / "edges.jsonl")
    if not nodes:
        log.error("没有图谱数据：%s 不存在或为空（先跑 20_build_graph.py）", graph_dir / "nodes.jsonl")
        return 2

    st = graph_stats(nodes, edges)

    meta_path = graph_dir / "graph_meta.json"
    sampling: dict = {}
    caliber = ""
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sampling = meta.get("sampling") or {}
            caliber = str(meta.get("抽样口径") or "")
            if not caliber and sampling:
                caliber = sampling_caliber_text(sampling)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("graph_meta.json 读取失败：%s: %s", type(exc).__name__, exc)

    payload = dict(st)
    payload["抽样口径"] = caliber
    payload["sampling"] = sampling
    write_json(graph_dir / "graph_stats.json", payload)

    # ---- 控制台摘要（真实计数）
    print("=" * 78)
    print("LightRAG 知识图谱统计（工单编号：人工智能NLP-RAG 项目-LightRAG 优化）")
    print("=" * 78)
    print(f"节点数：{st['nodes']}    边数：{st['edges']}    边权重合计：{st['edge_weight_total']}")
    print(f"实体类型：使用 {st['entity_types_used']} / 声明 {st['entity_types_declared']} 类")
    print(f"关系类型：使用 {st['relation_types_used']} / 声明 {st['relation_types_declared']} 类")
    print(f"多别名节点：{st['entity_multi_alias_nodes']}    类型冲突节点：{st['entity_type_conflicts']}")
    print(f"实体类型准确率（原始类型合法占比）：{_rate(st)}")
    print("-" * 78)
    print("实体类型分布：", json.dumps(st["entity_type_distribution"], ensure_ascii=False))
    print("关系类型分布：", json.dumps(st["relation_type_distribution"], ensure_ascii=False))
    print("-" * 78)
    print("度数最高的实体（Top 10）：")
    for row in st["top_nodes_by_degree"][:10]:
        print(f"  {row['name']}  [{row['type']}]  度数={row['degree']}  出现={row['freq']}")
    print("-" * 78)
    print("权重最高的关系（Top 10）：")
    for row in st["top_edges_by_weight"][:10]:
        print(f"  {row['head']} —{row['rel']}→ {row['tail']}  ×{row['weight']}")
    if caliber:
        print("-" * 78)
        print(caliber)

    if not args.no_viz:
        try:
            png = Path(cfg.path_str("screenshot_dir")) / DIST_PNG_NAME
            footnote = (
                f"节点 {st['nodes']} 个、关系 {st['edges']} 条"
                f"｜实体类型 {st['entity_types_used']}/{st['entity_types_declared']} 类"
                f"｜关系类型 {st['relation_types_used']}/{st['relation_types_declared']} 类"
                f"｜数据源 研发/data/graph/nodes.jsonl + edges.jsonl"
            )
            out = draw_type_distribution(st["entity_type_distribution"], st["relation_type_distribution"],
                                         png, cfg=cfg,
                                         title="招股说明书知识图谱 类型分布（真实计数）", footnote=footnote)
            log.info("类型分布图已出图：%s", out)
            print(f"\n类型分布图：{out}")
        except Exception as exc:  # noqa: BLE001
            log.warning("类型分布出图失败：%s: %s", type(exc).__name__, exc)
            print(f"\n[WARN] 类型分布出图失败：{type(exc).__name__}: {exc}")

    print(f"统计落盘：{graph_dir / 'graph_stats.json'}")
    return 0


def _rate(st: dict) -> str:
    total = int(st.get("entity_type_mention_total") or 0)
    exact = int(st.get("entity_type_exact_mentions") or 0)
    return f"{exact}/{total}（{100.0 * exact / total:.1f}%）" if total else "—"


if __name__ == "__main__":
    raise SystemExit(main())
