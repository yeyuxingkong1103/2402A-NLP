# -*- coding: utf-8 -*-
"""工单8 图谱可视化脚本：力导向布局 PNG + 自包含 SVG/HTML。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

产物（默认落在 ``研发/data/graph/``，PNG 另存到 ``测试/测试截图/``）：
    * ``测试/测试截图/知识图谱可视化.png``（中文命名，真实统计角标）
    * ``研发/data/graph/graph.svg`` / ``graph.html``（自包含，无外网依赖）
    * ``研发/data/graph/可视化统计.json``（节点/边/布局耗时等实测数字）

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/render_graph_viz.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/render_graph_viz.py --top-nodes 160 --label-min-degree 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

from app.graphrag.config import GRAPH_FILES, WORK_ORDER, ensure_dirs, get_config  # noqa: E402
from app.graphrag.store import graph_digest, load_graph  # noqa: E402
from app.graphrag.visualize import (force_directed_layout, render_html, render_png,  # noqa: E402
                                    render_svg, select_subgraph)

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 图谱可视化：{WORK_ORDER}")
    parser.add_argument("--top-nodes", type=int, default=None, help="绘制节点数（默认取配置 top_nodes=120）")
    parser.add_argument("--max-edges", type=int, default=None, help="最多绘制边数（默认取配置 400）")
    parser.add_argument("--label-min-degree", type=int, default=None, help="度数 ≥ 该值的节点才画标签")
    parser.add_argument("--iterations", type=int, default=None, help="力导向迭代轮数")
    parser.add_argument("--canvas", default=None, help="画布尺寸，形如 2400x1600")
    parser.add_argument("--out-png", default=None, help="PNG 输出路径（默认 测试/测试截图/知识图谱可视化.png）")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    viz = dict(cfg.viz)
    top_nodes = int(args.top_nodes or viz.get("top_nodes", 120))
    max_edges = int(args.max_edges or viz.get("max_edges", 400))
    label_min_degree = int(args.label_min_degree if args.label_min_degree is not None
                           else viz.get("label_min_degree", 2))
    iterations = int(args.iterations or viz.get("iterations", 320))
    canvas = args.canvas or "x".join(str(v) for v in (viz.get("canvas") or [2400, 1600]))
    width, height = (int(x) for x in str(canvas).lower().split("x"))
    out_png = Path(args.out_png) if args.out_png else (_ROOT / "测试" / "测试截图" / "知识图谱可视化.png")

    started = time.perf_counter()
    store = load_graph(cfg, logger=None)
    if not store.ready:
        print(f"[ERROR] 图谱不可用：{cfg.graph_dir}（请先跑 研发/scripts/build_graph.py）")
        return 2
    print(f"[1/4] 图谱加载：{store.node_count} 节点 / {store.edge_count} 边（{store.load_ms} ms）")

    nodes, edges = select_subgraph(store, top_nodes=top_nodes, max_edges=max_edges,
                                   min_edges=int(viz.get("min_edges", 120)),
                                   min_degree=int(viz.get("min_degree", 2)),
                                   seed=int(viz.get("random_seed", 42)))
    print(f"[2/4] 选取核心子图：{len(nodes)} 节点 / {len(edges)} 边（按度数 Top-N，边不足时自动放宽名额）")

    layout = force_directed_layout(nodes, edges, width=width, height=height, iterations=iterations,
                                   repulsion=float(viz.get("repulsion", 12000.0)),
                                   spring=float(viz.get("spring", 0.012)),
                                   damping=float(viz.get("damping", 0.86)),
                                   seed=int(viz.get("random_seed", 42)))
    print(f"      力导向布局完成：{layout.stats['layout_ms']} ms / {iterations} 轮")

    stats_full = graph_digest(store, max_nodes=1, max_edges=1)
    title = "工单8 金融知识图谱（9 份 A 股年报 · Graph RAG）"
    subtitle = (f"实体 {store.node_count} · 关系 {store.edge_count} · 提及边 {len(store.chunk_meta)} 块 "
                f"· 本图展示度数最高的 {len(nodes)} 个实体与 {len(edges)} 条关系")
    png_info = render_png(layout, out_png, title=title, subtitle=subtitle,
                          legend=bool(viz.get("legend", True)), label_min_degree=label_min_degree,
                          radius_min=int(viz.get("node_radius_min", 7)),
                          radius_max=int(viz.get("node_radius_max", 26)))
    print(f"[3/4] PNG：{png_info['png']}（{png_info['bytes'] // 1024} KB，节点 {png_info['nodes']}，"
          f"边 {png_info['edges_drawn']}，图例 {png_info['legend_rows']} 项）")

    overview = store.to_public_dict(max_nodes=top_nodes, max_edges=max_edges)
    svg_info = render_svg(layout, cfg.graph_file("svg"), title=title, subtitle=subtitle)
    html_info = render_html(layout, cfg.graph_file("html"), title=title, subtitle=subtitle,
                            full_graph=overview)
    print(f"[4/4] SVG：{svg_info['svg']}（外链 {svg_info['external_refs']}） | "
          f"HTML：{html_info['html']}（外链 {html_info['external_refs']}）")

    summary = {
        "work_order": WORK_ORDER,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_dir": str(cfg.graph_dir),
        "graph_nodes_total": store.node_count,
        "graph_edges_total": store.edge_count,
        "graph_chunks_total": len(store.chunk_meta),
        "graph_by_entity_type": stats_full["by_entity_type"],
        "graph_by_relation_group": stats_full["by_relation_group"],
        "layout": layout.stats,
        "png": png_info, "svg": svg_info, "html": html_info,
        "top_node_names": [n["name"] for n in nodes[:20]],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    out_json = Path(cfg.graph_dir) / "可视化统计.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      统计：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
