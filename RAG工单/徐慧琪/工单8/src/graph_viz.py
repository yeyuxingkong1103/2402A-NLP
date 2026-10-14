# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：知识图谱可视化（硬性要求1「知识图谱需可视化展示」）

双方案，保证任何环境下都能看到图：
  1. **交互式（主）**：生成自包含 HTML，用 vis.js（CDN 加载）做力导向布局；
     多类型实体用不同颜色节点、关系用带标签连线 —— 对齐工单图谱示例。
  2. **静态（兜底）**：networkx + matplotlib 渲染 PNG。
     无网络 / CDN 不可达时仍有可用产出（本机无 dot.exe，故不用 graphviz）。

安全：所有注入 HTML 的文本都做 JSON 转义（`<` → `\\u003c`），
防止实体名中的 `</script>` 破坏页面。
"""
from __future__ import annotations

import json
import socket
import urllib.request

from src.kg_schema import ENTITY_TYPES, RELATION_TYPES, node_color
from src.kg_store import KnowledgeGraph

VIS_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"


def network_available(timeout: float = 2.0) -> bool:
    """探测 CDN 可达性（仅用于决定是否提示用户走静态兜底）。"""
    try:
        socket.setdefaulttimeout(timeout)
        with urllib.request.urlopen(VIS_CDN, timeout=timeout):
            return True
    except Exception:                              # noqa: BLE001
        return False


def _json_for_script(obj) -> str:
    """把对象序列化为可安全嵌入 <script> 的 JSON。"""
    text = json.dumps(obj, ensure_ascii=False)
    return (text.replace("<", "\\u003c").replace(">", "\\u003e")
                .replace("&", "\\u0026").replace(" ", "\\u2028")
                .replace(" ", "\\u2029"))


def to_vis_html(nodes: list[dict], edges: list[dict], height: int = 620,
                title: str = "") -> str:
    """生成交互式图谱 HTML（自包含，vis.js 走 CDN）。"""
    vis_nodes = [{"id": n["id"], "label": n.get("label", n["id"]),
                  "group": n.get("type", "?"),
                  "color": n.get("color") or node_color(n.get("type", "?")),
                  "title": _node_tooltip(n),
                  "shape": "dot", "size": 14}
                 for n in nodes]
    vis_edges = [{"from": e["from"], "to": e["to"],
                  "label": e.get("label", ""),
                  "title": e.get("evidence", "") or e.get("label", ""),
                  "arrows": "to",
                  "font": {"size": 10, "align": "middle"}}
                 for e in edges]

    legend = "".join(
        f'<span class="lg"><i style="background:{node_color(t)}"></i>{t}</span>'
        for t in ENTITY_TYPES)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{_escape(title or "知识图谱可视化")}</title>
<script type="text/javascript" src="{VIS_CDN}"></script>
<style>
  body {{ margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif;
          background:#fafbfc; }}
  #hdr {{ padding:10px 16px; background:#fff; border-bottom:1px solid #e5e7eb; }}
  #hdr h3 {{ margin:0 0 6px; font-size:15px; }}
  #lg {{ display:flex; flex-wrap:wrap; gap:10px; font-size:12px; color:#444; }}
  .lg {{ display:inline-flex; align-items:center; gap:4px; }}
  .lg i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  #net {{ width:100%; height:{height}px; }}
  #tip {{ padding:8px 16px; font-size:12px; color:#666; }}
</style>
</head>
<body>
<div id="hdr">
  <h3>{_escape(title or "工单08 金融知识图谱")}（{len(vis_nodes)} 个实体 / {len(vis_edges)} 条关系）</h3>
  <div id="lg">{legend}</div>
</div>
<div id="net"></div>
<div id="tip">提示：滚轮缩放、拖拽平移；点击节点查看属性与来源。</div>
<script type="text/javascript">
  var nodes = new vis.DataSet({_json_for_script(vis_nodes)});
  var edges = new vis.DataSet({_json_for_script(vis_edges)});
  var container = document.getElementById('net');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    nodes: {{ borderWidth: 1.5, font: {{ size: 13 }} }},
    edges: {{ color: {{ color: '#b9bec7' }}, smooth: {{ type: 'continuous' }} }},
    physics: {{ stabilization: {{ iterations: 220 }}, barnesHut: {{
      gravitationalConstant: -9000, springLength: 130, springConstant: 0.04 }} }},
    interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true }}
  }};
  var network = new vis.Network(container, data, options);
</script>
</body>
</html>"""


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _node_tooltip(node: dict) -> str:
    attrs = node.get("attrs") or {}
    lines = [f"类型：{node.get('type', '?')}"]
    for k in ("metric", "value", "unit", "period", "yoy", "code", "industry",
              "source_doc", "page", "full"):
        if attrs.get(k) not in (None, "", []):
            lines.append(f"{k}：{attrs[k]}")
    return "\n".join(lines)


def filter_subgraph(kg: KnowledgeGraph, companies: list[str] | None = None,
                    types: list[str] | None = None, hops: int = 1,
                    max_nodes: int = 150) -> tuple[list[dict], list[dict]]:
    """按公司/类型/hop 过滤图谱，返回 (nodes, edges)（均为可视化就绪的 dict）。"""
    seeds: list[str] = []
    if companies:
        wanted = set(companies)
        seeds = [n for n in kg.nodes_of_type("Company")
                 if kg.g.nodes[n].get("name") in wanted]
    if not seeds:
        seeds = [n for n in kg.nodes_of_type("Company")][:3]
    if not seeds:
        seeds = kg.entity_ids()[:3]

    keep: set[str] = set()
    for s in seeds:
        keep.update(kg.neighbors(s, hops=hops, max_nodes=max_nodes).g.nodes)
        keep.add(s)
        if len(keep) >= max_nodes:
            break

    if types:
        keep = {n for n in keep if kg.g.nodes[n].get("type") in types}

    keep = set(list(keep)[:max_nodes])

    nodes = []
    for n in keep:
        d = kg.g.nodes[n]
        label = d.get("name", n)
        if d.get("type") == "Metric":
            a = d.get("attrs", {})
            if a.get("value") is not None:
                label = f"{a.get('metric', label)} {a['value']}{a.get('unit', '')}"
        nodes.append({"id": n, "label": label, "type": d.get("type", "?"),
                      "color": node_color(d.get("type", "?")),
                      "attrs": {k: v for k, v in d.get("attrs", {}).items()
                                if k in ("metric", "value", "unit", "period",
                                         "yoy", "source_doc", "page", "code",
                                         "industry", "full")}})

    edges = []
    for u, v, d in kg.g.edges(data=True):
        if u in keep and v in keep:
            edges.append({"from": u, "to": v, "label": d.get("rel_type", ""),
                          "evidence": (d.get("evidence") or "")[:200],
                          "source_doc": d.get("source_doc", ""),
                          "page": int(d.get("page", -1))})
    return nodes, edges


def to_matplotlib_png(nodes: list[dict], edges: list[dict], path: str,
                      dpi: int = 140) -> str:
    """离线兜底：networkx + matplotlib 静态渲染（无网络也能出图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    from matplotlib import font_manager
    # 中文字体：优先微软雅黑，缺失则退化为默认（图仍可出，字可能变方框）
    for fam in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
        if any(fam.lower() in f.name.lower()
               for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], label=n.get("label", n["id"]),
                   color=n.get("color") or node_color(n.get("type", "?")))
    for e in edges:
        if e["from"] in g and e["to"] in g:
            g.add_edge(e["from"], e["to"], label=e.get("label", ""))

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, k=0.9, iterations=120, seed=42)
    nx.draw_networkx_nodes(
        g, pos, node_size=420,
        node_color=[g.nodes[n].get("color", "#9AA0A6") for n in g.nodes])
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=9, alpha=0.5,
                           edge_color="#b9bec7")
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edge_labels(
        g, pos, font_size=6,
        edge_labels={(u, v): d.get("label", "") for u, v, d in g.edges(data=True)})
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
