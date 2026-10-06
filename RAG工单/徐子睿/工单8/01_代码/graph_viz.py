# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
# 关联工单：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：graph_viz —— 知识图谱可视化（pyvis，交互式 HTML）
# 说明：把 graph.json 渲染成可交互图谱（公司/行业/指标/年份 分色，边标关系与数值），
#       并支持按问题渲染「相关子图」。输出 static/graph.html 与 static/graph_query.html。
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import config                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")
GRAPH_JSON = os.path.join(config.DATA_DIR, "graph", "graph.json")

COLOR = {"公司": "#4c8dff", "行业": "#38d39f", "指标": "#f7b731", "年份": "#a55eea"}


def render(nodes, links, out_path, title="金融知识图谱", height="640px"):
    from pyvis.network import Network
    net = Network(height=height, width="100%", directed=True, bgcolor="#0f1115",
                  font_color="#e6e9ef", notebook=False, cdn_resources="in_line")
    for n in nodes:
        t = n.get("type", "指标")
        net.add_node(n["id"], label=n["id"], color=COLOR.get(t, "#98a1b2"),
                     title="%s（%s）" % (n["id"], t), shape="dot",
                     size=22 if t == "公司" else 14)
    for l in links:
        rel = l.get("rel", "")
        if rel == "披露指标":
            lbl = "%s %s" % (l.get("year", ""), l.get("value", ""))
        else:
            lbl = rel
        net.add_edge(l["source"], l["target"], label=lbl, title=lbl,
                     color="#3a4152", arrows="to")
    net.set_options(json.dumps({
        "physics": {"barnesHut": {"gravitationalConstant": -12000, "springLength": 160}},
        "edges": {"font": {"size": 11, "color": "#98a1b2"}},
        "interaction": {"hover": True, "navigationButtons": True},
    }))
    os.makedirs(STATIC, exist_ok=True)
    # 用 generate_html + utf-8 写入（pyvis 默认按系统编码，中文/版权符号会报错）
    try:
        html = net.generate_html(notebook=False)
    except Exception:  # noqa: BLE001
        html = net.html or ""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html or "")
    print("saved ->", out_path, len(html or ""), "bytes")


def main():
    d = json.load(open(GRAPH_JSON, encoding="utf-8"))
    # 全图（去掉“报告年度”边，避免过密；保留 属于行业 + 披露指标）
    links = [l for l in d["links"] if l.get("rel") != "报告年度"]
    render(d["nodes"], links, os.path.join(STATIC, "graph.html"), "金融知识图谱（全图）")

    # 按问题渲染子图
    import graph_rag
    q = sys.argv[1] if len(sys.argv) > 1 else "平安银行2019年末的拨备覆盖率是多少？"
    sg = graph_rag.subgraph(q)
    render(sg["nodes"], sg["links"], os.path.join(STATIC, "graph_query.html"),
           "问题相关子图：%s" % q)
    print("问题：", q, "｜ 子图节点 %d / 边 %d" % (len(sg["nodes"]), len(sg["links"])))


if __name__ == "__main__":
    main()
