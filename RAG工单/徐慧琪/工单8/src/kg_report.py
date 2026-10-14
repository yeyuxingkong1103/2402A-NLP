# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""模块：由图谱产物生成构建报告（docs/03 的数据章节）。"""
from __future__ import annotations

import io
import os
from collections import Counter

from src import config
from src.kg_dict import METRIC_TABLE
from src.kg_schema import ENTITY_TYPES, RELATION_TYPES
from src.kg_store import KnowledgeGraph


def write_report(path: str | None = None) -> str:
    path = path or os.path.join(config.DOCS_DIR, "03-知识图谱构建.md")
    kg = KnowledgeGraph.load()
    st = kg.stats()

    # 每份文档贡献的边数
    doc_edges: Counter = Counter()
    for _u, _v, d in kg.g.edges(data=True):
        if d.get("source_doc"):
            doc_edges[d["source_doc"]] += 1

    top_companies = []
    for n in kg.nodes_of_type("Company"):
        top_companies.append((kg.g.nodes[n]["name"], kg.g.degree(n)))
    top_companies.sort(key=lambda x: -x[1])

    lines = [
        "# 知识图谱构建（工单08 · 硬性要求1/3）", "",
        "> 本文件由 `python build_kg.py --report` 自动生成。", "",
        "## 一、图谱规模", "",
        f"- 实体节点：**{st['n_nodes']}**",
        f"- 关系边：**{st['n_edges']}**",
        f"- 覆盖文档：**{st['n_docs']}** 份年报",
        f"- 带 provenance 的边：**{st['edges_with_provenance']}** "
        f"（占比 {st['edges_with_provenance'] / max(st['n_edges'], 1):.1%}）", "",
        "## 二、实体类型分布", "",
        "| 实体类型 | 说明 | 数量 |", "| --- | --- | --- |",
    ]
    for t, n in sorted(st["by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{t}` | {ENTITY_TYPES.get(t, '—')} | {n} |")

    lines += ["", "## 三、关系类型分布", "",
              "| 关系类型 | 说明 | 数量 |", "| --- | --- | --- |"]
    for t, n in sorted(st["by_relation"].items(), key=lambda x: -x[1]):
        desc = RELATION_TYPES.get(t, {}).get("desc", "—")
        lines.append(f"| `{t}` | {desc} | {n} |")

    lines += ["", "## 四、各文档贡献", "",
              "| 来源文档 | 关系边数 |", "| --- | --- |"]
    for doc, n in doc_edges.most_common():
        lines.append(f"| {doc} | {n} |")

    lines += ["", "## 五、核心公司节点度数（Top 12）", "",
              "| 公司 | 关联边数 |", "| --- | --- |"]
    for name, deg in top_companies[:12]:
        lines.append(f"| {name} | {deg} |")

    lines += ["", "## 六、指标覆盖", "",
              f"词典定义指标 {len(METRIC_TABLE)} 个，"
              f"图谱中实际出现 "
              f"{len({kg.g.nodes[n]['attrs'].get('metric') for n in kg.nodes_of_type('Metric')})} 个。",
              "", "| 指标 | 图谱节点数 |", "| --- | --- |"]
    cnt = Counter(kg.g.nodes[n]["attrs"].get("metric")
                  for n in kg.nodes_of_type("Metric"))
    for m, n in cnt.most_common(30):
        lines.append(f"| {m or '—'} | {n} |")

    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
