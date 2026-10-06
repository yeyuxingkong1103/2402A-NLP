# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 模块：lightrag_rag/optimize_graph —— 知识图谱优化（工单 12 验收①「优化图谱构建中的实体、关系」）
# 说明：对 LightRAG 抽出的原始图谱做四项后处理，并输出优化前后统计对比：
#       ① 发行人别名归并：Company / The Company / Issuer / 发行人 / 本公司 / 兴图新科(有限) 等
#          统一归并到该文档对应的发行人全称（按实体来源块所属文档判定，避免跨文档串名）；
#       ② 实体类型规范化：中文自造类型（如「发行股数及占发行后总股本比例」）归入标准类型 data/concept；
#       ③ 清空噪声：剔除孤立节点（度=0）与无类型无描述的实体；
#       ④ 关系去重合并：同名同向边合并、描述去重截断。
#       产出 `data/lightrag_store/graph_optimized.graphml` + `lightrag_rag/graph_opt_stats.json`
#       + `static/lightrag_graph_opt.html`（可视化）。
# 用法（.venv_lightrag）：python lightrag_rag\optimize_graph.py
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(ROOT, "data", "lightrag_store", "graph_chunk_entity_relation.graphml")
DST = os.path.join(ROOT, "data", "lightrag_store", "graph_optimized.graphml")
OUT_HTML = os.path.join(ROOT, "static", "lightrag_graph_opt.html")
OUT_JSON = os.path.join(HERE, "graph_opt_stats.json")

# 两位发行人（两份招股书各一家）
ISSUERS = {"招股说明书1": "武汉兴图新科电子股份有限公司",
           "招股说明书2": "武汉力源信息技术股份有限公司"}
# 通用指代词（跨文档歧义 → 按来源块所属文档判定）
GENERIC = re.compile(r"^(the company|company|issuer|issuer company|本公司|发行人|公司|发行人（公司）|发行人\(公司\)|"
                     r"武汉兴图新科电子股份有限公司（发行人）)$", re.I)
# 明确的发行人简称 → 全称（仅这两家的写法，子公司/参股公司保持独立）
ISSUER_ALIAS = [
    (re.compile(r"^(兴图新科|兴图新科有限|武汉兴图新科电子有限公司|武汉兴图新科电子股份有限公司)$"), "招股说明书1"),
    (re.compile(r"^(力源信息|武汉力源信息技术股份有限公司|武汉力源信息)$"), "招股说明书2"),
]
TYPE_MAP = {"发行股数及占发行后总股本比例": "data", "发行前每股净资产": "data",
            "发行后每股净资产": "data", "发行方式": "concept", "发行对象": "concept",
            "承销方式": "concept"}


def chunk_doc_map():
    """chunk_id -> 文档名（靠文本里的【文档·第N页】标签）。"""
    p = os.path.join(ROOT, "data", "lightrag_store", "kv_store_text_chunks.json")
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for cid, v in d.items():
        t = (v.get("content") if isinstance(v, dict) else v) or ""
        m = re.search(r"【(.+?)·第\d+页】", t)
        if m:
            out[cid] = m.group(1)
    return out


def main():
    import networkx as nx
    g = nx.read_graphml(SRC)
    before = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
              "types": len(set((a.get("entity_type") or "") for _, a in g.nodes(data=True)))}
    cdoc = chunk_doc_map()

    def src_docs(a):
        docs = Counter()
        for cid in str(a.get("source_id") or "").split("<SEP>"):
            d = cdoc.get(cid.strip())
            if d:
                docs[d] += 1
        return docs

    # ---- ① 计算每个节点应归并到的规范名 ----
    canon = {}
    merged_alias = []
    for n, a in g.nodes(data=True):
        etype = (a.get("entity_type") or "").lower()
        if etype != "organization":
            continue
        if GENERIC.match(n):
            docs = src_docs(a)
            if docs:
                doc = docs.most_common(1)[0][0]
                canon[n] = ISSUERS.get(doc, n)
                merged_alias.append((n, canon[n]))
            continue
        for pat, doc in ISSUER_ALIAS:
            if pat.match(n):
                canon[n] = ISSUERS[doc]
                merged_alias.append((n, canon[n]))
                break

    H = nx.DiGraph()
    desc = {}

    def add_node(name, a):
        etype = a.get("entity_type") or ""
        etype = TYPE_MAP.get(etype, etype)
        if name in H:
            cur = H.nodes[name].get("entity_type") or ""
            H.nodes[name]["entity_type"] = cur or etype
            desc[name] = merge_desc(desc.get(name, ""), a.get("description") or "")
            return
        H.add_node(name, entity_type=etype, description=(a.get("description") or "")[:600],
                   source_id=a.get("source_id") or "", file_path=a.get("file_path") or "")
        desc[name] = a.get("description") or ""

    def merge_desc(x, y):
        if not y or y[:80] in x:
            return x
        return (x + " ｜ " + y)[:800]

    # ---- ② 建新图（别名归一 + 类型规范化）----
    for n, a in g.nodes(data=True):
        add_node(canon.get(n, n), a)
    for u, v, a in g.edges(data=True):
        U, V = canon.get(u, u), canon.get(v, v)
        if U == V:
            continue
        kw = (a.get("keywords") or api_desc(a) or "")[:120]
        if H.has_edge(U, V):
            old = H.edges[U, V].get("keywords") or ""
            H.edges[U, V]["keywords"] = (old if kw[:40] in old else (old + ";" + kw))[:300]
        else:
            H.add_edge(U, V, keywords=kw, description=(api_desc(a) or "")[:300])

    # ---- ③ 清空噪声 ----
    drop_iso = [n for n, d in H.degree() if d == 0]
    H.remove_nodes_from(drop_iso)
    drop_empty = [n for n, a in H.nodes(data=True)
                  if not (a.get("entity_type") or "").strip() and len(desc.get(n, "")) < 8]
    H.remove_nodes_from(drop_empty)

    for n in H.nodes:
        H.nodes[n]["description"] = desc.get(n, "")[:600]

    nx.write_graphml(H, DST)

    import viz_lightrag
    types = viz_lightrag.render(H, OUT_HTML)
    after = {"nodes": H.number_of_nodes(), "edges": H.number_of_edges(),
             "types": len(types), "isolated": sum(1 for _, d in H.degree() if d == 0)}
    out = {"before": before, "after": after,
           "merged_alias": [{"alias": x, "to": y} for x, y in merged_alias],
           "dropped_isolated": drop_iso, "dropped_empty": drop_empty,
           "entity_types_after": dict(types.most_common(12))}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("【图谱优化】节点 %d → %d ｜ 边 %d → %d ｜ 实体类型 %d → %d" %
          (before["nodes"], after["nodes"], before["edges"], after["edges"],
           before["types"], after["types"]))
    print("别名归并 %d 项：" % len(merged_alias),
          "、".join("%s→%s" % (x, y) for x, y in merged_alias[:8]), "…" if len(merged_alias) > 8 else "")
    print("剔除孤立节点 %d 个、空实体 %d 个" % (len(drop_iso), len(drop_empty)))
    print("优化后类型分布：", dict(types.most_common(8)))
    print("saved ->", DST, "|", OUT_HTML, "|", OUT_JSON)


def api_desc(a):
    return a.get("description") or ""


if __name__ == "__main__":
    main()
