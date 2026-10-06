# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
# 关联工单：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：graph_rag —— 图谱增强检索（Graph RAG）
# 说明：把问题里的「公司 / 指标 / 年份」实体链接到知识图谱，沿「公司-披露指标(年份)-数值」边
#       取到来源页，再回查知识库对应页的文本块作为上下文；图谱未覆盖时回退到向量+BM25 混合检索。
#       同时返回子图结构（节点/边），供前端可视化与「解析出的知识图谱结构」展示。
import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import config                     # noqa: E402
import engine                     # noqa: E402
import kb as kb_mod               # noqa: E402
from graph_build import INDUSTRY, INDICATORS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 图谱路径跟随 config.DATA_DIR（容器部署时挂载到 /data，见 Dockerfile）
GRAPH_JSON = os.path.join(config.DATA_DIR, "graph", "graph.json")

# 工单 09 开关：图谰证据页置顶策略（用于「优化前/优化后」对比）
#   v1  = 工单 08 初版：叙事句来源页一律置顶（最多 2 页）
#   opt = 工单 09 优化版：仅当“混合检索漏掉的页”且命中 ≥2 个指标时补充置顶（默认）
PREPEND_MODE = os.environ.get("GRAPH_MODE", "opt")

_CACHE = {"g": None}


def load_graph():
    if _CACHE["g"] is None:
        d = json.load(open(GRAPH_JSON, encoding="utf-8"))
        nodes = {n["id"]: n for n in d["nodes"]}
        _CACHE["g"] = {"nodes": nodes, "links": d["links"]}
    return _CACHE["g"]


def link_query(q):
    """实体链接：公司 / 指标 / 年份。"""
    co = [c for c in INDUSTRY if c in q]
    inds = [i for i in INDICATORS if i in q]
    years = re.findall(r"(20\d{2})", q)
    return co, inds, years


def subgraph(q):
    """取与问题相关的子图（用于可视化 + 结构展示）。"""
    g = load_graph()
    co, inds, years = link_query(q)
    keep = set(co)
    links = []
    for l in g["links"]:
        if l.get("rel") != "披露指标":
            continue
        if co and l["source"] not in co:
            continue
        if inds and l["target"] not in inds:
            continue
        if years and l.get("year") not in years:
            continue
        links.append(l)
        keep.add(l["source"])
        keep.add(l["target"])
    nodes = [g["nodes"][k] for k in keep if k in g["nodes"]]
    return {"nodes": nodes, "links": links}


def graph_pages(q):
    """图谱命中的来源页（按 (公司,指标,年份) 精确到页）。"""
    g = load_graph()
    co, inds, years = link_query(q)
    out = []
    for l in g["links"]:
        if l.get("rel") != "披露指标":
            continue
        if co and l["source"] not in co:
            continue
        if inds and l["target"] not in inds:
            continue
        if years and l.get("year") not in years:
            continue
        out.append((l["source"], l["target"], l.get("year"), l.get("value"), l.get("unit"),
                    l.get("page"), l.get("src", "table")))
    return out


def retrieve(q, top_k=5, use_graph=True):
    """Graph RAG 检索：以混合检索为主序，用知识图谱命中的来源页做加权（而非直接置顶，
    避免少量噪声三元组把错页摆到前面）；返回 (hits, conf, graph_info)。"""
    km = kb_mod.get_kb()
    co, inds, years = link_query(q)
    doc = engine.detect_doc(q) or (co[0] if co else None)

    gp = graph_pages(q)
    # 仅保留「值确实出现在该页」的三元组（表格行首列易串列），并建 (doc,page)->[chunks] 索引
    by_dp = {}
    for c in km.chunks:
        by_dp.setdefault((c["doc"], c["page"]), []).append(c)

    def _pick(comp, page, val):
        """在该页的块里选含该数值的块（没有则返回首块）。"""
        cs = by_dp.get((comp, page)) or []
        vt = (val or "").replace(",", "")
        if vt:
            for c in cs:
                if vt in c["text"].replace(",", ""):
                    return c
        return cs[0] if cs else None

    gp_ok = []
    pagescore = {}          # 叙事页 -> 该页已验证的三元组数
    for comp, ind, yr, val, unit, page, src in gp:
        cs = by_dp.get((comp, page))
        if not cs:
            continue
        vt = (val or "").replace(",", "")
        if not vt:          # 工单 09 优化：空值三元组不参与（容易把无关页拉进来）
            continue
        if not any(vt in c["text"].replace(",", "") for c in cs):
            continue
        gp_ok.append((comp, ind, yr, val, unit, page, src))
        if src == "narrative":
            pagescore[page] = pagescore.get(page, 0) + 1

    base, conf = engine.retrieve(q, top_k=top_k + 4, doc=doc)
    baserank = {h["page"]: i for i, h in enumerate(base)}

    def _ind_hits(page, comp):
        cs = by_dp.get((comp, page)) or []
        txt = " ".join(c["text"] for c in cs)
        return sum(1 for i in inds if i in txt)

    if PREPEND_MODE == "v1":
        # 工单 08 初版：叙事页按“已验证三元组数”降序置顶（最多 2 页）
        gorder = sorted(pagescore, key=lambda p: (-pagescore[p], baserank.get(p, 999)))[:2]
    else:
        # 工单 09 优化：① 只补充“混合检索完全没召回”的页；② 需命中 ≥2 个指标；仅 1 页
        cand = [p for p in pagescore if p not in baserank]
        gorder = []
        if cand and inds:
            best = max(cand, key=lambda p: (_ind_hits(p, doc or co[0]), pagescore[p]))
            if _ind_hits(best, doc or co[0]) >= 2:
                gorder = [best]
    gpages = set(pagescore)

    hits, seen = [], set()
    # ① 图谱证据页补充置顶
    for p in gorder:
        best = None
        for comp, ind, yr, val, unit, pg, src in gp_ok:
            if pg != p:
                continue
            c = _pick(comp, p, val)
            if c:
                best = c
                break
        if best is None:
            continue
        key = (best["doc"], best["page"], best["text"][:40])
        if key in seen:
            continue
        seen.add(key)
        h = dict(best)
        h.update({"rrf": 99.0, "dense_sim": 0.0, "source": "graph"})
        hits.append(h)
    # ② 混合检索结果：图谱页加权（不直接置顶）
    for h in base:
        key = (h["doc"], h["page"], h["text"][:40])
        if key in seen:
            continue
        seen.add(key)
        h = dict(h)
        if use_graph and h["page"] in gpages:
            h["rrf"] = (h.get("rrf") or 0.0) * 1.8
            h["source"] = "graph+hybrid"
        else:
            h["source"] = "hybrid"
        hits.append(h)
    hits.sort(key=lambda x: -(x.get("rrf") or 0.0))
    hits = hits[:top_k]

    info = {"entities": {"公司": co, "指标": inds, "年份": years},
            "mode": PREPEND_MODE,
            "graph_triples": [{"公司": a, "指标": b, "年份": y, "值": v, "单位": u, "页": p, "来源": s}
                              for a, b, y, v, u, p, s in gp_ok],
            "subgraph": subgraph(q)}
    return hits, conf, info


def answer(q, top_k=5, use_graph=True):
    """Graph RAG 端到端问答：图谱增强检索 → 同源 prompt 生成（与 engine.answer 同口径）。"""
    import prompts
    import llm
    qu = engine.query_understanding(q)
    hits, conf, info = retrieve(q, top_k=top_k, use_graph=use_graph)
    if not hits or conf < engine.REFUSE_SCORE:
        return {"answer": prompts.REFUSE_TEXT, "hits": [], "confidence": conf,
                "refused": True, "graph": info}
    ctx = prompts.build_context(hits)
    if len(ctx) > engine.MAX_CONTEXT_CHARS:
        ctx = ctx[:engine.MAX_CONTEXT_CHARS]
    msgs = prompts.rag_messages(q, hits, lang=qu["lang"])
    txt = engine.clean_answer(llm.chat(msgs, temperature=0.1))
    refused = ("未找到依据" in txt) or ("Not found" in txt)
    return {"answer": txt, "hits": [] if refused else hits, "confidence": conf,
            "refused": refused, "query_understanding": qu, "graph": info}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for q in ["平安银行2019年末的拨备覆盖率是多少？",
              "中国太保2021年实现营业收入和保险业务收入分别是多少？"]:
        r = answer(q)
        print("Q:", q)
        print("  实体:", r["graph"]["entities"])
        print("  图谱三元组:", r["graph"]["graph_triples"][:4])
        print("  命中:", [(h["page"], h["type"], h.get("source")) for h in r["hits"]])
        print("  A:", r["answer"][:200])
