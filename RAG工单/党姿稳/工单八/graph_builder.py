# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
知识图谱构建：
  1. 从金融年报语料中筛选"图谱价值高"的文本块（财务指标 / 风险管理 / 战略业务 / 股权治理等主题）；
  2. 用 LLM 做实体-关系抽取（实体类型：公司、指标、数值、人物、职位、策略、领域、年份）；
  3. 用规则模板补充结构化"公司-指标-数值"三元组（正则匹配金额/比率/年份，提升覆盖率与准确率）；
  4. 实体归一化去重后写入 networkx 有向图，并持久化为 data/graph.json（节点/边），供检索与可视化使用。
"""
import os
import re
import json
from collections import defaultdict

import networkx as nx

from config import (GRAPH_FILE, GRAPH_TOPIC_KEYWORDS, GRAPH_EXTRACT_CHUNKS, COMPANY_BY_CODE,
                    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)

from openai import OpenAI

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

ENTITY_TYPES = ["公司", "指标", "数值", "人物", "职位", "策略", "领域", "年份", "机构", "产品"]

EXTRACT_SYSTEM_PROMPT = """你是金融知识图谱抽取助手。请从给定的年报文本片段中抽取实体和关系。
实体类型限定为：公司、指标、数值、人物、职位、策略、领域、年份、机构、产品。
关系类型限定为：包含指标、指标数值为、任职于、担任、投资于、开展业务、实施策略、涉及领域、同比变化、关联公司。

只输出 JSON，不要任何解释，格式：
{"entities":[{"name":"...","type":"..."}],"relations":[{"head":"...","relation":"...","tail":"..."}]}
注意：head/tail 必须是 entities 中出现过的 name；没有则返回空数组。"""

# 规则模板：公司-指标-数值
PAT_AMOUNT = re.compile(r"(营业收入|净利润|归母净利润|归属于母公司股东的净利润|营业收入合计|利息净收入|手续费及佣金净收入|"
                        r"拨备覆盖率|不良贷款率|核心一级资本充足率|资本充足率|内含价值|新业务价值|总资产|"
                        r"总负债|每股收益|加权平均净资产收益率|保费收入)"
                        r"[^0-9%]{0,20}?([0-9][0-9,\.]{1,20}\s*(?:亿元|万元|百万元|元|%|个百分点))")
PAT_YEAR = re.compile(r"(20[0-2][0-9])\s*年")


def _salvage_json(raw):
    """LLM 输出被截断时的抢救式解析：正则直接提取实体与关系的键值对"""
    ents = [{"name": n, "type": t} for n, t in
            re.findall(r'\{\s*"name"\s*:\s*"([^"]{1,40})"\s*,\s*"type"\s*:\s*"([^"]{1,10})"\s*\}', raw)]
    rels = [{"head": h, "relation": r, "tail": t} for h, r, t in
            re.findall(r'\{\s*"head"\s*:\s*"([^"]{1,40})"\s*,\s*"relation"\s*:\s*"([^"]{1,12})"\s*,\s*"tail"\s*:\s*"([^"]{1,40})"\s*\}', raw)]
    return ents, rels


def _llm_extract(text, doc):
    """对单个块做 LLM 实体关系抽取（输出过长时按正则抢救解析）"""
    prompt = f"【文档】{doc}\n【文本片段】\n{text[:700]}\n\n请抽取实体与关系（JSON，最多 8 个实体、8 条关系）："
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=1200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        try:
            data = json.loads(raw)
            ents = [e for e in data.get("entities", []) if e.get("name") and e.get("type") in ENTITY_TYPES]
            rels = [r for r in data.get("relations", []) if r.get("head") and r.get("tail")]
        except json.JSONDecodeError:
            ents, rels = _salvage_json(raw)
            ents = [e for e in ents if e["type"] in ENTITY_TYPES]
            print(f"  [抢救解析] {doc}: 实体 {len(ents)} 关系 {len(rels)}")
        return ents, rels
    except Exception as e:
        print(f"  [抽取失败] {doc}: {repr(e)[:80]}")
        return [], []


def _rule_extract(text, doc):
    """规则抽取：公司-指标-数值 三元组 + 年份"""
    ents, rels = [], []
    for m in PAT_AMOUNT.finditer(text):
        ind, val = m.group(1), m.group(2).replace(" ", "")
        ents.append({"name": ind, "type": "指标"})
        ents.append({"name": val, "type": "数值"})
        rels.append({"head": doc, "relation": "包含指标", "tail": ind})
        rels.append({"head": ind, "relation": "指标数值为", "tail": val})
    for y in set(PAT_YEAR.findall(text)):
        ents.append({"name": f"{y}年", "type": "年份"})
        rels.append({"head": doc, "relation": "涉及年份", "tail": f"{y}年"})
    return ents, rels


def select_chunks(chunks, limit=GRAPH_EXTRACT_CHUNKS):
    """挑选图谱价值高的块：按主题关键词命中数排序，且每份文档限量，避免单文档垄断"""
    scored = []
    for i, c in enumerate(chunks):
        hits = sum(c["text"].count(k) for k in GRAPH_TOPIC_KEYWORDS)
        if hits > 0:
            scored.append((hits, i, c))
    scored.sort(key=lambda x: -x[0])
    picked, per_doc = [], defaultdict(int)
    cap = max(8, limit // max(len({c["doc"] for c in chunks}), 1) + 4)
    for hits, i, c in scored:
        if len(picked) >= limit:
            break
        if per_doc[c["doc"]] >= cap:
            continue
        per_doc[c["doc"]] += 1
        picked.append((i, c))
    return picked


def build_graph(chunks, limit=GRAPH_EXTRACT_CHUNKS, use_llm=True, save_path=GRAPH_FILE):
    """构建知识图谱，返回 (networkx图, 节点/边统计)"""
    picked = select_chunks(chunks, limit)
    print(f"[图谱] 选取高价值块 {len(picked)} 个用于抽取（共 {len(chunks)} 块）")

    G = nx.DiGraph()
    node_type, chunk_of_entity = {}, defaultdict(set)
    for n, (idx, c) in enumerate(picked, 1):
        doc = c["doc"]
        if doc not in G:
            G.add_node(doc, type="公司")
            node_type[doc] = "公司"
        rels = []
        if use_llm:
            ents, rels = _llm_extract(c["text"], doc)
            if n % 10 == 0 or n == len(picked):
                print(f"  [{n}/{len(picked)}] {doc} 抽取实体 {len(ents)} 关系 {len(rels)}")
        ents_r, rels_r = _rule_extract(c["text"], doc)
        for e in ents + ents_r:
            name = e["name"].strip()
            if not name or len(name) > 40:
                continue
            if name not in G:
                G.add_node(name, type=e["type"])
                node_type[name] = e["type"]
            chunk_of_entity[name].add(idx)
        for r in rels + rels_r:
            h, t, rel = r["head"].strip(), r["tail"].strip(), r.get("relation", "关联")
            if h and t and h != t:
                G.add_edge(h, t, relation=rel, doc=doc)
                chunk_of_entity[h].add(idx)
                chunk_of_entity[t].add(idx)

    data = {
        "nodes": [{"id": n, "type": G.nodes[n].get("type", "未知")} for n in G.nodes],
        "edges": [{"source": u, "target": v, "relation": d.get("relation", ""), "doc": d.get("doc", "")}
                  for u, v, d in G.edges(data=True)],
        "entity_chunks": {k: sorted(v) for k, v in chunk_of_entity.items()},
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[图谱] 节点 {G.number_of_nodes()}，边 {G.number_of_edges()} -> {save_path}")
    return G, data


def load_graph(save_path=GRAPH_FILE):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============ 主题层补充（规则，无需 LLM） ============
TOPIC_TYPE = {"战略": "策略", "绿色金融": "领域", "金融科技": "领域", "数字化转型": "策略",
              "风险管理": "策略", "信用风险": "领域", "股权投资": "策略", "子公司": "机构",
              "董事会": "机构", "战略": "策略"}


def augment_with_rules(chunks=None, limit=GRAPH_EXTRACT_CHUNKS, save_path=GRAPH_FILE):
    """
    在已有图谱基础上补充规则层：
      1) 直接命中主题关键词的块 -> 增加 主题实体（指标/策略/领域/机构）与"文档-涉及主题"关系；
      2) 对同一批块再跑一遍"公司-指标-数值"正则模板。
    这样保证图谱同时具备"实体层（LLM 抽取）"与"主题层（规则，覆盖稳定）"，
    使跨文档综合型问题也能链接到主题实体。
    """
    if chunks is None:
        from kb import load_chunks
        chunks = load_chunks()
    data = load_graph(save_path)
    nodes = {n["id"]: n["type"] for n in data["nodes"]}
    edges = {(e["source"], e["target"]): e for e in data["edges"]}
    ent_chunks = {k: set(v) for k, v in data.get("entity_chunks", {}).items()}

    picked = select_chunks(chunks, limit)
    added_n = added_e = 0
    for idx, c in picked:
        doc = c["doc"]
        text = c["text"]
        # 1) 主题层
        for kw in GRAPH_TOPIC_KEYWORDS:
            if kw in text:
                if kw not in nodes:
                    nodes[kw] = TOPIC_TYPE.get(kw, "指标")
                    added_n += 1
                ent_chunks.setdefault(kw, set()).add(idx)
                key = (doc, kw)
                if key not in edges:
                    edges[key] = {"source": doc, "target": kw,
                                  "relation": "涉及主题", "doc": doc}
                    added_e += 1
        # 2) 规则层
        _, rels = _rule_extract(text, doc)
        for r in rels:
            h, t = r["head"], r["tail"]
            for name, tp in ((h, "指标" if r["relation"] == "包含指标" else "公司"), (t, None)):
                if name and name not in nodes:
                    nodes[name] = tp or ("数值" if r["relation"] == "指标数值为" else "指标")
                    added_n += 1
            key = (h, t)
            if key not in edges:
                edges[key] = {"source": h, "target": t, "relation": r["relation"], "doc": doc}
                added_e += 1
            ent_chunks.setdefault(h, set()).add(idx)
            ent_chunks.setdefault(t, set()).add(idx)

    data["nodes"] = [{"id": k, "type": v} for k, v in nodes.items()]
    data["edges"] = list(edges.values())
    data["entity_chunks"] = {k: sorted(v) for k, v in ent_chunks.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[图谱] 规则层补充完成：新增节点 {added_n}，新增边 {added_e}；"
          f"现有节点 {len(data['nodes'])}，边 {len(data['edges'])}")


# ============ 覆盖率补齐（规则，无需 LLM/向量） ============
# 金融年报核心指标表：保证"公司-指标"这条最常用的检索链在图谱中有完整覆盖
FIN_INDICATORS = [
    "拨备覆盖率", "不良贷款率", "核心一级资本充足率", "一级资本充足率", "资本充足率",
    "营业收入", "净利润", "归属于母公司股东的净利润", "归母净利润", "利息净收入",
    "手续费及佣金净收入", "总资产", "总负债", "每股收益", "加权平均净资产收益率",
    "内含价值", "新业务价值", "保费收入", "资产质量", "逾期贷款", "净利息收益率",
    "成本收入比", "存款余额", "贷款余额", "投资收益", "公允价值变动损益",
    "每股净资产", "总股本", "净资产收益率", "赔付支出",
]


def ensure_coverage(chunks, save_path=GRAPH_FILE):
    """
    把图谱的"实体-文本块"映射补全到全语料范围：
      1) 文档节点 -> 该文档的全部文本块（公司名是最高频主体，必须全量覆盖）；
      2) 金融指标节点 -> 全语料中所有提及该指标的文本块。
    这样"公司 + 指标"两个种子实体的交集块就能被准确排到最前。
    """
    data = load_graph(save_path)
    nodes = {n["id"]: n["type"] for n in data["nodes"]}
    edges = {(e["source"], e["target"]): e for e in data["edges"]}
    ent_chunks = {k: set(v) for k, v in data.get("entity_chunks", {}).items()}

    added_n = added_e = 0
    # 公司简称 -> 该公司的全部文本块（年报正文多数以"本行/本公司"指代，简称直接匹配会严重低估）
    doc_of = {}
    for idx, c in enumerate(chunks):
        doc_of.setdefault(c["doc"], []).append(idx)
    for code, name in COMPANY_BY_CODE.items():
        docs = [d for d in doc_of if d.startswith(name)]
        if not docs:
            continue
        if name not in nodes:
            nodes[name] = "公司"
            added_n += 1
        for d in docs:
            ent_chunks.setdefault(name, set()).update(doc_of[d])
            key = (name, d)
            if key not in edges:
                edges[key] = {"source": name, "target": d, "relation": "发布报告", "doc": d}
                added_e += 1

    for idx, c in enumerate(chunks):
        doc, text = c["doc"], c["text"]
        if doc not in nodes:
            nodes[doc] = "公司"
            added_n += 1
        ent_chunks.setdefault(doc, set()).add(idx)
        for ind in FIN_INDICATORS:
            if ind in text:
                if ind not in nodes:
                    nodes[ind] = "指标"
                    added_n += 1
                ent_chunks.setdefault(ind, set()).add(idx)
                if (doc, ind) not in edges:
                    edges[(doc, ind)] = {"source": doc, "target": ind,
                                         "relation": "包含指标", "doc": doc}
                    added_e += 1

    data["nodes"] = [{"id": k, "type": v} for k, v in nodes.items()]
    data["edges"] = list(edges.values())
    data["entity_chunks"] = {k: sorted(v) for k, v in ent_chunks.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[图谱] 覆盖率补齐完成：新增节点 {added_n}，新增边 {added_e}；"
          f"现有节点 {len(data['nodes'])}，边 {len(data['edges'])}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from kb import load_chunks
    if "--coverage" in sys.argv:
        ensure_coverage(load_chunks())
    elif "--augment" in sys.argv:
        augment_with_rules(load_chunks())
        ensure_coverage(load_chunks())
    else:
        G, data = build_graph(load_chunks())
        for u, v, d in list(G.edges(data=True))[:15]:
            print(f"  {u} -[{d.get('relation')}]-> {v}")
        augment_with_rules(load_chunks())
        ensure_coverage(load_chunks())
