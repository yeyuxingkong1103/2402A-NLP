# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
# 关联工单：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：graph_build —— 金融知识图谱构建
# 说明：从 ccf_competition 年报语料（JSONL：page/type/inside）中抽取财务指标三元组
#       （公司, 指标, 数值@年份），并补充公司-行业、公司-报告年度等实体关系，构建 networkx 图。
#       抽取策略：① 叙事句正则（“2019年，本行实现营业收入1,379.58亿元…”）；
#                ② 表行解析（['指标','值1','值2',...] 形式，按页面年份列对齐）。
# 输出：data/graph/graph.json（节点/边）+ data/graph/graph.graphml（可入 Neo4j/Gephi）
import os
import re
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_DIR = os.path.join(ROOT, "data", "ccf", "txt")
OUT_DIR = os.path.join(ROOT, "data", "graph")

# 行业分类（实体：公司 → 行业）
INDUSTRY = {
    "平安银行": "银行", "招商银行": "银行", "邮储银行": "银行",
    "中国平安": "保险", "中国人寿": "保险", "中国太保": "保险",
    "中信证券": "证券", "招商证券": "证券", "国泰君安": "证券",
}

# 财务指标词典（长词优先，避免“净利润”吃掉“归属母公司净利润”）
INDICATORS = [
    "归属于母公司股东的净利润", "归属于本行股东的净利润", "归属于上市公司股东的净利润",
    "扣除非经常性损益后归属于母公司股东的净利润",
    "零售业务营业收入", "保险业务收入", "已赚保费", "保费收入", "总保费",
    "营业收入", "利息净收入", "非利息净收入", "净利润", "归母净利润",
    "不良贷款率", "拨备覆盖率", "资本充足率", "核心一级资本充足率",
    "加权平均净资产收益率", "净资产收益率", "基本每股收益", "每股收益",
    "总资产", "总负债", "股东权益", "净资产", "净息差", "净利差",
    "综合成本率", "总投资收益率", "偿付能力充足率", "内含价值", "新业务价值",
    "手续费及佣金净收入", "投资收益", "营业利润", "利润总额",
]

_NUM = r"(-?[\d,]+(?:\.\d+)?)\s*(亿元|万元|元|%|个百分点)?"
# 叙事句：年份 + 指标 + 数值
_NARR = re.compile(r"(20\d{2})\s*年[^。；\n]{0,25}?(%s)[^。；\n]{0,12}?%s" % ("|".join(map(re.escape, INDICATORS)), _NUM))
# 表行：['指标', '值', ...]
_TABROW = re.compile(r"^\s*\[\s*'([^']+)'\s*,(.*)\]")


def company_of(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("__")
    return parts[2] if len(parts) >= 4 else base


def year_of(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d{4})年年报", base)
    return m.group(1) if m else ""


def _clean_num(s, unit):
    v = s.replace(",", "")
    return v, (unit or "")


def extract_from_doc(path, doc, report_year):
    """返回 triples: [(指标, 值, 单位, 年份, page), ...]"""
    triples = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        text = (o.get("inside") or "").strip()
        page = int(o.get("page") or 0)
        if not text:
            continue
        # ① 叙事句
        for m in _NARR.finditer(text):
            yy, ind, val, unit = m.group(1), m.group(2), m.group(3), m.group(4)
            if ind not in INDICATORS:
                continue
            triples.append({"indicator": ind, "value": _clean_num(val, unit)[0],
                            "unit": unit or "", "year": yy, "page": page, "src": "narrative"})
        # ② 表行
        tm = _TABROW.match(text)
        if tm:
            head = tm.group(1).strip()
            for ind in INDICATORS:
                if head == ind or head.startswith(ind):
                    nums = re.findall(r"'-?([\d,]+(?:\.\d+)?)'", "[" + tm.group(2))
                    if nums:
                        triples.append({"indicator": ind, "value": nums[0].replace(",", ""),
                                        "unit": "", "year": report_year, "page": page, "src": "table"})
                    break
    return triples


def build():
    import networkx as nx
    G = nx.DiGraph()
    src_of = {}   # (doc, indicator, year) -> page（用于检索回溯源页）
    per_doc = {}
    for p in sorted(glob.glob(os.path.join(TXT_DIR, "*.txt"))):
        doc = company_of(p)
        ry = year_of(p)
        G.add_node(doc, type="公司", industry=INDUSTRY.get(doc, "其他"))
        G.add_node(INDUSTRY.get(doc, "其他"), type="行业")
        G.add_edge(doc, INDUSTRY.get(doc, "其他"), rel="属于行业")
        G.add_node(ry + "年", type="年份")
        G.add_edge(doc, ry + "年", rel="报告年度")
        trips = extract_from_doc(p, doc, ry)
        for t in trips:
            G.add_node(t["indicator"], type="指标")
            G.add_edge(doc, t["indicator"], rel="披露指标",
                       year=t["year"], value=t["value"], unit=t["unit"], page=t["page"],
                       src=t.get("src", "table"))
            key = (doc, t["indicator"], t["year"])
            if key not in src_of:
                src_of[key] = t["page"]
        per_doc[doc] = len(trips)
        print("  %-8s 三元组=%d" % (doc, len(trips)))
    os.makedirs(OUT_DIR, exist_ok=True)
    data = nx.node_link_data(G, edges="links")
    json.dump({"nodes": data.get("nodes"), "links": data.get("links"),
               "src_of": {"%s|%s|%s" % k: v for k, v in src_of.items()}},
              open(os.path.join(OUT_DIR, "graph.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    try:
        nx.write_graphml(G, os.path.join(OUT_DIR, "graph.graphml"))
    except Exception as e:  # noqa: BLE001
        print("  graphml skip:", e)
    print("图规模：节点 %d / 边 %d" % (G.number_of_nodes(), G.number_of_edges()))
    return G


if __name__ == "__main__":
    print("构建金融知识图谱 ...")
    build()
    print("saved ->", OUT_DIR)
