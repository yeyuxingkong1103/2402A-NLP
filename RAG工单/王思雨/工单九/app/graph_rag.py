# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 图谱层：NetworkX + JSON 构建金融知识图谱；DeepSeek 抽实体/关系（强制 JSON）；社区摘要；实体链接与邻域扩展
import json, re; import networkx as nx                                          # JSON 与文本归一化 / 图结构与社区发现（工单九指定轻量方案，不引入 Neo4j）
from . import config as C                                                       # 本单配置（含 08 的凭据与模型路径）
_M = {}                                                                         # 进程内单例表：图与社区摘要只建一次
EXTRACT_SYS = ("你是金融知识图谱抽取器。从给定年报片段中抽取实体与关系，只输出 JSON，不要解释。"      # 【优化·prompt 层】抽取提示词全文
               "实体类型限定：公司/人物/指标/机构/产品；关系类型限定：任职/披露指标/持股/子公司/合作/监管。"  # 闭合本体，避免 08 时期"实体"兜底类目泛滥
               "禁止把纯数值、百分比、日期当作实体；每个关系必须给出 evidence（原文短句）。" '输出格式：{"entities":[{"name":"","type":""}],"relations":[{"head":"","rel":"","tail":"","evidence":""}]}')  # 数值过滤前移到 prompt（08 靠事后正则补救）+ 强制 JSON 输出与证据回溯
EN = {"平安银行": "ping an bank", "中国平安": "ping an insurance", "招商银行": "china merchants bank", "邮储银行": "postal savings bank", "中信证券": "citic securities", "中国人寿": "china life", "中国太保": "china pacific", "招商证券": "cmb securities", "国泰君安": "guotai junan"}  # 英文问句的公司别名（工单要求支持中英文问答；仅覆盖 9 家语料公司，其余英文问句仍走多语言向量通道）
def _norm(s): return re.sub(r"\s+", "", C.fix(s))                               # 去全部空白 + 归一化字形：PDF 抽取会在中文词间插空格、并把"银行"输出成"银⾏"，不处理会让实体链接整体漏配
def graph():                                                                    # 图单例：首次调用时从磁盘建 NetworkX 图
    if "g" in _M: return _M["g"]                                                # 已建直接返回，避免重复读盘
    src = json.loads((C.GRAPH_NX if C.GRAPH_NX.exists() else C.ROOT08_GRAPH).read_text(encoding="utf-8"))  # 优先本单已建图，否则以 08 抽取结果为基础（不重复调大模型，省时省钱）
    g = nx.Graph()                                                              # 无向图：检索期两个方向都要能走
    for n in src["nodes"]: g.add_node(n["id"], type=n.get("type", ""), name=n.get("name", ""), freq=n.get("freq", 1), code=n.get("code", ""))  # 节点属性随图携带，实体链接与展示都要用
    for e in src["edges"]:                                                      # 逐条加边
        if e["s"] in g and e["t"] in g and e["s"] != e["t"]: g.add_edge(e["s"], e["t"], rel=e.get("rel", ""), w=e.get("w", 1), code=e.get("code", ""), page=e.get("page", 0), year=e.get("year", ""))  # 只加两端都在的合法边，自环丢弃
    _M["g"], _M["comm"] = g, src.get("communities", [])                          # 社区摘要随图落盘（未建社区时为空表，检索退化为纯邻域扩展）
    return g                                                                    # 返回图
def link(q):                                                                    # 实体链接：把问题锚到图上实体（中文名 + 英文别名），长名优先以免"银行"这类短名抢占
    qn, ql = _norm(q), q.lower()                                                # 中文口径（去空白）与英文口径（小写）
    return sorted([n for n in graph().nodes if len((nm := n.split(":", 1)[1])) >= 2 and (_norm(nm) in qn or (EN.get(nm) and EN[nm] in ql))], key=lambda x: -len(x.split(":", 1)[1]))  # 名称（≥2 字，防空名或单字全命中）命中即锚定
def context(q, hops=None):                                                      # 检索期一站式图上下文（检索层只调它）
    g, hops, ents, codes = graph(), hops or C.HOPS, link(q), []                  # 取图 / 跳数 / 命中实体 / 代码待收
    for e in ents:                                                              # 【修正·必读】只从"命中实体"收代码：2 跳扩展会把同业公司拉进来，若连邻居一起收，"唯一公司才硬约束"将永不触发（本机实测踩到，codes 恒为空）
        c = g.nodes[e].get("code", "") if e in g else ""                        # 节点上的公司代码（实测：只有"公司"类节点带 code，指标/实体为空）
        if c and c not in codes: codes.append(c)                                # 去重收集
    anchored = len(codes) == 1; expand, seen, frontier = [], set(ents), list(ents)  # 是否锚定唯一公司 / 扩展词 / 已访问 / 扩展前沿
    for _ in range(hops):                                                       # 【优化·检索层】BFS 扩展 HOPS 跳（08 为 1 跳，深一层以补多期聚合类召回）
        frontier = [nb for e in frontier if e in g for nb in g.neighbors(e) if nb not in seen and not seen.add(nb)]  # 取未访问邻居并同步标记
        expand += [nb.split(":", 1)[1] for nb in frontier if len(nb.split(":", 1)[1]) >= 2 and not (anchored and g.nodes[nb].get("type") == "公司")]  # 锚定唯一公司时剔掉公司类邻居名，否则同业公司名会被注入 BM25、把别家片段拉进来（直接伤 context_precision）
    sums = []                                                                   # 命中的社区摘要片段
    if C.USE_COMM:                                                              # 【优化·检索层】global 通道开关
        for cm in _M.get("comm", []):                                           # 逐社区
            if set(cm.get("members", [])) & set(ents): sums.append({"text": "【社区摘要】" + cm.get("summary", ""), "company": "社区摘要", "page": 0, "src": "community"})  # 与命中实体同社区才补充
    return {"entities": ents, "codes": codes if anchored else [], "expand": expand[:C.EXPAND_TERMS], "summaries": sums[:2], "hops": hops}  # 只命中唯一公司才敢硬约束，多公司交给软扩展（与 08 既定前提一致）；摘要封顶 2 条防稀释精度
def summarize(names):                                                           # 单社区摘要（GraphRAG 的 community summary，检索期作 global 素材）
    try:                                                                        # 摘要失败不能影响建图
        from 工单八.app.rag_core import llm                                     # 复用 08 的大模型客户端单例（不在本单新建连接）
        r = llm().chat.completions.create(model=C.C8.LLM_MODEL, temperature=0.0, max_tokens=160, messages=[{"role": "system", "content": "你是金融图谱分析器。用一句中文概括这组实体的共同主题，不超过 60 字。"}, {"role": "user", "content": "、".join(names)}])  # 低温短输出，控成本
        return (r.choices[0].message.content or "").strip()                     # 取摘要文本
    except Exception as e: print(f"[graph] 社区摘要失败：{type(e).__name__}"); return "、".join(names[:8])  # 降级：用成员名拼接，不阻断流程
def build_communities():                                                        # 社区发现 + 逐社区摘要（离线一次性，结果落图）
    try: parts = [sorted(p) for p in nx.community.greedy_modularity_communities(graph()) if len(p) >= C.COMM_MIN][: C.COMM_MAX]  # 模块度贪心划分，滤过小社区并封顶控成本
    except Exception as e: print(f"[graph] 社区发现失败：{type(e).__name__}"); parts = []  # 失败不致命：退化为无社区摘要
    return [{"members": p, "summary": summarize([graph().nodes[n].get("name", n) for n in p][:40])} for p in parts]  # 逐社区生成摘要（成员名截断防超长）
def extract(text, code=""):                                                     # 【优化·prompt 层】DeepSeek 抽实体与关系，强制 JSON 输出
    from 工单八.app.rag_core import llm                                         # 复用 08 的客户端单例
    r = llm().chat.completions.create(model=C.C8.LLM_MODEL, temperature=0.0, max_tokens=900, response_format={"type": "json_object"}, messages=[{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": text[:3000]}])  # 温度 0 保证可复现；json_object 避免自由文本解析漂移；正文截断控成本
    try: return json.loads(r.choices[0].message.content or "{}")                # 解析 JSON
    except Exception: return {"entities": [], "relations": []}                  # 解析失败退化为空，不抛断链路
def _ensure(g, name, typ="实体", code=""):                                       # 确保节点存在并返回 id（同名复用，避免同一实体被切碎成多节点）
    for n, a in g.nodes(data=True):                                             # 线性查找：图仅数百节点，成本可接受
        if a.get("name") == name: return n                                      # 同名直接复用（跨文档合并的关键）
    nid = f"{typ}:{name}"; g.add_node(nid, type=typ, name=name, freq=1, code=code); return nid  # 新建：id 形如 公司:平安银行，与 08 的 id 约定一致
def add(text, code=""):                                                         # 增量建图：抽取 → 并回图 → 落盘（/upload 与 /graph/text 共用）
    g, tr = graph(), extract(text, code)                                        # 取图 + 抽取三元组
    for d in tr.get("entities", []):                                            # 逐实体
        if d.get("name"): _ensure(g, d["name"], d.get("type") or "实体", code)   # 建/复用节点（类型缺省归入"实体"）
    for d in tr.get("relations", []):                                           # 逐关系
        if d.get("head") and d.get("tail"):                                     # 两端都要有名字
            s, t = _ensure(g, d["head"], "实体", code), _ensure(g, d["tail"], "实体", code)  # 确保两端节点存在
            g.add_edge(s, t, rel=d.get("rel", ""), w=1, code=code, page=0, year="", ev=d.get("evidence", ""))  # 建边并附证据原文
    save(); return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}  # 落盘并回报规模
def save():                                                                     # NetworkX 图 → JSON 落盘（与 08 的 graph.json 分开，互不污染）
    g = graph(); C.GRAPH_NX.write_text(json.dumps({                             # 写文件
        "nodes": [{"id": n, **a} for n, a in g.nodes(data=True)], "edges": [{"s": s, "t": t, **a} for s, t, a in g.edges(data=True)],  # 节点（id + 属性）/ 边（两端 + 属性）
        "communities": _M.get("comm", [])}, ensure_ascii=False), encoding="utf-8")  # 社区摘要随图落盘，下次启动直接可用
def stats():                                                                    # 图谱规模与类型分布（/health 与前端展示用）
    from collections import Counter; g = graph(); return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "communities": len(_M.get("comm", [])), "by_type": dict(Counter(a.get("type", "") for _, a in g.nodes(data=True)))}  # 规模 + 类型分布
if __name__ == "__main__":                                                      # python -m app.graph_rag build：离线建图谱与社区摘要
    import sys                                                                  # 取命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "build": _M["comm"] = build_communities(); save(); print(f"[graph] 完成：{stats()}")  # 先算社区摘要再落盘并打印规模
