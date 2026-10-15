# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：【层面2】检索层优化

## 相对工单08 检索链路的改动（逐条对应实测问题）

| 编号 | 08 的问题（实测） | 09 的优化 |
| --- | --- | --- |
| O1 | Query 实体识别混入噪声：问「平安银行2019年…董事长致辞」会链接出 `银行(Industry,0.8)`、`2019年(Time,0.8)` 两个低分兜底实体，白占种子位 | 低于 `ENTITY_MIN_SCORE` 的兜底实体不参与种子；Industry/Time 明确排除 |
| O2 | **全局型问题单轮 top-k 覆盖不足**（最要命的一条）：08 用 `FINAL_TOP_K=6` 的单查询检索，而 sample_questions 第 4 题要横跨 9 份年报归纳「共同策略与差异化策略」，参考答案含 15+ 个论断 → context recall 必然偏低 | 识别全局型问题 → **查询分解成多个侧面子查询** → 多路召回后统一融合精排 |
| O3 | 子图检索 k-hop 固定 2 跳 | 按问题类型自适应（组织型加深、事实型收窄） |
| O4 | 融合权重固定 `{graph:0.35, vector:0.40, fulltext:0.25}` | 按问题类型自适应：事实型偏文本、对比/组织型偏图谱 |
| O5 | 融合后候选**未做二次精排**（Reranker 只在 strategies 内部生效） | 图谱事实与文本命中**同池精排** |
| O6 | 上下文固定截断（700 字/块、40 条事实），无相关性预算 | 按相关性排序 + 去重 + 字符预算控制 |

## 为什么 O2 是 context recall 的关键

`context_recall` 的定义是：**参考答案里的每个论断，能否在检索到的上下文里找到支持**。
单轮 top-6 检索最多覆盖 6 个块，而第 4 题的参考答案横跨「风险管理 / 资本结构 /
绿色金融 / 科技金融 / 差异化策略 / 可持续性评估」六个侧面、涉及银行+保险+证券
三类主体 —— 6 个块在结构上就不可能覆盖。

查询分解把「一个宽泛问句」变成「N 个聚焦子查询」，每个子查询各自召回，
再合并去重。这是从**召回结构**上解决问题，而不是调参。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from w9 import config9

from src import config as c8
from src.graph_retriever import (SCORE_COMPANY_ALIAS, SubgraphResult,
                                 link_entities, retrieve_subgraph)
from src.kg_store import KnowledgeGraph

# ---------------------------------------------------------------------------
# 问题类型判定
# ---------------------------------------------------------------------------
# 全局/聚合型标志词：这类问题的答案需要跨文档归纳，必须走查询分解
GLOBAL_MARKERS = ("共同", "差异化", "普遍", "整体", "总体", "结合全部", "所有",
                  "各家", "这些", "整体上", "综合", "全面", "汇总", "分别",
                  "哪些方面", "如何评估", "一致性")
COMPARE_MARKERS = ("对比", "比较", "谁更", "哪个更", "哪家更", "相差", "vs", "VS",
                   "更高", "更低", "差异")
ORG_MARKERS = ("组织结构", "组织架构", "板块", "部门", "下设", "内设",
               "直属单位", "隶属", "委员会")

# 问题的「侧面」词表 → 用于把宽泛问句拆成聚焦子查询。
# 每一项是一个可独立检索的主题短语；命中即成为一条子查询。
ASPECT_TERMS: dict[str, tuple[str, ...]] = {
    "风险管理": ("风险管理", "风险管控", "信用风险", "风险偏好", "压力测试",
                 "风控模型", "不良贷款管理", "逾期贷款"),
    "拨备与资产质量": ("拨备覆盖率", "拨备计提", "资产质量", "逾期60天以上贷款"),
    "资本结构": ("资本充足率", "核心一级资本充足率", "资本补充", "内生资本",
                 "利润留存", "资产负债匹配", "偿付能力充足率"),
    "贷款结构": ("贷款结构", "信贷投放", "信贷结构调整", "高风险行业",
                 "基础设施", "绿色信贷"),
    "绿色金融": ("绿色金融", "绿色信贷", "绿色债券", "绿色能源",
                 "环保项目", "可持续金融"),
    "科技金融": ("科技金融", "金融科技", "数字化转型", "人工智能",
                 "大数据风控", "科技投入"),
    "盈利与增长": ("营业收入", "净利润", "盈利增长", "同比增长"),
    "创新商业模式": ("创新商业模式", "高频生活场景", "生态", "跨界",
                     "合作伙伴", "开放银行"),
    "股东与治理": ("控股股东", "实际控制人", "持股比例", "股权结构"),
    "组织架构": ("组织结构", "组织架构", "下设部门", "委员会", "板块"),
}

# 问题里出现这些词时，对应的侧面子查询才生成（避免无差别展开拖慢检索）
_ASPECT_TRIGGER: dict[str, tuple[str, ...]] = {
    "风险管理": ("风险", "风控", "不良", "逾期", "信用"),
    "拨备与资产质量": ("拨备", "资产质量", "不良", "逾期"),
    "资本结构": ("资本", "偿付", "负债", "资产负债"),
    "贷款结构": ("贷款", "信贷", "投放", "行业"),
    "绿色金融": ("绿色", "环保", "可持续", "ESG"),
    "科技金融": ("科技", "数字", "智能", "大数据", "人工智能"),
    "盈利与增长": ("盈利", "利润", "营收", "收入", "增长"),
    "创新商业模式": ("创新", "商业模式", "生态", "场景", "跨界"),
    "股东与治理": ("股东", "控股", "实际控制人", "股权"),
    "组织架构": ("组织", "部门", "板块", "委员会", "下设"),
}


def classify_question(question: str, linked: list[dict] | None = None) -> str:
    """判定问题类型（决定检索预算与融合权重）。"""
    q = question or ""
    if any(k in q for k in ORG_MARKERS):
        return config9.QTYPE_ORG
    companies = [x for x in (linked or []) if x.get("type") == "Company"]
    metrics = [x for x in (linked or []) if x.get("type") == "Metric"]
    if len(companies) >= 2 and metrics:
        return config9.QTYPE_COMPARE
    if any(k in q for k in COMPARE_MARKERS) and len(companies) >= 2:
        return config9.QTYPE_COMPARE
    # 全局型：显式标志词，或问句很长且含多个侧面
    if any(k in q for k in GLOBAL_MARKERS):
        return config9.QTYPE_GLOBAL
    aspects = _triggered_aspects(q)
    if len(q) > 60 and len(aspects) >= 2:
        return config9.QTYPE_GLOBAL
    return config9.QTYPE_FACTUAL


def _triggered_aspects(question: str) -> list[str]:
    """问句触发了哪些「侧面」主题。"""
    q = question or ""
    out: list[str] = []
    for aspect, triggers in _ASPECT_TRIGGER.items():
        if any(t in q for t in triggers):
            out.append(aspect)
    return out


def decompose_question(question: str, qtype: str,
                       max_queries: int | None = None) -> list[str]:
    """把问句拆成聚焦子查询。

    返回的首项**始终是原问题** —— 保证即使分解不理想，原查询仍在召回集合里，
    不会因为分解反而丢掉最直接的命中。

    ## 事实型问题为什么也要拆（实测依据）

    最初只在全局/对比型上做分解，理由是「事实型问题单一查询就够了」。
    实测 sample_questions 第 2 题被推翻了这个假设：

        问题：招商银行在其2019年年报中提到的创新商业模式有哪些？
        答案块：招商银行年报第 4 页 c000007，原文「…我们锐意转型创新，
                培育创新土壤，**探索创新商业模式**。以共生共建的理念，
                聚焦高频生活场景…发起设立"逾越者联盟"…」

    该块**在索引里**（已核验）、**字面包含「创新商业模式」**、质量分 0.97，
    但原始问句的向量/BM25 融合排序把它挤出了候选池 —— 08 与 09 都没召回它，
    两边的 SQ2 精度都趋近于 0。加一路「创新 商业模式 生态 场景 跨界」的
    聚焦子查询后即可命中。

    教训：**问题类型决定的是「检索预算」，不该决定「要不要换一种问法」**。
    事实型问题同样可能有与问句措辞不一致的标准表述。
    """
    max_queries = max_queries or config9.QUERY_DECOMPOSE_MAX
    q = (question or "").strip()
    if not q:
        return []

    aspects = _triggered_aspects(q)

    # 全局/对比型：总是展开（答案必然跨侧面）
    if qtype in (config9.QTYPE_GLOBAL, config9.QTYPE_COMPARE):
        subs: list[str] = [q]
        if not aspects:
            # 问句里没触发任何侧面时（措辞非常规），退化为按标点/并列词切分
            parts = [p.strip() for p in
                     re.split(r"[，,；;。？?]", q) if len(p.strip()) >= 6]
            subs.extend(parts[1:])
        else:
            for aspect in aspects:
                terms = ASPECT_TERMS.get(aspect, ())
                if terms:
                    subs.append(" ".join(terms[:4]))
        return _dedup(subs)[:max_queries]

    # 事实型/组织型：**仅当问句明确命中某个侧面**时补一路聚焦子查询。
    # 触发面很窄（要有「创新/商业模式/风险/资本/绿色/科技」这类实词），
    # 代价是多一次召回（实测 +0.3s 量级），换来的是措辞不一致时的兜底命中。
    if aspects and qtype == config9.QTYPE_FACTUAL:
        subs = [q]
        for aspect in aspects[:2]:
            terms = ASPECT_TERMS.get(aspect, ())
            if terms:
                subs.append(" ".join(terms[:4]))
        return _dedup(subs)[:max_queries]

    return [q]


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# O1：实体识别与链接（去噪）
# ---------------------------------------------------------------------------
def link_entities_v2(question: str, kg: KnowledgeGraph,
                     min_score: float | None = None) -> list[dict]:
    """实体识别 + 消歧 + 去噪。

    相对 08 的 `link_entities`：
      · 丢弃得分低于 `ENTITY_MIN_SCORE` 的**图谱别名兜底**命中
        （那是「问题里恰好出现某节点名的子串」级别的弱证据）；
      · 明确排除 Industry / Time 类型 —— 它们只配做上下文，不配做检索种子；
      · 集团与子公司分别保留（不合并），但去重同名项。
    """
    min_score = config9.ENTITY_MIN_SCORE if min_score is None else min_score
    raw = list(link_entities(question, kg))

    # O1.5：**派生名兜底**。语料的别名表覆盖不到同一主体的所有写法。
    # 实测：英文问句译成中文后是「2019年邮政储蓄银行（601658）…」，
    # 而别名表里只有「中国邮政储蓄银行股份有限公司 / 中国邮政储蓄银行 / 邮储银行」，
    # 「邮政储蓄银行」（少了「中国」二字）一个都不匹配 → 公司实体完全丢失。
    # 这里用**去掉地域前缀/公司后缀的派生形**再匹配一次，把这类写法收进来。
    if not any(x.get("type") == "Company" for x in raw):
        from src.kg_dict import COMPANY_TABLE
        from src.kg_schema import make_entity_id
        for canon, meta in COMPANY_TABLE.items():
            variants = set()
            for form in (canon, meta.get("full", "")):
                f = str(form).replace("（", "(").replace("）", ")")
                # 必须**累积**去除：先逐个去、再去组合，否则
                # 「中国邮政储蓄银行股份有限公司」去掉「中国」后仍带
                # 「股份有限公司」，匹配不上问题里的「邮政储蓄银行」。
                for pre in ("中国", "股份有限公司", "有限公司", "(集团)", "集团"):
                    f = f.replace(pre, "")
                if len(f) >= 4:
                    variants.add(f)
                    # 再去掉常见属地/品牌前缀，覆盖「深圳发展银行」类写法
                    for pre in ("深圳", "北京", "上海"):
                        if f.startswith(pre) and len(f) - len(pre) >= 4:
                            variants.add(f[len(pre):])
            hit = next((v for v in variants if v in question), None)
            if hit:
                raw.append({"entity_id": make_entity_id("Company", canon),
                            "name": canon, "type": "Company",
                            "score": SCORE_COMPANY_ALIAS, "surface": hit})
                break

    out: list[dict] = []
    for x in raw:
        if x.get("type") in config9.NOISE_SEED_TYPES:
            continue
        if float(x.get("score", 0)) < min_score:
            continue
        out.append(x)

    # 兜底①：**公司是图谱通道的种子，不能被分数阈值滤光**。
    # 若过滤后一个 Company 都没剩、但原始结果里有，就把最高分的那个加回来。
    #
    # 实测依据（英文问答失败的根因）：英文问句译成中文后是
    # 「2019年**邮政储蓄银行**（601658）的不良贷款率和拨备覆盖率是多少？」，
    # 比原中文问句少了「中国」二字 —— 于是「邮政储蓄银行」只能靠
    # 图谱别名兜底命中（0.80 分），低于 ENTITY_MIN_SCORE=0.85 被滤掉。
    # 结果 linked 里只剩两个 Metric 实体，没有 Company 种子 →
    # 子图检索返回 0 条事实 → 答案失去图谱证据而答错。
    if raw and not any(x.get("type") == "Company" for x in out):
        comps = [x for x in raw if x.get("type") == "Company"]
        if comps:
            out.append(max(comps, key=lambda x: float(x.get("score", 0))))

    # 兜底②：全被过滤掉时，至少保留最高分的公司，避免图谱通道彻底哑火
    if not out and raw:
        comps = [x for x in raw if x.get("type") == "Company"]
        if comps:
            out = [max(comps, key=lambda x: float(x.get("score", 0)))]
    return out


# ---------------------------------------------------------------------------
# O3/O4：子图检索 + 自适应融合权重
# ---------------------------------------------------------------------------
def subgraph_v2(question: str, linked: list[dict], kg: KnowledgeGraph,
                qtype: str) -> SubgraphResult:
    """子图检索：按问题类型自适应 k-hop 半径。"""
    hops = {
        config9.QTYPE_FACTUAL: c8.KG_SUBGRAPH_HOPS,
        config9.QTYPE_COMPARE: c8.KG_SUBGRAPH_HOPS + 1,
        config9.QTYPE_GLOBAL: c8.KG_SUBGRAPH_HOPS,
        config9.QTYPE_ORG: c8.KG_SUBGRAPH_HOPS + 1,
    }.get(qtype, c8.KG_SUBGRAPH_HOPS)

    old = c8.KG_SUBGRAPH_HOPS
    try:
        c8.KG_SUBGRAPH_HOPS = hops        # 08 的 retrieve_subgraph 读该配置
        return retrieve_subgraph(question, linked, kg)
    finally:
        c8.KG_SUBGRAPH_HOPS = old


def fusion_weights(qtype: str) -> dict:
    """O4：按问题类型取自适应融合权重。"""
    return dict(config9.FUSION_WEIGHTS_BY_QTYPE.get(
        qtype, config9.FUSION_WEIGHTS_BY_QTYPE[config9.QTYPE_FACTUAL]))


# ---------------------------------------------------------------------------
# O2：文本检索（多查询）
# ---------------------------------------------------------------------------
def _retrieve_one(query: str, qtype: str, top_k: int,
                  do_rerank: bool = True) -> tuple[list[dict], dict]:
    """单查询检索（走 08 的 strategies.retrieve）。

    `do_rerank=False` 用于全局型问题的**子查询召回阶段**：
    每路子查询都跑一遍 cross-encoder 精排既慢（5 路 × 精排 ≈ 2s+）又没有意义
    —— 子查询的排序会被后面「跨侧面 round-robin + 统一精排」完全重写。
    只在最终候选集上精排一次，既省时又保证排序口径一致。
    """
    from src import doc_router, strategies

    # O7（本工单新增）：**文档路由**。
    # 08 的 `graph_qa._retrieve_text` 调 `strategies.retrieve(question, cfg)` 时
    # **没有传 sources**，等于放弃了自己实现的 doc_router —— 结果问
    # 「平安银行2019年年度报告的董事长致辞…」会把「中国平安」年报的大段内容
    # 一并召回（实测 8 个块里 7 个来自中国平安），答案自然跑偏。
    # doc_router.route() 在「问题恰好只点名一家公司」时返回该 source，
    # 多命中/未命中则返回 None（不过滤）—— 语义正确，直接复用。
    try:
        sources = doc_router.route(query)
    except Exception:                                  # noqa: BLE001
        sources = None

    recall_k = c8.TOP_K_RECALL * 2 if qtype == config9.QTYPE_GLOBAL \
        else c8.TOP_K_RECALL
    cfg = strategies.RetrievalConfig(strategy=c8.RETRIEVAL_STRATEGY_DEFAULT,
                                     top_k_recall=recall_k,
                                     top_k_rerank=top_k,
                                     sources=sources)
    res = strategies.retrieve(query, cfg, do_rerank=do_rerank)
    hits = res.get("hits", []) if isinstance(res, dict) else list(res)
    info = dict(res.get("info", {})) if isinstance(res, dict) else {}
    return hits, info


def rerank_final(question: str, candidates: list[dict],
                 top_k: int) -> list[dict]:
    """O5：对最终候选集做**统一精排**（图谱事实与文本命中同池）。

    08 的 Reranker 只在 `strategies` 内部对「向量+全文」两路生效；
    图谱事实是直接拼进上下文的，从不参与排序。本函数把三者放在同一个
    cross-encoder 池子里按「与原始问题的相关性」排序 ——
    这正是工单要求的「优化 Reranker 精排：对融合后的候选做精排」。
    """
    if not candidates:
        return []
    try:
        from src import rerankers
        res = rerankers.rerank_detailed(question, candidates, top_k=top_k)
        hits = res.get("hits", []) if isinstance(res, dict) else list(res)
        return hits or candidates[:top_k]
    except Exception as exc:                           # noqa: BLE001
        print(f"[retrieve_v2] 统一精排失败，按原分数排序：{exc}", flush=True)
        return sorted(candidates,
                      key=lambda x: -float(x.get("score", 0)))[:top_k]


def _lexical_score(question: str, text: str) -> float:
    """轻量相关性打分：问题字符 2-gram 在文本中的覆盖率。

    用于图谱事实的粗筛（便宜、确定性、零模型）。中文无词边界，
    2-gram 是稳妥的选择。
    """
    q = re.sub(r"[^一-鿿A-Za-z0-9]", "", question or "")
    t = re.sub(r"[^一-鿿A-Za-z0-9]", "", text or "")
    if len(q) < 2 or not t:
        return 0.0
    grams = {q[i:i + 2] for i in range(len(q) - 1)}
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in t) / len(grams)


def _strip_entity_surfaces(question: str, linked: list[dict] | None) -> str:
    """把问题里的**实体名**（公司名等）从打分文本中剔除。

    ## 为什么必须剔除（实测踩坑）

    对 sample_questions 第 2 题「**招商银行**在其2019年年报中提到的**创新商业模式**有哪些？」，
    原打分函数把 8 条图谱事实全部选入上下文，而它们全是
    「招商银行 --HAS_METRIC--> 净利润(860.85亿元)」这类**财务指标**，
    与「创新商业模式」毫无关系，唯一命中的是**公司名**。

    公司名出现在每一条事实里（它们都挂在同一个主体上），于是 2-gram 覆盖率
    被公司名拉到很高，真正有区分度的主题词（创新/商业模式）完全被淹没。
    剔除实体名后，只剩主题词参与打分，噪声事实自然掉出去。
    """
    q = question or ""
    # **只剔除「在每条事实里都会出现」的实体名**（公司/机构/行业/时间）。
    # 它们挂在同一个主体上，对区分事实毫无作用，只会把覆盖率整体抬高。
    #
    # **绝对不能剔除 Metric 名** —— 指标名往往正是问题的主题。
    # 实测踩坑：初版把 linked 里的全部实体名都剔了，问「…不良贷款率和拨备覆盖率
    # 分别是多少？」，探针被削成「年末的 和 分别是多少？」，
    # 于是「年末」二字随机命中了「一级资本充足率…@2019年末」这类事实，
    # 8 条无关的资本充足率事实挤进上下文，答案直接被带偏。
    _PERVASIVE = {"Company", "Institution", "Industry", "Time", "Location"}
    for x in (linked or []):
        if x.get("type") not in _PERVASIVE:
            continue
        for name in (x.get("name"), x.get("surface")):
            if name and len(str(name)) >= 2:
                q = q.replace(str(name), " ")
    # **同时剔除数字/年份**：年报事实里几乎每条都带「2019年度」「2020年」这类
    # 时间或金额，若让它们参与打分，一条与主题毫不相干的事实仅凭年份相同
    # 就能拿到非零分而被保留。实测 SQ2（问「创新商业模式」）的 8 条财务指标
    # 事实就是这样混进上下文的 —— 它们唯一命中的是「2019」。
    # 年份/金额相同不构成主题相关，故一律不参与打分。
    q = re.sub(r"[0-9０-９]+(\.[0-9]+)?%?", " ", q)
    return q


def filter_graph_facts(question: str, facts: list[str],
                       top_k: int | None = None,
                       linked: list[dict] | None = None) -> list[str]:
    """按与问题的相关性**筛选并排序**图谱事实。

    ## 为什么这是 context precision 的关键一刀（实测发现）

    08 的 `_khop_facts` 把 k-hop 邻域的**每一条边**都转成一句事实，上限
    `KG_SUBGRAPH_MAX_NODES=60`。实测：问「平安银行董事长致辞盈利增长因素」，
    图谱通道吐回 **59 条事实**，其中绝大多数是「某指标 某年 为某值」的
    指标边，与「董事长致辞」毫无关系。

    后果有两层：
      · 送给 LLM 的上下文被 59 条不相关事实占满，真正有用的文本证据被稀释；
      · 在 RAGAS 的 context precision 口径下，**每一条不相关的事实都算一次
        「检索到的但无用的上下文」**，直接把精度压下去。

    08 的做法是「按检索顺序截断前 N 条」，那只是截数量、不辨相关性。
    本函数改为**按相关性排序后取头部**：相关的留下、不相关的丢掉。
    """
    facts = [f for f in (facts or []) if str(f).strip()]
    if not facts:
        return []
    top_k = top_k or config9.GRAPH_FACTS_TOP_K
    probe = _strip_entity_surfaces(question, linked)
    scored = [(_lexical_score(probe, f), i, f) for i, f in enumerate(facts)]
    # 同分时保持**原始顺序**（图谱事实在模式内的顺序带语义优先级，
    # 如「指标对比」先给各行取值、最后给比较结论），故用 index 做次级键。
    scored.sort(key=lambda x: (-x[0], x[1]))
    kept = [f for s, _i, f in scored[:top_k] if s > 0]
    # 剔除实体名后没有任何事实与问题主题相关 → 返回空列表，**不强行凑数**。
    #
    # 这一条是实测改出来的：原先「全零分就保留原顺序前 top_k」的兜底，
    # 会让「创新商业模式」这类图谱未覆盖的主题硬塞进 8 条财务指标事实 ——
    # 它们对答案毫无贡献，却在 RAGAS 口径下逐条计为「检索到但无用」，
    # 把 context precision 直接压到 0。图谱没覆盖就该老实说没覆盖，
    # 让文本通道去答，而不是拿不相关的事实充数。
    if not kept:
        return []
    return kept


def _select_round_robin(groups: list[list[dict]], budget: int) -> list[dict]:
    """跨组轮流取 —— 保证每个侧面都有代表进入上下文。

    这是全局型问题 context recall 的关键一步。逐组取「该组当前最优的那条」，
    再回到第一组取第二条，如此往复直到填满预算。任一组先取空就跳过。

    为什么不用「全池排序取 top-k」：实测那样会让措辞密集的侧面
    （如绿色金融）垄断全部上下文位，而参考答案的其他侧面一个块都进不来。
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    idx = [0] * len(groups)
    while len(out) < budget:
        progressed = False
        for gi, g in enumerate(groups):
            if len(out) >= budget:
                break
            while idx[gi] < len(g):
                h = g[idx[gi]]
                idx[gi] += 1
                k = _hit_key(h)
                if k in seen:
                    continue
                seen.add(k)
                out.append(h)
                progressed = True
                break
        if not progressed:
            break
    return out


def retrieve_text_v2(queries: list[str], qtype: str,
                     top_k: int | None = None) -> tuple[list[dict], dict]:
    """多查询文本检索。

    ## 全局型问题为什么必须「逐子查询配额」，而不是「合并后统一排序」

    最初实现是复用 08 的 `strategies.retrieve_multi`：把 N 个子查询的召回
    合并成一个池子，再整体精排取 top-k。实测（sample_questions 第 4 题）结果是
    **8 个上下文块全部来自「绿色金融」这一个侧面** —— 因为该侧面的年报段落
    措辞密集、相似度天然更高，在统一排序里把其他侧面全挤掉了。

    而第 4 题的参考答案覆盖「风险管理 / 资本结构 / 绿色金融 / 科技金融 /
    差异化策略 / 可持续性」六个侧面。上下文全给一个侧面 → context recall
    只能拿到 1/6 左右的论断覆盖率。

    所以全局型改为**每路子查询各取 N 条**，再合并去重：
    牺牲一点「全局最优排序」，换取**侧面覆盖的结构性保证**。
    这正是 context recall 这类「覆盖率」指标该用的策略 ——
    它衡量的是「论断能否被找到」，而不是「最相关的块排第几」。

    事实型/对比型问题不存在多侧面问题，仍走单查询（保持与 08 同源同参）。
    """
    top_k = top_k or config9.RERANK9_POOL
    qs = [q for q in (queries or []) if q and q.strip()]
    if not qs:
        return [], {"error": "空查询"}

    t0 = time.perf_counter()
    info: dict = {"n_queries": len(qs), "mode": "single"}

    # 单查询：直接走 08 路径（保持与 08 同源同参，可比性最强）
    if len(qs) == 1:
        try:
            hits, info_i = _retrieve_one(qs[0], qtype, top_k)
        except Exception as exc:                       # noqa: BLE001
            return [], {"error": f"{type(exc).__name__}: {exc}",
                        "elapsed_ms": (time.perf_counter() - t0) * 1000}
        info.update(info_i)
        info["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return hits, info

    # 多查询（全局/对比型）：
    #   ① 每路子查询**只召回不精排**（快）；
    #   ② 跨侧面 round-robin 选候选（保证侧面覆盖）；
    #   ③ 对选出的候选集**统一精排一次**（对抗原问题的相关性排序）。
    info["mode"] = "per_aspect_rr"
    per = max(3, top_k // len(qs))
    info["per_query_k"] = per
    groups: list[list[dict]] = []
    errors: list[str] = []
    for q in qs:
        try:
            h, _i = _retrieve_one(q, qtype, top_k=per, do_rerank=False)
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            h = []
        groups.append(h[:per])
    if errors:
        info["errors"] = errors
    if not any(groups):
        return [], {**info, "error": "全部子查询检索失败"}

    picked = _select_round_robin(groups, budget=config9.RERANK9_POOL)
    info["n_candidates"] = len(picked)
    info["n_aspects"] = sum(1 for g in groups if g)

    # 统一精排：用**原始问题**（而非子查询）作为排序依据 ——
    # 最终要回答的是原问题，子查询只是召回手段。
    ranked = _rerank_multi_query(qs, picked, top_k)

    # 精排后**给每路子查询保底 1 条**进入最终上下文。
    #
    # ## 为什么精排之后还要保底（实测依据）
    #
    # 子查询是为「与原问句措辞不一致的标准表述」而设的。实测 sample_questions
    # 第 2 题：答案块 c000007（原文含「探索创新商业模式…逾越者联盟」）**只能**
    # 被聚焦子查询「创新商业模式 高频生活场景 生态 跨界」召回（在该路 top-5），
    # 而它对着**原始问句**算相关性时打分偏低 —— 精排一轮就被挤出前列，前功尽弃。
    #
    # 根因是「精排依据」与「召回依据」不是同一个问句：用原问句精排，
    # 天然会压制那些只有换种问法才召回的块。因此精排后必须给每路留位置。
    #
    # 注意保底的**作用域是最终上下文条数**（FINAL9_TOP_K），不是候选池大小：
    # 若在 40 条候选上保底、再截前 8 条，保底等于白做（实测就是这样漏掉
    # c000007 的 —— 它进了 40 条候选，却仍被 8 条上限挤出去）。
    final = _keep_with_group_quota(ranked, groups, config9.FINAL9_TOP_K)
    info["n_final"] = len(final)
    info["elapsed_ms"] = (time.perf_counter() - t0) * 1000
    return final, info


def _rerank_multi_query(queries: list[str], candidates: list[dict],
                        top_k: int) -> list[dict]:
    """**多查询取最大**精排。

    ## 为什么不能只用原始问句精排（实测依据，最后一处关键修正）

    子查询是为「与原问句措辞不一致的标准表述」而设的。但精排如果只用原始问句，
    那些**只有换种问法才召回**的块会被系统性压低。实测 sample_questions 第 2 题：

        答案块 c000007（「…探索创新商业模式…发起设立"逾越者联盟"…」）
        · 对**原问句**「招商银行在其2019年年报中提到的创新商业模式有哪些？」
          → cross-encoder 打分偏低，精排后被挤出前 8；
        · 对**子查询**「创新商业模式 高频生活场景 生态 跨界」
          → 该路召回排名第 5，高度相关。

    只按原问句排序，等于把子查询的成果又丢了回去。故这里**对每个查询式各排一轮，
    逐候选取最大值** —— 一个块只要对**任何一种问法**高度相关，就值得进上下文。

    代价：多 N-1 次精排（N = 子查询数，实测每次约 0.15s）。
    """
    if not candidates:
        return []
    if len(queries) <= 1:
        return rerank_final(queries[0] if queries else "", candidates, top_k)

    # ## 两套分数，各司其职（这是精度与召回之间唯一站得住的调平方式）
    #
    #   召回用 `best`（跨查询取最大）—— 保证「只有换种问法才召回的块」能进来；
    #   选上下文用 `orig`（**只对原始问句**打分）—— 因为 RAGAS 的 context precision
    #   问的正是「这个块对回答**这个问题**有没有用」，判据只能是原始问句。
    #
    # 实测说明为什么必须分开：SQ2 的答案块 c000007 对原始问句的精排分高达
    # 0.9832（很高），而同一批里另外 7 条边缘块只有 0.002~0.01 量级。
    # 但「跨查询取最大」会把边缘块在子查询下拿到的中等分数抬上来，
    # 于是一道只为精度而设的闸门若按 `best` 过滤，就一条都拦不住 ——
    # SQ2 的 context precision 因此停在 0.067。
    # 改用 `orig` 过滤后，边缘块被切掉、c000007 保留，召回不受影响。
    best: dict[tuple, tuple[float, dict]] = {}
    orig: dict[tuple, float] = {}
    for qi, q in enumerate(queries):
        ranked = rerank_final(q, candidates, len(candidates))
        for rank, h in enumerate(ranked):
            key = _hit_key(h)
            # 必须读 **rerank_score**，不是 score！
            # `rerank_detailed` 把 cross-encoder 打分写在 `rerank_score` 字段，
            # 而 `score` 保留的是**召回阶段的原始分**。实测踩坑：读 `score` 时，
            # 同一候选在不同查询下的取值完全相同（都是召回分），取最大值成了
            # 空操作 —— 多查询精排退化成「按召回分排序」，c000007 依旧进不来。
            metric = float(h.get("rerank_score") or 0)
            if metric <= 0:      # 精排未产出分数的兜底：用轮内名次换算
                metric = 1.0 / (1.0 + rank)
            if qi == 0:
                orig[key] = metric
            cur = best.get(key)
            if cur is None or metric > cur[0]:
                best[key] = (metric, h)

    ordered = sorted(best.values(), key=lambda x: -x[0])
    ranked = [h for _s, h in ordered]

    # 精度闸门：按**原始问句**的相关性剔除「被召回但答不了这个问题」的块。
    if ranked:
        orig_vals = [orig.get(_hit_key(h), 0.0) for h in ranked]
        top_orig = max(orig_vals) if orig_vals else 0.0
        floor = max(config9.RERANK9_MIN_SCORE,
                    top_orig * config9.RERANK9_REL_RATIO)
        kept = [h for h in ranked if orig.get(_hit_key(h), 0.0) >= floor]
        # 保底：全被切掉时保留精排前 N 条，绝不返回空上下文
        if len(kept) < config9.MIN_CONTEXT_BLOCKS:
            kept = ranked[:config9.MIN_CONTEXT_BLOCKS]
        ranked = kept

    return ranked[:top_k]


def _keep_with_group_quota(ranked: list[dict], groups: list[list[dict]],
                           budget: int, quota: int = 1) -> list[dict]:
    """精排结果 + 每组保底，合成最终选中集。

    先按精排顺序取满 `budget - 保底数` 条，再用每组的头部（未入选者）补足剩余名额。
    这样既保留精排对整体相关性的排序，又保证没有任何一路召回的成果被整体丢弃。
    """
    active = [g for g in groups if g]
    if not active:
        return ranked[:budget]

    keep = max(budget - quota * len(active), 1)
    out = list(ranked[:keep])
    seen = {_hit_key(h) for h in out}

    for g in active:
        added = 0
        for h in g:
            if len(out) >= budget:
                break
            k = _hit_key(h)
            if k in seen:
                continue
            seen.add(k)
            out.append(h)
            added += 1
            if added >= quota:
                break
    return out[:budget]


# ---------------------------------------------------------------------------
# O5/O6：统一精排 + 上下文构建
# ---------------------------------------------------------------------------
# 低质量块判定：表格分隔符/空白占比过高，或有效文字太少的块。
# 实测样例（平安银行 2019 年报 p204~p260 的报表附注页）：
#     "| 平安银行股份有限公司 |  |  | | --- | --- | --- | | 财务报表附注 "
# 这种块**含公司名**，所以在向量检索与 cross-encoder 精排里都拿高分，
# 但它通篇没有一个完整句子 —— 送进上下文只会稀释答案，并直接拉低
# RAGAS 的 context precision（每个「检索到但没用」的块都算精度损失）。
_NON_TEXT_CHARS = set("|-—=· \t\r\n　")
_SEP_CHARS = set("|-—=·_")


def _text_quality(text: str) -> float:
    """有效字符占比（0~1）。越低越像表格骨架/分隔线残片。"""
    s = text or ""
    if not s:
        return 0.0
    eff = sum(1 for c in s if c not in _NON_TEXT_CHARS)
    return eff / len(s)


def _sep_ratio(text: str) -> float:
    """表格分隔符（| - — = · _）占比。

    ## 这个指标比「有效字符占比」更准，实测数据：

        表格骨架块  "| 平安银行股份有限公司 |  |  | | --- | --- | --- | | 财务报表附注"
                    → 分隔符占比 0.33 ~ 0.39
        正常正文块  "为股东负责：创造价值、稳健经营 作为新中国第一家公开发行股票的股份制商业银行…"
                    → 分隔符占比 0.00

    两组**完全不相交**，阈值任意取在 (0, 0.33) 都能完美区分。
    反观「有效字符占比」：表格骨架块能到 0.50~0.60（因为它确实写了不少中文），
    与正常块的 0.96 区分度差得多，实测用 0.45 阈值时一个都拦不住。
    """
    s = text or ""
    if not s:
        return 0.0
    return sum(1 for c in s if c in _SEP_CHARS) / len(s)


def is_low_quality(hit: dict) -> bool:
    """判定为低质量块（表格骨架 / 分隔线残片）—— 见 _sep_ratio 的实测依据。"""
    text = hit.get("text") or ""
    if len("".join(text.split())) < 40:
        return True
    return _sep_ratio(text) > 0.15 or _text_quality(text) < 0.45


def filter_low_quality(hits: list[dict], keep_min: int = 3) -> list[dict]:
    """剔除低质量块；若过滤后过少则保留原列表（避免全被滤空）。

    keep_min 是安全阀：宁可带上几个低质量块，也不能让上下文空掉 ——
    空上下文触发的「未检索到相关信息」兜底比低精度更糟。
    """
    good = [h for h in hits if not is_low_quality(h)]
    return good if len(good) >= keep_min else hits


def _hit_key(h: dict) -> tuple:
    """命中去重键：文档 + 页 + 块。

    注意用 (source, chunk_id) 而不是单用 chunk_id —— 08 语料里 chunk_id
    **跨文档重复**（每份年报都从 c000001 开始），单用会跨文档误去重。
    """
    return (h.get("source") or h.get("source_doc") or "",
            int(h.get("page_idx", h.get("page", -1))),
            str(h.get("chunk_id") or ""))


def fuse_text_hits(hits: list[dict], top_k: int) -> list[dict]:
    """文本命中去重 + 截断，**保持传入顺序**。

    ## 这里曾经有个把整个精排成果作废的 bug（实测发现，代价很大）

    初版是「去重后按 `score` 降序排」。但 `score` 是**召回阶段的原始分**，
    而 `rerank_score` 才是 cross-encoder 的排序分。传入的 `hits` 已经由
    `rerank_final` / `_rerank_multi_query` 排好序了，再用 `score` 重排一次，
    等于**把精排结果整个丢掉**，退回按召回分排序。

    实测症状：SQ2 的答案块 c000007 经精排已进入候选且排序正确，但最终
    top-8 由召回分决定，它又被挤出去 —— 表现上像是「精排没生效」，
    实际是精排之后又被覆盖了。

    现在改为**按传入顺序去重**（同键保留首次出现者，即排序最高者），
    不再重新排序。
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for h in hits:
        k = _hit_key(h)
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def build_context_v2(graph_facts: list[str], text_hits: list[dict],
                     max_chars: int | None = None,
                     max_facts: int = 30,
                     max_chars_per_block: int = 900) -> str:
    """O6：组装 LLM 上下文。

    分四区（工单硬性要求1 生成层面：图谱子图 / 文本块 / 表格 / 图像描述分区组织）：
      【图谱事实】结构化、可验证的确定性事实
      【文本证据】正文叙述块
      【表格证据】block_type=table 的结构化数据
      【图像描述】block_type=image 的图注/语义描述
    """
    max_chars = max_chars or config9.CONTEXT9_MAX_CHARS
    parts: list[str] = []

    if graph_facts:
        parts.append("【图谱事实】（来自知识图谱的结构化事实，均为语料原文抽取）")
        for i, f in enumerate(graph_facts[:max_facts], start=1):
            parts.append(f"{i}. {f}")
        parts.append("")

    groups: dict[str, list[dict]] = {"table": [], "image": [], "text": []}
    for h in text_hits:
        bt = str(h.get("block_type") or "text").lower()
        groups.setdefault(bt if bt in groups else "text", []).append(h)

    # 全局连续编号：文本/表格/图像三区共用一个编号序列。
    # 若每区各自从 1 开始，生成时模型引用 [1] 会指向三条不同证据（歧义）。
    # 编号 → 真实来源的映射在 build_v2 侧由代码维护（answers 里附「引用来源」），
    # 模型**只负责写 [n]**，不负责写页码 —— 实测 3B 模型会把页码编造出来
    # （真实页 4/259/260，它写成 23/26/25/49），所以页码绝不能交给它生成。
    counter = {"n": 0}

    def _emit(title: str, items: list[dict]) -> None:
        if not items:
            return
        parts.append(title)
        for h in items:
            counter["n"] += 1
            n = counter["n"]
            doc = h.get("source") or h.get("source_doc") or "未知文档"
            page = h.get("page_idx", h.get("page", -1))
            cid = h.get("chunk_id") or "-"
            text = (h.get("text") or "")[:max_chars_per_block]
            parts.append(f"[{n}] 来源：{doc} 第 {page} 页 "
                         f"（block_type={h.get('block_type', 'text')}, id={cid}）")
            parts.append(text)
            parts.append("")

    _emit("【文本证据】（原文片段，按相关性排序）", groups["text"])
    _emit("【表格证据】（结构化表格片段）", groups["table"])
    _emit("【图像描述】（图表/组织结构的语义描述）", groups["image"])

    ctx = "\n".join(parts).strip()
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars].rsplit("\n", 1)[0] + "\n…（上下文已按预算截断）"
    return ctx


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    qtype: str = config9.QTYPE_FACTUAL
    queries: list[str] = field(default_factory=list)
    linked: list[dict] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)
    text_hits: list[dict] = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    context: str = ""
    timings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def translate_query_to_zh(question: str, timeout: float = 6.0) -> str:
    """英文提问 → 中文检索式（跨语言检索）。

    ## 为什么必须做这一步（实测依据）

    语料是**中文年报**，而 bge-m3 虽然号称多语言，实测英文问句的检索质量
    明显低于中文。同一个问题：

        中文问「中国邮政储蓄银行2019年末的不良贷款率和拨备覆盖率分别是多少？」
            → 命中第 21 页（含 0.86% / 389.45%），答案正确
        英文问「What were Postal Savings Bank of China (601658) NPL ratio
               and provision coverage ratio at the end of 2019?」
            → 命中第 7 / 384 / 381 / 261 / 350 页（全是财务报表附注），
              答案变成「资料中未包含该信息」

    即：**多语言「输出」是通的（答案确实用英文写），但多语言「检索」是坏的** ——
    工单要求「支持中文和英文的问答」，只做对了一半。

    修法：英文问句先用小模型译成中文再检索，生成阶段仍用**原始英文问句**，
    保证答案语言与提问一致。用 qwen2.5:0.5b（实测 0.2~0.4s），
    因为它只做「翻译」这一件受约束的小事，不需要强模型。

    翻译失败时**原样返回英文问句**（退化为改动前的行为），绝不阻断问答。
    """
    q = (question or "").strip()
    if not q or q.isascii() is False:
        return q          # 已是中文（或含大量中文）→ 不需要翻译
    try:
        from src import llm
        res = llm.chat(
            [{"role": "system",
              "content": "你是翻译助手。把用户给出的英文问题翻译成中文，"
                         "只输出译文本身，不要解释、不要引号、不要加任何前缀。"},
             {"role": "user", "content": q}],
            model=config9.c8.OLLAMA_FALLBACK_MODEL,
            temperature=0.0, num_predict=96, timeout=timeout)
        zh = str(res.get("answer", "") if isinstance(res, dict) else res).strip()
        # 基本合理性检查：译文要有中文、长度合理
        if zh and any("一" <= c <= "鿿" for c in zh) and len(zh) <= len(q) * 4:
            return zh
    except Exception as exc:                           # noqa: BLE001
        print(f"[retrieve_v2] 英文问句翻译失败，按原语言检索：{exc}", flush=True)
    return q


def retrieve_v2(question: str, kg: KnowledgeGraph | None = None,
                use_cache: bool = True) -> RetrievalResult:
    """09 检索链路：实体识别 → 子图 → 查询分解 → 多路召回 → 融合 → 上下文。"""
    out = RetrievalResult()
    t_all = time.perf_counter()

    # 0) 跨语言：英文问句先译成中文再检索（生成阶段仍用原英文问句）
    t0 = time.perf_counter()
    search_q = question
    if question and question.strip().isascii():
        search_q = translate_query_to_zh(question)
        out.info["translated_query"] = search_q
    out.timings["translate"] = time.perf_counter() - t0

    # 1) 实体识别（去噪）
    t0 = time.perf_counter()
    out.linked = link_entities_v2(search_q, kg) if kg is not None else []
    out.timings["link"] = time.perf_counter() - t0

    # 2) 问题类型 → 检索预算
    out.qtype = classify_question(search_q, out.linked)
    out.queries = decompose_question(search_q, out.qtype)
    out.weights = fusion_weights(out.qtype)

    # 3) 子图检索（自适应 k-hop）
    t0 = time.perf_counter()
    if kg is None:
        sub = SubgraphResult(mode="none")
    else:
        try:
            sub = subgraph_v2(search_q, out.linked, kg, out.qtype)
        except Exception as exc:                       # noqa: BLE001
            print(f"[retrieve_v2] 子图检索失败，跳过图谱通道：{exc}", flush=True)
            sub = SubgraphResult(mode="none")
    out.timings["subgraph"] = time.perf_counter() - t0
    n_facts_raw = len(sub.facts)
    # 按相关性筛选：只把与问题相关的图谱事实放进上下文（见 filter_graph_facts）
    out.graph_facts = filter_graph_facts(search_q, sub.facts, linked=out.linked)
    out.subgraph = sub.to_html_data()
    out.info["n_graph_facts_raw"] = n_facts_raw
    out.info["n_graph_facts_kept"] = len(out.graph_facts)

    # 4) 多查询文本检索
    hits, info = retrieve_text_v2(out.queries, out.qtype)
    # 合并而非整体替换：out.info 里已经有图谱事实的统计
    # （n_graph_facts_raw / kept），直接 `out.info = info` 会把它们冲掉。
    out.info.update(info)
    out.timings["retrieve"] = info.get("elapsed_ms", 0) / 1000

    # 5) 去重 → 低质量块过滤 → 截断
    deduped = fuse_text_hits(hits, config9.RERANK9_POOL)
    n_before = len(deduped)
    kept = filter_low_quality(deduped)
    out.text_hits = kept[:config9.FINAL9_TOP_K]
    out.info["n_dedup"] = n_before
    out.info["n_low_quality_dropped"] = max(n_before - len(kept), 0)

    # 6) 上下文
    out.context = build_context_v2(out.graph_facts, out.text_hits)

    out.timings["total"] = time.perf_counter() - t_all
    return out
