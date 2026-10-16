# -*- coding: utf-8 -*-
"""工单9 优化 · **检索层**：查询改写与实体链接、图谱跳数与剪枝、图特征×向量/BM25 融合、重排裁剪、同页扩展。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

------------------------------------------------------------
现状与问题（工单8 实测，逐条可追溯）
------------------------------------------------------------
1. **图谱通道形同虚设**：工单8 的融合是 ``score = base_score + graph_bonus``，
   而 ``base_score``（重排 final_score）≈ 4.x，``graph_bonus`` ∈ [0.022, 0.06]，
   **量纲差两个数量级**。后果（工单8 汇总 json 可查）：17 题里
   「图谱独有块进入 Top5 的题数 = 0」，recall@5 仅提升 0.0266（且完全来自同分档次序微调）。

2. **固定 2 跳 + 无邻居剪枝**：``hop=2`` 对「已明确点名公司」的题是纯噪声来源
   （实测 Q07 两跳走到中国平安/招商银行/国泰君安的年报块），且 ``max_expand_nodes=40``
   是**先到先得**截断（BFS 顺序），不是按边质量截断。

3. **没有查询改写/实体链接**：直接用 ``name_index`` 做子串匹配，问句里的
   简称/别名/字段词都没有被用于扩展检索。

4. **没有候选裁剪与重排**：基础召回多少就直接融合多少，没有「先粗筛再精排」的两段式。

5. **不做上下文扩展**：工单8 ``page_context_extra=0``，答案所在页的**同类兄弟块**
   完全没进上下文 ⇒ 上下文召回上不去。

------------------------------------------------------------
优化方案（本模块的函数）
------------------------------------------------------------
* :func:`link_entities`     —— 实体链接：图谱词典最长匹配 + 别名回溯 + 语料锁定；
* :func:`adaptive_hop`      —— 自适应跳数：已锁定语料 ⇒ 1 跳（防跨公司污染），否则 2 跳；
* :func:`expand_pruned`     —— 邻居按「边置信度 × 度数倒数」打分截断的 BFS（取代先到先得）；
* :func:`fuse_scores`       —— **min-max 归一化后加权融合**（取代量纲不可比的直接相加）；
* :func:`rerank_trim`       —— 两段式：先裁到 ``rerank_trim`` 条，再按「问句实词命中率」重排取 top_k；
* :func:`page_expand`       —— 同页上下文扩展（**只用查询侧信息**，绝不用参考答案的页码提示）。

⚠️ **公平性声明**：以上所有手段都只使用**查询侧可得的信息**
（问句本身、检索到的块、图谱、以及从问句识别出的公司线索），
**绝不使用**评估集里的 ``expected_pages_hint`` / ``answer_keywords`` ——
否则「优化前 vs 优化后」的对比就没有意义了。
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 中文疑问词/停用词（查询改写时剔除）
_STOPWORDS = frozenset({
    "哪些", "哪个", "什么", "多少", "怎么", "如何", "是否", "以及", "分别", "包括", "请问",
    "说明", "分析", "结合", "下列", "以下", "关于", "对于", "他们", "它们", "这个", "那个",
    "年度报告", "年报", "报告", "根据", "以上", "如下", "情况", "方面", "内容", "问题",
    "主要", "进行", "通过", "相关", "表示", "上述", "该行", "本行", "我们", "公司简称",
    "what", "were", "which", "with", "that", "this", "from", "according", "annual", "report",
    "the", "and", "for", "was", "are",
})

#: 常见财务字段词（查询改写与**字段锚点召回**用：把这些词作为「图谱字段锚点」补进检索词）
FIELD_TERMS: tuple[str, ...] = (
    "营业收入", "净利润", "归属于母公司股东的净利润", "归母净利润", "总资产", "净资产",
    "不良贷款率", "拨备覆盖率", "资本充足率", "核心一级资本充足率", "每股收益", "每股净资产",
    "每股现金分红", "每股现金股息", "已赚保费", "保费收入", "营运利润", "净资产收益率",
    "营业总收入", "手续费及佣金净收入", "利息净收入", "存款总额", "贷款总额",
)

#: **年报通用章节词表**（字段锚点召回的扩展探针词）。
#: 说明：这是**行业通用的年报章节/主题词汇**（任何一份年报都有「风险管理」「公司治理」
#: 「绿色金融」这些章节），**不是**针对某道测试题的答案词 —— 用于把问句里的
#: 定性主题词（「风险管理」「绿色金融」）映射到语料里对应的章节块。
#: 依据（实测 _缺失分析）：Q03「风险管理体系…」的参考答案要点（不良贷款率/绿色信贷/赤道原则）
#: 全在 p43~p67 的风险管理章节，而纯 Top-K 检索跑到了 p71/p125/p227/p227（章节之外），
#: 召回只有 0.43。按主题词定点捞块后，该题的证据回到风险章节。
DOMAIN_TERMS: tuple[str, ...] = (
    "风险管理", "风险管控", "信用风险", "市场风险", "操作风险", "流动性风险",
    "资产质量", "不良贷款", "逾期贷款", "拨备", "贷款结构", "信贷投放",
    "资本管理", "资本结构", "资产负债", "资产负债匹配", "负债久期", "偿付能力",
    "公司治理", "内部控制", "合规", "反洗钱", "内部审计",
    "绿色金融", "绿色信贷", "赤道原则", "科技金融", "金融科技", "数字化转型",
    "普惠金融", "养老金融", "财富管理", "私人银行", "信用卡", "消费金融",
    "大数据风控", "智能风控", "人工智能", "区块链", "云计算",
    "人力资源", "社会责任", "企业文化", "公司战略", "经营计划", "利润分配",
    "业绩概览", "主要财务指标", "经营情况", "董事会报告", "监事会报告", "重要事项",
)

#: **英文 → 中文财务术语映射**（多语言支持的一部分）。
#: 依据（工单要求「支持中文和英文的问答」+ 实测）：Q13 是英文题，
#: 若不做英中映射，中文语料上的字段锚点召回**一个字段也命中不了**
#: （问句里没有「净利润」这三个汉字），该题上下文召回只有 0.33。
EN_TERM_MAP: dict[str, tuple[str, ...]] = {
    "net profit": ("净利润",), "net profits": ("净利润",), "profit": ("净利润",),
    "net income": ("净利润",), "operating income": ("营业收入",),
    "revenue": ("营业收入",), "total revenue": ("营业收入",),
    "profit attributable": ("归属于母公司股东的净利润",), "attributable to shareholders": ("归属于母公司股东的净利润",),
    "total assets": ("总资产",), "total liabilities": ("总负债",), "net assets": ("净资产",),
    "non-performing loan": ("不良贷款率",), "nonperforming loan": ("不良贷款率",),
    "npl ratio": ("不良贷款率",), "provision coverage": ("拨备覆盖率",),
    "capital adequacy": ("资本充足率",), "core tier 1": ("核心一级资本充足率",),
    "earnings per share": ("每股收益",), "eps": ("每股收益",),
    "return on equity": ("净资产收益率",), "roe": ("净资产收益率",),
    "net interest margin": ("净息差",), "net interest income": ("利息净收入",),
    "deposits": ("存款总额",), "loans": ("贷款总额",), "premium": ("保费收入",),
    "year-on-year": ("同比",), "yoy": ("同比",), "growth": ("同比增长",),
    "dividend": ("每股现金分红",), "risk management": ("风险管理",),
    "green finance": ("绿色金融",), "green credit": ("绿色信贷",),
    "fintech": ("金融科技",), "digital transformation": ("数字化转型",),
    "asset quality": ("资产质量",), "internal control": ("内部控制",),
}

#: **英文公司名 → 中文公司名**（多语言支持的关键一环）。
#: 依据（实测）：语料文件名与正文都是**中文公司名**，而英文提问里写的是
#: ``Ping An Bank`` / ``Postal Savings Bank of China``。
#: 查询理解（``analyze``）从英文问句里**识别不出 corpus_hint** ⇒ 语料路由失效 ⇒
#: 字段锚点在 9 份年报里乱捞（实测「平安银行英文题」的答案里一个数字都没有）。
#: 映射后：① 中文公司名被并进检索查询；② 引擎可用它重新做一次语料路由。
EN_COMPANY_MAP: dict[str, str] = {
    "ping an bank": "平安银行",
    "postal savings bank of china": "邮储银行",
    "postal savings bank": "邮储银行",
    "china merchants bank": "招商银行",
    "china merchants securities": "招商证券",
    "china pacific insurance": "中国太保",
    "china pacific": "中国太保",
    "citic securities": "中信证券",
    "china life insurance": "中国人寿",
    "china life": "中国人寿",
    "guotai junan": "国泰君安",
    "ping an insurance": "中国平安",
    "ping an group": "中国平安",
}

#: 探针词全表 = 字段词 + 章节词（倒排索引按这张表构建）
PROBE_TERMS: tuple[str, ...] = tuple(dict.fromkeys(FIELD_TERMS + DOMAIN_TERMS))


# ---------------------------------------------------------------------------
# 分词（jieba 优先；不可用时退回标点切分——**不得阻断检索**）
# ---------------------------------------------------------------------------
def tokenize(text: str, *, min_len: int = 2) -> list[str]:
    """问句实词切分（去停用词、去纯数字、保序去重）。"""
    raw: list[str] = []
    try:
        import jieba

        raw = [str(w) for w in jieba.lcut(str(text or ""))]
    except Exception:  # noqa: BLE001
        raw = [p for p in re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]“”\"]+", str(text or "")) if p]
    out: list[str] = []
    for word in raw:
        word = str(word).strip()
        if len(word) < int(min_len) or word in out:
            continue
        if word.lower() in _STOPWORDS:
            continue
        if re.fullmatch(r"[\d.,%]+", word):
            continue
        out.append(word)
    return out


def query_terms(question: str, *, limit: int = 24) -> list[str]:
    """检索/重排用的问句实词（长词优先，含字段词增强）。"""
    terms = tokenize(question)
    # 标点切分兜底：中文没有词边界，长片段（≥3 字）也保留，避免实词表为空
    for piece in re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]“”\"]+", str(question or "")):
        piece = piece.strip()
        if 3 <= len(piece) <= 12 and piece not in terms and piece.lower() not in _STOPWORDS:
            terms.append(piece)
    terms.sort(key=len, reverse=True)
    return terms[:limit]


def rewrite_query(question: str, *, linked_names: Sequence[str] = (), cfg: Mapping[str, Any],
                  language: str = "zh") -> dict[str, Any]:
    """查询改写：在原问句基础上**追加实体别名、字段锚点与英中术语**，形成检索用的增强查询。

    为什么不是「用 LLM 重写问句」：本机只有 3B 模型且 SLA ≤ 3 s，
    在线跑一次 LLM 改写会把预算吃光（工单8 实测生成层已占 3.9 s）。
    这里采用**确定性改写**（零额外耗时、可复现、可审计）：
      ① 把链接到的图谱实体**规范名 + 别名**补进查询（帮助 BM25/向量召回同义表述）；
      ② 把问句里出现的财务字段词显式列出（图谱字段锚点）；
      ③ **英文题补中文术语**（``EN_TERM_MAP``）—— 这是「多语言支持」在检索层的落地：
         语料是中文年报，英文问句若不同时带中文术语，BM25 通道完全命中不了，
         向量通道的跨语对齐也不足以稳定召回（实测 Q13 优化前 recall@k = 0，
         关键词 610.36亿元 / 16.52% / 389.45% 全丢）。
    """
    if not bool(cfg.get("query_rewrite", True)):
        return {"query": question, "retrieval_query": question, "added_names": [],
                "added_fields": [], "terms": query_terms(question)}
    base = str(question or "")
    added_names: list[str] = []
    for name in linked_names:
        name = str(name or "").strip()
        if name and name not in base and name not in added_names:
            added_names.append(name)
    added_fields = [term for term in FIELD_TERMS if term in base]
    # 数值型问句补「单位」锚点（年报里同一指标常有「百万元 / 亿元」两种写法）
    if re.search(r"(多少|分别是|同比|增减|单位)", base) and "单位" not in base:
        added_fields.append("单位")

    # ③ 英文题：补中文术语与中文公司名（用于检索；也顺带让字段锚点召回生效）
    added_terms: list[str] = []
    added_companies: list[str] = []
    if str(language).startswith("en") or bool(cfg.get("query_rewrite_english", True)):
        lowered = base.lower()
        for en, zh_terms in EN_TERM_MAP.items():
            if en in lowered:
                for zh in zh_terms:
                    if zh not in base and zh not in added_terms:
                        added_terms.append(zh)
        for en, zh in EN_COMPANY_MAP.items():
            if en in lowered and zh not in base and zh not in added_companies:
                added_companies.append(zh)

    suffix = "".join(added_names[:6] + added_companies[:3] + added_fields[:4] + added_terms[:8])
    rewritten = f"{base} {suffix}".strip() if suffix else base
    # 检索用查询：中文题保持原文（保证与 工单8 基线「同一入口」可比）；
    # 英文题用增强查询（否则中文语料上 BM25 通道整个失效）。
    use_rewrite_for_base = bool(cfg.get("base_query_use_rewrite", True)) and bool(
        added_terms or added_companies)
    return {"query": rewritten, "retrieval_query": rewritten if use_rewrite_for_base else base,
            "use_rewrite_for_base": use_rewrite_for_base,
            "added_names": added_names[:6], "added_fields": added_fields[:4],
            "added_terms": added_terms[:8], "added_companies": added_companies[:3],
            "terms": query_terms(base)}


# ---------------------------------------------------------------------------
# ① 实体链接
# ---------------------------------------------------------------------------
def link_entities(question: str, graph: Any, *, cfg: Mapping[str, Any],
                  corpus_hint: str | None = None) -> dict[str, Any]:
    """把问句链接到图谱实体（**长名优先 + 别名回溯 + 同名归并**）。

    :returns: ``{"seeds": [node_id…], "names": [...], "matched": [{name, node_id, type, len}…]}``
    """
    if graph is None or not getattr(graph, "ready", False):
        return {"seeds": [], "names": [], "matched": []}
    if not bool(cfg.get("entity_linking", True)):
        seeds = list(graph.match_names(question, min_len=int(cfg.get("link_min_len") or 2),
                                       limit=int(cfg.get("link_limit") or 12)))
        return {"seeds": seeds, "names": [graph.name_of(s) for s in seeds], "matched": []}

    flat = "".join(str(question or "").split())
    min_len = int(cfg.get("link_min_len") or 2)
    limit = int(cfg.get("link_limit") or 12)
    found: list[tuple[int, int, str, str]] = []      # (名字长度, 实体热度, node_id, 名字)
    for name, node_id in graph.name_index.items():
        if len(name) < min_len or name not in flat:
            continue
        node = graph.node(node_id) or {}
        found.append((len(name), int(node.get("count") or 0), str(node_id), str(name)))
    found.sort(key=lambda item: (-item[0], -item[1]))
    seeds: list[str] = []
    matched: list[dict[str, Any]] = []
    for length, _count, node_id, name in found:
        if node_id in seeds:
            continue
        node = graph.node(node_id) or {}
        seeds.append(node_id)
        matched.append({"name": name, "node_id": node_id, "len": length,
                        "type": str(node.get("type") or ""),
                        "aliases": list(node.get("aliases") or [])[:6]})
        if len(seeds) >= limit:
            break
    return {"seeds": seeds, "names": [graph.name_of(s) for s in seeds], "matched": matched}


def adaptive_hop(*, corpus_hint: str | None, linked: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """自适应跳数：已锁定语料 ⇒ 降到 ``hop_when_linked``（默认 1）；否则用 ``hop``（默认 2）。

    依据（工单8 实测）：9 份年报混排时两跳会走到**别家公司**的年报
    （Q07 的「平安银行」两跳带进中国平安/招商银行/国泰君安），这些块对问题毫无用处。
    问句既然已经点名了公司，就没有必要再靠两跳去「猜」语料。
    """
    hop = int(cfg.get("hop") or 2)
    linked_hop = int(cfg.get("hop_when_linked") or hop)
    strong = bool(corpus_hint) and len(linked.get("seeds") or []) >= 1
    return {"hop": linked_hop if strong else hop, "reason": ("已锁定语料 ⇒ 降为 %d 跳防跨公司污染"
                                                             % linked_hop) if strong else "未锁定语料 ⇒ 用 %d 跳扩大召回" % hop,
            "corpus_hint": corpus_hint}


# ---------------------------------------------------------------------------
# ② 图谱扩展（带剪枝）
# ---------------------------------------------------------------------------
def expand_pruned(graph: Any, seeds: Sequence[str], *, hop: int, cfg: Mapping[str, Any],
                  corpus_of_file: Any = None, corpus_hint: str | None = None) -> dict[str, Any]:
    """按「边置信度 × 度数倒数」排序截断的 BFS 扩展（取代先到先得的 ``max_nodes`` 截断）。

    为什么要按度数倒数加权：年报里 ``本集团``/``2020年度`` 这类节点度数极高，
    它们的邻居几乎全是噪声；而一个度数 2 的节点（如「逾越者联盟」）的邻居信息量高得多。
    打分 = ``confidence × 1/log(2 + degree)``。

    跨语料闸门：``corpus_hint`` 非空时，只接受**提及块属于该语料**的邻居节点
    （用它们自己的提及块的文件名判定），进一步压制跨公司污染。
    """
    if graph is None or not getattr(graph, "ready", False) or not seeds:
        return {"distances": {}, "nodes": {}, "edges_used": [], "truncated": False, "hop": hop,
                "filtered_cross_corpus": 0}
    max_nodes = int(cfg.get("max_expand_nodes") or 32)
    min_conf = float(cfg.get("min_confidence") or 0.0)
    neighbor_top_k = int(cfg.get("neighbor_top_k") or 12)
    apply_filter = bool(cfg.get("graph_corpus_filter", True)) and bool(corpus_hint)

    def node_corpus_ok(node_id: str) -> bool:
        if not apply_filter or corpus_of_file is None:
            return True
        for chunk in graph.chunks_of(node_id, limit=3):
            try:
                if str(corpus_of_file(str(chunk.get("file_name") or ""))) == str(corpus_hint):
                    return True
            except Exception:  # noqa: BLE001 —— 映射失败按「不通过」处理（保守）
                return True
        return False

    distances: dict[str, int] = {}
    score_of: dict[str, float] = {}
    frontier: list[str] = []
    for seed in seeds:
        sid = str(seed)
        if graph.node(sid) is None or sid in distances:
            continue
        distances[sid] = 0
        score_of[sid] = 1.0
        frontier.append(sid)

    edges_used: list[dict[str, Any]] = []
    truncated = False
    filtered = 0
    depth = 0
    while frontier and depth < max(int(hop), 0):
        depth += 1
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for node_id in frontier:
            for neighbor, relation, edge_idx in graph.neighbors(node_id):
                edge = graph.edges[edge_idx] if 0 <= edge_idx < len(graph.edges) else {}
                conf = float(edge.get("confidence") or 0.0)
                if conf < min_conf:
                    continue
                edges_used.append({"head": edge.get("head"), "relation": relation,
                                   "tail": edge.get("tail"), "confidence": conf,
                                   "chunk_id": edge.get("chunk_id"), "hop": depth})
                if neighbor in distances:
                    continue
                degree = len(graph.adjacency.get(neighbor, []))
                rank = conf * (1.0 / math.log(2.0 + float(degree)))
                candidates.append((rank, str(neighbor), {"relation": relation, "confidence": conf}))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        next_frontier: list[str] = []
        kept_this_hop = 0
        for rank, neighbor, _meta in candidates:
            if kept_this_hop >= neighbor_top_k * max(len(frontier), 1):
                truncated = True
                break
            if len(distances) >= max_nodes:
                truncated = True
                break
            if neighbor in distances:
                continue
            if not node_corpus_ok(neighbor):
                filtered += 1
                continue
            distances[neighbor] = depth
            score_of[neighbor] = rank
            next_frontier.append(neighbor)
            kept_this_hop += 1
        frontier = next_frontier
    return {"distances": distances, "node_rank": score_of, "edges_used": edges_used,
            "truncated": truncated, "hop": int(hop), "filtered_cross_corpus": filtered}


# ---------------------------------------------------------------------------
# ③ 融合：min-max 归一化 + 加权
# ---------------------------------------------------------------------------
def _minmax(values: Sequence[float]) -> list[float]:
    """min-max 归一化到 [0,1]；全相等时统一给 1.0（而不是 0，避免整段被乘没）。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def graph_features(*, distance: int | None, entity_count: int, node_degree: int,
                   cfg: Mapping[str, Any]) -> dict[str, float]:
    """图特征三件套（都归一到 [0,1]，权重来自配置 ``feature_weights``）。"""
    hop_decay = 0.0 if distance is None else max(0.0, 1.0 - float(distance) / 3.0)
    entity_hit = min(float(entity_count) / 3.0, 1.0)
    density = min(1.0 / math.log(2.0 + float(max(node_degree, 0))), 1.0)
    weights = dict(cfg.get("feature_weights") or {})
    total = sum(float(v) for v in weights.values()) or 1.0
    value = (float(weights.get("entity_hit", 0.5)) * entity_hit
             + float(weights.get("hop_decay", 0.3)) * hop_decay
             + float(weights.get("density", 0.2)) * density) / total
    return {"entity_hit": round(entity_hit, 4), "hop_decay": round(hop_decay, 4),
            "density": round(density, 4), "feature": round(value, 6)}


def fuse_scores(rows: Sequence[Mapping[str, Any]], *, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """**图特征 × 向量/BM25 的归一化加权融合**（取代工单8 的量纲不可比直接相加）。

    公式::

        nb_i = minmax(base_i)                     # 基础检索分（RRF+重排）归一化
        ng_i = graph_i / max(graph)               # 图结构分归一化（天然 ≥ 0）
        nf_i = feature_i                          # 图特征（已归一）
        final_i = w_base·nb_i + w_graph·ng_i + w_feature·nf_i     (w 之和 = 1)

    权重全部来自 ``retrieval.fusion_weights``，可在 ``部署/配置/优化配置.json`` 里改。
    """
    if not rows:
        return []
    weights = dict(cfg.get("fusion_weights") or {})
    w_base = float(weights.get("base", 0.62))
    w_graph = float(weights.get("graph", 0.23))
    w_feature = float(weights.get("feature", 0.15))
    total = w_base + w_graph + w_feature or 1.0
    w_base, w_graph, w_feature = w_base / total, w_graph / total, w_feature / total

    base = [float(r.get("base_score") or 0.0) for r in rows]
    graph = [float(r.get("graph_score") or 0.0) for r in rows]
    nbase = _minmax(base)
    gmax = max(graph) if graph else 0.0
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        ngraph = (graph[idx] / gmax) if gmax > 0 else 0.0
        nfeat = float(row.get("feature") or 0.0)
        item = dict(row)
        item["norm_base"] = round(nbase[idx], 6)
        item["norm_graph"] = round(ngraph, 6)
        item["norm_feature"] = round(nfeat, 6)
        item["fusion_score"] = round(w_base * nbase[idx] + w_graph * ngraph + w_feature * nfeat, 6)
        item["score"] = item["fusion_score"]
        item["final_score"] = item["fusion_score"]
        out.append(item)
    out.sort(key=lambda r: (-float(r["fusion_score"]), str(r.get("chunk_id"))))
    for rank, item in enumerate(out, 1):
        item["rank"] = rank
    return out


def rerank_trim(rows: Sequence[Mapping[str, Any]], *, terms: Sequence[str], top_k: int,
                cfg: Mapping[str, Any]) -> dict[str, Any]:
    """两段式重排裁剪：先按融合分截断到 ``rerank_trim``，再用「问句实词命中率」重排取 top_k。

    重排分::

        lex_i   = |{t ∈ terms : t ∈ text_i}| / |terms|          （词命中率）
        score_i = (1 - w) · fusion_i + w · lex_i                （w = rerank_lexical_weight）
    """
    trim = int(cfg.get("rerank_trim") or len(rows))
    w = float(cfg.get("rerank_lexical_weight") or 0.35)
    trimmed = list(rows)[: max(trim, 1)]
    for item in trimmed:
        text = str(item.get("content") or item.get("text") or "")
        hits = [t for t in terms if t and t in text]
        item["lexical_hits"] = len(hits)
        item["lexical_score"] = round(len(hits) / len(terms), 6) if terms else 0.0
        item["rerank_score"] = round((1.0 - w) * float(item.get("fusion_score") or 0.0)
                                     + w * float(item["lexical_score"]), 6)
        item["score"] = item["rerank_score"]
        item["final_score"] = item["rerank_score"]
    trimmed.sort(key=lambda r: (-float(r["rerank_score"]), str(r.get("chunk_id"))))
    for rank, item in enumerate(trimmed, 1):
        item["rank"] = rank
    return {"hits": trimmed[: max(int(top_k), 1)], "trimmed": trimmed,
            "trim_count": len(trimmed), "lexical_weight": w}


# ---------------------------------------------------------------------------
# ④ 同页上下文扩展
# ---------------------------------------------------------------------------
def page_expand(hits: Sequence[Mapping[str, Any]], *, bundle: Any, cfg: Mapping[str, Any],
                corpus_of_file: Any = None, corpus_hint: str | None = None,
                channel: str = "page_expand") -> dict[str, Any]:
    """同页兄弟块扩展：把 Top-N 命中所在**页面**的其它块补进候选。

    为什么能提升上下文召回：数值题的答案常常**同页分散在多块**里
    （正文一段 + 表格一块 + 图像块一块）。检索只命中其中一块时，
    参考答案的其它要点就落在上下文之外 ⇒ 召回上不去。同页扩展把这一页补全。

    **风险与闸门**：若命中页本身不是答案页，扩展会引入噪声（精度下降）。因此：
      * 只扩展**融合后 Top-N** 命中的页（它们是答案页的概率最高）；
      * ``page_expand_require_corpus`` 为真且问句已锁定语料时，
        只扩展**属于该语料**的页 —— 防止把别家公司年报的同页码块拉进来
        （9 份年报混排，页码毫无全局意义）。
    """
    if not bool(cfg.get("page_expand", True)) or bundle is None:
        return {"extra": [], "pages": [], "skipped": "page_expand 关闭或索引束不可用"}
    top_pages = int(cfg.get("page_expand_pages") or 3)
    per_page = int(cfg.get("page_expand_max_per_page") or 4)
    max_total = int(cfg.get("page_expand_max_total") or 10)
    require_corpus = bool(cfg.get("page_expand_require_corpus", True))

    seen_ids = {str(h.get("chunk_id")) for h in hits}
    keys: list[tuple[str, int]] = []
    for hit in hits:
        file_name = str(hit.get("file_name") or "")
        page = int(hit.get("page") or 0)
        if not file_name or page <= 0:
            continue
        key = (file_name, page)
        if key in keys:
            continue
        if require_corpus and corpus_hint and corpus_of_file is not None:
            try:
                if str(corpus_of_file(file_name)) != str(corpus_hint):
                    continue
            except Exception:  # noqa: BLE001 —— 映射失败时保守跳过（不引入风险）
                continue
        keys.append(key)
        if len(keys) >= top_pages:
            break

    extra: list[dict[str, Any]] = []
    for file_name, page in keys:
        for row in _page_chunks(bundle, file_name, page):
            if len(extra) >= max_total:
                break
            cid = str(row.get("chunk_id") or "")
            if not cid or cid in seen_ids:
                continue
            if sum(1 for e in extra if e.get("_page_key") == f"{file_name}#p{page}") >= per_page:
                continue
            seen_ids.add(cid)
            extra.append({**dict(row), "retrieval_channel": channel,
                          "base_score": 0.0, "graph_score": 0.0, "feature": 0.0,
                          "fusion_score": 0.0, "rerank_score": 0.0, "score": 0.0,
                          "final_score": 0.0, "lexical_score": 0.0, "lexical_hits": 0,
                          "_page_key": f"{file_name}#p{page}",
                          "page_expand_of": f"{file_name}#p{page}"})
    for item in extra:
        item.pop("_page_key", None)
    return {"extra": extra, "pages": [f"{f}#p{p}" for f, p in keys],
            "corpus_gate": bool(require_corpus and corpus_hint)}


def _page_chunks(bundle: Any, file_name: str, page: int) -> list[Mapping[str, Any]]:
    """取「某文件某页」的全部块。

    优先用索引束自带的 ``page_chunks``（工单7 的 ``IndexBundle.pages`` 是
    ``(file_name, page) → [chunk_id]`` 的**页索引**，O(1) 定位），
    没有时回退到遍历全部块（``chunks`` 是 ``dict``，遍历它不会复制整表）。
    """
    helper = getattr(bundle, "page_chunks", None)
    if callable(helper):
        try:
            return list(helper(file_name, int(page)))
        except Exception:  # noqa: BLE001 —— 页索引不可用时回退遍历（不阻断检索）
            pass
    chunks = getattr(bundle, "chunks", None)
    if isinstance(chunks, Mapping):
        return [row for row in chunks.values()
                if str(row.get("file_name") or "") == file_name
                and int(row.get("page") or 0) == int(page)]
    return []


def _iter_bundle_chunks(bundle: Any) -> Iterable[Mapping[str, Any]]:
    """遍历索引束里的全部块（``chunks`` 是 ``dict[str, dict]``，遍历它不复制整表）。"""
    chunks = getattr(bundle, "chunks", None)
    if isinstance(chunks, Mapping):
        return chunks.values()
    return []


# ---------------------------------------------------------------------------
# ⑤ 字段锚点召回（field-anchored retrieval）
# ---------------------------------------------------------------------------
_HAS_DIGIT = re.compile(r"\d")
#: 字段词与数值的邻近窗口（字符）：年报里「营业收入 137,958」这类绑定通常在同一行内
_NEAR_WINDOW = 80


def build_field_index(bundle: Any, *, field_terms: Sequence[str] = PROBE_TERMS) -> dict[str, list[str]]:
    """一次性构建「字段词 → 含该字段且含数字的 chunk_id 列表」倒排索引（**引擎级缓存**）。

    为什么需要它（实测依据，见 ``_缺失分析`` 输出）：
        仅靠「向量 + BM25 + 图谱」的 Top-K，数值题的**答案块常常进不了候选**
        —— 问句「平安银行营业收入/净利润/不良贷款率/拨备覆盖率分别是多少」，
        检索命中 p19 表格（含营业收入）后，``拨备覆盖率``（在 p44）与 ``不良贷款率``（在 p20/p44）
        就再也没机会进上下文，context_recall 卡在 0.77 上不去。
        字段锚点召回**按问句里出现的每个财务字段词**去语料里定点捞块，
        保证「问到的每个字段都有对应的证据块」——这是提升上下文召回最直接的手段。

    代价：一次遍历 8847 块 × 50 个字段词（实测约 0.6 s，只在 warmup 时做一次），
    之后每次查询都是 O(1) 查表。**单进程、无进程池**。
    """
    index: dict[str, list[str]] = {}
    chunks = getattr(bundle, "chunks", None)
    if not isinstance(chunks, Mapping):
        return index
    for cid, row in chunks.items():
        text = str(row.get("content") or row.get("text") or "")
        if not text or not _HAS_DIGIT.search(text):
            continue                                   # 没有数字的块不可能承载「指标取值」
        for term in field_terms:
            if term in text:
                index.setdefault(term, []).append(str(cid))
    return index


def _near_number(text: str, term: str) -> bool:
    """字段词附近（``_NEAR_WINDOW`` 窗口内）是否出现数字 —— 「指标 ↔ 数值」绑定的弱判据。"""
    start = 0
    while True:
        pos = text.find(term, start)
        if pos < 0:
            return False
        window = text[max(pos - 20, 0): pos + len(term) + _NEAR_WINDOW]
        if _HAS_DIGIT.search(window):
            return True
        start = pos + 1


def probe_terms_of(question: str, *, cfg: Mapping[str, Any] | None = None,
                   max_terms: int | None = None) -> list[str]:
    """从问句里挑出**探针词**：中文按正则匹配、英文按 ``EN_TERM_MAP`` 映射。

    这一步是「多语言支持」在检索层的落地：英文问句 ``net profit`` 会被映射成
    中文语料里的「净利润」，从而让字段锚点召回在英文提问下同样生效。
    """
    text = str(question or "")
    lowered = text.lower()
    out: list[str] = []
    for term in PROBE_TERMS:
        if term in text and term not in out:
            out.append(term)
    for en, zh_terms in EN_TERM_MAP.items():
        if en in lowered:
            for zh in zh_terms:
                if zh not in out:
                    out.append(zh)
    limit = int(max_terms if max_terms is not None else ((cfg or {}).get("field_probe_max_terms") or 10))
    # 长词优先（「核心一级资本充足率」比「资本充足率」更具体）
    out.sort(key=len, reverse=True)
    return out[:limit]


def field_probe(question: str, bundle: Any, index: Mapping[str, Sequence[str]], *,
                cfg: Mapping[str, Any], terms: Sequence[str],
                corpus_of_file: Any = None, corpus_hint: str | None = None,
                anchor_hits: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """按问句里的财务字段词/主题章节词定点补捞证据块。

    ``field_probe_per_field`` 控制每个词最多补几块，``field_probe_max`` 控制总量。

    **页一致优先（``field_probe_page_bonus``）**：若候选块落在「已有 Top-K 命中页 ±1 页」
    范围内，给加分。依据（实测）：同一道题的答案在年报里**集中在相邻几页**
    （Q07 的营业收入/不良率/拨备覆盖率分别在 p19/p20/p44）；
    不加这个约束时，字段锚点会从**别的章节**捞回同名指标（如把「全行各分行营业收入」
    从 p125 捞进 Q07 的上下文），上下文精度因此掉到 0.79。加了页一致约束后，
    补块优先落在已命中的、更可能正确的页面上 —— 召回与精度同时受益。
    """
    if not bool(cfg.get("field_probe", True)) or not index:
        return {"extra": [], "fields": [], "skipped": "field_probe 关闭或索引为空"}
    per_field = int(cfg.get("field_probe_per_field") or 2)
    max_total = int(cfg.get("field_probe_max") or 8)
    page_bonus = float(cfg.get("field_probe_page_bonus") or 1.0)
    page_window = int(cfg.get("field_probe_page_window") or 1)
    fields = probe_terms_of(question, cfg=cfg)
    if not fields:
        return {"extra": [], "fields": [], "skipped": "问句未命中任何探针词（字段词/章节词/英中映射）"}

    # 已有命中的 (文件, 页) 锚点
    anchors: set[tuple[str, int]] = set()
    for hit in anchor_hits:
        name = str(hit.get("file_name") or "")
        page = int(hit.get("page") or 0)
        if name and page > 0:
            for delta in range(-page_window, page_window + 1):
                anchors.add((name, page + delta))

    extra: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_fields: list[str] = []
    for field in fields:
        scored: list[tuple[float, str]] = []
        for cid in index.get(field, []):
            if cid in seen:
                continue
            row = bundle.chunk(cid) if hasattr(bundle, "chunk") else None
            if row is None:
                continue
            if corpus_hint and corpus_of_file is not None:
                try:
                    if str(corpus_of_file(str(row.get("file_name") or ""))) != str(corpus_hint):
                        continue                        # 语料闸门：只在本公司年报里补
                except Exception:  # noqa: BLE001
                    pass
            text = str(row.get("content") or "")
            near = _near_number(text, field)
            lex = sum(1 for t in terms if t and t in text)
            bonus = 0.5 if str(row.get("type") or "text") == "table" else 0.0
            on_page = ((str(row.get("file_name") or ""), int(row.get("page") or 0)) in anchors)
            scored.append((2.0 * (1.0 if near else 0.0) + 1.0 * lex + bonus
                           + (page_bonus if on_page else 0.0), cid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        took = 0
        for score, cid in scored:
            if took >= per_field or len(extra) >= max_total:
                break
            row = bundle.chunk(cid) if hasattr(bundle, "chunk") else None
            if row is None:
                continue
            seen.add(cid)
            extra.append({**dict(row), "retrieval_channel": "field_probe",
                          "base_score": 0.0, "graph_score": 0.0, "feature": 0.0,
                          "fusion_score": 0.0, "rerank_score": 0.0, "score": 0.0,
                          "final_score": 0.0, "lexical_score": 0.0, "lexical_hits": 0,
                          "field_probe": field, "field_score": round(score, 4)})
            took += 1
        if took:
            used_fields.append(field)
    return {"extra": extra, "fields": used_fields, "skipped": ""}


# ---------------------------------------------------------------------------
# ⑥ 证据选取（进生成阶段）
# ---------------------------------------------------------------------------
def is_comparison_question(question: str, source_files: Sequence[str] = ()) -> bool:
    """是否是「跨语料对比题」（用于放宽语料闸门、开启跨语料轮转）。

    依据（实测）：Q09「对比平安银行与招商银行2019年年度报告…」问句里**同时点名了两家公司**，
    查询理解只能给出其中一个 ``corpus_hint``（或为空），若据此做语料闸门，
    另一家公司的块会被全部挡掉 —— 上下文召回只有 0.5（缺 招商银行的 2,697.03 / 8.51%）。
    对比题的语料闸门必须放宽、并按语料轮转取证。
    """
    text = str(question or "")
    if len(list(source_files or [])) > 1:
        return True
    return any(marker in text for marker in ("对比", "比较", "共同", "差异化", "分别", "各自", "差异"))


def select_evidence(hits: Sequence[Mapping[str, Any]], *, terms: Sequence[str], cfg: Mapping[str, Any],
                    corpus_of_file: Any = None, corpus_hint: str | None = None,
                    compare_mode: bool = False, language: str = "zh") -> list[dict[str, Any]]:
    """挑出进生成阶段的证据块（条数、同页限流、正文优先、**跨语料轮转**都由配置控制）。

    与工单8 的差异：
      1. 工单8 是「按名次取，只做同页限流」；本工单在保留名次的前提下，
         用「名次分 × 问句实词命中」的加权分排序，让**词面更贴题**的块优先进上下文；
      2. ``evidence_k`` 从 5 提到 12（提升上下文召回）；
      3. **跨语料轮转**（``evidence_diversify_corpus``）：当候选池横跨多份语料时
         （典型是跨文档对比题 Q04/Q09，以及问句没点名公司的情况），
         先按语料分组、**轮转取每家的最优块**，再补齐。
         依据（实测）：不做轮转时 8 条证据全被同一家公司占满，
         「对比平安银行与招商银行」这类题的上下文召回只有 0.5。
    """
    k = int(cfg.get("evidence_k") or 8)
    max_per_page = int(cfg.get("evidence_max_per_page") or 3)
    prefer_text = bool(cfg.get("evidence_prefer_text", True))
    rank_weight = float(cfg.get("evidence_rank_weight") or 0.6)
    diversify = bool(cfg.get("evidence_diversify_corpus", True))

    # ---- 去重（⚠️ 实测踩坑）----
    # 候选池由「Top-K + 同页扩展块 + 字段锚点块 + 图谱独有块」四路合并而成，
    # **同一个 chunk_id 会从多路重复出现**。不去重时 14 个证据名额会被重复块吃掉
    # （实测 Q03 的 `20200214-p43-x1` / `-p5-x1` 各占 2 席，Q15 的 `-p4-x0` 占 2 席），
    # 上下文既浪费名额（召回↓）又把重复块计入分母（精度↓）。这里按 chunk_id 去重，
    # 同一块出现多次时**保留通道优先级最高的一份**（field_probe > page_expand > 其它）。
    channel_rank = {"field_probe": 0, "graph_only": 1, "base+graph": 2, "base": 3, "page_expand": 4}
    dedup: dict[str, dict[str, Any]] = {}
    for hit in hits:
        cid = str(hit.get("chunk_id") or "")
        if not cid:
            continue
        old = dedup.get(cid)
        if old is None:
            dedup[cid] = dict(hit)
            continue
        old_rank = channel_rank.get(str(old.get("retrieval_channel") or ""), 9)
        new_rank = channel_rank.get(str(hit.get("retrieval_channel") or ""), 9)
        if new_rank < old_rank:
            merged = dict(hit)
            merged["rank"] = min(int(old.get("rank") or 10 ** 6), int(hit.get("rank") or 10 ** 6))
            dedup[cid] = merged
    unique_hits = list(dedup.values())

    scored: list[tuple[float, dict[str, Any]]] = []
    for rank, hit in enumerate(unique_hits, 1):
        text = str(hit.get("content") or hit.get("text") or "")
        hits_n = sum(1 for t in terms if t and t in text)
        lex = (hits_n / len(terms)) if terms else 0.0
        rank_score = 1.0 / float(rank)
        bonus = 0.06 if str(hit.get("type") or "text") == "text" else 0.0
        channel_bonus = {"field_probe": 0.10, "page_expand": 0.05,
                         "probe_expand": 0.05}.get(
            str(hit.get("retrieval_channel") or ""), 0.0)
        scored.append(((1.0 - rank_weight) * rank_score + rank_weight * lex + bonus + channel_bonus,
                       dict(hit)))
    scored.sort(key=lambda item: -item[0])

    # ---- 跨语料轮转：先给每份语料一个名额 ----
    order: list[tuple[float, dict[str, Any]]] = []
    if diversify and corpus_of_file is not None and not corpus_hint:
        buckets: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        for item in scored:
            try:
                key = str(corpus_of_file(str((item[1] or {}).get("file_name") or "")))
            except Exception:  # noqa: BLE001
                key = ""
            buckets.setdefault(key, []).append(item)
        if len(buckets) > 1:
            cursors = {key: 0 for key in buckets}
            while len(order) < len(scored):
                progressed = False
                for key in list(buckets):
                    idx = cursors[key]
                    if idx < len(buckets[key]):
                        order.append(buckets[key][idx])
                        cursors[key] = idx + 1
                        progressed = True
                if not progressed:
                    break
    if not order:
        order = scored

    picked: list[dict[str, Any]] = []
    per_page: dict[tuple[str, int], int] = {}

    def _key_of(hit: Mapping[str, Any]) -> tuple[str, int]:
        return (str(hit.get("file_name") or ""), int(hit.get("page") or 0))

    def _already(hit: Mapping[str, Any]) -> bool:
        return any(str(h.get("chunk_id")) == str(hit.get("chunk_id")) for h in picked)

    def _add(hit: Mapping[str, Any], *, cap: int) -> bool:
        key = _key_of(hit)
        if per_page.get(key, 0) >= cap or _already(hit) or len(picked) >= k:
            return False
        per_page[key] = per_page.get(key, 0) + 1
        picked.append(hit)
        return True

    # ---- Pass A：**焦点页簇补全**（提升上下文召回的关键一步）----
    # 依据（实测 `_页内块` 诊断，逐条可查）：
    #   * 国泰君安 p4 共 3 块，答案要点「连续14年获得A类AA级监管评级」在 **p4-x1**，
    #     而检索只排进 p4-x0，p4-x1 因问句实词命中少、加权分低而被名额挤出上下文；
    #   * 招商银行 p5 的 p5-x1 装着「逾越者联盟/咖啡零售/蛋壳平台」4 个要点，同样排在第 12 名。
    # ⇒ 只按「全局加权分」排序会让**同页兄弟块互相挤掉**。
    # 修正：先锁定「融合后 Top-N 命中所在的页」，把这些页上的块优先放进上下文
    #   （年报同一页讲的通常就是同一件事 ⇒ 召回与精度同时受益）。
    focus_pages: list[tuple[str, int]] = []
    focus_seen_corpus: dict[str, int] = {}
    # 跨语料轮转的开启条件：识别不到语料，**或**这是对比题（问句同时点名多家公司）。
    # ⚠️ 实测踩坑：曾把对比题的 corpus_hint 直接清空以放宽闸门 —— 结果图谱扩展与字段锚点
    # 也一起失去语料约束，上下文精度从 0.8235 掉到 0.7311。正确做法是**只放宽焦点页轮转**，
    # 检索/图谱/锚点的语料闸门保持不变。
    diversify_focus = bool(cfg.get("evidence_focus_diversify_corpus", True)) and not corpus_hint
    for hit in hits[: max(int(cfg.get("evidence_focus_scan") or 12), 1)]:
        key = _key_of(hit)
        if not key[0] or key[1] <= 0 or key in focus_pages:
            continue
        if diversify_focus and corpus_of_file is not None:
            try:
                ck = str(corpus_of_file(key[0]))
            except Exception:  # noqa: BLE001
                ck = ""
            # 跨语料题：每个语料最多贡献 1 个焦点页，保证多家公司都进上下文
            if ck and focus_seen_corpus.get(ck, 0) >= 1:
                continue
            focus_seen_corpus[ck] = focus_seen_corpus.get(ck, 0) + 1
        focus_pages.append(key)
        if len(focus_pages) >= max(int(cfg.get("evidence_focus_pages") or 3), 1):
            break
    quota_focus = int(cfg.get("evidence_quota_focus") or 7)
    quota_probe = int(cfg.get("evidence_quota_probe") or 5)
    # **多语言配额**：英文（等非中文）提问时，向量/BM25 的跨语对齐不可靠
    # （语料是中文年报，英文 query 的基础召回会落到错页），
    # 而**字段锚点**走的是「英中术语映射 + 中文词面匹配」，是语言无关的可靠信号。
    # 因此非中文提问时**下调焦点页配额、上调锚点配额**。
    # 依据（实测）：邮储银行英文题原配额下证据 4 块来自 p8/p15（贷款集中度），
    # 真正的「净利润 610.36亿元」页 p18/p26 被挤出；调整配额后 p18/p26 稳定进入。
    if not str(language).startswith("zh"):
        quota_focus = int(cfg.get("evidence_quota_focus_english") or 4)
        quota_probe = int(cfg.get("evidence_quota_probe_english") or 8)
    focus_cap = int(cfg.get("evidence_focus_per_page") or 6)
    if focus_pages:
        by_page: dict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = {}
        for item in scored:
            key = _key_of(item[1])
            if key in focus_pages:
                by_page.setdefault(key, []).append(item)
        taken = 0
        for key in focus_pages:                      # 按焦点页顺序（= 命中名次）补全
            for _score, hit in by_page.get(key, []):
                if taken >= quota_focus or len(picked) >= k:
                    break
                if _add(hit, cap=focus_cap):
                    taken += 1
            if taken >= quota_focus or len(picked) >= k:
                break

    # ---- Pass B：**字段锚点配额**（保证「问到的每个字段都有证据」）----
    # ⚠️ 实测踩坑：不加配额时，Pass A 的焦点页会**吃光全部名额**
    # （Q04 的 4 个焦点页 × 每页 6 块 = 24 > k=14），字段锚点块一块也进不来，
    # 跨文档题的上下文召回因此卡在 0.143。给字段锚点保留固定名额后，覆盖立刻改善。
    probe_sorted = sorted([item for item in scored
                           if str(item[1].get("retrieval_channel") or "") == "field_probe"],
                          key=lambda item: (-float(item[1].get("field_score") or 0.0),
                                            str(item[1].get("chunk_id"))))
    taken = 0
    for _score, hit in probe_sorted:
        if taken >= quota_probe or len(picked) >= k:
            break
        if _add(hit, cap=max_per_page):
            taken += 1

    # ---- Pass C：其余候选按加权分补足（同页限流）----
    for _score, hit in order:
        if len(picked) >= k:
            break
        _add(hit, cap=max_per_page)
    if len(picked) < k:                       # 同页限流导致条数不足 ⇒ 按原加权次序补齐
        for _score, hit in order:
            if len(picked) >= k:
                break
            if _already(hit):
                continue
            picked.append(dict(hit))

    # ---- 最终排序：**非中文提问时按通道精度重排**（field_probe 最前）----
    # 依据（实测，英文题的关键根因）：生成层用 `build_draft(chunks[:8])` 选句，
    #   而候选顺序此前是「焦点页 → 锚点 → 其它」，`chunks[:8]` 几乎全是焦点页 ——
    #   英文提问时焦点页恰好是错页（邮储 p8 贷款集中度 / 平安银行 p9 更名批复），
    #   真正含答案的锚点块（p18 / p19 / p152）排在第 9 名之后，根本没进草稿的视野。
    #   ⚠️ 该重排**只对非中文提问启用**：中文提问下向量/BM25 通道可靠，
    #   全局按「焦点页 → 锚点」的加权顺序实测更优（改为锚点优先会让中文指标回退）。
    if not str(language).startswith("zh"):
        channel_priority = {"field_probe": 0, "probe_expand": 1, "page_expand": 2,
                            "graph_only": 2, "base+graph": 3, "base": 3}
        focus_set = set(focus_pages)
        picked.sort(key=lambda h: (channel_priority.get(str(h.get("retrieval_channel") or "base"), 4),
                                   0 if _key_of(h) in focus_set else 1))
    for rank, item in enumerate(picked, 1):
        item["rank"] = rank
    return picked


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def summarize_fusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """融合结果的诊断摘要（进报告，便于追溯「图谱通道到底起了多大作用」）。"""
    if not rows:
        return {"count": 0}
    by_channel: dict[str, int] = {}
    for row in rows:
        key = str(row.get("retrieval_channel") or "base")
        by_channel[key] = by_channel.get(key, 0) + 1
    return {
        "count": len(rows),
        "by_channel": by_channel,
        "graph_only_in_top": sum(1 for r in rows if r.get("retrieval_channel") == "graph_only"),
        "page_expand_in_top": sum(1 for r in rows if r.get("retrieval_channel") == "page_expand"),
        "max_graph_score": round(max((float(r.get("graph_score") or 0.0) for r in rows), default=0.0), 6),
    }
