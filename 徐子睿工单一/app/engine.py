# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：engine —— 问答引擎（Query 理解 + 混合检索 + 生成 + 拒答）
# 说明：核心链路。dense(bge-m3) 与 bm25 双路召回 -> 加权 RRF 融合 -> 去重 -> 取 top_k
#       -> 组装带页码上下文 -> LLM 生成；置信度过低则拒答。另提供纯 LLM 对照链路。

import re

from config import (RECALL_K, RRF_K, RRF_LAMBDA, TOP_K, REFUSE_SCORE,
                    MAX_CONTEXT_CHARS, FUSION_MODE, DENSE_W, SPARSE_W)
# 说明：查询词、术语改写均为中英双轨；口径消歧（金额 / 占比）同样中英双语生效。
import kb as kb_mod
import prompts
import llm

# ---------- Query 理解：规则化术语扩展（不依赖 LLM，可离线复现） ----------
REWRITE_RULES = [
    (r"军用领域的?收入|军用领域收入", "军用领域收入 国防客户销售收入 军品销售收入"),
    (r"占(主营)?业务?收入的?比重|占比", "占主营业务收入比重 占比"),
    (r"电子信息行业的?上游", "电子信息行业上游 电子元器件制造企业 机箱机柜 金属壳体制造企业"),
    (r"电子信息行业的?下游", "电子信息行业下游 终端用户 军队 政府机关 能源"),
    (r"重要供应商", "重要供应商 视频指挥领域 国防军队"),
    (r"国家科技进步一等奖|科技进步一等奖", "国家科技进步一等奖 情报 指挥 控制 通信网络一体化工程"),
    (r"注册资本", "注册资本 万元"),
    (r"法定代表人", "法定代表人"),
    (r"补充流动资金", "补充流动资金 募集资金投资项目"),
    (r"技术标准|规范", "视频指挥系统技术标准 视频技术规范"),
]

_INTENT_RULES = [
    ("数值", r"多少|几|金额|比例|比重|占比|万元|%|收入|资本"),
    ("列举", r"哪些|包括|涉及|有哪些"),
    ("事实", r"是谁|哪个|什么|哪一年|何时"),
]


def detect_lang(q):
    return "en" if re.search(r"[A-Za-z]{3,}", q) and not re.search(r"[\u4e00-\u9fff]", q) else "zh"


# ---------- 跨语（英→中）术语映射：英文提问时把关键术语映回中文，提升跨语检索命中 ----------
EN_GLOSSARY = [
    (r"registered capital", "注册资本"),
    (r"legal representative", "法定代表人"),
    (r"technical standard|technology standard", "视频指挥系统技术标准 视频技术规范"),
    (r"science and technology progress award|national science.{0,4}progress", "国家科技进步一等奖 情报 指挥 控制 通信网络一体化工程"),
    (r"military (revenue|income|sales|domain)|revenue from (the )?military|defense (revenue|income)", "军用领域收入"),
    (r"replenish|working capital|raised funds|proceeds", "补充流动资金 募集资金投资项目"),
    (r"upstream", "上游 电子元器件制造企业 机箱机柜 金属壳体"),
    (r"downstream", "下游 终端用户 军队 政府机关 能源"),
    (r"important supplier|key supplier", "重要供应商 视频指挥领域 国防军队"),
    (r"main business (revenue|income)", "主营业务收入"),
]
_COMPANY = "武汉兴图新科电子股份有限公司"
# 问“发行人工商登记类信息”时补公司全名（其余问题补全名会误召公司名称表 / 子公司表）
_CO_NEED = re.compile(
    r"registered capital|legal representative|registered address|date of establishment|"
    r"incorporated|business scope|company name|unified social credit|issuer's? name",
    re.I)
# 金额口径词（英）—— 与中文 _AMT_Q 对齐，保证英文问“金额”时同样触发口径消歧
_AMT_Q_EN = re.compile(
    r"how much|how many|amount|revenue|income|sales|capital|yuan|ten thousand|fund(?:s)?|proceeds",
    re.I)
# 占比口径词（英）
_RATIO_Q_EN = re.compile(r"proportion|percentage|ratio|percent|share of|accounted for", re.I)


# 跨语扩展的三种档位（供“优化前后”消融对比用：full = 02 优化版，basic = 01 交付版，off = 无）
XLING_MODES = ("off", "basic", "full")


def cross_lingual_expand(q, lang, mode="full"):
    """英文提问：抽取中文术语（不做逐词翻译，而是术语级映射），再补公司全名。
    mode="full"（02 优化版）：术语映射 + 口径词补齐 + 仅工商登记类问题补公司全名；
    mode="basic"（01 交付版）：术语映射 + 无条件补公司全名；
    mode="off"：不跨语扩展（英文原样检索）。
    金额 / 占比口径词额外拼入“万元 / 占比”等中文口径词，让中英双语的消歧规则一致生效。"""
    if lang != "en" or mode == "off":
        return ""
    extra = []
    low = q.lower()
    for pat, zh in EN_GLOSSARY:
        if re.search(pat, low):
            extra.append(zh)
    if mode == "full":
        amt_hit = bool(_AMT_Q_EN.search(q))
        ratio_hit = bool(_RATIO_Q_EN.search(q))
        if ratio_hit and not amt_hit:
            extra.append("占主营业务收入比重 占比")
        elif amt_hit and not ratio_hit:
            extra.append("收入金额 万元")
        # 仅当问题明确登记类信息时才补公司全名 —— 用于区分母公司 vs 子公司
        # （实测：无条件补全名会把“公司名称表格页”抬到首位，反而拉低命中；实测规则命中率 10/10）
        if _CO_NEED.search(q):
            extra.append(_COMPANY)
    else:  # basic：01 版行为
        extra.append(_COMPANY)
    return " " + " ".join(extra)


def _hit_caliber(query, caliber="full"):
    """判断问题口径：返回 (need_amt, need_ratio)。
    caliber="full"：中英双语同一套规则（02）；caliber="penalty"：仅中文规则且只降权（01）；
    caliber="off"：不做口径处理。"""
    if caliber == "off":
        return False, False
    if caliber == "penalty":
        amt = bool(_AMT_Q.search(query))
        ratio = bool(_RATIO_Q.search(query))
    else:
        amt = bool(_AMT_Q.search(query)) or bool(_AMT_Q_EN.search(query))
        ratio = bool(_RATIO_Q.search(query)) or bool(_RATIO_Q_EN.search(query))
    return (amt and not ratio), (ratio and not amt)


def query_understanding(query, xling="full"):
    lang = detect_lang(query)
    intent = "事实"
    for name, pat in _INTENT_RULES:
        if re.search(pat, query):
            intent = name
            break
    rewrites = []
    for pat, rep in REWRITE_RULES:
        if re.search(pat, query):
            rewrites.append(rep)
    expanded = query + cross_lingual_expand(query, lang, xling)
    if rewrites:
        expanded += " " + " ".join(rewrites)
    return {"lang": lang, "intent": intent, "expanded": expanded, "rewrites": rewrites}


def _rrf(dense, sparse, lam=RRF_LAMBDA, k=RRF_K):
    """加权 RRF 融合。dense/sparse: list[(idx, score)]"""
    fused = {}
    for rank, (idx, _s) in enumerate(dense):
        fused[idx] = fused.get(idx, 0.0) + lam / (k + rank + 1)
    for rank, (idx, _s) in enumerate(sparse):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def _weighted_sum(dense, sparse, wd=DENSE_W, ws=SPARSE_W):
    """归一化加权求和融合：保留分数的强弱信息，避免纯 RRF 丢失“分差”。"""
    fused = {}
    dmax = max([s for _i, s in dense], default=0.0) or 1.0
    smax = max([s for _i, s in sparse], default=0.0) or 1.0
    for i, s in dense:
        fused[i] = fused.get(i, 0.0) + wd * (s / dmax)
    for i, s in sparse:
        fused[i] = fused.get(i, 0.0) + ws * (s / smax)
    return sorted(fused.items(), key=lambda x: -x[1])


def fuse(dense, sparse, mode=None, **kw):
    mode = mode or FUSION_MODE
    if mode == "sum":
        return _weighted_sum(dense, sparse, **kw)
    return _rrf(dense, sparse, **kw)


# 口径消歧：问金额时压制“只含百分比”的块
_AMT_Q = re.compile(r"多少|金额|万元|收入|资本|资金|投资|募集|注册")
_RATIO_Q = re.compile(r"占比|比重|比例|百分之|%")
_AMT_NUM = re.compile(r"\d[\d,]*\.?\d*\s*万元")
AMT_PENALTY = 0.45
AMT_BOOST = 1.25      # 问金额时，命中“万元”数值块的加权


def _is_ratio_only(text):
    return text.count("%") >= 2 and not _AMT_NUM.search(text)


def retrieve(query, qu=None, top_k=TOP_K, mode=None, recall_k=None, caliber="full", **kw):
    """caliber：口径消歧档位（full=02 优化版 / penalty=01 交付版 / off=关），
    仅用于“优化前后”对比实验；线上默认 full。"""
    km = kb_mod.get_kb()
    qu = qu or query_understanding(query)
    rk = recall_k or 30
    dense = km.search_dense(qu["expanded"], k=rk)
    sparse = km.search_bm25(qu["expanded"], k=rk)
    fused = fuse(dense, sparse, mode=mode, **kw)
    # —— 口径消歧：问“金额”→ 压制只给百分比的块、抬升含“万元”数值的块；
    #    问“占比”→ 压制没有任何百分比的块。中英双语问题都生效（_hit_caliber）。 ——
    need_amt, need_ratio = _hit_caliber(query, caliber)
    if need_amt or need_ratio:
        adj = []
        for idx, sc in fused:
            t = km.get(idx)["text"]
            if need_amt:
                if _is_ratio_only(t):
                    sc *= AMT_PENALTY
                elif caliber != "penalty" and _AMT_NUM.search(t):
                    sc *= AMT_BOOST   # 仅 02 优化版新增：抬升含“万元”数值的块
            elif need_ratio and not _is_ratio_only(t) and "%" not in t:
                sc *= AMT_PENALTY
            adj.append((idx, sc))
        fused = sorted(adj, key=lambda x: -x[1])
    dense_sim = {i: s for i, s in dense}
    hits, seen_pages = [], set()
    for idx, fscore in fused:
        c = km.get(idx)
        # 同句多页去重：文本前 40 字相同且页码相近 -> 只留一条
        key = (c["text"][:40], c.get("section"))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        h = dict(c)
        h["rrf"] = fscore
        h["dense_sim"] = dense_sim.get(idx, 0.0)
        hits.append(h)
        if len(hits) >= top_k:
            break
    confidence = max([h["dense_sim"] for h in hits], default=0.0)
    return hits, confidence


def expand_hits(hits, span=1, max_chunks=8):
    """上下文邻块扩展：把命中块 id±span 且同章节的邻块一并送入生成，
    避免“金额句/占比句被切块切开”导致只拿到半句。"""
    km = kb_mod.get_kb()
    ids, out = set(), []
    for h in hits:
        for d in range(-span, span + 1):
            j = h["id"] + d
            if j < 0 or j >= len(km.chunks):
                continue
            nb = km.chunks[j]
            if nb.get("section") != h.get("section"):
                continue
            if j in ids:
                continue
            ids.add(j)
            nb2 = dict(nb)
            nb2.setdefault("dense_sim", h.get("dense_sim", 0.0))
            out.append(nb2)
            if len(out) >= max_chunks:
                return out
    return out


# 生成结果清洗：小模型偶尔会“继续写题”/回显提示词模板，在模板标记处截断
_CUT_MARKS = ("\n【用户问题】", "\n【Question】", "\n【检索到的文档片段】", "\n【Excerpts", "\n【作答要求】", "\n【Answer requirements】")


def clean_answer(txt):
    for m in _CUT_MARKS:
        i = txt.find(m)
        if i != -1:
            txt = txt[:i]
    return txt.strip()


def answer(query, top_k=TOP_K):
    qu = query_understanding(query)
    hits, conf = retrieve(query, qu, top_k)
    if not hits or conf < REFUSE_SCORE:
        return {"answer": prompts.REFUSE_TEXT, "hits": [], "confidence": conf,
                "refused": True, "query_understanding": qu}
    hits = expand_hits(hits, span=0, max_chunks=8)
    ctx = prompts.build_context(hits)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS]
    msgs = prompts.rag_messages(query, hits, lang=qu["lang"])
    txt = clean_answer(llm.chat(msgs, temperature=0.1))
    refused = ("未找到依据" in txt) or ("Not found" in txt)
    return {"answer": txt, "hits": [] if refused else hits, "confidence": conf,
            "refused": refused, "query_understanding": qu}


def answer_stream(query, top_k=TOP_K):
    """流式：先 yield 元信息，再 yield 文本片段。"""
    qu = query_understanding(query)
    hits, conf = retrieve(query, qu, top_k)
    if not hits or conf < REFUSE_SCORE:
        yield {"type": "meta", "hits": [], "confidence": conf, "refused": True}
        yield {"type": "delta", "text": prompts.REFUSE_TEXT}
        return
    yield {"type": "meta", "hits": hits, "confidence": conf, "refused": False,
           "query_understanding": qu}
    hits = expand_hits(hits, span=0, max_chunks=8)
    for piece in llm.chat(prompts.rag_messages(query, hits, lang=qu["lang"]),
                          temperature=0.1, stream=True):
        yield {"type": "delta", "text": piece}


def answer_plain_llm(query):
    """对照链路：不做检索，直接让同一 LLM 作答。"""
    txt = llm.chat(prompts.plain_messages(query), temperature=0.1)
    return {"answer": txt, "hits": [], "confidence": None, "refused": False, "mode": "llm_only"}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r = answer("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
    print("confidence:", r["confidence"], "| refused:", r["refused"])
    for h in r["hits"]:
        print("  hit p%s sim=%.3f %s" % (h["page"], h["dense_sim"], h["text"][:60]))
    print("ANSWER:", r["answer"][:500])
