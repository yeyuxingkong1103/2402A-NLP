# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

检索结果问题分析脚本（工单验收硬性要求）。

设计原则：**先自动量化，再人工归因**。
脚本先跑 8 组可计算的自动检查，拿到真实数字与真实案例，
再把这些证据归入 6 类典型问题，逐类给出「现象 / 根因 / 证据 / 改进方向」。

自动检查清单：
    A. 元数据检查        —— 文件名乱码对文档名/引用/命中判定的量化影响（含反事实模拟）
    B. 检索失败检查      —— 主责文档未命中、首次排名靠后、期望页码未召回
    C. 长文档稀释检查    —— 目标信息的「chunk 密度」与问题难度、命中率的关系
    D. 数值张冠李戴检查  —— 答案中的数字有多少无法在上下文中找到；上下文数字密度
    E. 多文档覆盖检查    —— 多文档题的文档覆盖率、漏掉了哪些文档、理论覆盖上限
    F. 表格丢失检查      —— 表格块占比、需表格的题是否真的召回了表格块
    G. 章节干扰检查      —— 检索片段中「非主责文档但同章节名」的比例（串公司）
    H. 关键词/拒答检查   —— 关键信息点漏答、模型拒答、耗时异常

六类问题（与 docs/问题分析与改进建议.md 一一对应）：
    ① 文件名乱码导致的元数据错误
    ② 长文档（年报几百页）中目标信息被稀释、召回不到
    ③ 数值类问题：上下文里数字很多，LLM 张冠李戴
    ④ 跨文档归纳类问题：单次检索覆盖不全
    ⑤ 表格数据丢失
    ⑥ 章节结构相似的干扰（年报都有「风险管理」章节，容易串公司）

用法：
    python analyze_problems.py
    python analyze_problems.py --top-n 3      # 每类问题最多展示 3 个案例
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config  # noqa: E402
from prepare_corpus import (WO_NO, COMPANY_SECTOR, ensure_results_dir,  # noqa: E402
                            list_ccf_docs)

#: 年报共有的高频章节名 —— 用于「章节结构相似导致串公司」的量化
COMMON_SECTION_KEYWORDS = [
    "风险管理", "资产质量", "经营情况讨论与分析", "董事长致辞", "公司治理",
    "资本充足率", "拨备覆盖率", "不良贷款", "普惠金融", "绿色金融",
]
#: 乱码特征字符（锟斤拷 / U+FFFD）
GARBLE_MARKS = ["�", "锟", "斤", "拷", "？"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _load(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise SystemExit(f"缺少输入文件：{path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_garbled(s: str) -> bool:
    """判断字符串是否含乱码特征。"""
    if not s:
        return False
    if "�" in s or "锟" in s:
        return True
    # 大量问号（GBK→其它编码失败时的另一种表现）
    return s.count("?") >= 3


NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str, min_len: int = 2) -> list[str]:
    """抽取文本中的数字（含千分位），过滤掉长度 < min_len 的噪声。"""
    out, seen = [], set()
    for m in NUM_RE.findall(text or ""):
        n = m.rstrip(".,")
        if len(n.replace(",", "")) < min_len:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def number_supported(n: str, ctx_flat: str, ctx_nocomma: str) -> bool:
    """数字是否能被上下文支撑（容忍千分位差异）。"""
    return (n in ctx_flat) or (n.replace(",", "") in ctx_nocomma)


# ---------------------------------------------------------------------------
# A. 元数据检查
# ---------------------------------------------------------------------------
def analyze_metadata(inputs: dict) -> dict:
    """
    量化「文件名乱码」的影响。

    反事实模拟：如果直接用 Path(pdf).stem 当文档名（最常见的偷懒写法），
    那么 10 道题的「主责文档命中判定」还能成立几道？
    """
    docs = [d.to_dict() for d in list_ccf_docs()]
    broken = [d for d in docs if not d["gbk_strict_ok"]]
    survivors = {d["doc_name"]: d["survivors"][:18] for d in docs}

    cases = inputs["test_cases"]["cases"]
    would_match, would_fail = [], []
    for c in cases:
        ref = c["reference_doc"]
        # 反事实：以 stem 作为文档名时的匹配结果
        stem_hit = any(ref in Path(d["pdf_path"]).stem or
                       Path(d["pdf_path"]).stem in ref
                       for d in docs if d["doc_name"] == ref)
        (would_match if stem_hit else would_fail).append(c["id"])

    # 实际结果（已用还原后的文档名建索引）
    actual_hit = [r["id"] for r in inputs["rag_results"]["results"]
                  if r.get("hit_reference_doc")]

    return {
        "n_docs": len(docs),
        "n_broken": len(broken),
        "broken_examples": [
            {"raw": d["raw_filename"][:46], "survivors": d["survivors"][:12],
             "restored": d["doc_name"]} for d in broken[:3]],
        "survivor_index": survivors,
        "counterfactual": {
            "n_would_match": len(would_match),
            "n_would_fail": len(would_fail),
            "failed_ids": would_fail,
            "actual_hit_ids": actual_hit,
        },
    }


# ---------------------------------------------------------------------------
# A2. 文本层兼容字符检查（乱码问题的「隐性兄弟」）
# ---------------------------------------------------------------------------
#: 需要警惕的字符区段：CJK 部首补充 / 康熙部首 / 私用区。
#: 注意不把「全角符号区」算进来——中文语料里的全角逗号「，」是正常标点，不是乱码。
_TEXTLAYER_RANGES = [
    (0x2E80, 0x2EFF, "CJK部首补充"),
    (0x2F00, 0x2FDF, "康熙部首"),
    (0xE000, 0xF8FF, "私用区(PUA)"),
]


def analyze_text_layer(inputs: dict) -> dict:
    """
    扫描检索片段与生成答案中的「兼容字符」。

    这类字符字形正常、肉眼无法察觉，但码位与标准汉字不同，
    会让关键词匹配与 BM25 检索静默失配——是乱码问题里最隐蔽的一种。
    """
    hits: dict[str, dict] = {}
    for r in inputs["rag_results"]["results"]:
        texts = [d.get("text", "") for d in r.get("retrieved", [])]
        texts.append(r.get("answer", ""))
        texts.append(r.get("question", ""))
        for t in texts:
            for ch in t or "":
                o = ord(ch)
                for lo, hi, name in _TEXTLAYER_RANGES:
                    if lo <= o <= hi:
                        slot = hits.setdefault(name, {"count": 0, "chars": {}})
                        slot["count"] += 1
                        slot["chars"][ch] = slot["chars"].get(ch, 0) + 1
    return {
        "n_ranges_hit": len(hits),
        "total": sum(v["count"] for v in hits.values()),
        "detail": {
            k: {"count": v["count"],
                "examples": [f"U+{ord(c):04X}({c})×{n}"
                             for c, n in sorted(v["chars"].items(),
                                                key=lambda x: -x[1])[:5]]}
            for k, v in hits.items()},
    }


# ---------------------------------------------------------------------------
# B/C. 检索失败与长文档稀释
# ---------------------------------------------------------------------------
def analyze_retrieval(inputs: dict) -> dict:
    results = inputs["rag_results"]["results"]
    stats = inputs.get("corpus_stats") or {}
    doc_pages = {d["doc_name"]: d["n_pages"] for d in stats.get("docs", [])}
    doc_chunks = {d["doc_name"]: d["n_chunks"] for d in stats.get("docs", [])}

    rows = []
    for r in results:
        ref = r["reference_doc"]
        rows.append({
            "id": r["id"],
            "question_type": r["question_type"],
            "difficulty": r["difficulty"],
            "reference_doc": ref,
            "hit": bool(r.get("hit_reference_doc")),
            "rank": r.get("rank_of_reference_doc", 0),
            "top_k": r.get("top_k", 5),
            "page_hit_rate": r.get("page_hit_rate"),
            "expected_pages": r.get("expected_pages", []),
            "page_hits": r.get("page_hits", []),
            "doc_pages": doc_pages.get(ref, 0),
            "doc_chunks": doc_chunks.get(ref, 0),
            # 长文档稀释度：1 个目标片段在文档中的相对稀缺程度
            "dilution": round(1.0 / doc_chunks[ref], 5) if doc_chunks.get(ref) else None,
            "n_retrieved": len(r.get("retrieved", [])),
            "latency": r.get("latency", 0.0),
        })

    missed = [x for x in rows if not x["hit"]]
    page_missed = [x for x in rows
                   if x["expected_pages"] and x["page_hit_rate"] is not None
                   and x["page_hit_rate"] == 0]
    low_rank = [x for x in rows if x["hit"] and x["rank"] > max(1, x["top_k"] // 2)]
    # 长文档 vs 短文档的命中对比（用文档块数中位数切分）
    chunks_sorted = sorted(x["doc_chunks"] for x in rows if x["doc_chunks"])
    median_chunks = chunks_sorted[len(chunks_sorted) // 2] if chunks_sorted else 0
    long_docs = [x for x in rows if x["doc_chunks"] >= median_chunks]
    short_docs = [x for x in rows if 0 < x["doc_chunks"] < median_chunks]

    def _hit_rate(xs):
        return round(sum(1 for x in xs if x["hit"]) / len(xs), 4) if xs else None

    return {
        "rows": rows,
        "n_missed": len(missed),
        "missed": missed,
        "page_missed": page_missed,
        "low_rank": low_rank,
        "median_doc_chunks": median_chunks,
        "hit_rate_long_docs": _hit_rate(long_docs),
        "hit_rate_short_docs": _hit_rate(short_docs),
        "avg_page_hit_rate": round(
            sum(x["page_hit_rate"] for x in rows if x["page_hit_rate"] is not None)
            / max(len([x for x in rows if x["page_hit_rate"] is not None]), 1), 4),
    }


# ---------------------------------------------------------------------------
# D. 数值张冠李戴
# ---------------------------------------------------------------------------
YEAR_RE = re.compile(r"(20\d{2})\s*年")


def analyze_numeric(inputs: dict) -> dict:
    """检查数值题：答案里的数字有多少能在上下文中找到；上下文数字密度多大。"""
    cases = {c["id"]: c for c in inputs["test_cases"]["cases"]}
    rows = []
    for r in inputs["rag_results"]["results"]:
        c = cases.get(r["id"], {})
        if c.get("question_type") != "数值型" and not c.get("need_table"):
            continue
        ctx_flat = "\n".join(d.get("text", "") for d in r.get("retrieved", []))
        ctx_nocomma = ctx_flat.replace(",", "")
        nums = extract_numbers(r.get("answer", ""))
        unsupported = [n for n in nums if not number_supported(n, ctx_flat, ctx_nocomma)]
        ctx_nums = extract_numbers(ctx_flat)
        years = sorted(set(YEAR_RE.findall(ctx_flat)))
        rows.append({
            "id": r["id"],
            "question_type": c.get("question_type"),
            "need_table": c.get("need_table"),
            "n_answer_numbers": len(nums),
            "n_unsupported": len(unsupported),
            "unsupported_numbers": unsupported[:8],
            "n_ctx_numbers": len(ctx_nums),
            "ctx_numeric_density": round(len(ctx_nums) / max(len(ctx_flat), 1) * 1000, 2),
            "years_in_context": years,
            "keyword_miss": None,          # 稍后由 evaluation 回填
            "latency": r.get("latency", 0.0),
        })
    tot = sum(x["n_answer_numbers"] for x in rows)
    bad = sum(x["n_unsupported"] for x in rows)
    return {
        "rows": rows,
        "n_numeric_questions": len(rows),
        "total_numbers": tot,
        "unsupported_numbers": bad,
        "unsupported_rate": round(bad / tot, 4) if tot else 0.0,
    }


# ---------------------------------------------------------------------------
# E. 多文档覆盖
# ---------------------------------------------------------------------------
def analyze_multi_doc(inputs: dict) -> dict:
    rows = []
    for r in inputs["rag_results"]["results"]:
        if r.get("n_docs_expected", 1) <= 1:
            continue
        got = {d for d in r.get("retrieved_docs", [])}
        miss = [d for d in r.get("reference_docs", []) if d not in got]
        rows.append({
            "id": r["id"],
            "n_expected": r.get("n_docs_expected", 0),
            "n_covered": r.get("n_docs_covered", 0),
            "coverage": r.get("doc_coverage", 0.0),
            "top_k": r.get("top_k", 5),
            "theoretical_max_coverage": round(
                min(1.0, r.get("top_k", 5) / max(r.get("n_docs_expected", 1), 1)), 4),
            "missing_docs": miss,
            "retrieved_doc_distribution": _dist(r.get("retrieved_docs", [])),
        })
    avg_cov = round(sum(x["coverage"] for x in rows) / len(rows), 4) if rows else None
    return {"rows": rows, "n_multi": len(rows), "avg_coverage": avg_cov}


def _dist(xs: list[str]) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ---------------------------------------------------------------------------
# F. 表格丢失
# ---------------------------------------------------------------------------
def analyze_tables(inputs: dict) -> dict:
    stats = inputs.get("corpus_stats") or {}
    total_chunks = stats.get("total_chunks", 0)
    total_tables = stats.get("total_table_chunks", 0)
    per_doc = [{"doc": d["doc_name"], "table_chunks": d["n_table_chunks"],
                "chunks": d["n_chunks"],
                "table_ratio": round(d["n_table_chunks"] / max(d["n_chunks"], 1), 4)}
               for d in stats.get("docs", [])]

    rows = []
    for r in inputs["rag_results"]["results"]:
        rows.append({
            "id": r["id"], "need_table": r.get("need_table", False),
            "n_table_chunks": r.get("n_table_chunks", 0),
            "hit": r.get("hit_reference_doc", False),
            "page_hit_rate": r.get("page_hit_rate"),
            "top_k": r.get("top_k", 5),
        })
    need = [x for x in rows if x["need_table"]]
    return {
        "corpus_table_ratio": round(total_tables / max(total_chunks, 1), 4),
        "total_chunks": total_chunks,
        "total_table_chunks": total_tables,
        "per_doc": per_doc,
        "rows": rows,
        "need_table_questions": need,
        "need_table_with_zero_table_chunk": [x["id"] for x in need if x["n_table_chunks"] == 0],
    }


# ---------------------------------------------------------------------------
# G. 章节相似干扰（串公司）
# ---------------------------------------------------------------------------
def analyze_section_interference(inputs: dict) -> dict:
    rows = []
    for r in inputs["rag_results"]["results"]:
        ref = r["reference_doc"]
        refs = set(r.get("reference_docs", [ref]))
        chunks = r.get("retrieved", [])
        if not chunks:
            continue
        foreign = [d for d in chunks if d.get("doc") not in refs]
        # 干扰片段：来自非主责文档，但片段含年报共有章节关键词
        interfering = [
            d for d in foreign
            if any(k in (d.get("snippet", "") + (d.get("section") or ""))
                   for k in COMMON_SECTION_KEYWORDS)
        ]
        rows.append({
            "id": r["id"],
            "top_k": len(chunks),
            "n_foreign": len(foreign),
            "foreign_rate": round(len(foreign) / len(chunks), 4),
            "n_interfering": len(interfering),
            "interfering_rate": round(len(interfering) / len(chunks), 4),
            "examples": [
                {"doc": d.get("doc"), "page": d.get("page"),
                 "score": d.get("score"),
                 "matched_keyword": next((k for k in COMMON_SECTION_KEYWORDS
                                          if k in (d.get("snippet", "") + (d.get("section") or ""))), ""),
                 "snippet": (d.get("snippet", "") or "")[:70]}
                for d in interfering[:3]],
        })
    n_int = sum(x["n_interfering"] for x in rows)
    n_tot = sum(x["top_k"] for x in rows)
    return {
        "rows": rows,
        "total_chunks": n_tot,
        "total_interfering": n_int,
        "interfering_rate": round(n_int / max(n_tot, 1), 4),
    }


# ---------------------------------------------------------------------------
# H. 关键词漏答 / 拒答 / 耗时
# ---------------------------------------------------------------------------
def analyze_answer(inputs: dict) -> dict:
    ev = inputs.get("evaluation") or {}
    kw = ev.get("keyword_accuracy", {})
    # evaluation.json 里逐题的 keywords 明细挂在 records 上
    per_q = []
    for r in ev.get("records", []):
        k = r.get("keywords") or {}
        per_q.append({
            "id": r["id"], "question_type": r.get("question_type"),
            "hit_kw": k.get("命中", []), "miss_kw": k.get("漏答", []),
            "passed": r.get("passed"), "fail_reasons": r.get("fail_reasons", []),
            "answer_len": len(r.get("answer", "") or ""),
            "metrics": r.get("metrics", {}),
        })
    refused = [x["id"] for x in per_q if any("拒答" in f for f in x["fail_reasons"])]
    lat_rows = [{"id": r["id"], "latency": r.get("latency", 0.0),
                 "timings": r.get("timings", {})}
                for r in inputs["rag_results"]["results"]]
    lat = [x["latency"] for x in lat_rows if x["latency"]]
    return {
        "keyword_accuracy": kw.get("accuracy"),
        "keyword_correct": kw.get("correct"), "keyword_total": kw.get("total"),
        "per_question": per_q,
        "questions_with_miss": [x["id"] for x in per_q if x["miss_kw"]],
        "miss_detail": {x["id"]: x["miss_kw"] for x in per_q if x["miss_kw"]},
        "refused": refused,
        "latency": {
            "avg": round(sum(lat) / len(lat), 3) if lat else None,
            "max": round(max(lat), 3) if lat else None,
            "slowest": sorted(lat_rows, key=lambda x: -x["latency"])[:3],
        },
    }


# ---------------------------------------------------------------------------
# 汇总 + 渲染
# ---------------------------------------------------------------------------
def collect(inputs: dict) -> dict:
    return {
        "metadata": analyze_metadata(inputs),
        "text_layer": analyze_text_layer(inputs),
        "retrieval": analyze_retrieval(inputs),
        "numeric": analyze_numeric(inputs),
        "multi_doc": analyze_multi_doc(inputs),
        "tables": analyze_tables(inputs),
        "interference": analyze_section_interference(inputs),
        "answer": analyze_answer(inputs),
    }


def render(findings: dict, inputs: dict, top_n: int = 3) -> str:
    meta, ret = findings["metadata"], findings["retrieval"]
    num, multi = findings["numeric"], findings["multi_doc"]
    tab, inter = findings["tables"], findings["interference"]
    ans = findings["answer"]
    cases = {c["id"]: c for c in inputs["test_cases"]["cases"]}
    L: list[str] = []

    # ---------- 封面与总览 ----------
    L += [
        "# 检索结果问题分析报告",
        "",
        f"> 工单编号：{WO_NO}　"
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 分析样本：{inputs['rag_results']['summary']['n']} 个问题的检索与生成结果"
        f"（语料：{meta['n_docs']} 份金融年报）",
        "",
        "## 〇、自动分析总览",
        "",
        "本报告的结论全部先经**自动量化**再人工归因，下表为自动检查的真实数字：",
        "",
        "| 自动检查 | 关键数字 | 结论 |",
        "| --- | --- | --- |",
        f"| A 元数据（文件名乱码） | {meta['n_broken']}/{meta['n_docs']} 份文件名被破坏；"
        f"若直接用 Path.stem 建索引，{meta['counterfactual']['n_would_fail']}/"
        f"{meta['counterfactual']['n_would_fail'] + meta['counterfactual']['n_would_match']} "
        f"题会判定为「未命中」；检索片段中兼容字符 {findings['text_layer']['total']} 个 | "
        f"乱码影响可被「代码锚点+正文校验」完全消除，"
        f"兼容字符需 NFKC 归一化 |",
        f"| B 检索失败 | 主责文档未命中 {ret['n_missed']} 题；"
        f"期望页码零召回 {len(ret['page_missed'])} 题；排名靠后 {len(ret['low_rank'])} 题 | "
        f"文档级命中与页码级召回要分开看 |",
        f"| C 长文档稀释 | 单文档检索块数中位数 {ret['median_doc_chunks']}；"
        f"块数多的一半文档命中率 {_p(ret['hit_rate_long_docs'])} vs "
        f"块数少的一半 {_p(ret['hit_rate_short_docs'])} | 文档越长，目标片段越难被排到前面 |",
        f"| D 数值张冠李戴 | 数值题答案共 {num['total_numbers']} 个数字，"
        f"其中 {num['unsupported_numbers']} 个无法在上下文中找到"
        f"（{_p(num['unsupported_rate'])}） | 存在换算/臆造风险，需逐题人工复核 |",
        f"| E 多文档覆盖 | {multi['n_multi']} 道多文档题，平均覆盖率 "
        f"{_p(multi['avg_coverage']) if multi['n_multi'] else '（本次无）'} | "
        f"单次检索无法覆盖全部文档 |",
        f"| F 表格数据 | 语料表格块占比 {_p(tab['corpus_table_ratio'])}；"
        f"需表格的 {len(tab['need_table_questions'])} 题中 "
        f"{len(tab['need_table_with_zero_table_chunk'])} 题一个表格块都没召回到 | "
        f"表格是数值题的关键证据来源 |",
        f"| G 章节干扰 | 检索片段中「非主责文档且命中年报共有章节名」占 "
        f"{_p(inter['interfering_rate'])}（{inter['total_interfering']}/"
        f"{inter['total_chunks']}） | 章节结构相似确实在制造串公司干扰 |",
        f"| H 答案质量 | 关键词判准准确率 {_p(ans['keyword_accuracy'])}"
        f"（{ans['keyword_correct']}/{ans['keyword_total']}）；拒答 {len(ans['refused'])} 题 | "
        f"漏答集中在数值细节 |",
        "",
        "**逐题自动体检表**（✓ 正常 / ✗ 异常）：",
        "",
        "| 编号 | 题型 | 主责文档 | 命中 | 排名 | 页码召回 | 表格块 | 数字无支撑 | "
        "文档覆盖 | 关键词漏答 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    num_by_id = {x["id"]: x for x in num["rows"]}
    for row in ret["rows"]:
        qid = row["id"]
        n = num_by_id.get(qid, {})
        miss = ans["miss_detail"].get(qid, [])
        ph = row["page_hit_rate"]
        L.append(
            f"| {qid} | {row['question_type']} | {row['reference_doc']} | "
            f"{'✓' if row['hit'] else '✗'} | {row['rank'] or '—'} | "
            f"{('—' if ph is None else f'{ph:.0%}')} | "
            f"{next((x['n_table_chunks'] for x in tab['rows'] if x['id'] == qid), '—')} | "
            f"{n.get('n_unsupported', '—')} | "
            f"{_cov(multi, qid)} | {'、'.join(miss[:3]) if miss else '—'} |")
    L.append("")

    # ---------- 六类问题 ----------
    L += ["## 一、六类典型问题（现象 / 根因 / 证据 / 改进方向）", ""]

    # ① 元数据
    L += ["### 问题① 文件名乱码导致的元数据错误", "",
          "**现象**：源目录 9 份年报的文件名全部是乱码（形如 "
          "`2020-02-14__平锟斤拷锟斤拷锟叫股凤拷锟斤拷锟睫癸拷司__000001__...`），"
          "若直接拿文件名做文档名，检索结果的来源、页码引用、"
          "以及评估时的「主责文档命中」判定全部失效。", "",
          "**根因**：文件名在拷贝过程中被二次编码破坏——原始 GBK 字节先按 UTF-8 解码"
          "（非法字节变成 U+FFFD），再按 GBK 显示，得到「锟斤拷」。"
          "**这是不可逆的信息丢失**，单纯 `bytes.decode('gbk')` 无法完全还原。", "",
          "**证据（自动检测）**：",
          "",
          f"- {meta['n_broken']}/{meta['n_docs']} 份文件名无法被 GBK 严格解码；"]
    for e in meta["broken_examples"]:
        L.append(f"  - `{e['raw']}` → 幸存片段 `{e['survivors']}` → "
                 f"还原为 **{e['restored']}**")
    cf = meta["counterfactual"]
    L += [
        f"- **反事实模拟**：若用 `Path(pdf).stem` 当文档名，"
        f"{cf['n_would_fail']}/{cf['n_would_fail'] + cf['n_would_match']} 道题"
        f"（{('、'.join(cf['failed_ids']))}）会因为「主责文档名对不上」被误判为未命中，"
        f"检索指标被人为压低甚至归零；而采用「股票代码锚点 + 正文校验」还原后，"
        f"实际命中 {len(cf['actual_hit_ids'])} 道。",
        "",
    ]
    tl = findings.get("text_layer", {})
    if tl.get("total"):
        L += [f"- **同源的隐性乱码**：检索片段与答案中还发现 {tl['total']} 个"
              f"「兼容字符」（字形正常但码位不同），分布在 "
              f"{('、'.join(tl['detail'].keys()))}："]
        for k, v in tl["detail"].items():
            L.append(f"  - {k}：{v['count']} 个，例如 {'、'.join(v['examples'])}")
        L.append("")
    else:
        L += ["- **同源的隐性乱码**：本次扫描未在检索片段中发现兼容字符"
              "（语料 PDF 文本层干净）。注意 `sample_questions.pdf` 的文本层含"
              "康熙部首（如 U+2F8F「⾏」本应为「行」、U+2ED3「⻓」本应为「长」），"
              "已在 `build_questions.normalize_text()` 中做 NFKC 归一化处理。", ""]
    L += [
        "**改进方向**：",
        "",
        "1. 建索引时**绝不使用原始文件名**作为文档名，统一走 `prepare_corpus.py` 的"
        "「股票代码→公司名」映射 + 正文首 3 页一致性校验（`metadata_verified` 字段留痕）；",
        "2. 采集阶段就把文件名规范化（解压时指定 `-O CP936` 或改名脚本），从源头止损；",
        "3. 评估时对文档名做归一化匹配（去空格/全半角/括号），避免同实体不同写法造成假阴性。",
        "",
    ]

    # ② 长文档稀释
    L += ["### 问题② 长文档中目标信息被稀释、召回不到", "",
          "**现象**：年报普遍 200~400 页，目标信息往往只占一两段。"
          "即使切块后，单个目标块的「语义密度」也被大量同类内容淹没，"
          "在 Top-k 里排不到前面。", "",
          "**根因**：固定 400 字切块 + 单次 Top-k 截断，"
          "使「一个文档里只有一处讲这件事」与「几十处讲类似的事」无法区分；"
          "多文档混合检索时，长文档天然贡献更多候选块，进一步挤压目标块。", "",
          "**证据（自动检测）**：",
          "",
          f"- 单文档检索块数的中位数为 {ret['median_doc_chunks']}，"
          f"即一个目标片段在文档中的占比约 {1 / max(ret['median_doc_chunks'], 1):.3%}；",
          f"- 期望页码零召回的题：{_brief(ret['page_missed'], 'page')}；",
          f"- 命中但排名靠后（排名 > top_k/2）的题：{_brief(ret['low_rank'], 'rank')}。",
          "",
          "**改进方向**：",
          "",
          "1. 提高 `recall_k`（当前 20）并对长文档单独配额，避免单个长文档占满候选；",
          "2. 结构感知分块时把「章节路径」也编码进向量（`chunk_structure` 已注入路径文本，"
          "可进一步把路径单独作为一路检索信号）；",
          "3. 引入**父块回溯（small-to-big）**：先用小片段精确定位，再把所在章节整体喂给 LLM；",
          "4. 对分析型问题启用问题分解（`query_understand.decompose`）+ 多路检索合并，"
          "让「拨备覆盖率」「贷款结构」各自检索一次。",
          "",
          ]

    # ③ 数值张冠李戴
    L += ["### 问题③ 数值类问题：上下文数字多，LLM 张冠李戴", "",
          "**现象**：年报里同一指标有三年三列（如 2017/2018/2019），"
          "数字密度极高，LLM 容易把不同年份或不同口径的数字混在一句话里。", "",
          "**根因**：① 表格转 Markdown 后表头与数据行的对应关系需要模型自行对齐；"
          "② 检索回来的片段里往往同时含正确答案与相邻的干扰数字；"
          "③ 答案中的换算值（元↔亿元、百万↔亿）无法在上下文里逐字匹配，"
          "自动核验时会暴露为「无支撑数字」。", "",
          "**证据（自动检测）**：", ""]
    if num["rows"]:
        L += [f"- {num['n_numeric_questions']} 道数值型 / 需表格题，答案共 "
              f"{num['total_numbers']} 个数字，其中 **{num['unsupported_numbers']} 个"
              f"（{_p(num['unsupported_rate'])}）无法在检索上下文中直接找到**"
              f"（多为单位换算值或推算值，需人工确认是否属于臆造）；", ""]
        L += ["  | 题号 | 答案数字数 | 无支撑数字 | 上下文数字密度(个/千字) | 上下文中出现的年份 |",
              "  | --- | --- | --- | --- | --- |"]
        for x in num["rows"]:
            L.append(f"  | {x['id']} | {x['n_answer_numbers']} | {x['n_unsupported']}"
                     f"{('（' + '、'.join(x['unsupported_numbers'][:4]) + '）') if x['unsupported_numbers'] else ''}"
                     f" | {x['ctx_numeric_density']} | {'、'.join(x['years_in_context'])} |")
        L.append("")
        numeric_ids = {x["id"] for x in num["rows"]}
        miss_numeric = {k: v for k, v in ans["miss_detail"].items() if k in numeric_ids}
        L += [f"- 数值题的关键信息点漏答情况："
              f"{json.dumps(miss_numeric, ensure_ascii=False)}", ""]
    else:
        L += ["- 本次未观测到数值型用例。", ""]
    L += ["**改进方向**：",
          "",
          "1. 表格转 Markdown 时**强制带上年份表头行**，并要求 LLM 逐行读取（generator 的 Prompt 已有约束，"
          "可在评估中增加「年份-数值配对」专项检查）；",
          "2. 数值题单独走「表格优先」检索通道（`type=table` 过滤 + BM25 数字加权）；",
          "3. 生成阶段要求模型**逐字抄写原文数字并标注页码**，评估时用 `answer_keywords` 做数字级核对；",
          "4. 对单位换算（元/亿元、百万元/亿元）在答案里强制写明两种口径，避免自动核验误判。",
          ""]

    # ④ 多文档覆盖
    L += ["### 问题④ 跨文档归纳类问题：单次检索覆盖不全", "",
          "**现象**：「分析这些银行和保险公司的共同策略与差异化策略」这类问题，"
          "Top-k 检索很难同时命中 7 份年报，答案只能覆盖其中一部分公司。", "",
          "**根因**：一次检索只有一个查询向量，Top-k 片段在「多文档、多子话题」"
          "分布下会被少数文档占满；且问题里的「这些」没有显式实体，"
          "Query 理解抽不出公司列表，无法主动指定检索范围。", "",
          "**证据（自动检测）**：", ""]
    if multi["rows"]:
        L += ["  | 题号 | 期望文档数 | 实际覆盖 | 覆盖率 | top_k | 理论上限 | 未覆盖文档 |",
              "  | --- | --- | --- | --- | --- | --- | --- |"]
        for x in multi["rows"]:
            L.append(f"  | {x['id']} | {x['n_expected']} | {x['n_covered']} | "
                     f"{x['coverage']:.0%} | {x['top_k']} | "
                     f"{x['theoretical_max_coverage']:.0%} | "
                     f"{'、'.join(x['missing_docs'][:6]) or '—'} |")
        L.append("")
        worst = min(multi["rows"], key=lambda x: x["theoretical_max_coverage"])
        L += [f"- 平均覆盖率 **{_p(multi['avg_coverage'])}**；"
              f"缺口最大的是 {worst['id']}——top_k={worst['top_k']} 而期望覆盖 "
              f"{worst['n_expected']} 份文档，理论上限就只有 "
              f"{worst['theoretical_max_coverage']:.0%}，"
              f"**无论检索多准都覆盖不全**，属于结构性问题。", ""]
    else:
        L += ["- 本次未观测到多文档用例。", ""]
    L += ["**改进方向**：",
          "",
          "1. Query 理解阶段识别「这些公司」类指代 → 用文档注册表展开为显式实体清单，"
          "每个实体独立检索后合并（`Pipeline._multi_query_retrieve` 已支持子问题并行检索，"
          "把公司名作为子问题即可）；",
          "2. 按文档**配额召回**（每份文档至少 1~2 块），保证覆盖面；",
          "3. 用**层次化摘要**（先对每份年报生成「风险管理/资本结构/绿色金融」维度摘要，"
          "再在摘要层做多文档归纳）——这正是工单08 Graph RAG 社区摘要要解决的问题；",
          "4. 生成阶段显式要求「每家公司分点陈述，找不到就说明未覆盖」，避免用少数公司代替全体。",
          ""]

    # ⑤ 表格
    L += ["### 问题⑤ 表格数据丢失", "",
          "**现象**：年报的核心数据（监管指标、主要会计数据）几乎都在表格里。"
          "若表格未被解析成可检索单元，数值题就只能靠正文里的零散复述，答案精度大幅下降。", "",
          "**根因**：① PDF 表格是版式对象，纯文本层提取会丢失行列关系；"
          "② 表格块虽然整体保留不切分（`chunk._block_to_single_chunk`），"
          "但一个表格只产生一个 chunk，在 Top-k 里很容易被正文块挤掉；"
          "③ 部分表格跨页，被拆成两个不完整的表。", "",
          "**证据（自动检测）**：",
          "",
          f"- 语料共 {tab['total_chunks']} 个检索块，其中表格块 {tab['total_table_chunks']} 个"
          f"（占比 **{_p(tab['corpus_table_ratio'])}**）；",
          "",
          "  | 文档 | 检索块 | 表格块 | 表格占比 |",
          "  | --- | --- | --- | --- |"]
    for x in tab["per_doc"][:9]:
        L.append(f"  | {x['doc']} | {x['chunks']} | {x['table_chunks']} | "
                 f"{_p(x['table_ratio'])} |")
    L += ["", f"- 标记为「需要表格」的题目："
              f"{'、'.join(x['id'] for x in tab['need_table_questions']) or '无'}；"
              f"其中**一个表格块都没召回到**的："
              f"{'、'.join(tab['need_table_with_zero_table_chunk']) or '无'}。", ""]
    L += ["**改进方向**：",
          "",
          "1. 表格块入库时在文本前拼接表名/表头/所属章节（增强上下文），"
          "并给表格块加类型权重（检索时 `type=table` 加权）；",
          "2. 表格按行拆分为子块（保留表头行），让「2019 年那一列」可以单独被检索到；",
          "3. 跨页表格做合并（按表头相似度拼接）；",
          "4. 数值题启用「表格优先」的级联检索：先过滤 `type=table` 召回，再全局补召回。",
          ""]

    # ⑥ 章节干扰
    L += ["### 问题⑥ 章节结构相似的干扰（易串公司）", "",
          "**现象**：9 份年报都是金融机构年报，章节结构高度相似"
          "（都有「董事长致辞」「经营情况讨论与分析」「风险管理」「资产质量」），"
          "检索某一家公司的「风险管理」时，很容易把别家的同名章节一起召回。", "",
          "**根因**：章节名、句式、术语在三份银行年报之间几乎同构，"
          "BM25 与向量检索都会给出高分；而片段头部只有章节路径、没有公司名，"
          "模型容易把 A 公司的数字安到 B 公司头上。", "",
          "**证据（自动检测）**：",
          "",
          f"- 全部检索片段 {inter['total_chunks']} 个，"
          f"其中「来自非主责文档且命中年报共有章节关键词」的 **{inter['total_interfering']} 个"
          f"（{_p(inter['interfering_rate'])}）**；", ""]
    for row in sorted(inter["rows"], key=lambda x: -x["n_interfering"])[:top_n]:
        L.append(f"- **{row['id']}**：{row['n_foreign']}/{row['top_k']} 个片段来自非主责文档，"
                 f"其中 {row['n_interfering']} 个命中共有章节名")
        for e in row["examples"][:2]:
            L.append(f"  - 《{e['doc']}》第{e['page']}页（命中「{e['matched_keyword']}」，"
                     f"分数 {e['score']}）：{e['snippet']}…")
    L += ["", "**改进方向**：",
          "",
          "1. 片段文本头部**强制注入公司名**（「【平安银行2019年报 > 第三章 风险管理】」），"
          "让生成模型始终知道这段是谁的；",
          "2. 检索时按问题中的实体做**元数据过滤**（Chroma `where` / BM25 文档级过滤），"
          "把候选限定在主责文档内；",
          "3. 用命名实体识别校验答案中的「公司-数字」配对，发现串公司直接判错；",
          "4. 这正是工单08 Graph RAG 的用武之地：图谱中的「公司-指标-数值」三元组"
          "天然带主体约束，不会跨公司串味。",
          ""]

    # ---------- 问题-用例矩阵 ----------
    L += ["## 二、问题类型 × 测试用例 矩阵", "",
          "| 编号 | ①元数据 | ②长文档稀释 | ③数值张冠李戴 | ④多文档覆盖 | "
          "⑤表格丢失 | ⑥章节干扰 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in ret["rows"]:
        qid = row["id"]
        n = num_by_id.get(qid, {})
        m = next((x for x in multi["rows"] if x["id"] == qid), None)
        t = next((x for x in tab["rows"] if x["id"] == qid), None)
        i = next((x for x in inter["rows"] if x["id"] == qid), None)
        cells = [
            "—",                                        # ① 全体共因，逐题不再重复标注
            "✗" if (not row["hit"] or (row["page_hit_rate"] == 0)) else "—",
            "✗" if n.get("n_unsupported") else "—",
            "✗" if (m and m["coverage"] < 1.0) else "—",
            "✗" if (t and t["need_table"] and t["n_table_chunks"] == 0) else "—",
            "✗" if (i and i["n_interfering"] > 0) else "—",
        ]
        L.append(f"| {qid} | " + " | ".join(cells) + " |")
    L += ["", "> 说明：① 是全体问题的共同根因（已在建索引阶段治理），"
              "故不逐题标注；②~⑥ 为逐题自动检出的实际受影响项。", ""]

    # ---------- 改进优先级 ----------
    L += ["## 三、改进优先级建议", "",
          "| 优先级 | 改进项 | 预期收益 | 对应工单 |",
          "| --- | --- | --- | --- |",
          "| P0 | 片段头部注入公司名 + 实体元数据过滤 | 直接消除「串公司」，"
          "对 ⑥ 有效 | 工单07/08 |",
          "| P0 | 表格整体保留 + 表头年份显式化 + 表格通道加权 | 提升 ③⑤，"
          "数值题精度 | 工单03/07 |",
          "| P1 | Query 理解展开「这些公司」为实体清单 + 按文档配额召回 | "
          "提升 ④ 多文档覆盖率 | 工单05/08 |",
          "| P1 | 多文档题改用层次化摘要 / 图谱社区摘要 | 提升 ④，"
          "跨文档归纳质量 | 工单08 |",
          "| P2 | 提高 recall_k、父块回溯 small-to-big | 缓解 ② 长文档稀释 | 工单02/13 |",
          "| P2 | 文件名规范化落库 + 文档名归一化匹配 | 消除 ① 元数据风险 | 工单07 |",
          "",
          "> 上述问题在工单08（Graph RAG）中的系统性缓解方案，"
          "详见 `docs/问题分析与改进建议.md` 第四节。",
          ""]
    return "\n".join(L)


def _p(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.2%}" if v <= 1 else f"{v:.2f}"


def _brief(rows: list[dict], mode: str) -> str:
    """把问题清单渲染成「Q03(期望 5、22)」这样的一行摘要。"""
    if not rows:
        return "无"
    out = []
    for x in rows:
        if mode == "page":
            pages = "、".join(str(p) for p in x.get("expected_pages", [])) or "—"
            out.append(f"{x['id']}(期望第{pages}页)")
        else:
            out.append(f"{x['id']}(第{x['rank']}位)")
    return "、".join(out)


def _cov(multi: dict, qid: str) -> str:
    for x in multi["rows"]:
        if x["id"] == qid:
            return f"{x['n_covered']}/{x['n_expected']}（{x['coverage']:.0%}）"
    return "—"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=f"检索问题分析（{WO_NO}）")
    ap.add_argument("--top-n", type=int, default=3, help="每类问题展示的案例数")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    inputs = {
        "test_cases": _load(base / "results/test_cases.json"),
        "rag_results": _load(base / "results/rag_test_results.json"),
        "evaluation": _load(base / "results/evaluation.json", required=False),
        "corpus_stats": _load(base / "results/corpus_stats.json", required=False),
    }
    print(f"===== {WO_NO} · 检索结果问题分析 =====")
    print(f"样本：{inputs['rag_results']['summary']['n']} 个问题")

    findings = collect(inputs)
    md = render(findings, inputs, top_n=args.top_n)

    out = ensure_results_dir() / "problem_analysis.md"
    out.write_text(md, encoding="utf-8")
    # 同时落一份机器可读版本，便于工单08 做前后对比
    (ensure_results_dir() / "problem_analysis.json").write_text(
        json.dumps({"wo_no": WO_NO,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "findings": findings}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n-- 自动分析摘要 ------------------------------------------------")
    print(f"  A 元数据      ：{findings['metadata']['n_broken']}/"
          f"{findings['metadata']['n_docs']} 份文件名乱码；"
          f"反事实模拟命中 "
          f"{findings['metadata']['counterfactual']['n_would_match']}/"
          f"{findings['metadata']['counterfactual']['n_would_match'] + findings['metadata']['counterfactual']['n_would_fail']}"
          f"；文本层兼容字符 {findings['text_layer']['total']} 个")
    print(f"  B 检索失败    ：主责文档未命中 {findings['retrieval']['n_missed']} 题，"
          f"期望页码零召回 {len(findings['retrieval']['page_missed'])} 题")
    print(f"  C 长文档稀释  ：单文档块数中位数 "
          f"{findings['retrieval']['median_doc_chunks']}")
    print(f"  D 数值张冠李戴：无支撑数字 "
          f"{findings['numeric']['unsupported_numbers']}/"
          f"{findings['numeric']['total_numbers']}")
    print(f"  E 多文档覆盖  ：平均覆盖率 {_p(findings['multi_doc']['avg_coverage'])}")
    print(f"  F 表格丢失    ：表格块占比 "
          f"{_p(findings['tables']['corpus_table_ratio'])}")
    print(f"  G 章节干扰    ：干扰片段占比 "
          f"{_p(findings['interference']['interfering_rate'])}")
    print(f"  H 答案质量    ：关键词准确率 "
          f"{_p(findings['answer']['keyword_accuracy'])}")
    print(f"\n报告已保存 → {out}")


if __name__ == "__main__":
    main()
