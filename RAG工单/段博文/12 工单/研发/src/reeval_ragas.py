# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
从已保存的 compare_results.json 重新计算 RAGAS 指标（修正评分口径，无需重跑检索）。

口径修正（公平对比）：
    1. context_precision / context_recall：两链路统一基于"检索到的源文本块"
       评估——传统 RAG 是 8 个 chunk；LightRAG 取其 Sources（Document Chunks）
       中的文本块（同为 top-8，同切块、同 embedding），剔除实体/关系 JSON
       与标题，避免把图增强结构当作检索块造成精度低估。
    2. faithfulness：答案命中关键词可归因于"完整上下文（LightRAG 含图结构）"，
       因为答案就是用完整上下文生成的。
    3. answer_relevancy：答案对参考答案关键词的覆盖（不变）。
    4. LightRAG 额外统计图结构证据（实体/关系）命中关键词的情况，作为
       "图证据覆盖"单独展示，不并入 precision。
"""

import json
import re
from collections import Counter

from config import RESULT_DIR

FILE = RESULT_DIR / "compare_results.json"


def parse_json_block(text: str) -> list:
    """解析 LightRAG ```json 块，返回记录列表（整体/逐对象 raw_decode）。"""
    t = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    dec = json.JSONDecoder()
    out, idx = [], 0
    while idx < len(t):
        while idx < len(t) and t[idx] not in "{[":
            idx += 1
        if idx >= len(t):
            break
        try:
            obj, end = dec.raw_decode(t, idx)
            out.extend(obj if isinstance(obj, list) else [obj])
            idx = end
        except Exception:
            idx += 1
    return out


def _norm(s: str) -> str:
    return str(s).replace(",", "").replace("，", "").replace(" ", "")


def _hit(text: str, keywords) -> int:
    t = _norm(text)
    return sum(1 for k in keywords if _norm(k) in t)


def lightrag_source_chunks(raw_contexts: list) -> list:
    """从 LightRAG 原始上下文提取 Sources 文本块（有序）。"""
    chunks = []
    for seg in raw_contexts:
        if '"reference_id"' in seg and '"content"' in seg:
            for rec in parse_json_block(seg):
                if isinstance(rec, dict) and rec.get("content"):
                    chunks.append(rec["content"])
    return chunks


def precision_recall(keywords, chunks):
    """基于源文本块的 context_precision（排名加权）与 context_recall。"""
    n = len(chunks)
    if n == 0:
        return 0.0, 0.0
    rel_ranks = [i for i, c in enumerate(chunks, 1) if _hit(c, keywords) > 0]
    if rel_ranks:
        weight_sum = sum(1.0 / r for r in range(1, n + 1))
        precision = sum(1.0 / r for r in rel_ranks) / weight_sum
    else:
        precision = 0.0
    recall = _hit(" ".join(chunks), keywords) / len(keywords)
    return round(precision, 4), round(recall, 4)


def graph_evidence_hits(raw_contexts, keywords):
    """统计实体/关系图证据中命中关键词的记录数（图增强证据覆盖）。"""
    ent_recs, rel_recs = [], []
    for seg in raw_contexts:
        if '"entity1"' in seg and '"entity2"' in seg:
            rel_recs = parse_json_block(seg)
        elif '"entity"' in seg and '"type"' in seg:
            ent_recs = parse_json_block(seg)
    ent_hit = sum(1 for r in ent_recs
                  if _hit(json.dumps(r, ensure_ascii=False), keywords) > 0)
    rel_hit = sum(1 for r in rel_recs
                  if _hit(json.dumps(r, ensure_ascii=False), keywords) > 0)
    return {"entities_total": len(ent_recs), "entities_hit": ent_hit,
            "relations_total": len(rel_recs), "relations_hit": rel_hit}


def main():
    data = json.loads(FILE.read_text(encoding="utf-8"))
    results = data["results"]

    for row in results:
        kw = row["keywords"]

        # ---- 传统 RAG：chunks 即检索块，precision/recall 重算（口径不变，保持一致）----
        rag_chunks = row["rag"]["contexts"]
        p, rc = precision_recall(kw, rag_chunks)
        row["rag"]["context_precision"] = p
        row["rag"]["context_recall"] = rc

        # ---- LightRAG：基于 Sources 文本块评 precision/recall ----
        lr = row["lightrag"]
        source_chunks = lightrag_source_chunks(lr["contexts"])
        p, rc = precision_recall(kw, source_chunks)
        lr["context_precision"] = p
        lr["context_recall"] = rc
        lr["source_chunks_count"] = len(source_chunks)

        # faithfulness：答案命中关键词在完整上下文（含图）中可归因的比例
        full_ctx = " ".join(lr["contexts"])
        ans = lr["answer"]
        ans_hits = [k for k in kw if _norm(k) in _norm(ans)]
        lr["faithfulness"] = round(
            sum(1 for k in ans_hits if _hit(full_ctx, [k]) > 0) / len(ans_hits),
            4) if ans_hits else 0.0
        lr["answer_relevancy"] = round(len(ans_hits) / len(kw), 4)

        # 传统 RAG faithfulness/relevancy 同步重算（保持函数一致）
        rfull = " ".join(rag_chunks)
        rh = [k for k in kw if _norm(k) in _norm(row["rag"]["answer"])]
        row["rag"]["faithfulness"] = round(
            sum(1 for k in rh if _hit(rfull, [k]) > 0) / len(rh), 4) if rh else 0.0
        row["rag"]["answer_relevancy"] = round(len(rh) / len(kw), 4)

        # 图证据覆盖（单独展示，不并入主指标）
        lr["graph_evidence"] = graph_evidence_hits(lr["contexts"], kw)

    # ---- 汇总 ----
    n = len(results)
    summary = {"n": n}
    for side in ("rag", "lightrag"):
        summary[side] = {
            m: round(sum(r[side][m] for r in results) / n, 4)
            for m in ("context_precision", "context_recall",
                      "faithfulness", "answer_relevancy")
        }
        summary[side]["overall"] = round(
            sum(summary[side][m] for m in
                ("context_precision", "context_recall", "faithfulness",
                 "answer_relevancy")) / 4, 4)
        summary[side]["avg_search_time"] = round(
            sum(r[side]["search_time"] for r in results) / n, 2)

    data["summary"] = summary
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"RAGAS 指标对比（{n} 题，修正后口径）")
    print(f"{'指标':<22}{'传统RAG':>10}{'LightRAG':>10}")
    for m in ("context_precision", "context_recall", "faithfulness",
              "answer_relevancy", "overall"):
        print(f"{m:<22}{summary['rag'][m]:>10.4f}{summary['lightrag'][m]:>10.4f}")
    print(f"{'平均检索耗时(s)':<20}{summary['rag']['avg_search_time']:>10.2f}"
          f"{summary['lightrag']['avg_search_time']:>10.2f}")


if __name__ == "__main__":
    main()
