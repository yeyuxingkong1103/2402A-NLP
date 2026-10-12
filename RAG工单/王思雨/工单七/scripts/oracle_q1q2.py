# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— Q01/Q02 的「指代澄清」上限实测（oracle 上界实验）
# 背景：Q01/Q02 题面是 `xx银行`（占位符），**没有公司名**，所以"实体约束检索"对它们不触发（见 entity_filter_probe）。
#       评审建议"只对 Q01/Q02 跑优化后 RAGAS 压成本"。但直接跑 = 什么都没改，指标不会动。
#       正确的做法是测**上限**：假设指代已被澄清（用 ground_truth 反推的主体公司做 oracle 过滤），
#       看答案与指标能到什么水平。这回答的是"加一个查询澄清步骤值不值"，而不是"当前系统表现"。
# ⚠ 本实验是 ORACLE（用了标准答案反推的公司），**不是可上线的配置**，只是收益上限。n=2，无噪声底，仅供指示。
# 用法：D:/myconda1/python.exe scripts/oracle_q1q2.py
import sys, json, pathlib, time                                       # 读配置/计时
ROOT = pathlib.Path(__file__).resolve().parent.parent                  # 项目根
sys.path.insert(0, str(ROOT))                                          # 可 import app
from app import config as C                                            # 配置
import jieba                                                           # BM25 分词
jieba.setLogLevel(60)                                                  # 静音

ORACLE = {"Q01": "000001", "Q02": "600036"}                            # 用 ground_truth 反推的主体公司（平安银行 / 招商银行）

def retrieve_filtered(q, code, top_n):                                 # 元数据过滤版检索（与 entity_filter_probe 同实现）
    from app.rag_core import embed, milvus, fuse, rerank                # 复用生产件
    import pickle                                                       # 读 BM25
    vh = [dict(h["entity"], score=float(h["distance"]), src="vec") for h in milvus().search(
        C.MILVUS_COLLECTION, data=[list(embed([q])[0])], limit=C.VEC_TOP_K, filter=f'code == "{code}"',
        output_fields=["text", "company", "year", "page", "code"],
        search_params={"metric_type": "COSINE"}, consistency_level="Strong")[0]]
    d = pickle.loads((C.DATA / "bm25.pkl").read_bytes())                 # {bm25, recs}
    sc = d["bm25"].get_scores([w for w in jieba.lcut(q) if w.strip()])    # 全语料打分
    bh = sorted((dict(d["recs"][i], score=float(sc[i]), src="bm25") for i in range(len(d["recs"]))
                 if d["recs"][i]["code"] == code and sc[i] > 0), key=lambda x: -x["score"])[: C.BM25_TOP_K]
    return rerank(q, fuse(vh, bh), n=top_n)                              # 融合 + 重排

def main():                                                             # 入口
    from app.rag_core import answer, retrieve, build_context, generate    # 问答链路
    from app.evaluate import ragas_metrics                                # 复用同一条评估实现
    qs = {q["id"]: q for q in json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]}
    rows = []                                                             # 构造 RAGAS 能吃的行
    for qid, code in ORACLE.items():                                      # 只跑 Q01/Q02
        q = qs[qid]                                                       # 题目
        comp = C.CODE2NAME[code]                                          # 主体公司
        base_hits, _ = retrieve(q["question"], mode="hybrid")              # 基线检索（对比）
        hits = retrieve_filtered(q["question"], code, C.RERANK_TOP_N)      # oracle 过滤后检索
        t0 = time.time()                                                  # 计时
        ans, _ = generate(q["question"], build_context(hits))              # 用过滤后的上下文重新生成答案
        rows.append(dict(id=qid, question=q["question"], ground_truth=q["ground_truth"], answer=ans,
                         latency_ms=round((time.time() - t0) * 1000), contexts=[h["text"] for h in hits],
                         base_docs=sorted({h["company"] for h in base_hits}),
                         oracle_docs=sorted({h["company"] for h in hits}), error=""))
        print(f"[{qid}] oracle主体={comp}｜基线来源 {sorted({h['company'] for h in base_hits})}"
              f" → oracle来源 {sorted({h['company'] for h in hits})}")     # 来源对比（可核验）
        print(f"      答案: {ans[:200]}")                                  # 答案文本（人工可核验：是否还多答别家）
    print("\n跑 RAGAS（仅 2 题，指标仅供指示，不构成达标结论）…")
    m = ragas_metrics(rows)                                                # 4 指标
    out = {"rows": rows, "ragas": m, "note": "ORACLE 上界实验：用 ground_truth 反推主体公司，非可上线配置；n=2 无噪声底"}
    (C.RESULTS / "oracle_q1q2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("oracle 2 题指标:", {k: v for k, v in m.items() if not k.startswith("per_sample_")})   # 汇总
    return out                                                            # 返回

if __name__ == "__main__":                                                # 直接运行
    main()
