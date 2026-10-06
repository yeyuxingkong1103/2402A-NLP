# -*- coding: utf-8 -*-
# 【批量评测入口 · run_evaluation.py】真实运行10题：检索→作答/拒答→RAGAS四指标，落盘JSON与CSV
# 工单编号：人工智能NLP-RAG-功能测试及评估

"""功能测试与评估主入口（真实运行，非模拟）。

执行流程：
1. dataset_loader 加载语料（ccf_competition.zip 优先，缺失回退招股书 PDF）；
2. rag_pipeline 构建/复用 TF-IDF+BM25 混合索引；
3. 逐题真实检索与抽取作答（不可回答题验证拒答），打点记录逐题耗时；
4. ragas_evaluator 离线计算 faithfulness/answer_relevancy/
   context_precision/context_recall 四项指标；
5. 产出：
   - ``03_测试/检索结果.json``：每题 query/answer/contexts/拒答标记/耗时；
   - ``03_测试/ragas评估结果.csv``：四项指标逐题 + 均值（含题型与拒答列）。

运行方式（PowerShell）：
    cd 02_研发/src
    python run_evaluation.py
"""
import json
import os
import sys
import time

import pandas as pd

from dataset_loader import load_corpus
from rag_pipeline import build_or_load_index
from ragas_evaluator import evaluate_one

# 路径定位：src -> 02_研发 -> 工单根目录
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DEV_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
QUESTIONS_PATH = os.path.join(DEV_DIR, "questions.json")
TEST_DIR = os.path.join(ROOT_DIR, "03_测试")
CACHE_PATH = os.path.join(DEV_DIR, "index_cache", "index.pkl")

RETRIEVAL_JSON = os.path.join(TEST_DIR, "检索结果.json")
RAGAS_CSV = os.path.join(TEST_DIR, "ragas评估结果.csv")


def _evidence_to_dict(ev) -> dict:
    """把检索证据对象序列化为可落盘 JSON 的字典。

    :param ev: Evidence 对象
    :return: JSON 友好的字典
    """
    return {
        "块编号": ev.chunk_id,
        "来源文档": ev.doc_name,
        "页码": ev.page_no,
        "融合分": ev.score,
        "TFIDF名次": ev.tfidf_rank,
        "BM25名次": ev.bm25_rank,
        "文本": ev.text,
    }


def main() -> int:
    """批量评测主流程。

    :return: 进程退出码（0 正常）
    """
    os.makedirs(TEST_DIR, exist_ok=True)
    print("=" * 72)
    print("RAG 功能测试及评估（工单编号：人工智能NLP-RAG-功能测试及评估）")
    print("=" * 72)

    # 1. 加载语料（zip 缺失会打印放置说明并自动回退）
    corpus = load_corpus(DATA_DIR)
    documents = corpus["documents"]

    # 2. 构建/加载混合索引
    t_build = time.perf_counter()
    pipeline = build_or_load_index(documents, CACHE_PATH)
    build_seconds = round(time.perf_counter() - t_build, 2)
    print(f"[流程] 索引就绪，共 {len(pipeline.chunks)} 块（构建/加载用时 {build_seconds} 秒）")

    # 3. 读取 10 题问题集
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        question_bank = json.load(f)
    samples = question_bank["问题列表"]
    print(f"[流程] 载入问题集：{QUESTIONS_PATH}，共 {len(samples)} 题")

    # 4. 逐题真实运行
    records = []
    metric_rows = []
    refusal_count = 0
    max_latency = 0.0
    t_all = time.perf_counter()
    for sample in samples:
        qid = sample["题号"]
        question = sample["问题"]
        result = pipeline.answer(question)
        if result.refused:
            refusal_count += 1
        max_latency = max(max_latency, result.latency_s)

        contexts = [_evidence_to_dict(e) for e in result.evidences]
        retrieved_texts = [e.text for e in result.evidences]
        metrics = evaluate_one(sample, result.answer, result.refused,
                               retrieved_texts)

        print(f"[{qid}|{sample['题型']}] 拒答={result.refused} "
              f"耗时={result.latency_s}s | {result.answer[:48]}")

        records.append({
            "题号": qid,
            "题型": sample["题型"],
            "query": question,
            "answer": result.answer,
            "gold_answer": sample.get("gold_answer", ""),
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "应拒答": not sample.get("可回答", True),
            "latency_s": result.latency_s,
            "top_score": result.top_score,
            "query_coverage": result.coverage,
            "rare_term_hit": result.rare_term_hit,
            "contexts": contexts,
        })

        row = {
            "题号": qid,
            "题型": sample["题型"],
            "问题": question,
            "系统是否拒答": "是" if result.refused else "否",
            "faithfulness": metrics["faithfulness"],
            "answer_relevancy": metrics["answer_relevancy"],
            "context_precision": metrics["context_precision"],
            "context_recall": metrics["context_recall"],
            "四指标均值": round(
                (metrics["faithfulness"] + metrics["answer_relevancy"]
                 + metrics["context_precision"] + metrics["context_recall"]) / 4,
                4),
            "逐题耗时秒": result.latency_s,
            "指标口径": metrics["口径"],
        }
        metric_rows.append(row)

    total_seconds = round(time.perf_counter() - t_all, 2)

    # 5. 指标汇总
    metric_names = ["faithfulness", "answer_relevancy",
                    "context_precision", "context_recall"]
    means = {m: round(sum(r[m] for r in metric_rows) / len(metric_rows), 4)
             for m in metric_names}
    overall = round(sum(means.values()) / 4, 4)
    mean_row = {
        "题号": "均值",
        "题型": "全部10题",
        "问题": "—",
        "系统是否拒答": f"{refusal_count}题拒答",
        "faithfulness": means["faithfulness"],
        "answer_relevancy": means["answer_relevancy"],
        "context_precision": means["context_precision"],
        "context_recall": means["context_recall"],
        "四指标均值": overall,
        "逐题耗时秒": round(sum(r["latency_s"] for r in records) / len(records), 4),
        "指标口径": "算术平均（不可回答题按拒答口径计入）",
    }

    # 6. 落盘检索结果 JSON
    payload = {
        "工单编号": "人工智能NLP-RAG-功能测试及评估",
        "评测时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "语料来源": corpus["source"],
        "语料说明": (
            "本次因 ccf_competition.zip 缺失，使用 data/招股说明书1.pdf 回退语料，"
            "演示同一套RAG测试与RAGAS评估流程；zip到位后删除回退PDF重跑即可。"
            if corpus["source"] == "fallback_pdf"
            else "已使用工单指定 ccf_competition.zip 竞赛语料。"),
        "文档数": corpus["doc_count"],
        "总页数": corpus["page_count"],
        "检索块数": len(pipeline.chunks),
        "检索方案": "TF-IDF(cosine)+BM25双路召回，RRF融合取并，归一化线性分+查询词覆盖度精排",
        "拒答策略": ("Top1融合分阈值0.24 / 查询词加权覆盖度阈值0.34 / "
                   "高IDF专有词命中率阈值0.30，任一不过即拒答"),
        "题量": len(samples),
        "拒答题数": refusal_count,
        "评测总耗时秒": total_seconds,
        "逐题最大耗时秒": max_latency,
        "指标均值": means,
        "四指标总均值": overall,
        "逐题结果": records,
    }
    with open(RETRIEVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 7. 落盘 RAGAS 评估 CSV（utf-8-sig 便于 Excel 直接打开）
    df = pd.DataFrame(metric_rows + [mean_row])
    df.to_csv(RAGAS_CSV, index=False, encoding="utf-8-sig")

    print("-" * 72)
    print(f"[结果] 拒答题数：{refusal_count}/{len(samples)}；"
          f"逐题最大耗时：{max_latency}s（验收线 3s）")
    print(f"[结果] RAGAS 四项均值：{means}；总均值：{overall}")
    print(f"[产出] {RETRIEVAL_JSON}")
    print(f"[产出] {RAGAS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
