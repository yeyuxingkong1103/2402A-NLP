# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""
基准测试：同一批查询分别跑 基线/优化 两条检索链路，输出分阶段耗时 JSON。

用法：
    python benchmark.py                # 默认 10 条查询
    python benchmark.py --profile      # 附带对基线做一次 cProfile 应用级性能分析
"""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

from config import MILVUS_COLLECTION
from logger import get_logger
from perf_pipeline import (warmup, baseline_search, optimized_search,
                           optimized_search_cached)

logger = get_logger(__name__)

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

# 法律知识库（rag_legal，1076 chunks）测试查询：覆盖事实查找/法条/流程类
QUERIES = [
    "劳动合同中试用期最长可以约定多久",
    "交通事故造成他人受伤需要赔偿哪些费用",
    "公司拖欠工资劳动者可以申请劳动仲裁吗",
    "离婚时夫妻共同债务如何认定和分割",
    "遗嘱继承和法定继承哪个效力优先",
    "合同违约后守约方可以主张哪些违约责任",
    "工伤认定的申请时限是多久",
    "民间借贷的利率超过多少不受法律保护",
    "消费者购买到假冒伪劣商品如何维权",
    "房屋买卖合同中没有约定违约金怎么办",
]


def run_one(mode: str, query: str):
    if mode == "baseline":
        return baseline_search(query, MILVUS_COLLECTION)
    return optimized_search_cached(query, MILVUS_COLLECTION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="对基线做一次 cProfile 分析")
    ap.add_argument("--repeat", type=int, default=1, help="每查询重复次数（>1 可观察缓存命中）")
    args = ap.parse_args()

    logger.info(f"测试集合：{MILVUS_COLLECTION}，查询数：{len(QUERIES)}")
    warmup(MILVUS_COLLECTION)

    # 预热首查（模型前向预热，不计入统计）
    baseline_search("预热查询", MILVUS_COLLECTION)
    optimized_search("预热查询", MILVUS_COLLECTION)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        baseline_search(QUERIES[0], MILVUS_COLLECTION)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        (RESULT_DIR / "cprofile_baseline.txt").write_text(s.getvalue(), encoding="utf-8")
        logger.info("cProfile 结果已保存 results/cprofile_baseline.txt")

    records = []
    for q in QUERIES:
        for rep in range(args.repeat):
            for mode in ("baseline", "optimized"):
                docs, t = run_one(mode, q)
                records.append({
                    "query": q, "repeat": rep, "mode": mode,
                    "timings": {k: round(v, 4) for k, v in t.items() if k.startswith("t_")},
                    "n_candidates": t.get("n_rerank_candidates", 0),
                    "cache_hit": t.get("cache_hit", 0),
                    "n_results": len(docs),
                    "top_score": round(docs[0].metadata.get("score", 0), 4) if docs else 0,
                })
                logger.info(f"[{mode}] rep{rep} '{q[:14]}...' 总耗时 {t['t_total']:.3f}s")

    out = RESULT_DIR / "benchmark_results.json"
    out.write_text(json.dumps({"collection": MILVUS_COLLECTION,
                               "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "records": records}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    logger.info(f"结果已保存：{out}")

    # 汇总打印
    for mode in ("baseline", "optimized"):
        ts = [r["timings"]["t_total"] for r in records if r["mode"] == mode and r["repeat"] == 0]
        ts.sort()
        print(f"{mode:>10}: 平均 {sum(ts)/len(ts):.3f}s  P95 {ts[int(len(ts)*0.95)-1]:.3f}s  "
              f"最大 {ts[-1]:.3f}s  全部<3S: {all(x < 3 for x in ts)}")


if __name__ == "__main__":
    main()
