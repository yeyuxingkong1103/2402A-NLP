# -*- coding: utf-8 -*-
"""
问答引擎 V7 - 评估整合
工单编号: 人工智能 NLP-RAG-功能测试及评估

核心: 调用 V6 混合检索 → 评估框架 → 报告生成
"""
import os, sys, time, json, logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluator
import test_suite
import report_generator

logger = logging.getLogger(__name__)

# 引入 V6
_V6_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "工单6", "研发"))
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)


def _call_v6(query: str) -> Dict:
    """调用 V6 混合检索"""
    try:
        import qa_engine_v6
        return qa_engine_v6.answer_question(query)
    except Exception as e:
        logger.warning(f"V6 不可用: {e}, 降级")
        return {"rag_answer": "", "retrieval_results": [], "response_time": 99}


def run_single_evaluation(test_item: Dict) -> Dict:
    """评估单个问题"""
    qid = test_item["id"]
    question = test_item["question"]
    ref = test_item["reference"]
    ref_kws = test_item["ref_keywords"]
    relevant_kws = test_item.get("relevant_chunk_keywords", [])

    # 1. 调用 RAG
    t0 = time.time()
    v6_result = _call_v6(question)
    elapsed = time.time() - t0

    retrieved = v6_result.get("retrieval_results", [])
    rag_answer = v6_result.get("rag_answer", "")
    context_chunks = [r.get("text", "") for r in retrieved[:5]]

    # 2. 构造相关 chunk id (用关键词匹配)
    relevant_ids = set()
    for r in retrieved:
        text = r.get("text", "")
        if any(kw in text for kw in relevant_kws):
            relevant_ids.add(str(r.get("id", "")))

    # 3. 检索评估
    ret_eval = evaluator.RetrievalEvaluator(top_k=5)
    ret_metrics = ret_eval.evaluate(retrieved, relevant_ids)

    # 4. QA 评估
    qa_eval = evaluator.QAEvaluator()
    qa_metrics = qa_eval.evaluate(rag_answer, ref, ref_kws, context_chunks)

    return {
        "id": qid,
        "question": question,
        "type": test_item["type"],
        "difficulty": test_item["difficulty"],
        "retrieved": retrieved,
        "rag_answer": rag_answer,
        "reference": ref,
        "response_time": round(elapsed, 3),
        "retrieval": ret_metrics,
        "qa": qa_metrics,
    }


def run_full_evaluation() -> Dict:
    """运行完整评估 (10 个问题)"""
    suite = test_suite.get_test_suite()
    all_results = []
    response_times = []

    print(f"{'='*60}\nRAG 功能测试及评估 ({len(suite)} 题)\n{'='*60}")

    for item in suite:
        print(f"\n[Q{item['id']}][{item['type']}] {item['question'][:40]}...")
        try:
            result = run_single_evaluation(item)
            all_results.append(result)
            response_times.append(result["response_time"])
            ret = result["retrieval"]
            qa = result["qa"]
            print(f"  P@5={ret['precision@k']:.2f} MRR={ret['mrr']:.2f} "
                  f"COV={qa['keyword_coverage']:.2f} 时间={result['response_time']:.2f}s")
        except Exception as e:
            logger.exception(f"Q{item['id']} 评估失败")
            print(f"  评估失败: {e}")

    # 批量评估
    ret_eval = evaluator.RetrievalEvaluator()
    qa_eval = evaluator.QAEvaluator()
    perf_eval = evaluator.PerformanceEvaluator()

    retrieval_metrics = ret_eval.evaluate_batch(all_results)
    qa_metrics = qa_eval.evaluate_batch(all_results)
    perf_metrics = perf_eval.evaluate(response_times)

    # 问题分析
    problem_analysis = _analyze_problems(all_results)

    # 生成报告
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "测试报告.md")
    md_path = os.path.abspath(md_path)
    md = report_generator.generate_markdown_report(
        all_results, retrieval_metrics, qa_metrics, perf_metrics, problem_analysis)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nMarkdown 报告: {md_path}")

    # JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "评估结果.json")
    json_path = os.path.abspath(json_path)
    save_json_payload = {
        "results": [{
            "id": r["id"], "question": r["question"],
            "type": r["type"], "difficulty": r["difficulty"],
            "retrieval": r["retrieval"], "qa": r["qa"],
            "response_time": r["response_time"],
        } for r in all_results],
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "perf_metrics": perf_metrics,
        "problem_analysis": problem_analysis,
    }
    report_generator.save_json_results(save_json_payload, json_path)

    return {
        "results": all_results,
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "perf_metrics": perf_metrics,
        "problem_analysis": problem_analysis,
    }


def _analyze_problems(results: List[Dict]) -> List[Dict]:
    """问题分析: 找出检索效果差的问题并分析原因"""
    problems = []
    for r in results:
        ret = r["retrieval"]
        qa = r["qa"]

        issues = []
        reason = ""
        suggestion = ""

        if ret.get("precision@k", 1) < 0.3:
            issues.append("Precision@5 < 0.3")
            reason = "检索 Top-5 中相关块太少, 可能是嵌入模型不匹配或查询扩展不足"
            suggestion = "尝试切换混合检索 + 更多同义词扩展"
        if ret.get("mrr", 1) < 0.3:
            issues.append("MRR < 0.3")
            reason = "第一个相关块排名太靠后, 重排算法需要优化"
            suggestion = "启用 LLM 重排器或调整融合权重"
        if qa.get("keyword_coverage", 1) < 0.5:
            issues.append("关键词覆盖率 < 0.5")
            reason = "答案缺少关键信息, 可能检索召回不足或 LLM 生成有问题"
            suggestion = "增大检索 Top-K 数量或优化 LLM Prompt"
        if qa.get("hallucination_ratio", 0) > 0.3:
            issues.append("幻觉比例 > 0.3")
            reason = "答案中有较多编造内容, LLM 未严格遵循检索上下文"
            suggestion = "优化 Prompt, 要求 LLM 仅基于提供的片段回答"

        if issues:
            problems.append({
                "id": r["id"],
                "question": r["question"],
                "issues": issues,
                "reason": reason,
                "suggestion": suggestion,
                "retrieval": ret,
                "qa": qa,
            })

    if not problems:
        # 即使没有严重问题, 也添加简要分析
        for r in results:
            cov = r["qa"].get("keyword_coverage", 1)
            if cov >= 0.7 and cov < 0.9:
                problems.append({
                    "id": r["id"],
                    "question": r["question"],
                    "issues": ["关键词覆盖率中等"],
                    "reason": "部分参考关键词未命中, 可能是同义词变体",
                    "suggestion": "扩展同义词表, 如 '注册资本'→'股本'"
                })

    return problems


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_full_evaluation()
    print(f"\n{'='*60}")
    print(f"检索评估: {json.dumps(result['retrieval_metrics'], indent=2)}")
    print(f"QA 评估: {json.dumps(result['qa_metrics'], indent=2)}")
    print(f"性能评估: {json.dumps(result['perf_metrics'], indent=2)}")
