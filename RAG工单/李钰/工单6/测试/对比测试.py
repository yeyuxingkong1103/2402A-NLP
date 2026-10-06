# -*- coding: utf-8 -*-
"""
检索策略对比测试 - V6
工单编号: 人工智能 NLP-RAG-混合检索任务

对比 vector / fulltext / hybrid 三种策略 + 3 种融合 + 3 种重排
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

# 测试用例 (来自工单5验收标准的 5 轮对话 + 旧问题)
TEST_QUERIES = [
    {"q": "武汉兴图新科电子股份有限公司注册资本是多少?", "type": "fact"},
    {"q": "报告期内武汉兴图新科来自军用领域的收入分别是多少?", "type": "financial"},
    {"q": "武汉力源信息技术股份有限公司本次发行股数是多少?", "type": "fact"},
    {"q": "武汉力源组织结构图中销售部有哪些下属部门?", "type": "structure"},
    {"q": "募集资金拟投资哪些项目?", "type": "investment"},
    {"q": "与公司存在控制关系的关联方是谁?", "type": "relation"},
    {"q": "报告期内军用领域收入占主营业务的比重?", "type": "ratio"},
    {"q": "公司参与制定了哪个技术标准?", "type": "standard"},
]


def run_comparison():
    import qa_engine_v6

    strategies = ["vector", "fulltext", "hybrid"]
    fusions = ["weighted", "rrf", "vote"]
    rerankers = ["tfidf", "llm", "feedback", "none"]

    print(f"{'='*70}\nV6 检索策略对比测试\n{'='*70}")

    # 1. 策略对比
    print(f"\n--- 1. 检索策略对比 (hybrid + weighted + tfidf重排) ---")
    for s in strategies:
        times = []
        for item in TEST_QUERIES:
            t0 = time.time()
            r = qa_engine_v6.answer_question(item["q"], strategy=s)
            times.append(time.time() - t0)
        avg_time = sum(times) / len(times)
        print(f"  {s:10s}: 平均 {avg_time:.3f}s | 总 {len(TEST_QUERIES)} 题")

    # 2. 融合方法对比
    print(f"\n--- 2. 融合方法对比 (hybrid + 3种融合) ---")
    for f in fusions:
        import config_v6
        config_v6.FUSION_METHOD = f
        times = []
        for item in TEST_QUERIES[:3]:
            t0 = time.time()
            r = qa_engine_v6.answer_question(item["q"], strategy="hybrid")
            times.append(time.time() - t0)
        avg_time = sum(times) / len(times)
        print(f"  {f:10s}: 平均 {avg_time:.3f}s | 样本 3 题")

    # 3. 重排器对比
    print(f"\n--- 3. 重排器对比 (hybrid + weighted + 4种重排) ---")
    for rr in rerankers:
        try:
            import config_v6
            config_v6.RERANKER = rr
            times = []
            top1_scores = []
            for item in TEST_QUERIES[:3]:
                r = qa_engine_v6.answer_question(item["q"], strategy="hybrid")
                times.append(r["response_time"])
                if r["retrieval_results"]:
                    top1_scores.append(r["retrieval_results"][0]["score"])
            avg_time = sum(times) / len(times)
            avg_score = sum(top1_scores) / max(len(top1_scores), 1)
            print(f"  {rr:10s}: 平均 {avg_time:.3f}s | Top1 分 {avg_score:.4f}")
        except Exception as e:
            print(f"  {rr:10s}: 不可用 - {e}")

    print(f"\n{'='*70}\n对比完成\n{'='*70}")


if __name__ == "__main__":
    run_comparison()
