import json
import re
import requests
import time

DEEPSEEK_API_KEY = "sk-6e04da2625084194a695de2b19c13ad2"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== 测试数据集 ==========
test_cases = [
    {
        "question": "Alice今年多大？",
        "ground_truth": "20岁",
        "keywords": ["20", "二十"],
    },
    {
        "question": "Alice最好的朋友叫什么？",
        "ground_truth": "小红",
        "keywords": ["小红"],
    },
    {
        "question": "Alice喜欢什么？",
        "ground_truth": "星空、诗歌、看星星",
        "keywords": ["星空", "诗歌", "星星"],
    },
    {
        "question": "Alice不喜欢什么环境？",
        "ground_truth": "吵闹的环境",
        "keywords": ["吵闹", "喧闹"],
    },
    {
        "question": "Alice冬天喜欢做什么？",
        "ground_truth": "看星星",
        "keywords": ["看星星", "星星"],
    },
]

def call_chat_api(question):
    """调用本地API获取回答和检索上下文"""
    url = "http://127.0.0.1:8000/api/chat"
    payload = {
        "username": "xiaoming",
        "character_name": "Alice",
        "query": question
    }
    resp = requests.post(url, json=payload, timeout=60)
    data = resp.json()
    return data["data"]["answer"], data["data"]["retrieved_docs"]

def calc_context_recall(contexts, keywords):
    """上下文召回率：标准答案关键词有多少出现在检索上下文中"""
    context_text = " ".join(contexts)
    hit = 0
    for kw in keywords:
        if kw in context_text:
            hit += 1
    return hit / len(keywords) if keywords else 0

def calc_context_precision(contexts, keywords):
    """上下文精确率：检索的上下文有多少包含关键词"""
    if not contexts:
        return 0
    relevant = 0
    for ctx in contexts:
        if any(kw in ctx for kw in keywords):
            relevant += 1
    return relevant / len(contexts)

def calc_answer_accuracy(answer, keywords):
    """回答准确率：回答是否包含标准答案关键词"""
    for kw in keywords:
        if kw in answer:
            return 1.0
    return 0.0

def calc_answer_relevancy(answer, question):
    """回答相关性：回答是否非空且有内容（简单规则）"""
    if not answer or len(answer.strip()) < 2:
        return 0.0
    # 排除明显的拒答
    refuse_patterns = ["不知道", "不清楚", "无法回答", "没有提到", "抱歉"]
    if any(p in answer for p in refuse_patterns):
        return 0.3
    return 1.0

def llm_judge(question, answer, ground_truth):
    """用大模型做忠实度判断：回答是否基于事实，无幻觉"""
    prompt = f"""
请判断以下回答是否准确回答了问题，且没有编造信息。
问题：{question}
标准答案：{ground_truth}
系统回答：{answer}
请只回答"准确"或"不准确"，不要解释。
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        result = resp.json()["choices"][0]["message"]["content"].strip()
        return 1.0 if "准确" in result else 0.0
    except:
        return 0.5

def run_eval():
    print("=" * 60)
    print("  RAG系统评测报告（简化版RAGAS）")
    print("=" * 60)
    print(f"测试问题数：{len(test_cases)}")
    print()

    results = []
    total_scores = {
        "context_recall": 0,
        "context_precision": 0,
        "answer_accuracy": 0,
        "answer_relevancy": 0,
        "faithfulness": 0,
    }

    for i, case in enumerate(test_cases):
        print(f"--- 问题{i+1}：{case['question']} ---")
        start = time.time()
        answer, contexts = call_chat_api(case["question"])
        elapsed = time.time() - start

        context_recall = calc_context_recall(contexts, case["keywords"])
        context_precision = calc_context_precision(contexts, case["keywords"])
        answer_accuracy = calc_answer_accuracy(answer, case["keywords"])
        answer_relevancy = calc_answer_relevancy(answer, case["question"])
        faithfulness = llm_judge(case["question"], answer, case["ground_truth"])

        result = {
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": answer,
            "context_count": len(contexts),
            "context_recall": round(context_recall, 4),
            "context_precision": round(context_precision, 4),
            "answer_accuracy": round(answer_accuracy, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "faithfulness": round(faithfulness, 4),
            "elapsed": round(elapsed, 2),
        }
        results.append(result)

        for k in total_scores:
            total_scores[k] += result[k]

        print(f"  回答：{answer}")
        print(f"  上下文召回率：{context_recall:.2f} | 精确率：{context_precision:.2f}")
        print(f"  回答准确率：{answer_accuracy:.2f} | 相关性：{answer_relevancy:.2f} | 忠实度：{faithfulness:.2f}")
        print(f"  耗时：{elapsed:.2f}s")
        print()

    # 计算平均分
    n = len(test_cases)
    avg_scores = {k: round(v / n, 4) for k, v in total_scores.items()}

    print("=" * 60)
    print("  评测总览（平均分）")
    print("=" * 60)
    print(f"  上下文召回率 Context Recall:    {avg_scores['context_recall']:.4f}")
    print(f"  上下文精确率 Context Precision: {avg_scores['context_precision']:.4f}")
    print(f"  回答准确率 Answer Accuracy:     {avg_scores['answer_accuracy']:.4f}")
    print(f"  回答相关性 Answer Relevancy:    {avg_scores['answer_relevancy']:.4f}")
    print(f"  忠实度 Faithfulness:            {avg_scores['faithfulness']:.4f}")
    print("=" * 60)

    # 保存结果
    output = {
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": n,
        "average_scores": avg_scores,
        "details": results,
    }
    with open("ragas_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评测报告已保存：ragas_eval_result.json")

if __name__ == "__main__":
    run_eval()
