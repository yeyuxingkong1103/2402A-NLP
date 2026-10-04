"""RAGAS 评测脚本。

依赖后端运行；contexts 通过 /ops/milvus/search 获取真实检索片段。
用法：``python evaluate_ragas.py``
"""
import os

os.environ.setdefault("OPENAI_API_KEY", os.getenv("LLM_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))

import requests  # noqa: E402
from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

TEST_DATA = [
    {"question": "猎捕国家重点保护野生动物怎么处罚？",
     "ground_truth": "根据《野生动物保护法》第四十八条，没收猎获物、猎捕工具和违法所得，吊销特许猎捕证，"
                     "并处猎获物价值二倍以上二十倍以下罚款；构成犯罪的，依法追究刑事责任。"},
    {"question": "人工繁育野生动物需要办理什么手续？",
     "ground_truth": "根据《野生动物保护法》第二十五条，人工繁育国家重点保护野生动物实行许可制度，"
                     "应当经省级以上野生动物保护主管部门批准，取得人工繁育许可证。"},
    {"question": "吃野生动物违法吗？",
     "ground_truth": "根据《野生动物保护法》第三十一条，禁止食用国家重点保护野生动物和国家保护的"
                     "有重要生态、科学、社会价值的陆生野生动物。"},
]


def get_answer(question: str) -> str:
    resp = requests.post(
        f"{BASE}/chat",
        json={"user_id": "eval", "role_id": 1, "message": question},
        timeout=180,
    )
    return resp.json().get("reply", "")


def get_contexts(question: str, top_k: int = 5) -> list[str]:
    resp = requests.post(
        f"{BASE}/ops/milvus/search",
        json={"query": question, "domain": "law", "top_k": top_k},
        timeout=120,
    )
    return [row.get("text", "") for row in resp.json()]


def build_dataset() -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in TEST_DATA:
        print("评测:", item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(get_answer(item["question"]))
        contexts = get_contexts(item["question"])
        rows["contexts"].append(contexts or [""])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


if __name__ == "__main__":
    print("=== RAGAS 评测开始（后端需在 8000 端口运行）===")
    dataset = build_dataset()
    print("=== 计算指标 ===")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("=== 结果 ===")
    print(result)
    result.to_pandas().to_csv("ragas_result.csv", index=False, encoding="utf-8-sig")
    print("已保存 ragas_result.csv")
