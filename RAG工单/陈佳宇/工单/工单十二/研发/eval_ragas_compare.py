# eval_ragas_compare.py 工单12 RAGAS评估脚本，对比两套RAG
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision


def load_batch_result(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(records, answer_key):
    """组装ragas数据集格式
    answer_key可选：ans_normal_rag / ans_lightrag
    """
    samples = []
    for r in records:
        samples.append({
            "question": r["question"],
            "answer": r[answer_key],
            "contexts": [r[answer_key]],
            "ground_truth": ""
        })
    return Dataset.from_list(samples)


if __name__ == "__main__":
    records = load_batch_result("./batch_query_result.json")

    print("===== 普通Naive-RAG 评估开始 =====")
    ds_normal = build_dataset(records, "ans_normal_rag")
    result_normal = evaluate(
        dataset=ds_normal,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision]
    )
    print(result_normal)

    print("\n===== LightRAG混合图谱RAG 评估开始 =====")
    ds_light = build_dataset(records, "ans_lightrag")
    result_light = evaluate(
        dataset=ds_light,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision]
    )
    print(result_light)

    out_data = {
        "normal_rag": result_normal.to_dict(),
        "lightrag": result_light.to_dict()
    }
    with open("./ragas_compare_result.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print("\n✅评估指标保存至 ragas_compare_result.json")
