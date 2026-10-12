"""转 LLaMA-Factory ShareGPT 格式"""
import os
import json
from sklearn.model_selection import train_test_split

IN_PATH = "/root/autodl-tmp/ragflow/finetune_data/intermediate_qa.jsonl"
OUT_DIR = "/root/autodl-tmp/ragflow/finetune_data"


def convert(item):
    return {
        "messages": [
            {"role": "user", "content": f"<image>{item['question']}"},
            {"role": "assistant", "content": item["answer"]},
        ],
        "images": [item["image"]],
    }


def main():
    data = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(convert(json.loads(line)))

    train, val = train_test_split(data, test_size=0.1, random_state=42)

    with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for d in train:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
        for d in val:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    dataset_info = {
        "imdr_visual": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练集 {len(train)} 条，验证集 {len(val)} 条")
    print(f"   输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
