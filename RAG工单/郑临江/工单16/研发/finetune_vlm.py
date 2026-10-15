"""工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单。数据转换、LoRA配置和启动命令。"""

import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ROOT / "original_problems/documents"
MODEL = ROOT / "models/MinerU2.5-1.2B"


def prepare():
    from train_lora import prepare as source_prepare

    datasets = source_prepare()
    DATA.mkdir(exist_ok=True)
    for name, rows in datasets.items():
        out = [
            {
                "image": row["image"],
                "question": row["question"],
                "answer": row["answer"],
                "reference": row["reference"],
                "document": row["document"],
                "pdf_page": row["pdf_page"],
                "options": row["options"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row["image"]},
                            {
                                "type": "text",
                                "text": row["question"]
                                + "\n"
                                + "\n".join(row["options"]),
                            },
                        ],
                    },
                    {"role": "assistant", "content": row["answer"]},
                ],
            }
            for row in rows
        ]
        (DATA / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
            encoding="utf-8",
        )
    (DATA / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "splits": {k: len(v) for k, v in datasets.items()},
                "model": str(MODEL),
                "format": "实际PNG图像+选择题messages",
                "split_unit": "专利文档，无交叉",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_config():
    cfg = {
        "model_name_or_path": str(MODEL),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_seq_length": 2048,
        "save_steps": 50,
        "output_dir": str(ROOT / "models/industrial-vlm-lora"),
    }
    (ROOT / "data/lora_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cfg


if __name__ == "__main__":
    prepare()
    print(write_config())
