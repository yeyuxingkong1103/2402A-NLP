"""工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单。
真实Qwen2-VL(MinerU) LoRA；专利分组隔离数据，不训练评估问题。
"""

import json, random, re, time, hashlib, os
from pathlib import Path
import torch
import pymupdf
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/finetune"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "models/MinerU2.5-1.2B"


def prepare():
    rows = [
        json.loads(x)
        for x in (ROOT / "original_problems/questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    grouped = {}
    for row in rows:
        if row["document"] in {"CN100342976C.pdf", "CN100347506C.pdf"}:
            continue
        grouped.setdefault(row["document"], []).append(row)
    names = list(grouped)
    random.Random(42).shuffle(names)
    datasets = {}
    for split, docs in [
        ("train", names[:10]),
        ("validation", names[10:12]),
        ("test", names[12:16]),
    ]:
        result = []
        for name in docs:
            # 每篇选1文本题和1图文题，数据本来就是四选一，不编造开放答案。
            chosen = [
                next(
                    (r for r in grouped[name] if r.get("group") == g), grouped[name][0]
                )
                for g in [1, 2]
            ]
            with pymupdf.open(ROOT / "original_problems/documents" / name) as pdf:
                for row in chosen:
                    m = re.search(r"第\s*(\d+)\s*页", row["question"])
                    page = min(len(pdf), int(m.group(1))) if m else 1
                    image = OUT / f"{Path(name).stem}-p{page}.png"
                    pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(image)
                    result.append(
                        {
                            **row,
                            "image": str(image.relative_to(ROOT)),
                            "pdf_page": page,
                            "reference": next(
                                (
                                    o[3:].strip()
                                    for o in row["options"]
                                    if o.startswith(row["answer"] + ".")
                                ),
                                row["answer"],
                            ),
                        }
                    )
        datasets[split] = result
        (OUT / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in result) + "\n",
            encoding="utf-8",
        )
    return datasets


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data = prepare()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True, min_pixels=56 * 56, max_pixels=224 * 224
    )
    processor.tokenizer.padding_side = "right"

    def encode(row, training=False):
        prompt = (
            row["question"]
            + "\n"
            + "\n".join(row["options"])
            + "\n仅回答正确选项的大写字母A、B、C或D。"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(ROOT / row["image"]).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        if training:
            full = processor(
                text=[text + row["answer"] + processor.tokenizer.eos_token],
                images=[image],
                return_tensors="pt",
            )
            labels = full["input_ids"].clone()
            labels[:, : inputs["input_ids"].shape[1]] = -100
            full["labels"] = labels
            inputs = full
        return {k: v.to("cuda") for k, v in inputs.items()}

    def evaluate(rows):
        model.eval()
        results = []
        for row in rows:
            with torch.inference_mode():
                inputs = encode(row)
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            decoded = processor.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            match = re.search(r"[ABCD]", decoded)
            pred = match.group() if match else None
            results.append(
                {
                    "document": row["document"],
                    "question": row["question"],
                    "group": row.get("group"),
                    "expected": row["answer"],
                    "prediction": pred,
                    "raw_output": decoded,
                    "correct": pred == row["answer"],
                }
            )
        return {
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "count": len(results),
            "samples": results,
            "terminology_accuracy": sum(
                r["correct"] for r in results if r["group"] == 1
            )
            / max(1, sum(r["group"] == 1 for r in results)),
            "drawing_reasoning_accuracy": sum(
                r["correct"] for r in results if r["group"] == 2
            )
            / max(1, sum(r["group"] == 2 for r in results)),
        }

    before = evaluate(data["test"])
    print("BASELINE", before["accuracy"], flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    logs = []
    for epoch in range(1, 3):
        model.train()
        losses = []
        for index, row in enumerate(data["train"]):
            batch = encode(row, True)
            loss = model(**batch).loss
            (loss / 4).backward()
            losses.append(float(loss.detach().cpu()))
            if (index + 1) % 4 == 0 or index + 1 == len(data["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                opt.zero_grad(set_to_none=True)
            print("train", epoch, index, float(loss.detach().cpu()), flush=True)
        logs.append(
            {"epoch": epoch, "mean_loss": sum(losses) / len(losses), "losses": losses}
        )
    model.save_pretrained(ROOT / "models/industrial-vlm-lora")
    processor.save_pretrained(ROOT / "models/industrial-vlm-lora")
    model.config.use_cache = True
    after = evaluate(data["test"])
    report = {
        "model": "MinerU2.5-1.2B/Qwen2VL",
        "method": "真实LoRA q_proj/v_proj rank8 alpha16",
        "epochs": 2,
        "device": torch.cuda.get_device_name(0),
        "splits": {k: len(v) for k, v in data.items()},
        "document_leakage": False,
        "train_documents": sorted({r["document"] for r in data["train"]}),
        "test_documents": sorted({r["document"] for r in data["test"]}),
        "before": before,
        "after": after,
        "training": logs,
        "measurable_terminology_improvement": after["terminology_accuracy"]
        - before["terminology_accuracy"],
        "limits": "小样本专利分组隔离测试；图像降分辨率适配8GB显存；基座为文档VLM，未保证提升超过5%。不能用训练集指标代替测试。",
    }
    (ROOT.parent / "测试/真实微调结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINISHED", before["accuracy"], after["accuracy"], flush=True)


if __name__ == "__main__":
    main()
