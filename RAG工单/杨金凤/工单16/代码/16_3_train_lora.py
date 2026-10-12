# 16_3_train_lora.py - 加 labels 版
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_DIR = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
DATA_DIR = "/root/autodl-tmp/ragflow/finetune_data"
OUTPUT_DIR = "/root/autodl-tmp/ragflow/finetune_output"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

MAX_IMAGE_SIZE = 768


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


class IMDRDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(os.path.join(DATA_DIR, item["images"][0])).convert("RGB")
        image = resize_image(image)

        user_text = item["messages"][0]["content"].replace("<image>", "").strip()
        answer = item["messages"][1]["content"]

        # 用户部分（不加 generation prompt，因为要拼 assistant 答案）
        messages_user = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": user_text}]},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages_user, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + answer

        inputs = self.processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
        # 不做 squeeze，保留 [1, ...] 维度，collate_fn 会处理

        # labels = input_ids，屏蔽 prompt 部分
        prompt_ids = self.processor(text=[prompt_text], return_tensors="pt")["input_ids"][0]
        labels = inputs["input_ids"].clone()
        labels[:, :len(prompt_ids)] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def collate_fn(batch):
    # batch_size=1，直接返回第一个样本，避免 default_data_collator 破坏 image_grid_thw 形状
    return batch[0]


def main():
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = IMDRDataset(TRAIN_FILE, processor)
    val_ds = IMDRDataset(VAL_FILE, processor)
    print(f"训练集 {len(train_ds)} 条，验证集 {len(val_ds)} 条")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, "lora_final")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"\n✅ LoRA 权重已保存到：{save_path}")


if __name__ == "__main__":
    main()
