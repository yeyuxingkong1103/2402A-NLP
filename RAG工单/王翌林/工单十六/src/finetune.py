# -*- coding: utf-8 -*-
"""
工单十六：Qwen2-VL LoRA 微调脚本（自包含，基于 transformers + peft）
功能：加载基座模型 → 构建图文 SFT 数据 → LoRA 微调 → 保存适配器。
显存友好：4bit 量化 / fp16 / gradient checkpointing / 冻结视觉塔 / batch=1。
"""
import os, sys, json, argparse, math
from pathlib import Path
from loguru import logger
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_DIR = "/home/dabaie/models/Qwen2-VL-2B-Instruct"
IMAGE_ROOT = "/home/dabaie/code/工单/工单十六/data/images"
DATA_DIR = "/home/dabaie/code/工单/工单十六/data"
OUTPUT_DIR = "/home/dabaie/code/工单/工单十六/finetune_output"

MAX_PIXELS = 512 * 512  # 控制图像分辨率，降低显存
IGNORE_INDEX = -100


def make_collator(processor):
    """自定义 collator：用 tokenizer 补齐文本，透传 pixel_values/image_grid_thw。"""
    tokenizer = processor.tokenizer

    def collate(features):
        # 文本字段补齐
        text_keys = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")
        to_pad = {k: [f[k] for f in features] for k in text_keys if k in features[0]}
        padded = tokenizer.pad(to_pad, padding=True, return_tensors="pt")
        # 视觉字段：batch=1 时直接堆叠
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "image_attention_mask"):
            vals = [f[k] for f in features if k in f]
            if vals:
                padded[k] = torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
        return padded

    return collate


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class VlmSFTDataset(Dataset):
    def __init__(self, records, processor, image_root, training=True):
        self.records = records
        self.processor = processor
        self.image_root = image_root
        self.training = training

    def __len__(self):
        return len(self.records)

    def _build_messages(self, rec):
        q = rec["question"]
        a = rec["answer"]
        content = []
        if rec.get("image"):
            content.append({"type": "image"})
        content.append({"type": "text", "text": q})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": a}]},
        ]
        return messages

    def __getitem__(self, idx):
        rec = self.records[idx]
        processor = self.processor
        messages = self._build_messages(rec)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        images = []
        if rec.get("image"):
            img_path = os.path.join(self.image_root, rec["image"])
            if os.path.isfile(img_path):
                im = Image.open(img_path).convert("RGB")
                # 限制图像最大边长，控制显存与内存
                max_side = 384
                w, h = im.size
                if max(w, h) > max_side:
                    s = max_side / max(w, h)
                    im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
                images.append(im)

        batch = processor(
            text=[text], images=images if images else None,
            padding=True, return_tensors="pt",
        )
        input_ids = batch["input_ids"][0]
        labels = input_ids.clone()
        # 仅对 assistant 回答部分计算 loss：找到最后一轮回答起始
        # 简化处理：mask 掉 prompt（question 及之前），保留 answer
        prompt_text = processor.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_ids = processor(text=[prompt_text], images=images if images else None,
                               return_tensors="pt")["input_ids"][0]
        plen = len(prompt_ids)
        if plen < len(labels):
            labels[:plen] = IGNORE_INDEX
        else:
            labels[:] = IGNORE_INDEX
        item = {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"][0],
            "labels": labels,
        }
        if "mm_token_type_ids" in batch:
            item["mm_token_type_ids"] = batch["mm_token_type_ids"][0]
        # 视觉字段保留完整张量（不做 [0]，否则破坏 patch/grid 维度）
        if "pixel_values" in batch:
            item["pixel_values"] = batch["pixel_values"]
            if "image_grid_thw" in batch:
                item["image_grid_thw"] = batch["image_grid_thw"]
            if "image_attention_mask" in batch:
                item["image_attention_mask"] = batch["image_attention_mask"]
        return item


def build_model(use_4bit=True):
    logger.info(f"加载模型: {MODEL_DIR}")
    kwargs = dict(torch_dtype=torch.float16, device_map="cuda")
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("使用 4bit 量化加载")
        except Exception as e:
            logger.warning(f"4bit 不可用，回退 fp16: {e}")
            use_4bit = False
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # 冻结视觉编码器
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        logger.info("已冻结视觉编码器 visual")
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, processor, use_4bit


def build_lora(model, use_4bit):
    target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_rows = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    eval_rows = load_jsonl(os.path.join(DATA_DIR, "eval.jsonl"))
    if args.max_samples > 0:
        train_rows = train_rows[:args.max_samples]
    logger.info(f"训练样本 {len(train_rows)} | 评估样本 {len(eval_rows)}")

    model, processor, use_4bit = build_model(use_4bit=not args.no_4bit)
    model = build_lora(model, use_4bit)

    train_ds = VlmSFTDataset(train_rows, processor, IMAGE_ROOT, training=True)
    eval_ds = VlmSFTDataset(eval_rows, processor, IMAGE_ROOT, training=False)

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0,
        logging_steps=2,
        save_strategy="no",
        eval_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    collator = make_collator(processor)

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    logger.info("开始微调...")
    trainer.train()
    final_loss = trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None
    logger.info(f"训练完成，最后 loss={final_loss}")

    # 保存 LoRA 适配器
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    # 记录训练曲线
    with open(os.path.join(OUTPUT_DIR, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    logger.info(f"适配器已保存到 {adapter_dir}")
    print(f"FINAL_LOSS={final_loss}")


if __name__ == "__main__":
    main()
