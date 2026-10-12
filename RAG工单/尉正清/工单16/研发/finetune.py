# 工单16：人工智能NLP-RAG-微调专用视觉语言模型
"""Qwen2-VL-2B 的 LoRA 微调（4-bit QLoRA，训练循环自己写）

## 为什么不用 Trainer / trl

本机 GPU 只有 **6GB**（RTX 3060 Laptop），而且这个项目的环境有个老毛病：
深度学习库的版本互相锁（见工单14/11 的过程记录）。`trl` 会连带把
transformers 的版本要求拉进来，风险不值当。训练循环本身不复杂 ——
前向、反向、梯度累积、按步打印 loss，一百来行，全部可控。

## 视觉塔一定要训（本工单最关键的结论）

配置里的 `lora.vision_lora` 控制视觉塔要不要一起挂 LoRA。**必须开**。

把它冻死的第一版训完之后，分选项看准确率是这样的：

| 标准答案 | A | B | C | D |
|---|---|---|---|---|
| 微调前 | 25% | 69% | 68% | 28% |
| 视觉塔冻死 | 25% | **85%** | 62% | **0%** |

B 涨了、D 掉到 0% —— 模型学到的是「答案多半在 B」的**位置先验**，根本没在读图。
把训练集的选项打乱之后，这份「提升」立刻归零（+3.31% → +0.00%）。
只有让视觉塔也可训，增益才是真的（见 `测试/评估报告.md` 第三节）。

## 加载方式

    conda activate rag_gd2
    python finetune.py --config finetune_config.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 6GB 显存跑 LoRA 会贴着上限，PyTorch 默认的缓存分配器容易碎片化：
# 实测跑到第 40 步时报 "memory allocation failed with OOM ... while trying to
# allocate 564MB (free: 0)"，之后每步从 96 秒掉到 336 秒（一直在重试分配）。
# expandable_segments 让分配器按需扩展段、允许拆分移动，能显著缓解碎片。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).parent


# ---------------------------------------------------------------- 数据

def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_example(processor, row, max_length):
    """把 (图, 问题, 答案) 变成模型输入，并把问题部分从 loss 里屏蔽掉。

    只让模型对**答案**负责 —— 否则它会去学怎么复述题面。
    """
    from PIL import Image

    img = Image.open(row["image"]).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["question"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
    ]

    # 只含问题和图片的版本，用来算出「答案从第几个 token 开始」
    prompt_msgs = messages[:-1]
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False,
                                                add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)

    enc = processor(text=[full_text], images=[img], return_tensors="pt",
                    padding=False, truncation=True, max_length=max_length)
    prompt_len = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_len), enc["input_ids"].shape[1])

    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100          # 题面与图像 token 不计 loss
    enc["labels"] = labels

    # 只有这几个字段带 batch 维（恒为 1），压掉。
    # **pixel_values 不能压** —— 它是 [总 patch 数, C*T*P*P]，本来就没有 batch 维，
    # 压一下就把 2944 个 patch 砍成 1 个，视觉塔会报
    # "split_with_sizes expects split_sizes to sum exactly to 1"。
    # image_grid_thw（[图数,3]）同理不能压，压成 [3] 会解不出 (t,h,w)。
    squeeze = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
    return {k: (v[0] if k in squeeze and hasattr(v, "shape") and v.dim() > 0 else v)
            for k, v in enc.items()}


class JsonlDataset:
    """惰性加载：图像在 __getitem__ 里才读。

    一上来就把 600 张图全解码进内存会吃掉好几个 G —— 6GB 显存的机器上
    再去跟 GPU 抢内存，很容易 OOM。
    """

    def __init__(self, rows, processor, max_length):
        self.rows, self.processor, self.max_length = rows, processor, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return build_example(self.processor, self.rows[i], self.max_length)


class Collator:
    """把一批样本 padding 到等长。"""

    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, batch):
        import torch

        maxlen = max(len(b["input_ids"]) for b in batch)
        keys = ["input_ids", "labels", "attention_mask", "mm_token_type_ids"]
        pad_value = {"labels": -100}
        padded = {k: [] for k in keys if k in batch[0]}
        for b in batch:
            n = maxlen - len(b["input_ids"])
            for k in padded:
                v = b[k]
                fill = pad_value.get(k, self.pad if k != "attention_mask" else 0)
                padded[k].append(torch.cat([v, torch.full((n,), fill, dtype=v.dtype)]))

        out = {k: torch.stack(v) for k, v in padded.items()}

        # Qwen2-VL 的 pixel_values 是**把一批里所有图展平后拼在一起**的
        # （形状 [总 patch 数, C*T*P*P]），image_grid_thw 逐图描述网格。
        # 这两个都没有 batch 维，所以用 cat 而不是 stack。
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out


# ---------------------------------------------------------------- 模型

def load_model(cfg):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    m = cfg["model"]
    processor = AutoProcessor.from_pretrained(
        m["base"],
        min_pixels=cfg["image"]["min_pixels"],
        max_pixels=cfg["image"]["max_pixels"],
    )

    kwargs = {"attn_implementation": m.get("attn_implementation", "sdpa"),
              "torch_dtype": torch.bfloat16}
    if m.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, m.get("bnb_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = Qwen2VLForConditionalGeneration.from_pretrained(m["base"], **kwargs)

    # 视觉塔的属性路径随 transformers 版本变：4.x 是 model.visual，5.x 是 model.model.visual
    visual = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise SystemExit("找不到视觉塔（model.visual / model.model.visual 都没有）")
    n_vis = sum(p.numel() for p in visual.parameters())

    lora = cfg["lora"]
    targets = list(lora["target_modules"])

    if lora.get("vision_lora"):
        # 视觉塔也挂 LoRA。peft 的 target_modules 传**列表时按叶子名精确匹配**
        # （传字符串才会当正则），所以这里直接给叶子名 ——
        # `qkv` / `proj` 只有视觉塔的注意力层用，不会误伤语言侧
        # （语言侧叫 q_proj / o_proj）。patch_embed.proj 也会被带上，
        # 那是个 3×1280 的切图卷积，参数量可忽略。
        targets += list(lora.get("vision_modules", ["qkv", "proj"]))
        print(f"  视觉塔可训（{n_vis / 1e6:.1f}M 参数，注意力投影挂 LoRA）")
    else:
        for p in visual.parameters():
            p.requires_grad = False
        print(f"  视觉塔已冻结（{n_vis / 1e6:.1f}M 参数）")

    if m.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True))

    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    ))

    # 确认 LoRA 挂到了哪些模块（视觉塔有没有真的被覆盖）
    named = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    in_vis = [n for n in named if ".visual." in n]
    print(f"  LoRA 模块总数 {len(named)}，其中视觉塔 {len(in_vis)} 个")
    if lora.get("vision_lora") and not in_vis:
        raise SystemExit("配置要求训视觉塔，但 LoRA 一个都没挂上去 —— 检查 vision_modules 的匹配规则")
    return model, processor


# ---------------------------------------------------------------- 训练

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "finetune_config.json"))
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟测试）")
    ap.add_argument("--max-steps", type=int, default=0, help="跑够 N 步就停（冒烟测试）")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    import torch
    from torch.optim import AdamW

    t_train = cfg["train"]
    torch.manual_seed(t_train["seed"])

    print("=== 加载模型 ===", flush=True)
    t0 = time.time()
    model, processor = load_model(cfg)
    model.print_trainable_parameters()
    print(f"  加载耗时 {time.time() - t0:.0f}s", flush=True)

    print("\n=== 准备数据 ===", flush=True)
    rows = read_jsonl(HERE / cfg["data"]["train"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"  训练样本 {len(rows)} 条", flush=True)

    collate = Collator(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
    loader = torch.utils.data.DataLoader(
        JsonlDataset(rows, processor, t_train["max_length"]),
        batch_size=t_train["batch_size"], shuffle=True, collate_fn=collate)

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=t_train["lr"], weight_decay=t_train["weight_decay"])
    steps_per_epoch = max(1, len(loader) // t_train["grad_accum"])
    total_steps = steps_per_epoch * t_train["epochs"]
    warmup = max(1, int(total_steps * t_train.get("warmup_ratio", 0)))

    def lr_at(step):
        if step < warmup:
            return t_train["lr"] * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return t_train["lr"] * max(0.0, 1 - prog)     # 线性衰减

    print(f"\n=== 开始训练 ===\n  每轮 {steps_per_epoch} 步 × {t_train['epochs']} 轮 "
          f"= {total_steps} 步（batch {t_train['batch_size']} × 累积 {t_train['grad_accum']}）",
          flush=True)

    logs, gstep, t0 = [], 0, time.time()
    model.train()
    for epoch in range(t_train["epochs"]):
        run_loss, n_run = 0.0, 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / t_train["grad_accum"]
            loss.backward()
            run_loss += out.loss.item()
            n_run += 1

            if (i + 1) % t_train["grad_accum"] == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], t_train["max_grad_norm"])
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                if gstep % t_train["logging_steps"] == 0 or gstep == 1:
                    avg = run_loss / n_run
                    logs.append({"step": gstep, "epoch": epoch + 1, "loss": round(avg, 4),
                                 "lr": round(lr_at(gstep), 8), "sec": round(time.time() - t0, 1)})
                    print(f"  epoch {epoch + 1}  step {gstep:4d}/{total_steps}  "
                          f"loss {avg:.4f}  lr {lr_at(gstep):.2e}  "
                          f"已用 {time.time() - t0:.0f}s", flush=True)
                    run_loss, n_run = 0.0, 0

                if t_train.get("save_steps") and gstep % t_train["save_steps"] == 0:
                    model.save_pretrained(HERE / cfg["output"])
                    print(f"    [存档] step {gstep}", flush=True)

                if args.max_steps and gstep >= args.max_steps:
                    print(f"  [冒烟测试] 跑够 {args.max_steps} 步，提前结束", flush=True)
                    break
        else:
            continue          # 内层正常跑完 → 进下一轮
        break                 # 内层是 break 出来的 → 结束训练

    out_dir = HERE / cfg["output"]
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (HERE / "results_train_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=1), encoding="utf-8")

    first = logs[0]["loss"] if logs else None
    last = logs[-1]["loss"] if logs else None
    print(f"\n=== 训练完成 ===\n  adapter 存到 {out_dir}")
    print(f"  首/末记录 loss：{first} -> {last}")
    print(f"  总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
