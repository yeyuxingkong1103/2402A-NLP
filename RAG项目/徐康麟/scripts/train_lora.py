#!/usr/bin/env python3
"""LoRA 微调小基座（peft + transformers，**不需要 LLaMA-Factory**）。

为什么要微调（B 方案，用户裁决）：P8 的天花板是**法条级精度** —— 通用重排 + 向量检索排不进
top-5，而"再调参"已被 v3c 证伪。微调的目标**不是灌知识**（知识在检索里），而是教模型：
只依据给定条文作答 / 带条号引用 / 没据就说没据。

关键实现点（都是坑）：
* **只在答案段算 loss**（``legal_rag/sft.py::encode_chat`` 掩掉提示词 token），否则在学复述；
* LoRA 用 ``target_modules="all-linear"``（对 Qwen 系稳妥），bf16；
* 训练数据必须来自 ``scripts/build_sft_data.py``（已过"引用有据 + 不与评测集重叠"两关）。

用法（云端，vLLM 已停以腾显存）：
    python scripts/train_lora.py --model /root/autodl-tmp/models/qwen3-4b \
        --train data/sft/train.jsonl --val data/sft/val.jsonl --out /root/autodl-tmp/lora/legal-v1 \
        --epochs 3 --lr 1e-4 --lora-r 16 --batch 1 --grad-accum 8

起服务（vLLM 直接挂 LoRA，便于跑评测）：
    vllm serve <base> --enable-lora --lora-modules legal=/path/to/adapter ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.sft import encode_chat, load_jsonl, supervision_summary  # noqa: E402
from legal_rag.sft_report import StepLogger  # noqa: E402


def iter_rows(dataset, limit: int = 0):
    """逐行取样本，**兼容 datasets 的两种行为**（真机踩到）。

    * 迭代 ``Dataset`` 本身 ⇒ 每项是 dict（``{"input_ids": [...], "labels": [...]}``）；
    * 但 ``dataset[:n]`` 在部分版本里返回的是 **dict-of-lists**：此时迭代它拿到的是
      **键字符串**（``"input_ids"``/``"labels"``），``row["labels"]`` 会炸
      ``TypeError: string indices must be integers`` —— 训练路径第一次真跑就是这么挂的
      （dry-run 走不到这段，所以一直没暴露）。

    统一走 ``select`` + 迭代，两个版本都对。
    """
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
    for row in dataset:
        yield row


def build_dataset(paths: list[Path], tokenizer, *, max_len: int):
    """读 JSONL → tokenize（只在答案上学）。torch 版 Dataset 在这里才 import。"""
    from datasets import Dataset

    rows: list[dict] = []
    skipped = 0
    for path in paths:
        if not path.is_file():
            print(f"  [跳过] 数据文件不存在：{path}")
            continue
        for record in load_jsonl(path):
            messages = record.get("messages") or []
            try:
                encoded = encode_chat(messages, tokenizer, max_len=max_len)
            except (ValueError, KeyError, TypeError) as exc:
                skipped += 1
                print(f"  [跳过] 样本不可用（{type(exc).__name__}: {exc}）：{str(messages)[:80]}")
                continue
            rows.append(encoded)
    print(f"  可用样本 {len(rows)} 条（跳过 {skipped}）")
    return Dataset.from_list(rows)


def make_chunked_loss(chunk_tokens: int):
    """返回一个 HF Trainer 的 ``compute_loss_func``：把 logits 按序列**分块**算交叉熵。

    为什么必须分块（2026-09-28 实测两次 OOM）：`cross_entropy` 拿到
    ``[1, L, 151936]`` 的 logits 后会展开 fp32 中间量，单次申请 **17.79 GiB**；

    * `--max-len 8192`、无分块 ⇒ 直接 OOM；
    * `--max-len 6144`、无分块 ⇒ 前 114 步正常（峰值 19.55 GB），到某一步仍请求 17.79 GiB ⇒ OOM。

    也就是说**窗口上限由 vocab logits 决定，不由激活显存决定**。分块后峰值 ≈
    ``chunk_tokens × vocab × 4`` 字节（512 token 时约 0.31 GB），长度再也不影响损失显存。

    数值上与默认实现等价：对每个非 ignore 位置求 CE，最后除以**监督 token 总数**
    （与 `CrossEntropyLoss(reduction="mean")` 同一口径，梯度累积下的缩放仍由 Trainer 负责）。
    """
    def _chunked_loss(outputs, labels, num_items_in_batch=None, ignore_index=-100, **kwargs):  # noqa: ANN001
        import torch                                  # noqa: PLC0415
        import torch.nn.functional as functional      # noqa: PLC0415

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        vocab = shift_logits.size(-1)
        total = None
        count = 0
        for start in range(0, shift_labels.shape[1], chunk_tokens):
            piece_logits = shift_logits[:, start:start + chunk_tokens, :]
            piece_labels = shift_labels[:, start:start + chunk_tokens]
            piece = functional.cross_entropy(
                piece_logits.reshape(-1, vocab), piece_labels.reshape(-1),
                ignore_index=ignore_index, reduction="sum")
            total = piece if total is None else total + piece
            count += int((piece_labels != ignore_index).sum())
        if total is None or count == 0:
            # 一个监督 token 都没有：返回 0，但**保持计算图**（否则 backward 会报错）
            return logits.sum() * 0.0
        return total / count

    return _chunked_loss


def make_collator(pad_token_id: int):
    """批内右填充；label 的 pad 用 -100（不参与 loss）。"""
    def _collate(batch: list[dict]) -> dict:
        import torch

        width = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, attention = [], [], []
        for row in batch:
            pad = width - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [pad_token_id] * pad)
            labels.append(list(row["labels"]) + [-100] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention)}
    return _collate


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT（法律 RAG 小基座）")
    parser.add_argument("--model", required=True, help="基座模型目录（本地路径）")
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--val", default="data/sft/val.jsonl")
    parser.add_argument("--out", required=True, help="adapter 输出目录")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-samples", type=int, default=0, help="调试用：只取前 N 条")
    parser.add_argument("--max-trimmed-share", type=float, default=0.10,
                        help="允许「训练时裁掉证据」的样本占比上限；超了就拒绝训练（exit 4）")
    parser.add_argument("--max-starved-share", type=float, default=0.05,
                        help="允许「只剩 <=1 个监督 token」的样本占比上限；超了拒绝训练（exit 5）")
    parser.add_argument("--ignore-supervision-gate", action="store_true",
                        help="明知提示会被裁/监督会丢，仍要训练（会写进报告）")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="用时间换显存；max-len 拉大时基本必须开")
    parser.add_argument("--chunked-loss-tokens", type=int, default=0,
                        help="把交叉熵按这个 token 数分块（0=不分块）。长序列必开："
                             "不分块时 cross_entropy 会为 L×vocab 的 logits 申请十几 GB")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读数据 + 打印统计，不加载模型（本地也能跑）")
    args = parser.parse_args()

    train_path, val_path = Path(args.train), Path(args.val)
    if not train_path.is_file():
        print(f"!! 训练数据不存在：{train_path}（先跑 scripts/build_sft_data.py）", file=sys.stderr)
        return 2
    if args.dry_run:
        rows = load_jsonl(train_path)
        val_rows = load_jsonl(val_path) if val_path.is_file() else []
        print(f"dry-run：train {len(rows)} 条 / val {len(val_rows)} 条")
        if rows:
            sample = rows[0]
            print(f"  样例 roles={[m.get('role') for m in sample.get('messages', [])]}")
            print(f"  样例问题：{str(sample.get('messages', [{}])[1].get('content'))[:60]}")
            print(f"  证据条数：{len(sample.get('evidence') or [])}")
        return 0

    import torch                                  # noqa: PLC0415 - 只有真训练才需要
    from peft import LoraConfig, get_peft_model   # noqa: PLC0415
    from transformers import (                    # noqa: PLC0415
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    )

    started = time.perf_counter()
    print(f"基座：{args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    print("准备数据：")
    train_ds = build_dataset([train_path], tokenizer, max_len=args.max_len)
    val_ds = build_dataset([val_path], tokenizer, max_len=args.max_len) if val_path.is_file() else None
    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        print(f"  （--max-samples 生效：只训 {len(train_ds)} 条）")

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lora)
    if args.grad_checkpointing:
        # LoRA 场景要这一句：只有 adapter 可训练，输入不带 grad 时 checkpointing 会失效
        model.enable_input_require_grads()
        print("梯度检查点：开（用时间换显存，长序列才装得下）")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数 {trainable:,} / 总参数 {total:,}（{trainable / max(total, 1):.4%}）")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        # ⚠️ 评估批大小必须显式给 1：默认 8 时 eval 会把 `[8, L, 151936]` 的 logits
        # 交给 accelerate 转 fp32（`convert_to_fp32`），8192 窗口下要 **34 GiB** ⇒ OOM。
        # 2026-09-28 因为这个默认值连崩三次，而且都发生在**epoch 边界**（看起来像训练崩了）。
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=args.grad_checkpointing,
        seed=args.seed,
        report_to=[],
        eval_strategy="no" if val_ds is None else "epoch",
    )
    # ---- 监控留痕（用户要求："微调记得监控、记录数据"）----
    from transformers import TrainerCallback  # noqa: PLC0415

    avg_tokens = (sum(len(row["input_ids"]) for row in iter_rows(train_ds))
                  / max(len(train_ds), 1)) if len(train_ds) else args.max_len
    # ---- 监督信号自检（2026-09-28 事故的闸门）----
    # 事故：证据进提示后提示平均 3032 token，而 max_len=2048，旧代码右截断把答案切光，
    # 912 条里 514 条只剩 ≤1 个监督 token —— 白训 20 分钟还看不出来。
    sup = supervision_summary(iter_rows(train_ds))
    if len(train_ds):
        print(f"[自检] 监督信号：中位 {sup['supervised_median']} token / p05 {sup['supervised_p05']}；"
              f"只剩 <=1 个监督 token 的样本 {sup['starved_rows']}/{sup['rows']}"
              f"（{sup['starved_share']:.1%}）；"
              f"训练时裁掉证据的样本 {sup['prompt_trimmed_rows']}/{sup['rows']}"
              f"（{sup['prompt_trimmed_share']:.1%}）")
        if not args.ignore_supervision_gate:
            if sup["starved_share"] > args.max_starved_share:
                print(f"!! 监督信号被截断吃掉：{sup['starved_share']:.1%} 的样本只剩 <=1 个监督 "
                      f"token（上限 {args.max_starved_share:.1%}）。这会白烧 GPU："
                      f"先加大 --max-len，或修 encode_chat 的截断策略。", file=sys.stderr)
                return 5
            if sup["prompt_trimmed_share"] > args.max_trimmed_share:
                print(f"!! 训练提示被裁：{sup['prompt_trimmed_share']:.1%} 的样本裁掉了证据"
                      f"（上限 {args.max_trimmed_share:.1%}）—— 训练形状与推理形状不一致，"
                      f"学到的东西不能指望在推理时用上。加大 --max-len，或显式传 "
                      f"--ignore-supervision-gate 并把这个事实写进报告。", file=sys.stderr)
                return 4
    supervised = float(sup.get("supervised_mean") or 0.0)

    def _vram_gb() -> float | None:
        try:
            return torch.cuda.max_memory_allocated() / 1024 ** 3
        except Exception:  # noqa: BLE001 - 读显存失败不该影响训练
            return None

    step_logger = StepLogger(out_dir / "train_log.jsonl", batch_size=args.batch,
                             grad_accum=args.grad_accum, avg_tokens=int(avg_tokens),
                             vram_fn=_vram_gb)

    class _Monitor(TrainerCallback):
        """薄薄一层：把 Trainer 的 log 事件交给 :class:`StepLogger` 落盘。"""

        def on_log(self, _args, state, _control, logs=None, **kwargs):  # noqa: ANN001
            if logs:
                step_logger.record(step=state.global_step, epoch=state.epoch, logs=dict(logs))
            return _control

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=make_collator(tokenizer.pad_token_id),
                      compute_loss_func=(make_chunked_loss(args.chunked_loss_tokens)
                                         if args.chunked_loss_tokens > 0 else None),
                      callbacks=[_Monitor()])
    result = trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    curve = step_logger.summary()
    try:
        import peft as _peft
        import transformers as _tf

        env = {"torch": torch.__version__, "transformers": _tf.__version__,
               "peft": getattr(_peft, "__version__", "?"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    except Exception as exc:  # noqa: BLE001 - 环境信息拿不到不该让训练失败
        env = {"error": f"{type(exc).__name__}: {exc}"}

    report = {
        "base_model": args.model,
        "train_samples": len(train_ds),
        "val_samples": 0 if val_ds is None else len(val_ds),
        "epochs": args.epochs, "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "max_len": args.max_len,
        "avg_tokens_per_sample": round(avg_tokens, 1),
        "avg_supervised_tokens": round(supervised, 1),
        # 监督信号自检的完整留痕（截断事故的现场证据）
        "supervision": sup,
        "gradient_checkpointing": bool(args.grad_checkpointing),
        "trainable_params": trainable, "total_params": total,
        "train_loss": getattr(result, "training_loss", None),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        # 以下三项来自**实测留痕**（train_log.jsonl），不是估计
        "peak_vram_gb": curve.get("peak_vram_gb"),
        # 注：早期 run 的 tokens_per_second 是"累计 token / 本区间秒数"的假口径（偏虚高），
        # 端到端口径 tokens_per_second_overall = 总 token / 总墙钟 才跨 run 可比。
        "tokens_per_second": curve.get("tokens_per_second_mean"),
        "tokens_per_second_overall": curve.get("tokens_per_second_overall"),
        "throughput_points_skipped": curve.get("throughput_points_skipped"),
        "curve": curve,
        "env": env,
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 训练完成 ===")
    for key, value in report.items():
        if key != "curve":
            print(f"  {key}: {value}")
    print(f"  逐步日志：{out_dir / 'train_log.jsonl'}（{curve.get('points')} 个记录点）")
    print(f"adapter 已存：{out_dir}")
    print("下一步：vllm serve <base> --enable-lora --lora-modules legal=<adapter> "
          "→ 再用 scripts/eval_answers.py 跑同一套题对比 → "
          "python scripts/report_finetune.py 生成详细报告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
