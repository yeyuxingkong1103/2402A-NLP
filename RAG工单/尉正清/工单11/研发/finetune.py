# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""第二步：微调 Embedding 模型（命令行入口 + 训练循环）

**为什么不用 SentenceTransformerTrainer**：它要求较新的 transformers
（内部调用 `Trainer(processing_class=...)`，该参数 4.46 才引入），
而本环境被 FlagEmbedding 1.3.3 硬锁在 transformers==4.44.2。
升级 transformers 会把工单 01-10 那套系统一起搞坏，不能动。
所以这里自己写训练循环 —— MNR 损失本身就是一个交叉熵，见 `train_data.py`。

数据管道与损失函数在 `train_data.py`，这里只管流程控制：优化器、学习率、
早停、回滚、保存。

用法：python finetune.py [--epochs N] [--batch N] [--lr X] [--freeze-layers N]
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from sentence_transformers import SentenceTransformer

import data as D
import retrieval as R
import train_data as TD
from config import (BASE_MODEL, DEV_RATIO, EPOCHS, EVAL_EVERY, FINETUNED_MODEL,
                    FREEZE_LAYERS, LEARNING_RATE, PATIENCE, TRAIN_BATCH,
                    TRAIN_MAX_LEN, WARMUP_RATIO)

OUT = D.DATA_DIR
LOGS = Path(__file__).parent.parent / "测试" / "results"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--output", default=str(FINETUNED_MODEL))
    ap.add_argument("--dev-ratio", type=float, default=DEV_RATIO,
                    help="从训练 query 里切多少比例做 dev")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="dev loss 连续多少次没改善就停")
    ap.add_argument("--keep-last", action="store_true",
                    help="不按 dev loss 回滚，直接用最后一版权重。"
                         "只用 batch 内负例时 dev loss 与检索质量脱钩，"
                         "**正式训练要开这个**，见 config 的早停说明")
    ap.add_argument("--freeze-layers", type=int, default=FREEZE_LAYERS,
                    help="冻结 embedding + 底部 N 层编码器，只训顶部。"
                         "训练样本少（约 5.6k）时全参数微调会把预训练学到的"
                         "通用语义带偏，冻住底部能保留通用能力、只做领域适配。"
                         "传 0 = 全参数微调（实测会低于基线）")
    ap.add_argument("--max-len", type=int, default=TRAIN_MAX_LEN,
                    help="训练截断长度")
    ap.add_argument("--use-gen", action="store_true",
                    help="把 gen_qa.py 生成的问答对并进训练集（扩大训练数据量）")
    ap.add_argument("--use-hard", dest="use_hard", action="store_true", default=None,
                    help="用难负例训练。**在 FiQA 上会掉 0.09**，见问题 7")
    ap.add_argument("--no-hard", dest="use_hard", action="store_false",
                    help="只用 batch 内负例（默认）")
    ap.add_argument("--temperature", type=float, default=None,
                    help="MNR 的缩放系数的倒数；默认取 train_data.TEMPERATURE")
    return ap.parse_args()


def freeze_bottom(model, n_layers):
    """冻结 embedding 与底部 n_layers 层编码器。"""
    bert = model[0].auto_model
    for p in bert.embeddings.parameters():
        p.requires_grad = False
    for i, layer in enumerate(bert.encoder.layer):
        if i < n_layers:
            for p in layer.parameters():
                p.requires_grad = False
    total = len(bert.encoder.layer)
    print(f"冻结 embedding + 底部 {n_layers} 层编码器，只训顶部 {total - n_layers} 层")


def main():
    args = parse_args()
    # 这两个开关住在 train_data 里（数据管道和损失函数都要用），命令行覆盖它。
    TD.configure(use_hard=args.use_hard, temperature=args.temperature)

    print("=" * 68)
    print("第二步：微调 Embedding 模型")
    print("=" * 68)
    print(R.gpu_report())

    corpus = D._read_jsonl(OUT / "corpus.jsonl")
    _, queries, qrels = D.load_fiqa()
    rel = D.relevance_map(qrels)
    train_q, _ = D.split_queries(rel)
    train_q, dev_q = D.split_train_dev(train_q, args.dev_ratio)
    loader, dev_loader, n, n_dev = TD.build_loaders(corpus, args.batch, dev_q,
                                                    use_gen=args.use_gen)

    print(f"\n训练样本 {n:,} 条 | dev {n_dev:,} 条"
          f"（dev 覆盖 {len(dev_q)} 个 query，与训练 query 不重叠）")
    print(f"基座 {BASE_MODEL.name}")
    print(f"超参：epochs≤{args.epochs} batch={args.batch} lr={args.lr} "
          f"warmup={WARMUP_RATIO} max_len={args.max_len} "
          f"temperature={TD.TEMPERATURE} 早停 patience={args.patience}")
    print(f"负例来源："
          f"{'难负例 + batch 内负例' if TD.USE_HARD_NEGATIVES else '仅 batch 内负例'}")

    model = SentenceTransformer(str(BASE_MODEL))
    model.max_seq_length = args.max_len      # 训练用短序列，见 config 的说明
    if args.freeze_layers > 0:
        freeze_bottom(model, args.freeze_layers)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数 {sum(p.numel() for p in trainable):,} / "
          f"{sum(p.numel() for p in model.parameters()):,}")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_at(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    @torch.no_grad()
    def dev_loss():
        """在 dev 样本上算一遍 loss（不反向）。

        ⚠️ **这个值不能用来早停**：只用 batch 内负例训练时，模型把向量空间
        摊开会让 batch 内那些「别的问题的正确答案」更难分开，dev loss 因此
        从第一次评估起就单调上涨，与检索质量脱钩 —— 真按它早停会回滚到第
        25 步，等于交一个没训过的模型。详见 优化/过程问题记录.md 问题 10。

        仍然保留它，是因为换成标注密集的语料、用上难负例之后它是有效信号，
        代码留个兜底；本工单的正式训练用 `--keep-last --patience 1000` 关掉。
        """
        model.eval()
        tot, cnt = 0.0, 0
        for batch in dev_loader:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = TD.batch_loss(model, batch)
            tot += loss.item() * len(batch["anchor"])
            cnt += len(batch["anchor"])
        model.train()
        return tot / max(cnt, 1)

    print(f"\n开始训练：最多 {total_steps} 步（warmup {warmup_steps} 步），"
          f"每 {EVAL_EVERY} 步看一次 dev")
    history, step, t0 = [], 0, time.time()
    best_dev, best_state, best_step, bad = float("inf"), None, 0, 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        run_loss, run_n = 0.0, 0
        for batch in loader:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = TD.batch_loss(model, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()

            step += 1
            run_loss += loss.item()
            run_n += 1

            if step % EVAL_EVERY == 0 or step == total_steps:
                tr_loss = run_loss / run_n
                dv = dev_loss()
                el = time.time() - t0
                if dv < best_dev - 1e-4:
                    best_dev, best_step, bad = dv, step, 0
                    best_state = copy.deepcopy(model.state_dict())
                    flag = "  ← dev 新低，记下"
                else:
                    bad += 1
                    flag = f"  （dev 连续 {bad} 次没降）"
                print(f"  epoch {epoch} step {step}/{total_steps} "
                      f"train={tr_loss:.4f} dev={dv:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({el:.0f}s){flag}",
                      flush=True)
                history.append({"epoch": epoch, "step": step,
                                "train_loss": round(tr_loss, 4),
                                "dev_loss": round(dv, 4),
                                "lr": sched.get_last_lr()[0]})
                run_loss, run_n = 0.0, 0
                if bad >= args.patience:
                    print(f"  dev loss 连续 {bad} 次没改善，早停在第 {step} 步")
                    stopped_early = True
                    break
        if stopped_early:
            break

    # 回滚到 dev loss 最低的那一版 —— 但**只在 dev loss 是有效信号时**才这么做。
    # 本工单它是无效的（见 dev_loss 的说明），所以正式训练走 --keep-last。
    if args.keep_last:
        print(f"\n保留最后一版权重（第 {step} 步，dev loss="
              f"{history[-1]['dev_loss']:.4f}；最佳 dev loss 在第 {best_step} 步"
              f"={best_dev:.4f}）")
    elif best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n已回滚到第 {best_step} 步的权重（dev loss={best_dev:.4f}）")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"模型已保存到 {out_dir}")

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "train_log.json").write_text(json.dumps({
        "base_model": str(BASE_MODEL), "output": str(out_dir),
        "n_samples": n, "n_dev": n_dev, "dev_queries": len(dev_q),
        "freeze_layers": args.freeze_layers,
        "max_epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "use_hard_negatives": TD.USE_HARD_NEGATIVES,
        "use_gen_pairs": bool(args.use_gen),
        "warmup_ratio": WARMUP_RATIO, "temperature": TD.TEMPERATURE,
        "max_steps": total_steps, "steps_run": step,
        "stopped_early": stopped_early,
        "best_step": best_step, "best_dev_loss": round(best_dev, 4),
        "kept_last": bool(args.keep_last),
        "elapsed_seconds": round(time.time() - t0, 1),
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练日志已写入 {LOGS / 'train_log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
