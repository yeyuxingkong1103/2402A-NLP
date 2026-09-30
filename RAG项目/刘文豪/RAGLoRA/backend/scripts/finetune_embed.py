# -*- coding: utf-8 -*-
"""bge-m3 嵌入微调（对比学习）。

⚠️ 为什么不用 peft / LoRA
==========================
环境里的 `peft 0.19.1` **装了但用不了** —— 它 import 时就报：

    ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'

因为 peft 0.19 需要更新的 transformers，而本项目锁定 `transformers==4.39.3`
（升级会破坏版本链 —— 本项目已因 FlagEmbedding / ragas 踩过两次同类问题）。

因此改用**层冻结**作为参数高效方案：只训练最后 N 层 + 池化，
其余层 `requires_grad = False`。效果与 LoRA 同类（都是只训练极小一部分参数），
且**零新增依赖**。

⚠️ 池化方式：用 mean，与生产 embed.py 一致
==========================================
`bge-m3/1_Pooling/config.json` 声明的是 CLS 池化，而 `embed.py` 用的是 mean。
已实测两者在本语料上**命中率完全相同**（75.0% vs 75.0%），差异不显著。
**但训练与推理必须用同一种** —— 否则微调的收益会被池化不匹配抵消。
既然生产推理用 mean，训练也用 mean。

训练目标
--------
InfoNCE（批内负样本）：同一 batch 内，其他样本的 positive 充当负例。
这是 `sentence_transformers.losses.MultipleNegativesRankingLoss` 的做法，
这里手写（因为要复用 embed.py 的池化实现，而非 ST 的模型封装）。

用法
----
    python scripts/finetune_embed.py --epochs 3
    python scripts/finetune_embed.py --dry-run          # 只跑 10 步验证通路
    python scripts/finetune_embed.py --trainable-layers 0   # 只训池化层（最省）
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402

log = get_logger("finetune")

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
PAIRS = EVAL_DIR / "embed_pairs.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "finetune" / "out"
MODEL_DIR = Path(config.EMBED_MODEL_PATH)


def load_pairs(limit: int = 0) -> list[tuple[str, str]]:
    rows = []
    for line in PAIRS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, p = (d.get("query") or "").strip(), (d.get("positive") or "").strip()
        if q and p:
            rows.append((q, p))
    if limit:
        rows = rows[:limit]
    return rows


def build_model(trainable_layers: int):
    """加载 bge-m3 并冻结大部分参数。

    `trainable_layers=0`  -> 只训练池化（这里 mean 池化无参数，等效于不训练）
    `trainable_layers=2`  -> 只训练最后 2 层 transformer + 最终 LayerNorm
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)

    for p in model.parameters():
        p.requires_grad = False

    if trainable_layers > 0:
        layers = model.encoder.layer
        for layer in layers[-trainable_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        # bge-m3 是 XLM-RoBERTa：最后一层后面还有个 LayerNorm，也应该放开
        if hasattr(model, "encoder") and hasattr(model.encoder, "LayerNorm"):
            for p in model.encoder.LayerNorm.parameters():
                p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("可训练参数 %s / %s（%.2f%%）",
             f"{n_train/1e6:.1f}M", f"{n_total/1e6:.1f}M", n_train / n_total * 100)
    return model


def mean_pool(hidden, mask):
    """与 embed.py 完全一致的 mean pooling。

    ⚠️ 必须与推理侧用同一种池化，否则微调白做（见模块 docstring）。
    """
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_batch(model, tok, texts, device, max_len=256):
    enc = tok(list(texts), padding=True, truncation=True,
              max_length=max_len, return_tensors="pt").to(device)
    hidden = model(**enc).last_hidden_state
    vec = mean_pool(hidden, enc["attention_mask"])
    return F.normalize(vec, p=2, dim=1)


def train(args) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("设备: %s", device)

    pairs = load_pairs(args.limit)
    log.info("训练对 %d 条", len(pairs))
    if len(pairs) < 32:
        log.error("训练对太少，无法训练")
        return 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = build_model(args.trainable_layers).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        log.error("没有可训练参数 —— trainable_layers 至少给 1")
        return 1

    # ⚠️ 用 SGD 而非 AdamW：bge-m3 有 5.7 亿参数，AdamW 的
    #    动量+方差状态是参数量的 2 倍（fp32 下约 4.5GB），
    #    在本机显存/内存下不可行。只训最后几层时，SGD 已足够。
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        nb = 0
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i + args.batch_size]
            if len(batch) < 2:
                continue
            qs = [b[0] for b in batch]
            ps = [b[1] for b in batch]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                qv = encode_batch(model, tok, qs, device, args.maxlen)
                pv = encode_batch(model, tok, ps, device, args.maxlen)
                # InfoNCE：批内负样本。logits[i,j] 表示第 i 个问题与第 j 个正例的相似度，
                # 对角线为正样本，其余为负样本。
                logits = qv @ pv.T / args.temperature
                labels = torch.arange(len(batch), device=device)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            nb += 1
            step += 1

            if step % 10 == 0:
                log.info("epoch %d step %d | loss %.4f | %.1f step/s",
                         epoch + 1, step, total_loss / max(nb, 1),
                         step / max(time.time() - t0, 1e-9))
            if args.dry_run and step >= 10:
                log.info("--dry-run：跑满 10 步即退出")
                return 0

        log.info("epoch %d 完成 | 平均 loss %.4f", epoch + 1, total_loss / max(nb, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    log.info("微调模型已保存到 %s", OUT_DIR)
    log.info("总耗时 %.0fs | 共 %d 步", time.time() - t0, step)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="批内负样本数 = batch_size-1，太小则对比信号弱")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--trainable-layers", type=int, default=2,
                    help="放开最后 N 层（0=只训池化，本方案下等于不训）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 对（调试）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
