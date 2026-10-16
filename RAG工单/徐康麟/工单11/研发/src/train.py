# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""微调训练：CPU 原生 PyTorch 训练循环（无 accelerate / peft / sentence-transformers Trainer）。

流程：数据集与模型加载 -> 定义损失函数 -> 定义训练参数 -> 逐步训练 -> 记录损失曲线 -> 落盘。

产出：
  研发/data/train_log.jsonl   逐步训练日志（损失/学习率/正负例相似度/耗时）
  研发/data/loss_curve.png    损失曲线
  研发/data/finetuned/        微调后的模型（含 工单11_微调信息.json）
  部署/日志/训练_*.log
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import Config  # noqa: E402
from encoder import EmbeddingModel, set_threads  # noqa: E402
from losses import build_loss  # noqa: E402
from plot import line_chart  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def build_samples(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """从数据集中拆出三元组与带分数的句子对。"""
    triplets, pairs = [], []
    for r in rows:
        fmt = r.get("格式")
        if fmt == "三元组" and r.get("negative"):
            triplets.append({"anchor": r["anchor"], "positive": r["positive"], "negative": r["negative"],
                             "负例分数": float(r.get("负例分数", 0.0)), "id": r.get("id")})
        elif fmt == "相似度分数对":
            pairs.append({"a": r["anchor"], "b": r.get("negative") or r["positive"],
                          "score": float(r.get("score", 1.0)), "id": r.get("id")})
    return triplets, pairs


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


def cosine_lr(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.inference_mode()
def dev_loss(model: EmbeddingModel, dev_triplets: list[dict], loss_mod, kind: str,
             batch_size: int, max_length: int) -> dict:
    if not dev_triplets:
        return {}
    model.eval()
    tot, nb = 0.0, 0
    pos_sims, neg_sims = [], []
    for i in range(0, len(dev_triplets), batch_size):
        b = dev_triplets[i : i + batch_size]
        a = model.encode([x["anchor"] for x in b], batch_size=batch_size, max_length=max_length)
        p = model.encode([x["positive"] for x in b], batch_size=batch_size, max_length=max_length)
        n = model.encode([x["negative"] for x in b], batch_size=batch_size, max_length=max_length)
        at, pt, nt = (torch.from_numpy(x) for x in (a, p, n))
        sc = torch.tensor([x["负例分数"] for x in b], dtype=torch.float32)
        tot += float(step_loss(loss_mod, kind, at, pt, nt, sc))
        nb += 1
        pos_sims.extend((at * pt).sum(-1).tolist())
        neg_sims.extend((at * nt).sum(-1).tolist())
    model.train()
    ps, ns = np.asarray(pos_sims), np.asarray(neg_sims)
    return {"dev_loss": round(tot / max(1, nb), 6),
            "dev_正例相似度": round(float(ps.mean()), 6) if ps.size else None,
            "dev_负例相似度": round(float(ns.mean()), 6) if ns.size else None,
            "dev_相似度间隔": round(float(ps.mean() - ns.mean()), 6) if ps.size and ns.size else None}


def step_loss(loss_mod, kind: str, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor,
              neg_scores: torch.Tensor, pos_scores: torch.Tensor | None = None) -> torch.Tensor:
    """按损失类型组装一步的损失（正例分数固定为 1.0）。"""
    if pos_scores is None:
        pos_scores = torch.ones_like(neg_scores)
    if kind == "triplet":
        return loss_mod(a, p, n)
    if kind == "matryoshka-triplet":
        return loss_mod(a, p, n)
    if kind == "cosine":
        return 0.5 * (loss_mod(a, p, pos_scores) + loss_mod(a, n, neg_scores))
    if kind == "contrastive":
        return 0.5 * (loss_mod(a, p, torch.ones_like(neg_scores))
                      + loss_mod(a, n, torch.zeros_like(neg_scores)))
    if kind == "combined":
        lt = loss_mod.triplet(a, p, n)
        lc = 0.5 * (loss_mod.cosine(a, p, pos_scores) + loss_mod.cosine(a, n, neg_scores))
        return loss_mod.w_t * lt + loss_mod.w_c * lc
    raise ValueError(f"未知损失类型键：{kind}")


def train(cfg: Config, *, 最大步数: int | None = None, 批大小: int | None = None,
          学习率: float | None = None, 输出目录: str | None = None,
          损失类型: str | None = None, 静默: bool = False) -> dict:
    t_cfg = dict(cfg.hyper["训练"])
    if 最大步数 is not None:
        t_cfg["最大步数"] = int(最大步数)
    if 批大小 is not None:
        t_cfg["批大小"] = int(批大小)
    if 学习率 is not None:
        t_cfg["学习率"] = float(学习率)
    l_cfg = dict(cfg.hyper["损失函数"])
    if 损失类型:
        l_cfg["类型"] = 损失类型

    seed = int(cfg.hyper_get("数据集", "随机种子", default=20260826))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_threads(int(t_cfg.get("线程数", 8)))

    max_length = int(cfg.hyper_get("数据集", "最大序列长度", default=128))
    batch_size = int(t_cfg.get("批大小", 16))
    max_steps = int(t_cfg.get("最大步数", 168))
    lr = float(t_cfg.get("学习率", 3e-5))
    wd = float(t_cfg.get("权重衰减", 0.01))
    grad_clip = float(t_cfg.get("梯度裁剪", 1.0))
    warmup = float(t_cfg.get("预热比例", 0.1))
    budget = float(t_cfg.get("时间预算秒", 1500))
    eval_every = int(t_cfg.get("每N步评估", 25))
    ckpt_every = int(t_cfg.get("检查点间隔", 40))

    ds_dir = cfg.path("数据集目录")
    train_rows = read_jsonl(ds_dir / "train.jsonl")
    dev_rows = read_jsonl(ds_dir / "dev.jsonl")
    triplets, _pairs = build_samples(train_rows)
    dev_triplets, _ = build_samples(dev_rows)
    if not triplets:
        raise SystemExit("!! 训练集为空，请先运行 研发/src/generate_dataset.py")
    # 降采样：CPU 时间预算约束下的抽样口径（随机种子固定，可复现）
    cap = int(t_cfg.get("训练样本上限", 0) or 0)
    if cap and len(triplets) > cap:
        random.Random(seed).shuffle(triplets)
        triplets = triplets[:cap]
        print(f"[训练] 降采样：训练三元组截取前 {cap} 条（随机种子 {seed}，可复现）")
    dev_triplets = dev_triplets[:64]
    print(f"[训练] 训练三元组 {len(triplets)} 条；验证三元组 {len(dev_triplets)} 条；"
          f"批大小 {batch_size}；最大步数 {max_steps}；学习率 {lr}")

    base_model = str(cfg.path("底座模型"))
    model = EmbeddingModel(base_model, max_length=max_length,
                           trainable_layers=str(t_cfg.get("可训练层", "最后2层+池化")),
                           device=str(t_cfg.get("设备", "cpu")))
    info = model.apply_freeze_policy(str(t_cfg.get("可训练层", "最后2层+池化")))
    print(f"[训练] 冻结策略 = {t_cfg.get('可训练层')} -> {info}")

    loss_mod, kind, loss_desc = build_loss(l_cfg)
    print(f"[训练] 损失函数 = {loss_desc}（类型键 {kind}）")

    # 优化器：LayerNorm / bias 不做权重衰减
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    log_path = cfg.path("微调模型目录").parent / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    timer = Timer()
    history: list[dict] = []
    order = list(range(len(triplets)))
    epoch = 0
    stopped_reason = "达到最大步数"
    running = {"loss": 0.0, "n": 0}
    model.train()

    try:
        for step in range(max_steps):
            if step % len(order) == 0:
                random.shuffle(order)
                epoch += 1
            idx = order[step % len(order) : step % len(order) + batch_size]
            if len(idx) < batch_size:
                idx = idx + order[: batch_size - len(idx)]
            batch = [triplets[i] for i in idx]

            cur_lr = cosine_lr(step, max_steps, lr, warmup)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            enc_a = model.tokenizer([x["anchor"] for x in batch], padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt")
            enc_p = model.tokenizer([x["positive"] for x in batch], padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt")
            enc_n = model.tokenizer([x["negative"] for x in batch], padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt")
            va = model(enc_a["input_ids"], enc_a["attention_mask"])
            vp = model(enc_p["input_ids"], enc_p["attention_mask"])
            vn = model(enc_n["input_ids"], enc_n["attention_mask"])
            neg_scores = torch.tensor([x["负例分数"] for x in batch], dtype=torch.float32)
            loss = step_loss(loss_mod, kind, va, vp, vn, neg_scores)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimizer.step()

            lv = float(loss.detach())
            running["loss"] += lv
            running["n"] += 1
            rec = {
                "step": step + 1,
                "epoch": epoch,
                "loss": round(lv, 6),
                "loss_滑动均值": round(running["loss"] / running["n"], 6),
                "lr": round(cur_lr, 8),
                "正例相似度": round(float((va.detach() * vp.detach()).sum(-1).mean()), 6),
                "负例相似度": round(float((va.detach() * vn.detach()).sum(-1).mean()), 6),
                "耗时秒": round(timer.elapsed, 2),
            }
            rec["相似度间隔"] = round(rec["正例相似度"] - rec["负例相似度"], 6)
            if dev_triplets and ((step + 1) % eval_every == 0 or step == 0):
                rec.update(dev_loss(model, dev_triplets, loss_mod, kind, batch_size, max_length))
            history.append(rec)
            log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_fh.flush()
            if not 静默 and ((step + 1) % 10 == 0 or step == 0):
                print(f"[训练] step {step+1}/{max_steps} loss={rec['loss']:.5f} "
                      f"间隔={rec['相似度间隔']:.4f} lr={cur_lr:.2e} 用时={rec['耗时秒']}s")
            if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
                # 内存紧张环境下定期落盘，避免中途 OOM 丢失全部训练成果
                ck_dir = (Path(输出目录) if 输出目录 else cfg.path("微调模型目录"))
                model.save(ck_dir, meta={"检查点步数": step + 1, "检查点损失": lv, "状态": "训练中"})
                print(f"[训练] 已写入检查点 -> {ck_dir}（step {step+1}）")
            if timer.elapsed > budget:
                stopped_reason = f"触发时间预算 {budget}s（已完成 {step+1} 步）"
                print(f"[训练] {stopped_reason}，提前停止")
                break
    except BaseException as exc:  # noqa: BLE001 —— 内存不足等异常也要保住已有成果
        stopped_reason = f"异常中断：{type(exc).__name__}: {exc}"
        print(f"[训练][错误] {stopped_reason}")
        try:
            ck_dir = Path(输出目录) if 输出目录 else cfg.path("微调模型目录")
            model.save(ck_dir, meta={"状态": "异常中断", "已完成步数": len(history), "错误": str(exc)})
            print(f"[训练] 已将中断前状态落盘 -> {ck_dir}")
        except Exception as exc2:  # noqa: BLE001
            print(f"[训练][错误] 中断落盘也失败：{exc2}")
        raise
    finally:
        log_fh.close()

    elapsed = timer.elapsed
    steps_done = len(history)
    epochs_done = round(steps_done * batch_size / max(1, len(triplets)), 3)
    first_loss = history[0]["loss"] if history else None
    last_loss = history[-1]["loss"] if history else None
    tail = [h["loss"] for h in history[-max(1, steps_done // 10):]] if history else []
    summary = {
        "训练时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "底座模型": base_model,
        "冻结策略": t_cfg.get("可训练层"),
        "参数统计": info,
        "损失函数": loss_desc,
        "损失类型键": kind,
        "损失配置": l_cfg,
        "超参": t_cfg,
        "训练三元组数": len(triplets),
        "验证三元组数": len(dev_triplets),
        "实际步数": steps_done,
        "实际轮次": epochs_done,
        "停止原因": stopped_reason,
        "总耗时秒": round(elapsed, 1),
        "每步平均秒": round(elapsed / max(1, steps_done), 3),
        "初始损失": first_loss,
        "末步损失": last_loss,
        "末段平均损失": round(sum(tail) / len(tail), 6) if tail else None,
        "损失下降": round(first_loss - last_loss, 6) if (first_loss is not None and last_loss is not None) else None,
        "损失下降比例": round((first_loss - last_loss) / first_loss, 6) if first_loss else None,
        "最大序列长度": max_length,
        "批大小": batch_size,
        "学习率": lr,
        "设备": "cpu",
        "线程数": int(t_cfg.get("线程数", 8)),
    }

    # 损失曲线
    curve_png = cfg.path("微调模型目录").parent / "loss_curve.png"
    line_chart(curve_png, title="工单11 Embedding 微调训练损失曲线",
               subtitle=f"{loss_desc}｜{steps_done} 步 / {epochs_done} 轮｜批大小 {batch_size}｜CPU {t_cfg.get('线程数')} 线程",
               series=[("训练损失(逐步)", [h["step"] for h in history], [h["loss"] for h in history]),
                       ("训练损失(滑动均值)", [h["step"] for h in history], [h["loss_滑动均值"] for h in history])],
               xlabel="训练步数", ylabel="损失值")

    out_dir = Path(输出目录) if 输出目录 else cfg.path("微调模型目录")
    model.save(out_dir, meta=summary)
    (out_dir / "训练摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[训练] 模型已保存 -> {out_dir}")
    print(f"[训练] 日志 -> {log_path}；曲线 -> {curve_png}")
    print(f"[训练] 初始损失 {summary['初始损失']} -> 末段平均 {summary['末段平均损失']}；"
          f"下降 {summary['损失下降']}（{summary['损失下降比例']}）")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单11 微调训练")
    ap.add_argument("--最大步数", type=int, default=None)
    ap.add_argument("--批大小", type=int, default=None)
    ap.add_argument("--学习率", type=float, default=None)
    ap.add_argument("--损失类型", type=str, default=None,
                    help="triplet / cosine / contrastive / matryoshka-triplet / triplet+cosine")
    ap.add_argument("--输出目录", type=str, default=None)
    ap.add_argument("--静默", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = Config()
    cfg.ensure_dirs()
    kw = {k: v for k, v in vars(args).items() if v is not None and k != "静默"}
    summary = train(cfg, 静默=args.静默, **kw)
    print(json.dumps({k: summary[k] for k in ("实际步数", "实际轮次", "初始损失", "末段平均损失",
                                              "损失下降比例", "总耗时秒", "停止原因")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
