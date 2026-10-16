# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""CPU 训练吞吐标定：为「训练必须在 10~25 分钟内跑完」提供实测依据。

输出 研发/data/性能标定.json，训练脚本据此自动收缩步数（预算保护）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from config import Config  # noqa: E402


def bench(model, tok, seq_len: int, batch: int, back_ward: bool = True, iters: int = 3) -> dict:
    text = "平安银行2019年年度报告显示，报告期末不良贷款率为1.65%，拨备覆盖率183.12%。" * 8
    enc = tok([text] * batch, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    model.train(back_ward)
    # 预热
    for _ in range(1):
        out = model(**enc).last_hidden_state[:, 0]
        if back_ward:
            out.sum().backward()
            model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(**enc).last_hidden_state[:, 0]
        if back_ward:
            out.sum().backward()
            model.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / iters
    return {
        "序列长度": seq_len,
        "批大小": batch,
        "反向": back_ward,
        "单步秒": round(dt, 4),
        "每秒序列数": round(batch / dt, 2),
    }


def main() -> int:
    cfg = Config()
    threads = int(cfg.hyper_get("训练", "线程数", default=8))
    torch.set_num_threads(threads)
    torch.set_grad_enabled(True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_dir = str(cfg.path("底座模型"))
    print(f"[标定] 线程数 = {threads}；模型 = {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[标定] 参数量 = {n_param/1e6:.2f} M；隐藏维度 = {model.config.hidden_size}；层数 = {model.config.num_hidden_layers}")

    results = {"线程数": threads, "参数量M": round(n_param / 1e6, 2),
               "隐藏维度": model.config.hidden_size, "层数": model.config.num_hidden_layers,
               "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
               "torch": torch.__version__, "标定项": []}
    for seq_len, batch in ((64, 24), (128, 24), (128, 48), (128, 96)):
        row = bench(model, tok, seq_len, batch, back_ward=True)
        results["标定项"].append(row)
        print(f"[标定] seq={seq_len} batch={batch} 单步 {row['单步秒']}s  {row['每秒序列数']} seq/s")

    # 推理吞吐（评估阶段用）：batch 64 / seq 128
    with torch.inference_mode():
        row = bench(model, tok, 128, 64, back_ward=False, iters=3)
    results["推理_128_64"] = row
    print(f"[标定] 推理 seq=128 batch=64 每秒 {row['每秒序列数']} seq/s")

    out = cfg.path("微调模型目录").parent / "性能标定.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标定] 已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
