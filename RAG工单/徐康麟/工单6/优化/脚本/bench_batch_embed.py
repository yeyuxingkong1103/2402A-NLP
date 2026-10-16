# -*- coding: utf-8 -*-
"""T7 批处理基准：嵌入「逐条调用」vs「批量打包」，量出往返次数与耗时差异。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_batch_embed.py --n 48 --batch-size 16

做什么：
    * 从索引里取 ``--n`` 条真实文本（图像块 / 正文块混合，取索引真实语料，不造假数据）；
    * **逐条**模式：每 1 条调用一次嵌入后端（``batch_size=1``）；
    * **批量**模式：按 ``--batch-size`` 打包调用（``batch_size=N``）；
    * 两种模式产出的向量做**逐行余弦一致性**校验（``min``/``max``），确保批处理不改变语义；
    * 报告往返次数、总耗时、每条耗时与节省比例，落盘 ``优化/评估结果/批处理基准.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

from app.core.cache import batch_embed                       # noqa: E402
from app.core.config import get_config                        # noqa: E402
from app.core.embedder import embed_texts                     # noqa: E402
from app.core.logging_conf import get_logger                  # noqa: E402


def pick_texts(n: int) -> list[str]:
    """从索引里取真实文本（优先图像块与表格块，再补正文块）。"""
    index_dir = ROOT / "研发" / "data" / "index" / "bge-m3_1024" / "chunks_indexed.jsonl"
    rows = [json.loads(line) for line in index_dir.read_text(encoding="utf-8").splitlines() if line.strip()]
    priority = {"image": 0, "table": 1, "text": 2}
    rows.sort(key=lambda r: (priority.get(str(r.get("type")), 3), str(r.get("chunk_id"))))
    texts = [str(r.get("content") or "")[:400] for r in rows[:n]]
    return texts or ["武汉兴图新科电子股份有限公司注册资本是多少？"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T7 批处理嵌入基准")
    parser.add_argument("--n", type=int, default=48, help="样本条数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=str(ROOT / "优化" / "评估结果" / "批处理基准.json"))
    args = parser.parse_args(argv)
    cfg = get_config()
    log = get_logger("bench_batch_embed")
    texts = pick_texts(max(4, int(args.n)))
    size = max(1, int(args.batch_size))

    # 逐条：每条一次调用（用同一个嵌入后端，仅 batch_size=1）
    single = embed_texts(texts, cfg=cfg, batch_size=1, logger=log)
    single_vecs = np.asarray(single.vectors, dtype=np.float32)
    # 批量：打包调用
    batched = embed_texts(texts, cfg=cfg, batch_size=size, logger=log)
    batched_vecs = np.asarray(batched.vectors, dtype=np.float32)

    # 一致性：逐行余弦
    def _norm(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    cos = (np.sum(_norm(single_vecs) * _norm(batched_vecs), axis=1)
           if single_vecs.shape == batched_vecs.shape else np.array([float("nan")]))
    single_ms = float(getattr(single, "elapsed_ms", 0.0))
    batched_ms = float(getattr(batched, "elapsed_ms", 0.0))
    # 同时给出 cache.batch_embed 的通用实现（供索引侧复用）的往返统计
    generic = batch_embed(texts, lambda window: embed_texts(window, cfg=cfg, batch_size=len(window),
                                                            logger=log).vectors,
                          batch_size=size, logger=log).to_dict()

    payload = {
        "work_order": "人工智能NLP-RAG-图像内容解析及检索优化",
        "n_texts": len(texts), "batch_size": size,
        "single": {"elapsed_ms": single_ms, "per_item_ms": round(single_ms / len(texts), 2),
                   "round_trips": len(texts)},
        "batched": {"elapsed_ms": batched_ms, "per_item_ms": round(batched_ms / len(texts), 2),
                    "round_trips": (len(texts) + size - 1) // size},
        "speedup_x": round(single_ms / batched_ms, 2) if batched_ms else None,
        "round_trips_saved": len(texts) - ((len(texts) + size - 1) // size),
        "cosine_single_vs_batched": {"min": float(np.min(cos)) if cos.size else None,
                                     "max": float(np.max(cos)) if cos.size else None},
        "batch_embed_helper": generic,
        "note": "同一嵌入后端（本地 Ollama bge-m3:latest）；批量只改「每次送多少条」，不改语义",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[T7 批处理基准] {len(texts)} 条文本 / batch_size={size}")
    print(f"  逐条：{single_ms:.0f} ms（{payload['single']['per_item_ms']} ms/条，往返 {len(texts)} 次）")
    print(f"  批量：{batched_ms:.0f} ms（{payload['batched']['per_item_ms']} ms/条，"
          f"往返 {payload['batched']['round_trips']} 次）")
    print(f"  加速 {payload['speedup_x']}×，少 {payload['round_trips_saved']} 次往返；"
          f"逐行余弦 min={payload['cosine_single_vs_batched']['min']:.6f} "
          f"max={payload['cosine_single_vs_batched']['max']:.6f}")
    print(f"  落盘 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
