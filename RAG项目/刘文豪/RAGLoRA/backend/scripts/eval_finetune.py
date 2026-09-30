# -*- coding: utf-8 -*-
"""微调前后对比：bge-m3 原模型 vs 微调模型。

**必须实测**：训练 loss 下降只说明「更拟合训练对」，
不等于「检索变好」—— 2390 个训练对相对 5.7 亿参数模型是很小的数据量，
过拟合导致检索退化的风险是实打实的。所以一切以检索指标为准。

对比口径
--------
* 同一批语料、同一批问题、同一种池化（mean，与生产 embed.py 一致）
* 只比 **dense 单路** —— 微调只动了 dense 分支，sparse 分支未训练

用法
----
    python scripts/eval_finetune.py                # 默认 800 条语料
    python scripts/eval_finetune.py --corpus 2000 --split train   # 在训练集上测（会虚高）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402

log = get_logger("eval_ft")

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
BASE_MODEL = Path(config.EMBED_MODEL_PATH)
FT_MODEL = Path(__file__).resolve().parents[1] / "finetune" / "out"


def load_corpus(limit: int):
    from app.services import milvus_store
    c = milvus_store.get_client()
    rows, off = [], 0
    while True:
        r = c.query(collection_name="kb_legal", filter="",
                    output_fields=["text", "law_name", "source"],
                    limit=1000, offset=off)
        if not r:
            break
        rows.extend(r)
        off += len(r)
        if limit and len(rows) >= limit:
            break
        if len(r) < 1000:
            break
    rows = rows[:limit] if limit else rows
    return ([(x.get("text") or "") for x in rows],
            [(x.get("law_name") or x.get("source") or "") for x in rows])


def encode(model_dir: Path, texts: list[str], device: str,
           batch: int = 32, max_len: int = 256) -> np.ndarray:
    """与 embed.py 相同的 mean pooling + L2 归一化。"""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).eval().to(device)
    if device == "cuda":
        model = model.half()

    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(vec.float(), p=2, dim=1).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out)


def hit_rate(vecs, sources, q_vecs, expects, topk):
    sims = q_vecs @ vecs.T
    hits = 0
    detail = []
    for i, exp in enumerate(expects):
        idx = np.argsort(-sims[i])[:topk]
        got = " ".join(sources[j] for j in idx)
        ok = any(e.lower() in got.lower() for e in exp)
        hits += ok
        detail.append(ok)
    return hits / len(expects) * 100 if expects else 0.0, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=800)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--report", default=str(EVAL_DIR / "finetune_report.md"))
    args = ap.parse_args()

    if not (FT_MODEL / "config.json").exists():
        print(f"✗ 找不到微调模型 {FT_MODEL}，请先跑 scripts/finetune_embed.py")
        return 1

    qa = json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))
    questions = [x["q"] for x in qa["legal"]]
    expects = [x.get("expect_sources", []) for x in qa["legal"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"微调前后对比 | 语料 {args.corpus} 条 | 问题 {len(questions)} 条 | "
          f"top-{args.topk} | {device}")
    print("=" * 72)

    texts, sources = load_corpus(args.corpus)
    print(f"载入语料 {len(texts)} 条\n")

    res = {}
    for name, path in (("原模型", BASE_MODEL), ("微调后", FT_MODEL)):
        t = time.time()
        vecs = encode(path, texts, device)
        q_vecs = encode(path, questions, device)
        hr, detail = hit_rate(vecs, sources, q_vecs, expects, args.topk)
        res[name] = {"hit": hr, "detail": detail, "sec": round(time.time() - t, 1)}
        print(f"  {name}: 来源命中率 {hr:.1f}%（{round(hr/100*len(expects))}/{len(expects)}）"
              f"  {time.time()-t:.0f}s")

    print()
    print("-" * 72)
    delta = res["微调后"]["hit"] - res["原模型"]["hit"]
    print(f"  变化：{delta:+.1f} 个百分点")
    # 逐题看谁改善谁变差
    d0, d1 = res["原模型"]["detail"], res["微调后"]["detail"]
    better = [i for i in range(len(d0)) if d1[i] and not d0[i]]
    worse = [i for i in range(len(d0)) if d0[i] and not d1[i]]
    print(f"  改善 {len(better)} 题，变差 {len(worse)} 题")
    for i in better:
        print(f"    ↑ {questions[i][:38]}")
    for i in worse:
        print(f"    ↓ {questions[i][:38]}")

    lines = ["# bge-m3 嵌入微调评估", "",
             f"> 语料 {len(texts)} 条（kb_legal）　问题 {len(questions)} 条　top-{args.topk}",
             f"> 训练对 2390 条　可训练参数 25.2M / 567.8M（4.44%）　3 epoch / 225 步 / 127 秒",
             "",
             "## 对比口径", "",
             "- 同一语料、同一问题、同一池化（mean，与生产 `embed.py` 一致）",
             "- 只比 **dense 单路** —— 微调只动了 dense 分支，sparse 未训练",
             "- **在测试语料上评测**，非训练集（训练 loss 不等于检索效果）",
             "",
             "## 结果", "",
             "| 模型 | 来源命中率 |", "|---|---:|",
             f"| 原 bge-m3 | {res['原模型']['hit']:.1f}% |",
             f"| 微调后 | {res['微调后']['hit']:.1f}% |",
             f"| **变化** | **{delta:+.1f} pp** |",
             "",
             f"改善 {len(better)} 题，变差 {len(worse)} 题。", ""]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
