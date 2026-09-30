# -*- coding: utf-8 -*-
"""嵌入模型横向对比：bge-m3 / m3e-base / bert-base-chinese。

清单要求「Embedding-Model（BGE-m3，微调）优化」，
对比是微调的前置 —— 得先知道基线在哪，微调后才有可比对象。

对比范围刻意收窄以保证可比
--------------------------
只比 **dense 单路**。原因：三个模型里**只有 bge-m3 能产出 sparse 向量**
（它是多向量模型，带 learned sparse 分支），另两个是纯 dense 模型。
若让 bge-m3 走混合检索、另两个走单路，那是拿「两条路」比「一条路」，
比出来的是「混合检索更好」而不是「模型更好」—— 这不是本对比要回答的问题。

评测用**语料内检索**：问题来自 `qa_set.json`，语料取 kb_legal 的一个子集。
子集大小可选 —— 全量 14319 条 × 3 个模型太慢，而检索命中率在小样本上已能区分模型。

用法
----
    python scripts/compare_embeddings.py                  # 默认 3000 条语料
    python scripts/compare_embeddings.py --corpus 1000    # 更快
    python scripts/compare_embeddings.py --corpus 0       # 全量（很慢）
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402

log = get_logger("cmp_embed")

MODELS_DIR = Path(config.EMBED_MODELS_DIR)
# 注意 m3e-base 多套了一层同名目录
MODELS = {
    "bge-m3": MODELS_DIR / "bge-m3",
    "m3e-base": MODELS_DIR / "m3e-base" / "m3e-base",
    "bert-base-chinese": MODELS_DIR / "bert-base-chinese",
}
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def load_corpus(limit: int) -> tuple[list[str], list[str]]:
    """取 kb_legal 的语料子集，返回 (texts, sources)。"""
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
    texts = [(x.get("text") or "") for x in rows]
    sources = [(x.get("law_name") or x.get("source") or "") for x in rows]
    return texts, sources


def encode_all(model_dir: Path, texts: list[str], batch: int = 32,
               max_len: int = 512) -> np.ndarray:
    """用 transformers 编码，mean-pooling + L2 归一化（与项目现有 embed.py 同法）。

    刻意不复用 `embed.py` —— 它写死了 bge-m3 的 sparse 分支，
    而本对比要能加载任意 dense 模型。
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model = model.eval().to("cpu")

    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.float().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(out)


def evaluate(vecs: np.ndarray, sources: list[str],
             questions: list[str], expects: list[list[str]],
             q_vecs: np.ndarray, topk: int) -> dict:
    """dense 单路检索，算来源命中率。"""
    sims = q_vecs @ vecs.T                       # 已归一化 -> 点积即余弦
    hits, lat = 0, []
    for i, exp in enumerate(expects):
        t0 = time.time()
        idx = np.argsort(-sims[i])[:topk]
        lat.append((time.time() - t0) * 1000)
        got = " ".join(sources[j] for j in idx)
        if any(e.lower() in got.lower() for e in exp):
            hits += 1
    return {"hit_rate": hits / len(expects) * 100 if expects else 0.0,
            "hits": hits, "n": len(expects),
            "search_ms": statistics.mean(lat) if lat else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=3000, help="语料条数（0=全量）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--maxlen", type=int, default=256,
                    help="编码截断长度。实测 bge-m3 在 CPU 上跑 max_len=512 只有 "
                         "3 条/秒（3000 条要 16 分钟），降到 256 可显著加速")
    ap.add_argument("--report", default=str(EVAL_DIR / "embed_compare.md"))
    args = ap.parse_args()

    qa = json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))
    questions = [x["q"] for x in qa["legal"]]
    expects = [x.get("expect_sources", []) for x in qa["legal"]]

    print("=" * 74)
    print(f"嵌入模型对比 | 语料 {args.corpus or '全量'} 条 | 问题 {len(questions)} 条 | top-{args.topk}")
    print("=" * 74)

    t0 = time.time()
    texts, sources = load_corpus(args.corpus)
    print(f"载入语料 {len(texts)} 条（{time.time()-t0:.1f}s）\n")

    results = {}
    for name, path in MODELS.items():
        if not (path / "config.json").exists():
            print(f"✗ {name}: 路径无效 {path}")
            continue
        print(f"▶ {name}")
        try:
            t = time.time()
            vecs = encode_all(path, texts, max_len=args.maxlen)
            enc_s = time.time() - t
            t = time.time()
            q_vecs = encode_all(path, questions, max_len=args.maxlen)
            print(f"    编码 {vecs.shape[0]} 条 ({vecs.shape[1]}维) 用时 {enc_s:.1f}s "
                  f"({vecs.shape[0]/enc_s:.0f} 条/秒)")
            r = evaluate(vecs, sources, questions, expects, q_vecs, args.topk)
            r.update({"dim": int(vecs.shape[1]), "enc_s": round(enc_s, 1),
                      "enc_rate": round(vecs.shape[0] / enc_s, 1),
                      "model_size_mb": round(sum(f.stat().st_size for f in path.glob("*")
                                                 if f.is_file()) / 1024 / 1024)})
            results[name] = r
            print(f"    来源命中率: {r['hit_rate']:.1f}%  ({r['hits']}/{r['n']})")
        except Exception as e:
            print(f"    ✗ 失败: {type(e).__name__}: {str(e)[:150]}")
        print()

    if not results:
        print("没有可对比的模型")
        return 1

    print("=" * 74)
    print(f"{'模型':<22} {'维度':>5} {'来源命中':>9} {'编码速度':>10}")
    print("-" * 74)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["hit_rate"]):
        print(f"{name:<22} {r['dim']:>5} {r['hit_rate']:>8.1f}% {r['enc_rate']:>8.0f} 条/秒")

    lines = ["# 嵌入模型对比", "",
             f"> 语料 {len(texts)} 条（kb_legal）　问题 {len(questions)} 条　top-{args.topk}",
             "",
             "## ⚠️ 对比口径",
             "",
             "**只比 dense 单路。** 三个模型里只有 bge-m3 能产出 sparse 向量，",
             "另两个是纯 dense 模型。若让 bge-m3 走混合检索而另两个走单路，",
             "比出来的是「混合检索更好」而不是「模型更好」—— 那不是本对比要回答的问题。",
             "",
             "## 结果", "",
             "| 模型 | 维度 | 来源命中率 | 编码速度 | 模型体积 |", "|---|---:|---:|---:|---:|"]
    for name, r in sorted(results.items(), key=lambda x: -x[1]["hit_rate"]):
        lines.append(f"| {name} | {r['dim']} | **{r['hit_rate']:.1f}%** "
                     f"| {r['enc_rate']:.0f} 条/秒 | {r['model_size_mb']} MB |")
    lines += ["", "## 结论", ""]
    best = max(results.items(), key=lambda x: x[1]["hit_rate"])
    lines.append(f"- 命中率最高：**{best[0]}**（{best[1]['hit_rate']:.1f}%）")
    lines.append(f"- 生产链路实际用的是 **bge-m3 + 混合检索**，本表只反映其 dense 单路表现。")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
