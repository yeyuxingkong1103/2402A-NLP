# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""评估器主程序：微调前后在「留出测试集」与「工单7 真实索引」上的检索效果对照。

三重证据链（全部可追溯到 json/csv 产物）：
  1) 留出测试集（语料自造、页级留出、训练中从未出现）：
     Recall@1/5/10、MRR、nDCG@10；
  2) 相似度指标：正例对平均余弦相似度 vs 负例对平均余弦相似度（间隔越大越好）；
  3) 端到端真实检索：把工单7 索引的 8847 个真实分块全部重新编码，
     对 17 道真实测试问题做 Top-K 检索对照（微调前 vs 微调后），
     并以工单7 生产模型 bge-m3（1024 维）作为参照列。

产出：
  研发/评估/评估结果_微调前.json、评估结果_微调后.json
  优化/评估结果/对比表.csv、逐query检索对照.csv、对比.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from encoder import EmbeddingModel, set_threads  # noqa: E402
from evaluator import (bootstrap_ci, ranks_from_gold_sets, ranks_from_scores,  # noqa: E402
                       retrieval_metrics, similarity_stats)
from text_utils import keyword_hit_count, normalize_text  # noqa: E402


# ------------------------------------------------------------------ 数据加载
def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def load_index_pool(cfg: Config) -> dict:
    """加载工单7 真实索引分块（只读）。"""
    idx_dir = cfg.path("工单7索引目录")
    chunks_path = idx_dir / "chunks_indexed.jsonl"
    ids_path = idx_dir / "ids.json"
    if not chunks_path.exists():
        raise SystemExit(f"!! 找不到工单7 索引分块：{chunks_path}")
    rows = read_jsonl(chunks_path)
    ids = [r["chunk_id"] for r in rows]
    if ids_path.exists():
        try:
            idx_ids = json.loads(ids_path.read_text(encoding="utf-8")).get("ids", [])
            if idx_ids and set(idx_ids) != set(ids):
                print(f"[评估][警告] ids.json 与 chunks_indexed.jsonl 不一致：{len(idx_ids)} vs {len(ids)}")
            else:
                ids = idx_ids or ids
                by_id = {r["chunk_id"]: r for r in rows}
                rows = [by_id[i] for i in ids if i in by_id]
                ids = [r["chunk_id"] for r in rows]
        except Exception as exc:  # noqa: BLE001
            print(f"[评估][警告] 读取 ids.json 失败，按分块文件顺序：{exc}")
    pool = {
        "ids": ids,
        "texts": [str(r.get("content") or "") for r in rows],
        "file_names": [str(r.get("file_name") or "") for r in rows],
        "pages": [int(r.get("page") or 0) for r in rows],
        "types": [str(r.get("type") or "") for r in rows],
        "sections": [str(r.get("section") or "") for r in rows],
        "norm": [normalize_text(r.get("content") or "") for r in rows],
    }
    print(f"[评估] 真实索引分块 {len(pool['ids'])} 个（{idx_dir}）")
    return pool


def subset_pool(pool: dict, keep: Sequence[int]) -> dict:
    """按索引子集重建检索池，并记录原始索引（供 bge-m3 参照向量对齐）。"""
    keys = ("ids", "texts", "file_names", "pages", "types", "sections", "norm")
    out = {k: [pool[k][i] for i in keep] for k in keys}
    out["原始索引"] = [int(i) for i in keep]
    return out


def load_holdout(cfg: Config, pool: dict) -> tuple[list[str], list[int], list[dict]]:
    """留出测试集：查询 + 唯一金标准分块。"""
    path = cfg.path("数据集目录") / "留出测试集.jsonl"
    rows = read_jsonl(path)
    pos = {cid: i for i, cid in enumerate(pool["ids"])}
    queries, gold_idx, kept = [], [], []
    missing = 0
    for r in rows:
        gi = pos.get(r["gold_chunk_id"])
        if gi is None:
            missing += 1
            continue
        queries.append(r["query"])
        gold_idx.append(gi)
        kept.append(r)
    print(f"[评估] 留出测试集 {len(queries)} 条（金标准不在索引中的 {missing} 条已跳过）")
    return queries, gold_idx, kept


def load_real_questions(cfg: Config) -> list[dict]:
    path = cfg.path("工单7测试问题集")
    return read_jsonl(path)


def build_gold_sets(questions: Sequence[dict], pool: dict, *, ratio: float = 0.34) -> tuple[list[list[int]], list[dict]]:
    """按「判分关键词命中数 + 来源文档」反推金标准分块集合。

    规则（可追溯、可复现）：
      - 只在该题的 ``source_file``（多文档用 ';' 分隔）范围内匹配；
      - 命中关键词数 >= max(2, ceil(ratio * 关键词总数)) 视为金标准；
      - 若该阈值下无金标准（题目过难），降级为命中 >= 1，并在明细中标注。
    """
    gold_sets: list[list[int]] = []
    details: list[dict] = []
    for q in questions:
        files = {f.strip() for f in str(q.get("source_file") or "").split(";") if f.strip()}
        kws = q.get("answer_keywords") or []
        thr = max(2, math.ceil(ratio * len(kws))) if kws else 1
        hits_per: list[tuple[int, int]] = []
        for i, cid in enumerate(pool["ids"]):
            if files and pool["file_names"][i] not in files:
                continue
            cnt, _hits = keyword_hit_count(pool["norm"][i], kws)
            if cnt > 0:
                hits_per.append((cnt, i))
        gold = [i for c, i in hits_per if c >= thr]
        fallback = False
        if not gold:
            gold = [i for c, i in hits_per]
            fallback = True
        gold_sets.append(gold)
        details.append({
            "id": q.get("id"), "question": q.get("question"),
            "命中关键词数阈值": thr, "金标准分块数": len(gold),
            "候选分块数(命中>=1)": len(hits_per), "降级为阈值1": fallback,
            "来源文档数": len(files), "关键词数": len(kws),
        })
    return gold_sets, details


# ------------------------------------------------------------------ 检索评估
def eval_query_set(query_vecs: np.ndarray, doc_vecs: np.ndarray, gold_index_sets: Sequence[Sequence[int]],
                   query_texts: Sequence[str], *, ks: Sequence[int], ndcg_k: int,
                   pool: dict, gold_detail: bool = True) -> dict:
    ranks, gold_counts = ranks_from_gold_sets(query_vecs, doc_vecs, gold_index_sets)
    metrics = retrieval_metrics(ranks, ks=ks, ndcg_k=ndcg_k)
    # 关键词覆盖@K（真实问题专用，留出集无关键词时自然为空）
    per_query = []
    if gold_detail:
        for i, gs in enumerate(gold_index_sets):
            sims = doc_vecs @ query_vecs[i]
            order = np.argsort(-sims, kind="stable")[: max(ks)]
            per_query.append({
                "序号": i + 1, "查询": query_texts[i], "首个命中排名": (None if not math.isfinite(ranks[i]) else int(ranks[i])),
                "金标准分块数": int(gold_counts[i]),
                "Top1分块": pool["ids"][int(order[0])] if len(order) else None,
                "Top1文档": pool["file_names"][int(order[0])] if len(order) else None,
                "Top1页码": pool["pages"][int(order[0])] if len(order) else None,
                "TopK分块": [pool["ids"][int(j)] for j in order],
            })
    metrics["逐query"] = per_query
    metrics["命中排名列表"] = [None if not math.isfinite(r) else int(r) for r in ranks]
    return metrics


def eval_holdout(model: EmbeddingModel, queries: list[str], gold_idx: list[int], doc_vecs: np.ndarray,
                 pool: dict, *, ks: Sequence[int], ndcg_k: int, batch: int, max_length: int) -> dict:
    qv = model.encode(queries, batch_size=batch, max_length=max_length)
    ranks = ranks_from_scores(qv, doc_vecs, gold_idx)
    metrics = retrieval_metrics(ranks, ks=ks, ndcg_k=ndcg_k)
    metrics["Recall@1 自助法95%CI"] = list(bootstrap_ci([1.0 if r <= 1 else 0.0 for r in ranks]))
    metrics["Recall@5 自助法95%CI"] = list(bootstrap_ci([1.0 if r <= 5 else 0.0 for r in ranks]))
    metrics["命中排名列表"] = [None if not math.isfinite(r) else int(r) for r in ranks]
    metrics["查询列表"] = list(queries)
    metrics["金标准下标"] = [int(g) for g in gold_idx]
    return metrics


def eval_similarity(model: EmbeddingModel, rows: Sequence[dict], *, batch: int, max_length: int) -> dict:
    """正例对 / 负例对余弦相似度对比（用测试集三元组）。"""
    tri = [r for r in rows if r.get("格式") == "三元组" and r.get("negative")]
    if not tri:
        return {}
    pos_sims, neg_sims = [], []
    for i in range(0, len(tri), batch):
        b = tri[i : i + batch]
        va = model.encode([x["anchor"] for x in b], batch_size=batch, max_length=max_length)
        vp = model.encode([x["positive"] for x in b], batch_size=batch, max_length=max_length)
        vn = model.encode([x["negative"] for x in b], batch_size=batch, max_length=max_length)
        pos_sims.extend((va * vp).sum(-1).tolist())
        neg_sims.extend((va * vn).sum(-1).tolist())
    out = similarity_stats(pos_sims, neg_sims)
    out["三元组条数"] = len(tri)
    out["负例类型分布"] = _counter([x.get("负例类型") for x in tri])
    return out


def _counter(vals: Iterable[Any]) -> dict:
    d: dict[str, int] = {}
    for v in vals:
        d[str(v)] = d.get(str(v), 0) + 1
    return d


# --------------------------------------------------------------- bge-m3 参照
def ollama_embed(texts: Sequence[str], model: str = "bge-m3:latest", batch: int = 16,
                 timeout: int = 120) -> np.ndarray | None:
    """调用本机 Ollama 生成参照向量（失败返回 None，不裸崩）。"""
    import requests

    out: list[list[float]] = []
    try:
        for i in range(0, len(texts), batch):
            chunk = list(texts[i : i + batch])
            try:
                r = requests.post("http://127.0.0.1:11434/api/embed",
                                  json={"model": model, "input": chunk}, timeout=timeout)
                if r.status_code == 200:
                    out.extend(r.json().get("embeddings", []))
                    continue
            except Exception:  # noqa: BLE001
                pass
            for t in chunk:  # 退回单条接口
                r = requests.post("http://127.0.0.1:11434/api/embeddings",
                                  json={"model": model, "prompt": t}, timeout=timeout)
                r.raise_for_status()
                out.append(r.json()["embedding"])
    except Exception as exc:  # noqa: BLE001
        print(f"[评估][警告] Ollama 参照向量生成失败，跳过 bge-m3 参照：{exc}")
        return None
    v = np.asarray(out, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


# ------------------------------------------------------------------ 主流程
def compare(cfg: Config, *, 跳过端到端: bool = False, 缓存向量: bool = True,
            模型标签: str = "微调后", 模型目录: str | None = None) -> dict:
    t0 = time.perf_counter()
    e_cfg = cfg.hyper["评估"]
    ks = list(e_cfg.get("topk", [1, 5, 10]))
    real_ks = list(e_cfg.get("真实问题topk", [1, 3, 5, 10]))
    ndcg_k = int(e_cfg.get("ndcg_k", 10))
    batch = int(e_cfg.get("端到端编码批大小", 64))
    max_length = int(cfg.hyper_get("数据集", "最大序列长度", default=128))
    set_threads(int(cfg.hyper_get("训练", "线程数", default=8)))

    eval_dir = cfg.path("评估目录")
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir = cfg.path("对比结果目录")
    cmp_dir.mkdir(parents=True, exist_ok=True)

    pool = load_index_pool(cfg)
    hold_q, hold_gold, _hold_rows = load_holdout(cfg, pool)
    questions = load_real_questions(cfg)
    gold_sets, gold_detail = build_gold_sets(questions, pool)
    print(f"[评估] 真实问题 {len(questions)} 道；金标准分块数 "
          f"{[d['金标准分块数'] for d in gold_detail]}")

    test_rows = read_jsonl(cfg.path("数据集目录") / "test.jsonl")
    dev_rows = read_jsonl(cfg.path("数据集目录") / "dev.jsonl")
    sim_rows = test_rows or dev_rows

    # ---------------- 检索池抽样（时间预算约束下的口径） ----------------
    max_chunks = int(e_cfg.get("端到端最大分块数", 0) or 0)
    if max_chunks and len(pool["ids"]) > max_chunks:
        import random as _random
        must = set(int(g) for g in hold_gold)
        for gs in gold_sets:
            must |= {int(g) for g in gs}
        rng = _random.Random(int(cfg.hyper_get("数据集", "随机种子", default=20260826)))
        rest = [i for i in range(len(pool["ids"])) if i not in must]
        rng.shuffle(rest)
        keep = sorted(must | set(rest[: max(0, max_chunks - len(must))]))
        remap = {old: new for new, old in enumerate(keep)}
        pool = subset_pool(pool, keep)
        hold_gold = [remap[g] for g in hold_gold]
        gold_sets = [[remap[g] for g in gs if g in remap] for gs in gold_sets]
        print(f"[评估] 检索池抽样：{len(keep)} 块（含全部 {len(must)} 个金标准分块 + 随机抽样）；"
              f"原始索引 {len(remap)} 个已重映射")

    models = [("微调前", str(cfg.path("底座模型"))),
              (模型标签, str(模型目录 or cfg.path("微调模型目录")))]
    results: dict[str, Any] = {}
    for tag, mdir in models:
        if not Path(mdir).exists():
            print(f"[评估][警告] 模型目录不存在，跳过：{mdir}")
            continue
        print(f"\n===== 评估 {tag}（{mdir}）=====")
        mt = time.perf_counter()
        model = EmbeddingModel(mdir, max_length=max_length, device="cpu")
        res: dict[str, Any] = {
            "模型标签": tag, "模型路径": mdir, "向量维度": model.hidden_size,
            "最大序列长度": max_length, "设备": "cpu",
        }
        cache = eval_dir / f"向量_{tag}.npy"
        if 跳过端到端:
            res["端到端"] = {"跳过": True}
            doc_vecs = None
        elif 缓存向量 and cache.exists():
            doc_vecs = np.load(cache)
            print(f"[评估] 复用缓存向量 {cache} {doc_vecs.shape}")
        else:
            t1 = time.perf_counter()
            doc_vecs = model.encode(pool["texts"], batch_size=batch, max_length=max_length,
                                    progress_cb=lambda i, n: print(f"\r[评估] 编码索引 {i}/{n}", end="") if i % (batch * 20) == 0 else None)
            print()
            print(f"[评估] 索引编码耗时 {time.perf_counter()-t1:.1f}s")
            np.save(cache, doc_vecs)
        if doc_vecs is not None:
            res["留出测试集"] = eval_holdout(model, hold_q, hold_gold, doc_vecs, pool, ks=ks,
                                            ndcg_k=ndcg_k, batch=batch, max_length=max_length)
            res["真实问题"] = eval_query_set(
                model.encode([q["question"] for q in questions], batch_size=batch, max_length=max_length),
                doc_vecs, gold_sets, [q["question"] for q in questions],
                ks=real_ks, ndcg_k=ndcg_k, pool=pool)
            res["真实问题"]["关键词覆盖@10"] = keyword_coverage(
                model.encode([q["question"] for q in questions], batch_size=batch, max_length=max_length),
                doc_vecs, questions, pool, k=10)
        res["相似度指标"] = eval_similarity(model, sim_rows, batch=batch, max_length=max_length)
        res["评估耗时秒"] = round(time.perf_counter() - mt, 1)
        results[tag] = res
        (eval_dir / f"评估结果_{tag}.json").write_text(
            json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[评估] {tag} 完成，用时 {res['评估耗时秒']}s -> 研发/评估/评估结果_{tag}.json")

    # ------------------- bge-m3 参照 -------------------
    ref = None
    if e_cfg.get("bge-m3参照", True) and not 跳过端到端:
        try:
            vec_path = cfg.path("工单7索引目录") / "vectors.npy"
            if vec_path.exists() and hold_q:
                bge_docs = np.load(vec_path).astype("float32")
                if pool.get("原始索引"):
                    bge_docs = bge_docs[pool["原始索引"]]
                bge_docs = bge_docs / np.clip(np.linalg.norm(bge_docs, axis=1, keepdims=True), 1e-9, None)
                qv = ollama_embed(hold_q)
                if qv is not None and len(qv) == len(hold_q):
                    ref = {"模型": "bge-m3:latest（工单7 生产模型，未微调）", "向量维度": int(bge_docs.shape[1])}
                    ref["留出测试集"] = retrieval_metrics(
                        ranks_from_scores(qv, bge_docs, hold_gold), ks=ks, ndcg_k=ndcg_k)
                    rq = ollama_embed([q["question"] for q in questions])
                    if rq is not None:
                        ref["真实问题"] = retrieval_metrics(
                            ranks_from_gold_sets(rq, bge_docs, gold_sets)[0], ks=real_ks, ndcg_k=ndcg_k)
                        ref["真实问题"]["关键词覆盖@10"] = keyword_coverage(
                            rq, bge_docs, questions, pool, k=10)
                    print(f"[评估] bge-m3 参照：留出集 Recall@1={ref['留出测试集']['Recall@1']} "
                          f"MRR={ref['留出测试集']['MRR']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[评估][警告] bge-m3 参照失败：{exc}")

    # ------------------- 对比表 -------------------
    rows, summary = build_comparison(results, ref, ks, real_ks, ndcg_k)
    csv_path = cmp_dir / "对比表.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["指标"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[评估] 对比表 -> {csv_path}")

    per_q_path = cmp_dir / "逐query检索对照.csv"
    write_per_query(per_q_path, results, questions, hold_q, hold_gold, pool, gold_detail)

    out = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "总耗时秒": round(time.perf_counter() - t0, 1),
        "留出测试集条数": len(hold_q),
        "真实问题数": len(questions),
        "索引分块数": len(pool["ids"]),
        "金标准明细": gold_detail,
        "微调前": results.get("微调前"),
        "微调后": results.get("微调后"),
        "bge_m3参照": ref,
        "对比摘要": summary,
    }
    (cmp_dir / "对比.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "对比.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def keyword_coverage(query_vecs: np.ndarray, doc_vecs: np.ndarray, questions: Sequence[dict],
                     pool: dict, *, k: int = 10) -> float:
    """Top-K 检索结果对判分关键词的覆盖率（越接近 1 越好）。"""
    vals = []
    for i, q in enumerate(questions):
        kws = q.get("answer_keywords") or []
        if not kws:
            continue
        sims = doc_vecs @ query_vecs[i]
        order = np.argsort(-sims, kind="stable")[:k]
        best = 0
        for j in order:
            cnt, _ = keyword_hit_count(pool["norm"][int(j)], kws)
            best = max(best, cnt)
        vals.append(best / len(kws))
    return round(float(np.mean(vals)), 6) if vals else 0.0


def _g(d: dict, *keys, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return default
        cur = cur[key]
    return cur


def build_comparison(results: dict, ref: dict | None, ks: Sequence[int], real_ks: Sequence[int],
                     ndcg_k: int) -> tuple[list[dict], dict]:
    before, after = results.get("微调前", {}), results.get("微调后", {})
    specs = []
    for k in ks:
        specs.append((f"留出测试集", f"Recall@{k}", ("留出测试集", f"Recall@{k}"), "越高越好"))
    specs += [
        ("留出测试集", "MRR", ("留出测试集", "MRR"), "越高越好"),
        ("留出测试集", f"nDCG@{ndcg_k}", ("留出测试集", f"nDCG@{ndcg_k}"), "越高越好"),
        ("留出测试集", "平均命中排名", ("留出测试集", "平均命中排名"), "越低越好"),
        ("相似度指标", "正例对平均相似度", ("相似度指标", "正例对平均相似度"), "越高越好"),
        ("相似度指标", "负例对平均相似度", ("相似度指标", "负例对平均相似度"), "越低越好"),
        ("相似度指标", "相似度间隔", ("相似度指标", "相似度间隔"), "越高越好"),
        ("相似度指标", "正负例判别准确率", ("相似度指标", "正负例判别准确率"), "越高越好"),
    ]
    for k in real_ks:
        specs.append(("真实问题(工单7)", f"命中@{k}", ("真实问题", f"Recall@{k}"), "越高越好"))
    specs += [
        ("真实问题(工单7)", "MRR", ("真实问题", "MRR"), "越高越好"),
        ("真实问题(工单7)", "关键词覆盖@10", ("真实问题", "关键词覆盖@10"), "越高越好"),
    ]

    def ref_val(key):
        if not ref:
            return None
        if key[0] == "留出测试集":
            return _g(ref, "留出测试集", key[1])
        if key[0] == "真实问题":
            return _g(ref, "真实问题", key[1])
        return None

    rows, summary = [], {"提升指标": [], "下降指标": [], "持平指标": []}
    for group, label, key, direction in specs:
        b = _g(before, *key)
        a = _g(after, *key)
        r = ref_val(key)
        if b is None or a is None:
            continue
        delta = round(a - b, 6)
        rate = round(delta / abs(b), 6) if b not in (0, None) else None
        good = (delta > 0) if direction == "越高越好" else (delta < 0)
        verdict = "提升" if (good and abs(delta) > 1e-9) else ("下降" if (abs(delta) > 1e-9) else "持平")
        row = {
            "指标组": group, "指标": label, "方向": direction,
            "微调前": b, "微调后": a,
            "bge-m3参照": r,
            "变化": delta, "变化率": rate, "结论": verdict,
        }
        rows.append(row)
        (summary["提升指标"] if verdict == "提升" else summary["下降指标"] if verdict == "下降" else summary["持平指标"]).append(label)
    return rows, summary


def write_per_query(path: Path, results: dict, questions: Sequence[dict], hold_q: Sequence[str],
                    hold_gold: Sequence[int], pool: dict, gold_detail: Sequence[dict]) -> None:
    b = results.get("微调前", {}).get("真实问题", {}).get("逐query", [])
    a = results.get("微调后", {}).get("真实问题", {}).get("逐query", [])
    rows = []
    for i in range(max(len(b), len(a))):
        rb = b[i] if i < len(b) else {}
        ra = a[i] if i < len(a) else {}
        br, ar = rb.get("首个命中排名"), ra.get("首个命中排名")
        rows.append({
            "题号": (questions[i].get("id") if i < len(questions) else i + 1),
            "查询": (questions[i].get("question") if i < len(questions) else ""),
            "金标准分块数": rb.get("金标准分块数", ra.get("金标准分块数")),
            "微调前首个命中排名": br, "微调后首个命中排名": ar,
            "排名变化(正=改善)": (None if (br is None or ar is None) else br - ar),
            "微调前命中@10": int(br is not None and br <= 10),
            "微调后命中@10": int(ar is not None and ar <= 10),
            "微调前Top1文档": rb.get("Top1文档"), "微调后Top1文档": ra.get("Top1文档"),
        })
    # 留出测试集逐查询
    hb = results.get("微调前", {}).get("留出测试集", {})
    ha = results.get("微调后", {}).get("留出测试集", {})
    if hb.get("命中排名列表") and ha.get("命中排名列表"):
        for i, q in enumerate(hold_q):
            rb = hb["命中排名列表"][i] if i < len(hb["命中排名列表"]) else None
            ra = ha["命中排名列表"][i] if i < len(ha["命中排名列表"]) else None
            rows.append({
                "题号": f"留出-{i+1:03d}", "查询": q, "金标准分块数": 1,
                "微调前首个命中排名": rb, "微调后首个命中排名": ra,
                "排名变化(正=改善)": (None if (rb is None or ra is None) else rb - ra),
                "微调前命中@10": int(rb is not None and rb <= 10),
                "微调后命中@10": int(ra is not None and ra <= 10),
                "微调前Top1文档": None, "微调后Top1文档": None,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["题号"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[评估] 逐query对照 -> {path}（{len(rows)} 行）")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单11 微调前后评估")
    ap.add_argument("--跳过端到端", action="store_true", help="只做留出测试集与相似度指标（快）")
    ap.add_argument("--不缓存", action="store_true")
    ap.add_argument("--模型标签", default="微调后", help="评估标签（用于输出文件名与缓存名）")
    ap.add_argument("--模型目录", default=None, help="被评估的微调模型目录（默认取配置）")
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = Config()
    cfg.ensure_dirs()
    out = compare(cfg, 跳过端到端=args.跳过端到端, 缓存向量=not args.不缓存,
                  模型标签=args.模型标签, 模型目录=args.模型目录)
    s = out.get("对比摘要", {})
    print("\n===== 对比摘要 =====")
    print("提升：", s.get("提升指标"))
    print("下降：", s.get("下降指标"))
    print("持平：", s.get("持平指标"))
    print(f"总耗时 {out['总耗时秒']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
