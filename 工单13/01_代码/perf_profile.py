# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统优化 | 人工智能NLP-RAG-混合检索任务
# 模块：evaluation/perf_profile —— RAG 分阶段耗时剖析（工单 13）
# 说明：对检索链路逐阶段打点（Query理解 / 嵌入 / 向量召回 / BM25 / 融合 / 重排 / 上下文组装 / 生成），
#       跑题集统计 mean/median/p95，产出 `evaluation/性能剖析结果.json` 与 `.md`。
# 用法：python evaluation/perf_profile.py [--n 20] [--gen] [--cprofile]
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine       # noqa: E402
import kb as kb_mod  # noqa: E402
import llm          # noqa: E402
import prompts      # noqa: E402

T = {}


def _add(name, ms):
    T.setdefault(name, []).append(ms)


class _Stop:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        _add(self.name, (time.perf_counter() - self.t0) * 1000)


def patch():
    """对关键函数打点（外层总时长 - 内层 = 自身开销）。"""
    o_embed = llm.embed

    def embed(texts, **kw):
        with _Stop("2_embed(HTTP)"):
            return o_embed(texts, **kw)
    llm.embed = embed

    o_dense = kb_mod.KB.search_dense

    def search_dense(self, *a, **kw):
        with _Stop("3_向量召回(含embed)"):
            return o_dense(self, *a, **kw)
    kb_mod.KB.search_dense = search_dense

    o_bm25 = kb_mod.KB.search_bm25

    def search_bm25(self, *a, **kw):
        with _Stop("4_BM25检索"):
            return o_bm25(self, *a, **kw)
    kb_mod.KB.search_bm25 = search_bm25

    o_fuse = engine.fuse

    def fuse(*a, **kw):
        with _Stop("5_融合(RRF)"):
            return o_fuse(*a, **kw)
    engine.fuse = fuse

    o_qu = engine.query_understanding

    def query_understanding(q, **kw):
        with _Stop("1_Query理解"):
            return o_qu(q, **kw)
    engine.query_understanding = query_understanding

    o_ctx = prompts.build_context

    def build_context(hits, **kw):
        with _Stop("6_上下文组装"):
            return o_ctx(hits, **kw)
    prompts.build_context = build_context

    o_chat = llm.chat

    def chat(*a, **kw):
        with _Stop("7_生成(LLM)"):
            return o_chat(*a, **kw)
    llm.chat = chat


def questions(n):
    qs = []
    p1 = os.path.join(HERE, "eval_questions.json")
    p1b = os.path.join(HERE, "eval_questions_en.json")
    for p in (p1, p1b):
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            items = d.get("questions") or d.get("items") or []
            for it in items:
                q = it.get("question") or it.get("q")
                if q:
                    qs.append({"id": it.get("id"), "q": q})
    return qs[:n] if n else qs


def stats(v):
    v = sorted(v)
    return {"n": len(v), "mean": round(statistics.mean(v), 1),
            "median": round(statistics.median(v), 1),
            "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 1),
            "max": round(max(v), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gen", action="store_true", help="同时测生成（默认只测检索，验收口径是检索 ≤3s）")
    ap.add_argument("--out", default="性能剖析结果")
    a = ap.parse_args()
    patch()
    km = kb_mod.get_kb()
    print("KB chunks =", len(km.chunks), "| embed =", km.embed_name())
    qs = questions(a.n)
    print("题目数 =", len(qs), "| 是否含生成 =", a.gen)

    tot = []
    for i, it in enumerate(qs, 1):
        t0 = time.perf_counter()
        with _Stop("0_检索总时长"):
            hits, conf = engine.retrieve(it["q"], top_k=5)
        rt = (time.perf_counter() - t0) * 1000
        tot.append(rt)
        if a.gen and hits and conf >= engine.REFUSE_SCORE:
            engine.answer(it["q"])
        print("  [%2d/%d] %6.0f ms  top1=p%-3s conf=%.3f  %s" %
              (i, len(qs), rt, hits[0]["page"] if hits else "-", conf, it["q"][:34]))

    out = {"kb_chunks": len(km.chunks), "embed_model": km.embed_name(),
           "n_questions": len(qs), "with_gen": a.gen, "stages": {}, "total_ms": stats(tot)}
    for k in sorted(T):
        out["stages"][k] = stats(T[k])
    print("\n==== 分阶段耗时（ms）====")
    for k in sorted(out["stages"]):
        s = out["stages"][k]
        print("  %-22s mean=%7.1f  median=%7.1f  p95=%7.1f  max=%7.1f  (n=%d)" %
              (k, s["mean"], s["median"], s["p95"], s["max"], s["n"]))
    s = out["total_ms"]
    print("  %-22s mean=%7.1f  median=%7.1f  p95=%7.1f  max=%7.1f" %
          ("检索总时长", s["mean"], s["median"], s["p95"], s["max"]))
    json.dump(out, open(os.path.join(HERE, a.out + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved -> evaluation/%s.json" % a.out)


if __name__ == "__main__":
    main()
