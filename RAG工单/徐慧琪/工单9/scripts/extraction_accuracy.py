# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：实体/关系抽取准确率评估（工单硬性要求1(1)）

## 工单要求

硬性要求1(1) 明确要「对比优化前后：**实体抽取准确率、关系抽取准确率、图谱覆盖度**」。
图谱覆盖度已由节点/边统计给出（08 = 560/873，09 = 594/912），
但准确率需要**人工标注**，无法只靠程序算出来。

## 本脚本怎么做（并说明它为什么是「抽样」而不是「全量」）

1. 从 09 图谱的 **LLM 层**（`attrs.layer == "llm"`）分层抽样若干实体与关系；
2. 对每一条，导出**判定所必需的全部材料**：实体名/类型、关系的头尾与
   `evidence`、以及该 evidence 所在块的**原文**；
3. 输出成一张复核表（Markdown + JSON），由人工逐条判定「正确 / 错误」；
4. 判定结果回填到 `data/eval/extraction_review.json` 后，本脚本算出准确率。

**为什么必须人工判**：LLM 抽取的错法是「语义错」——
实体共现但关系不成立（如用「某人在别家公司任独董」的证据去支撑
「该人在本公司任职」）。这类错误在字面上完全成立，任何基于字符串匹配的
自动判定都会把它判成「对」。实测已确认这种假阳性真实存在
（见 `docs/02-优化方案.md` §2.1 的 P1/P5 与闸门4/5 的由来）。

用法：
    python scripts/extraction_accuracy.py --sample 20     # 导出复核表
    python scripts/extraction_accuracy.py --score         # 回填后算准确率
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w9 import compat, config9  # noqa: E402

REVIEW_JSON = os.path.join(config9.EVAL9_DIR, "extraction_review.json")
REVIEW_MD = os.path.join(config9.DOCS9, "12-抽取准确率复核表.md")


def _load_kg():
    from src.kg_store import KnowledgeGraph
    return KnowledgeGraph.load(config9.KG9_PICKLE)


def _chunk_texts() -> dict:
    """(来源文档, chunk_id) → 原文。chunk_id 跨文档重复，必须用二元组作键。"""
    from w9.kg_merge import load_chunk_texts
    return load_chunk_texts()


def sample(n: int, seed: int = 42) -> dict:
    """分层抽样：LLM 层实体与关系各抽 n 条（不足则全取）。"""
    kg = _load_kg()
    texts = _chunk_texts()
    rnd = random.Random(seed)

    ents = []
    for nid, d in kg.g.nodes(data=True):
        attrs = d.get("attrs") or {}
        if attrs.get("layer") != "llm":
            continue
        ents.append({
            "id": nid, "name": d.get("name"), "type": d.get("type"),
            "aliases": d.get("aliases") or [],
            "source_doc": attrs.get("source_doc", ""),
            "page": attrs.get("page", -1),
        })

    rels = []
    for u, v, d in kg.g.edges(data=True):
        if (d.get("attrs") or {}).get("layer") != "llm":
            continue
        cid = (d.get("attrs") or {}).get("chunk_id", "")
        src = d.get("source_doc", "")
        body = texts.get((src, str(cid)), "")
        rels.append({
            "head": kg.g.nodes[u].get("name"), "head_id": u,
            "rel": d.get("rel_type"), "tail": kg.g.nodes[v].get("name"),
            "tail_id": v,
            "evidence": d.get("evidence", ""),
            "source_doc": src, "page": int(d.get("page", -1)),
            "chunk_id": cid,
            # 截取 evidence 前后的原文窗口，供人工判断「这句话到底在说什么」
            "source_window": _window(body, d.get("evidence", "")),
        })

    rnd.shuffle(ents)
    rnd.shuffle(rels)
    return {"entities": ents[:n], "relations": rels[:n],
            "n_entity_total": len(ents), "n_relation_total": len(rels)}


def _window(body: str, evidence: str, span: int = 140) -> str:
    """取 evidence 在原文中的上下文窗口。找不到则退回原文开头。"""
    if not body:
        return ""
    body_c = "".join(body.split())
    ev_c = "".join(str(evidence or "").split())
    idx = body_c.find(ev_c) if ev_c else -1
    if idx < 0:
        return body[:span * 2]
    lo = max(idx - span, 0)
    return body_c[lo: idx + len(ev_c) + span]


def render_md(data: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# 实体/关系抽取准确率 — 人工复核表")
    A("")
    A("> **工单编号**：人工智能NLP-RAG-Graph RAG优化任务")
    A(f"> 抽样范围：09 图谱的 **LLM 抽取层**"
      f"（实体共 {data['n_entity_total']} 条、关系共 {data['n_relation_total']} 条）")
    A(f"> 抽样量：实体 {len(data['entities'])} 条、关系 {len(data['relations'])} 条")
    A("")
    A("**判定口径**：")
    A("")
    A("- 实体「正确」= 该名称确实在原文中以该类型出现（含正确的指代消解/别名归一）；")
    A("- 关系「正确」= 原文的这句话**确实在陈述** head 与 tail 之间的该种关系"
      "（方向也要对）。")
    A("")
    A("> ⚠️ 关系判定必须看语义，不能只看「证据里有没有这两个名字」——")
    A("> 「某人在**别家公司**任独董」这类证据字面成立但关系不成立。")
    A("")

    A("## 一、实体抽样")
    A("")
    A("| # | 实体 | 类型 | 来源 | 页 | 判定 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for i, e in enumerate(data["entities"], 1):
        A(f"| E{i} | {e['name']} | {e['type']} | {e['source_doc'][:34]} | "
          f"{e['page']} | ☐ 对 / ☐ 错 |")
    A("")

    A("## 二、关系抽样（含原文证据窗口）")
    A("")
    for i, r in enumerate(data["relations"], 1):
        A(f"### R{i}. {r['head']} --{r['rel']}--> {r['tail']}")
        A("")
        A(f"- 来源：{r['source_doc']} 第 {r['page']} 页（id={r['chunk_id']}）")
        A(f"- evidence：`{r['evidence']}`")
        A(f"- 原文上下文：…{r['source_window'][:260]}…")
        A(f"- 判定：☐ 对 / ☐ 错　理由：")
        A("")

    A("## 三、回填与算分")
    A("")
    A("把判定结果填入 `data/eval/extraction_review.json` 的 `verdict.entities` /")
    A("`verdict.relations`（值为 `\"correct\"` 或 `\"wrong\"`），然后运行：")
    A("")
    A("```bash")
    A("python scripts/extraction_accuracy.py --score")
    A("```")
    return "\n".join(L)


def score() -> dict:
    """读取回填的判定结果，算准确率（含 Wilson 置信区间，样本量小时提示不确定性）。"""
    if not os.path.isfile(REVIEW_JSON):
        return {"error": f"未找到 {REVIEW_JSON}，请先运行 --sample"}
    with io.open(REVIEW_JSON, encoding="utf-8") as f:
        d = json.load(f)
    v = d.get("verdict") or {}

    def _acc(kind: str) -> dict:
        items = d.get(kind) or []
        marks = v.get(kind) or {}
        judged = {k: x for k, x in marks.items() if x in ("correct", "wrong")}
        n = len(judged)
        c = sum(1 for x in judged.values() if x == "correct")
        if not n:
            return {"n_judged": 0, "n_total": len(items), "accuracy": None,
                    "note": "尚未回填判定"}
        p = c / n
        # Wilson 95% 区间：小样本下比 p 本身更有说明力
        z = 1.96
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        return {"n_judged": n, "n_total": len(items), "n_correct": c,
                "accuracy": round(p, 4),
                "wilson_95": [round(max(center - half, 0), 4),
                              round(min(center + half, 1), 4)]}

    return {"entities": _acc("entities"), "relations": _acc("relations"),
            "note": "准确率来自**人工抽样判定**；样本量小，请一并看 Wilson 区间。"}


def main() -> int:
    ap = argparse.ArgumentParser(description="实体/关系抽取准确率（人工抽样）")
    ap.add_argument("--sample", type=int, default=0, help="抽样条数并导出复核表")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", action="store_true", help="按已回填的判定算准确率")
    a = ap.parse_args()

    if a.score:
        print(json.dumps(score(), ensure_ascii=False, indent=2))
        return 0

    if not a.sample:
        ap.print_help()
        return 1

    data = sample(a.sample, a.seed)
    data["verdict"] = data.get("verdict", {"entities": {}, "relations": {}})
    os.makedirs(config9.EVAL9_DIR, exist_ok=True)
    with io.open(REVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.makedirs(config9.DOCS9, exist_ok=True)
    with io.open(REVIEW_MD, "w", encoding="utf-8") as f:
        f.write(render_md(data))
    print(f"已写出 {REVIEW_JSON}\n已写出 {REVIEW_MD}")
    print(f"（LLM 层共 {data['n_entity_total']} 实体 / "
          f"{data['n_relation_total']} 关系，抽样各 {a.sample}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
