# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：【层面1】Prompt v2 定向补抽

## 为什么要「定向」而不是「全量重抽」

全量重抽要覆盖 7757 块，本机 qwen2.5:3b 实测约 4~6 小时 —— 与工单「2 人日」
的工时预估不成比例，且边际收益递减：图谱真正要补的是**能回答问题的关系**，
而不是把每一页年报都过一遍。

因此本模块做两件事：
  1. **复用** 08 已抽的 349 块结果（那是已经花掉的算力，不重复投入）；
  2. **定向补抽**与工单评测问题相关的高价值块 —— 判定依据是
     块文本命中了「问题主题词表」（董事长致辞、风险管理、拨备覆盖率、
     绿色金融、科技金融、资本结构、创新商业模式…）。

这样既满足工单「基于 08 优化、不另起炉灶」的要求，又把算力压在真正影响
评测指标的块上。

## 断点续抽

每块完成即追加 JSONL，进程中断后 --resume 从已完成块继续，不丢进度。
（与 08 `kg_extract_llm` 相同策略：崩溃安全靠单行 append 的原子性。）
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time

from w9 import config9

from src import config as c8
from src import llm
from src.kg_prompt import (build_entities_messages, build_relations_messages,
                           parse_llm_json, to_entities_relations)
from src.kg_schema import Entity, Relation

# Prompt 加载器用 09 自己的：v2 Prompt 在 工单9/w9/prompts/，
# 08 的 load_prompt 只会去 工单8/prompts/ 找（见 w9/prompts_loader.py 说明）。
# build_*_messages 由 08 提供（它们接受 prompt dict 参数，可直接喂 v2）。
from w9.prompts_loader import load_prompt

# ---------------------------------------------------------------------------
# 高价值块的主题词表（依据工单评测问题集）
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "董事长致辞": ("董事长致辞", "致股东", "行长致辞", "致辞"),
    "风险管理": ("风险管理", "风险管控", "全面风险管理", "风险偏好",
                 "信用风险", "市场风险", "操作风险", "压力测试", "风控"),
    "资产质量": ("不良贷款", "逾期", "拨备", "资产质量", "五级分类", "核销"),
    "资本结构": ("资本充足率", "核心一级资本", "资本补充", "内生资本",
                 "利润留存", "资产负债匹配", "偿付能力"),
    "绿色金融": ("绿色金融", "绿色信贷", "绿色债券", "环保", "ESG",
                 "可持续金融", "碳中和", "绿色能源"),
    "科技金融": ("科技金融", "金融科技", "数字化转型", "人工智能", "大数据",
                 "区块链", "云计算", "信息科技投入"),
    "创新商业模式": ("创新", "商业模式", "生态", "场景", "跨界",
                     "逾越者联盟", "合作伙伴", "开放银行", "平台"),
    "股东治理": ("控股股东", "实际控制人", "持股比例", "股权结构",
                 "股东结构", "发起人", "关联方"),
    "董监高": ("董事", "监事", "高级管理人员", "任职", "聘任", "履历"),
    "贷款结构": ("贷款结构", "信贷投放", "贷款投向", "行业集中度",
                 "基础设施", "对公贷款", "零售贷款"),
    "盈利增长": ("营业收入", "净利润", "盈利", "增长", "同比", "增速"),
}


def _topic_hits(text: str) -> list[str]:
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in text for k in kws)]


def load_chunks() -> list[dict]:
    path = config9.WO8_CHUNKS
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def select_blocks(limit: int | None = None,
                  exclude_done: bool = True) -> list[dict]:
    """选出高价值块：命中主题词表最多者优先，同分则长块优先（信息量大）。

    排除 08 已抽过的块（按 (source, chunk_id) 判定，单用 chunk_id 会跨文档串味）。
    """
    chunks = load_chunks()
    done: set[tuple[str, str]] = set()
    if exclude_done and os.path.isfile(config9.WO8_LLM_RESULTS):
        for rec in _iter_jsonl(config9.WO8_LLM_RESULTS):
            cid = str(rec.get("chunk_id") or "")
            src = ""
            for e in (rec.get("entities") or []):
                if isinstance(e, dict) and isinstance(e.get("attrs"), dict):
                    src = e["attrs"].get("source_doc") or ""
                    if src:
                        break
            done.add((src, cid))

    scored: list[tuple[int, int, dict]] = []
    for c in chunks:
        text = c.get("text") or ""
        if len(text) < 60:
            continue
        if (c.get("source", ""), str(c.get("chunk_id"))) in done:
            continue
        hits = _topic_hits(text)
        if not hits:
            continue
        scored.append((len(hits), len(text), {**c, "_topics": hits}))

    # 命中主题数降序 → 文本长度降序
    scored.sort(key=lambda x: (-x[0], -x[1]))
    picked = [c for _h, _l, c in scored]
    return picked[:limit] if limit else picked


def _iter_jsonl(path: str):
    if not os.path.isfile(path):
        return
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_done_keys() -> set[tuple[str, str]]:
    """已补抽完成的块键（断点续抽）。"""
    out: set[tuple[str, str]] = set()
    for rec in _iter_jsonl(config9.KG9_EXTRACT_V2):
        out.add((str(rec.get("source") or ""), str(rec.get("chunk_id") or "")))
    return out


def append_result(rec: dict) -> None:
    parent = os.path.dirname(config9.KG9_EXTRACT_V2)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with io.open(config9.KG9_EXTRACT_V2, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call_llm(messages: list[dict], **kw) -> str:
    res = llm.chat(messages, timeout=config9.EXTRACT_V2_TIMEOUT_S,
                   num_predict=kw.get("num_predict", 2048),
                   temperature=0.0)
    return str(res.get("answer", "")) if isinstance(res, dict) else str(res)


def _dedupe_entities(ents: list[Entity]) -> tuple[list[Entity], int, int]:
    """折叠重复实体，返回 (去重后, 原始条数, 去重条数)。

    ## 为什么必须有这一步（实测结论，重要）

    qwen2.5:3b 在这个抽取任务上会**严重退化**：反复输出同一个实体名直到
    触及 num_predict 上限。实测（2026-10-15，本机）：

        输入：招商银行 2019 年报第 5 页「董事长致辞」（干净散文，无表格噪声）
        v1（工单08 Prompt）：实体 8 条 → 去重后 1 条（「招商银行」×8），关系 0
        v2（本工单 Prompt）：实体 14 条 → 去重后 2 条，关系 1

    两种 Prompt 都退化，说明**这不是 Prompt 写得好不好的问题，而是
    3B 参数量的模型能力上限** —— 它无法稳定完成「多实体 + 严格 JSON」的
    结构化抽取。08 的 Prompt 里已经写了三条针对退化的禁令
    （「每个实体只列一次」「严禁进入重复循环」「输出到最后一个大括号立即停止」），
    模型全部无视。

    工程上的正确应对不是继续调 Prompt，而是**在代码层兜住**：
    折叠重复、并用退化率量化模型可靠度，把结论如实写进报告。
    """
    seen: dict[tuple[str, str], Entity] = {}
    for e in ents:
        key = (e.type, e.name)
        if key not in seen:
            seen[key] = e
        else:                                    # 合并别名
            for a in e.aliases:
                if a not in seen[key].aliases:
                    seen[key].aliases.append(a)
    out = list(seen.values())
    return out, len(ents), len(out)


def extract_one(chunk: dict, prompt: dict | None = None,
                call_llm=None) -> dict:
    """两段式抽取单块：① 实体 → ② 关系（把已接受实体喂回去做端点白名单）。"""
    prompt = prompt or load_prompt("kg_extract_v2")
    call = call_llm or _call_llm

    # ① 实体
    ents: list[Entity] = []
    try:
        raw = call(build_entities_messages(chunk, prompt), num_predict=1024)
        payload = parse_llm_json(raw)
        ents, _ = to_entities_relations(payload, chunk)
    except Exception as exc:                       # noqa: BLE001
        return {"error": f"实体抽取失败：{type(exc).__name__}: {exc}",
                "chunk_id": chunk.get("chunk_id"), "source": chunk.get("source")}

    ents, n_raw, n_uniq = _dedupe_entities(ents)
    # 退化率：重复条数占比。>0.5 视为该块抽取不可靠
    degenerate = n_raw > 3 and (n_raw - n_uniq) / max(n_raw, 1) > 0.5
    # 只剩一个实体时无法构成任何关系，关系调用纯属浪费 → 跳过
    if len(ents) < 2:
        return {
            "chunk_id": chunk.get("chunk_id"), "source": chunk.get("source"),
            "page_idx": chunk.get("page_idx"), "topics": chunk.get("_topics", []),
            "prompt_version": 2,
            "entities": [e.to_dict() for e in ents], "relations": [],
            "n_raw": n_raw, "n_uniq": n_uniq, "degenerate": degenerate,
            "rel_error": "", "skipped_relations": "去重后实体不足 2 个",
        }

    # ② 关系（喂回去重后的实体）
    rels: list[Relation] = []
    try:
        raw2 = call(build_relations_messages(chunk, ents, prompt),
                    num_predict=1024)
        payload2 = parse_llm_json(raw2)
        payload2["entities"] = [
            {"name": e.name, "type": e.type, "aliases": e.aliases,
             "attrs": e.attrs} for e in ents]
        _, rels = to_entities_relations(payload2, chunk)
    except Exception as exc:                       # noqa: BLE001
        rels = []
        rel_err = f"{type(exc).__name__}: {exc}"
    else:
        rel_err = ""

    return {
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "page_idx": chunk.get("page_idx"),
        "topics": chunk.get("_topics", []),
        "prompt_version": 2,
        "entities": [e.to_dict() for e in ents],
        "relations": [r.to_dict() for r in rels],
        "n_raw": n_raw, "n_uniq": n_uniq, "degenerate": degenerate,
        "rel_error": rel_err,
    }


def run(limit: int = 200, resume: bool = True, verbose: bool = True) -> dict:
    """定向补抽主流程。"""
    prompt = load_prompt("kg_extract_v2")
    blocks = select_blocks(limit=None)
    done = load_done_keys() if resume else set()
    todo = [b for b in blocks
            if (b.get("source", ""), str(b.get("chunk_id"))) not in done]
    todo = todo[:limit]

    if verbose:
        print(f"[extract_v2] 候选高价值块 {len(blocks)}，已完成 {len(done)}，"
              f"本轮待抽 {len(todo)}", flush=True)
    t0 = time.perf_counter()
    ok = err = n_ent = n_rel = 0
    for i, ch in enumerate(todo, 1):
        rec = extract_one(ch, prompt)
        append_result(rec)
        if rec.get("error"):
            err += 1
        else:
            ok += 1
            n_ent += len(rec.get("entities") or [])
            n_rel += len(rec.get("relations") or [])
        if verbose and (i % 10 == 0 or i == len(todo)):
            el = time.perf_counter() - t0
            eta = (el / i) * (len(todo) - i)
            print(f"[extract_v2] {i}/{len(todo)}  实体 {n_ent}  关系 {n_rel}  "
                  f"失败 {err}  已用 {el:.0f}s  预计剩余 {eta:.0f}s", flush=True)
    out = {"blocks": len(todo), "ok": ok, "error": err,
           "entities": n_ent, "relations": n_rel,
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "out": config9.KG9_EXTRACT_V2}
    if verbose:
        print(f"[extract_v2] 完成：{out}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prompt v2 定向补抽")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--stats", action="store_true", help="只统计候选块，不抽取")
    a = ap.parse_args()
    if a.stats:
        bs = select_blocks(limit=None)
        from collections import Counter
        c = Counter(t for b in bs for t in b["_topics"])
        print(f"候选高价值块：{len(bs)}（08 已抽的已排除）")
        for k, v in c.most_common():
            print(f"  {k}: {v}")
    else:
        run(limit=a.limit, resume=not a.no_resume)
