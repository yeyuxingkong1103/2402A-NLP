# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：知识图谱构建入口（硬性要求1）

分层构建：
  第1层（本模块 build_layer1）—— 词典 + 正则，全量块，秒级，确定性；
  第2层（build_layer2）      —— LLM 定向关系抽取，见 kg_extract_llm。

构建流程：抽取 → 消歧归一 → 入图 → 原子持久化。
跨文档融合由 entity_id（`{类型}:{规范名}`）天然保证：同名同类型即同节点。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

from src import config
from src.kg_dedup import dedup_entities, remap_relations
from src.kg_extract import iter_extract
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def load_chunks() -> list[dict]:
    """读取复用工单07 的分块数据。"""
    path = os.path.join(config.CHUNK_DIR, "chunks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少分块数据 {path}\n"
            f"请先执行 Task 1 的数据复制步骤，或运行 scripts/setup_project.py")
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def build_layer1(chunks: list[dict], limit: int | None = None,
                 verbose: bool = True) -> KnowledgeGraph:
    """第1层：词典+正则全量抽取 → 消歧归一 → 建图。"""
    t0 = time.time()
    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    n_blocks_with_entities = 0

    for i, got in enumerate(iter_extract(chunks, limit=limit), start=1):
        if got["entities"]:
            n_blocks_with_entities += 1
        all_entities.extend(got["entities"])
        all_relations.extend(got["relations"])
        if verbose and i % 1000 == 0:
            print(f"  [layer1] 已处理 {i}/{len(chunks) if limit is None else limit} 块",
                  flush=True)

    merged = dedup_entities(all_entities)
    rels = remap_relations(all_relations, merged)

    kg = KnowledgeGraph()
    for e in merged.values():
        try:
            kg.add_entity(e)
        except ValueError as exc:                # noqa: PERF203
            print(f"  [layer1] 跳过非法实体：{exc}", file=sys.stderr)
    n_added = sum(1 for r in rels if kg.add_relation(r))

    if verbose:
        st = kg.stats()
        print(f"[layer1] 完成：{len(chunks)} 块（有实体 {n_blocks_with_entities}）→ "
              f"实体 {st['n_nodes']}（原始 {len(all_entities)}，归一后 "
              f"{len(merged)}）、关系 {st['n_edges']}（候选 {len(rels)}），"
              f"耗时 {time.time() - t0:.1f}s", flush=True)
    return kg


def build_layer2(kg: KnowledgeGraph, chunks: list[dict], resume: bool = True,
                 limit: int | None = None, verbose: bool = True) -> dict:
    """第2层：LLM 定向关系抽取并合并进图谱。

    流程：候选块选择（kg_select） → LLM 两段式抽取（kg_extract_llm）
          → 消歧归一（kg_dedup） → 合并入图（KnowledgeGraph.merge）
    """
    from src import kg_extract_llm, kg_select

    selected = kg_select.select_chunks(chunks, limit=limit)
    if verbose:
        print(f"[layer2] 候选块 {len(selected)} / {len(chunks)}", flush=True)

    out = kg_extract_llm.extract_llm(selected, resume=resume, verbose=verbose)

    # 归一 + 入图
    # 关键：把「图中已存在的节点」并入 merged，否则 LLM 抽到的
    # 「老实体 → 新实体」这类边会因为端点不在 merged 里而被 remap 丢弃。
    existing = {nid: kg.get_entity(nid) for nid in kg.entity_ids()}
    merged = {**existing, **dedup_entities(out["entities"])}

    sub = KnowledgeGraph()
    for e in merged.values():
        try:
            sub.add_entity(e)
        except ValueError:
            continue
    for r in remap_relations(out["relations"], merged):
        sub.add_relation(r)

    stat = kg.merge(sub)
    if verbose:
        print(f"[layer2] 合并入图：新增节点 {stat['added_nodes']}、"
              f"新增边 {stat['added_edges']}", flush=True)
    return {**out, **stat}


def save_all(kg: KnowledgeGraph) -> dict:
    """持久化：pickle + graphml + jsonl + stats。"""
    paths = kg.save()
    kg.dump_jsonl()
    st = kg.stats()
    with io.open(config.KG_STATS, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return {**paths, "stats_path": config.KG_STATS, "stats": st}


def build_all(chunks: list[dict] | None = None, with_llm: bool = False,
              limit: int | None = None, verbose: bool = True) -> dict:
    """完整构建：第1层（必做） + 第2层（可选）。"""
    if chunks is None:
        chunks = load_chunks()
    kg = build_layer1(chunks, limit=limit, verbose=verbose)
    if with_llm:
        build_layer2(kg, chunks, limit=limit, verbose=verbose)
    return save_all(kg)
