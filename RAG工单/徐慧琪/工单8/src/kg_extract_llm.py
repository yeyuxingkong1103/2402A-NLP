# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：第2层抽取 —— LLM 定向关系抽取（硬性要求3）

在 kg_select 选出的高价值块上跑 qwen2.5:3b，抽取第1层词典覆盖不到的
长尾关系（控股链、任职、投资、发行等）。

两段式调用（本模块的关键工程决策）：
  qwen2.5:3b 单次同时输出实体+关系时，会在实体数组里退化重复、在
  num_predict 处被截断，**一个关系对象都来不及输出**（Task 10 探针：5 块
  关系全为 0）。故拆成两次有界小输出调用：
    ① 实体调用：最多 8 个实体，输出短 → 模型来不及退化即自然收尾；
    ② 关系调用：把①**已接受**的实体清单喂回去，最多 6 条关系 ——
       端点检查因此有东西可匹配，evidence 闸门照常生效。
  截断抢救（parse_llm_json/salvage）与三道防幻觉闸门（to_entities_relations）
  完全复用，不因拆分而放松。

工程要点：
  · **断点续抽**：每块完成即追加 JSONL checkpoint，进程中断后 --resume 不丢进度；
  · **抽取结果全量持久化**（工单08修正）：checkpoint 只是「已完成块索引」
    （计数），resume 时凭它无法还原抽取产物；故成功块同时把 Entity/Relation
    全量序列化追加到 KG_LLM_RESULTS，resume 直接从结果文件重建累积结果，
    否则「越跑到后面、重启丢得越多」；
  · **单块失败不中断整批**（容错要求7）：失败记录 error，继续下一块；
  · **可注入 call_llm**：便于单元测试不依赖 Ollama；
  · 三道防幻觉闸门在 kg_prompt.to_entities_relations 中执行。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

from src import config
from src.kg_prompt import (build_entities_messages, build_relations_messages,
                           parse_llm_json, to_entities_relations)
from src.kg_schema import Entity, Relation

_ckpt_cache: set[str] = set()


def load_checkpoint() -> set[str]:
    """读取已完成的 chunk_id 集合。"""
    path = config.KG_LLM_CKPT
    done: set[str] = set()
    if os.path.isfile(path):
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    _ckpt_cache.clear()
    _ckpt_cache.update(done)
    return done


def append_checkpoint(record: dict) -> None:
    """追加一条完成记录（JSONL，崩溃安全：单行追加自带原子性）。"""
    parent = os.path.dirname(config.KG_LLM_CKPT)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with io.open(config.KG_LLM_CKPT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if "chunk_id" in record:
        _ckpt_cache.add(record["chunk_id"])


def append_results(record: dict) -> None:
    """追加一条**抽取结果**（实体/关系已序列化）。

    与 checkpoint 分离：checkpoint 保持「只记计数」的旧格式（兼容暂停任务
    已写下的历史行），本文件才是 resume 时还原产物的数据源。
    崩溃安全：单行追加（O_APPEND 下单次 write 带 \\n）自带原子性，
    绝不整文件重写。
    """
    parent = os.path.dirname(config.KG_LLM_RESULTS)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with io.open(config.KG_LLM_RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results(path: str | None = None) -> list[dict]:
    """读回抽取结果 JSONL，重建 Entity/Relation 对象。

    返回 [{"chunk_id", "entities": [Entity...], "relations": [Relation...]}]。
    同一 chunk_id 出现多次时**后写者胜**（结果先写、checkpoint 后写，
    二者之间崩溃会导致重抽并再追加一份，以最后一次为准即可自愈）。
    文件不存在/行损坏 → 跳过（历史 checkpoint 无对应结果文件属预期情况）。
    """
    path = path or config.KG_LLM_RESULTS
    by_chunk: dict[str, dict] = {}
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec["chunk_id"]
                raw_entities = rec.get("entities") or []
                raw_relations = rec.get("relations") or []
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            try:
                entities = [
                    Entity(entity_id=str(e.get("entity_id", "")),
                           name=str(e.get("name", "")),
                           type=str(e.get("type", "")),
                           aliases=[str(a) for a in (e.get("aliases") or [])],
                           attrs=dict(e.get("attrs") or {}))
                    for e in raw_entities]
                relations = [
                    Relation(head_id=str(r.get("head_id", "")),
                             rel_type=str(r.get("rel_type", "")),
                             tail_id=str(r.get("tail_id", "")),
                             evidence=str(r.get("evidence", "")),
                             source_doc=str(r.get("source_doc", "")),
                             page=int(r.get("page", -1)),
                             confidence=float(r.get("confidence", 1.0)),
                             attrs=dict(r.get("attrs") or {}))
                    for r in raw_relations]
            except (AttributeError, TypeError, ValueError):
                continue                   # 单条记录畸形 → 跳过，不拖垮整个文件
            by_chunk[cid] = {"chunk_id": cid, "entities": entities,
                             "relations": relations}
    return list(by_chunk.values())


def _results_record(chunk_id: str, entities, relations) -> dict:
    """Entity/Relation 对象 → 可 JSON 序列化的结果记录。"""
    return {"chunk_id": chunk_id,
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations]}


def _default_call_llm(messages: list[dict]) -> str:
    """默认走 Ollama；延迟导入避免测试时拉起 GPU。

    注意：上游 `llm.chat()` 返回 dict（{"answer","elapsed","model",...}，
    见 src/llm.py），本函数统一转成字符串，
    使 call_llm 的契约恒为 `messages -> str`。
    """
    from src import llm
    res = llm.chat(messages, temperature=0.0, num_predict=1024,
                   timeout=config.KG_LLM_TIMEOUT_S)
    return str(res.get("answer", "")) if isinstance(res, dict) else str(res)


def _accepted_entity_payload(entities) -> list[dict]:
    """已接受的 Entity → payload dict，供关系调用的端点匹配重建 payload。"""
    return [{"name": e.name, "type": e.type, "aliases": list(e.aliases),
             "attrs": {}} for e in entities]


def _extract_two_calls(chunk: dict, call_llm) -> tuple[list, list]:
    """两段式抽取：①实体 → ②关系（喂回已接受实体名）。异常上抛给重试循环。"""
    payload_e = parse_llm_json(call_llm(build_entities_messages(chunk)))
    entities, _ = to_entities_relations(payload_e, chunk)

    if not entities:
        # 没有已接受实体 → 关系端点必然匹配不上，省一次 LLM 调用。
        return entities, []

    raw = call_llm(build_relations_messages(chunk, entities))
    payload_r = parse_llm_json(raw)
    # 关系调用的 payload 用①已接受的实体重建：端点检查严格限定在它们之中，
    # 关系调用自身若又输出实体也一律忽略（防重复计入）。
    payload_merged = {"entities": _accepted_entity_payload(entities),
                      "relations": payload_r.get("relations") or []}
    _, relations = to_entities_relations(payload_merged, chunk)
    return entities, relations


def extract_one(chunk: dict, call_llm=None) -> dict:
    """单块两段式 LLM 抽取，带重试。任何异常都转成 error 字段，不向外抛。"""
    call_llm = call_llm or _default_call_llm
    cid = chunk.get("chunk_id", "")

    last_err: str | None = None
    for attempt in range(config.KG_LLM_RETRY + 1):
        try:
            entities, relations = _extract_two_calls(chunk, call_llm)
            return {"chunk_id": cid, "entities": entities, "relations": relations,
                    "error": None, "attempts": attempt + 1}
        except Exception as exc:                       # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < config.KG_LLM_RETRY:
                time.sleep(0.5 * (attempt + 1))
    return {"chunk_id": cid, "entities": [], "relations": [],
            "error": last_err, "attempts": config.KG_LLM_RETRY + 1}


def extract_llm(selected: list[dict], resume: bool = True, limit: int | None = None,
                call_llm=None, verbose: bool = True) -> dict:
    """批量抽取。

    selected：kg_select.select_chunks 的输出（[{"chunk","score","reasons"}]）。
    返回 {"entities","relations","n_ok","n_fail","n_skipped",
          "n_skipped_no_results"}。

    resume=True 时，checkpoint 中已完成的块不再调用 LLM，而是从
    KG_LLM_RESULTS 读回其持久化产物并入返回值 —— 跨进程的
    「抽到哪、图就能并到哪」由此成立。历史 checkpoint 无对应结果
    （旧版只写了计数）的块按跳过处理并记日志，不抛异常。
    """
    done = load_checkpoint() if resume else set()
    persisted = ({rec["chunk_id"]: rec for rec in load_results()}
                 if resume else {})
    all_entities, all_relations = [], []
    n_ok = n_fail = n_skipped = n_no_result = 0
    t0 = time.time()

    todo = selected[:limit] if limit is not None else selected
    total = len(todo)

    for i, item in enumerate(todo, start=1):
        chunk = item["chunk"]
        cid = chunk.get("chunk_id", "")
        if cid in done:
            n_skipped += 1
            rec = persisted.get(cid)
            if rec is None:
                # 兼容旧 checkpoint：计了数但产物从未落盘，该块结果不可恢复。
                n_no_result += 1
                if verbose:
                    print(f"  [layer2] 块 {cid} 在 checkpoint 中但无持久化结果"
                          f"（旧版断点，产物已丢失），按跳过处理；"
                          f"如需重抽请从 {config.KG_LLM_CKPT} 删除该行",
                          file=sys.stderr)
            else:
                all_entities.extend(rec["entities"])
                all_relations.extend(rec["relations"])
            continue

        got = extract_one(chunk, call_llm=call_llm)
        if got["error"] is None:
            n_ok += 1
            all_entities.extend(got["entities"])
            all_relations.extend(got["relations"])
            # 先写结果、后写完成索引：两步之间崩溃只会重抽该块（load_results
            # 后写者胜），绝不会「索引说做完了、产物却没落盘」。
            append_results(_results_record(cid, got["entities"], got["relations"]))
            append_checkpoint({"chunk_id": cid, "n_entities": len(got["entities"]),
                               "n_relations": len(got["relations"]),
                               "error": None})
        else:
            n_fail += 1
            append_checkpoint({"chunk_id": cid, "n_entities": 0, "n_relations": 0,
                               "error": got["error"]})
            if verbose:
                print(f"  [layer2] 块 {cid} 抽取失败：{got['error']}",
                      file=sys.stderr)

        if verbose and i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate if rate else 0
            print(f"  [layer2] {i}/{total}  ok={n_ok} fail={n_fail} "
                  f"skip={n_skipped}  已用 {elapsed / 60:.1f}min  "
                  f"预计剩余 {eta / 60:.1f}min", flush=True)

    if verbose:
        extra = (f"（其中 {n_no_result} 块无持久化结果、产物不可恢复）"
                 if n_no_result else "")
        print(f"[layer2] 完成：成功 {n_ok} / 失败 {n_fail} / 跳过 {n_skipped}{extra}，"
              f"实体 {len(all_entities)}、关系 {len(all_relations)}，"
              f"耗时 {(time.time() - t0) / 60:.1f}min", flush=True)
    return {"entities": all_entities, "relations": all_relations,
            "n_ok": n_ok, "n_fail": n_fail, "n_skipped": n_skipped,
            "n_skipped_no_results": n_no_result}
