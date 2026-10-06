#!/usr/bin/env python3
"""长期记忆**重复/膨胀**体检（B-9 后续：召回会被近似重复记录稀释）。

为什么需要它：`message_id_for(user_id, session_id, role, content)` 是**内容寻址**的。
用户消息的内容就是他自己的问题（同一会话里重问 ⇒ 同一个 ID，幂等）；
但**助手消息的内容是生成出来的**，逐字会变 ⇒ 同一话题每问一次就多写一条新记录。
实机现象：同一个评测账号连灌两遍同样的 10 轮背景对话，长期记忆 **36 → 50 条**。

本脚本**只读**，给出可判定的体检读数：

* 某用户下按 ``role`` 分组的记录数；
* **完全重复**（正文逐字相同）的记录数与占比；
* **同问不同答**（同一会话里 `role=user` 正文相同、但对应助手记录多份）的组数；
* 估计的"有效信息量"（去重后还剩多少条）。

⚠️ Milvus Lite 是**单进程文件库**：跑之前必须停 API，否则拿不到锁
（或静默回落到内存库 ⇒ 读到空库、给出假结论）。

用法（云端）::

    bash -c 'source /root/env_cloud.sh; python scripts/audit_memory_duplicates.py \\
        --user eval-recall-ab --out eval/results/memory-duplicates.json'
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import setup_utf8_stdout  # noqa: E402


def _load_records(user_id: str) -> tuple[Any, list[Any]]:
    """打开长期记忆 collection 并读该用户的全部记录（含 store 本身，供调用方核对后端）。"""
    from legal_rag.config import RagConfig
    from legal_rag.embedding.base import build_embedder
    from legal_rag.memory.longterm import DEFAULT_MEMORY_COLLECTION, memory_doc_id
    from legal_rag.store.base import build_store

    config = RagConfig.from_env()
    if str(config.vector_store).strip().lower() == "memory":
        raise SystemExit("!! VECTOR_STORE=memory（没 source 云端 env）⇒ 读到的是空内存库，"
                         "结论不可信。请先 source /root/env_cloud.sh")
    # ⚠️ 必须给 embedder：否则 store 拿不到实测维度 ⇒ **静默降级成内存库**（空库假通过）
    embedder = build_embedder(config.embedding_provider, config.embedding_model,
                              config.embedding_dim, config=config)
    store = build_store(config.vector_store, persist_dir=config.index_dir,
                        collection=DEFAULT_MEMORY_COLLECTION, config=config,
                        embedder=embedder)
    if getattr(store, "name", "") == "memory":
        raise SystemExit("!! 记忆库回落到内存（多半是 API 没停、Milvus Lite 抢锁）"
                         "⇒ 看到的是空库，结论不可信")
    rows = list(store.all_chunks({"doc_id": memory_doc_id(user_id)}))
    return store, rows


def _meta_of(chunk: Any) -> dict:
    try:
        return json.loads(getattr(chunk, "summary", "") or "{}")
    except (TypeError, ValueError):
        return {}


def _session_of(chunk: Any) -> str:
    """从 ``memory://u/<user>/s/<session>/m/<mid>`` 里取会话（**按下标 5**）。

    ⚠️ 这里踩过一次 off-by-one：``split("/")`` 后下标 4 是字面量 ``"s"``，
    会话在下标 5；取错会得到"所有记录都在会话 s 里"的假读数。
    """
    parts = str(getattr(chunk, "source", "") or "").split("/")
    return parts[5] if len(parts) > 5 else "?"


def main() -> int:
    setup_utf8_stdout()          # 把 stdout/stderr 切成 UTF-8，避免 GBK 控制台丢中文
    parser = argparse.ArgumentParser(description="长期记忆重复/膨胀体检（只读）")
    parser.add_argument("--user", required=True, help="按用户维度体检（服务端派生的取值）")
    parser.add_argument("--out", default="", help="把体检读数写成 JSON")
    parser.add_argument("--examples", type=int, default=3, help="每组重复示例打印几条")
    parser.add_argument("--dump", action="store_true",
                        help="逐条打印（会话/轮次/角色/message_id/正文前 40 字），用于把条数对平")
    args = parser.parse_args()

    store, rows = _load_records(args.user)
    print(f"store={getattr(store, 'name', '?')}  用户={args.user}  记录数={len(rows)}")
    if not rows:
        print("（该用户没有记忆记录——要么没沉淀，要么用户维度不对）")
        return 0

    by_role = collections.Counter()
    text_groups: dict[str, list[Any]] = collections.defaultdict(list)
    # 会话 -> 用户消息正文 -> 助手记录（用于找"同问多答"）
    per_session_questions: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    source_of: dict[str, str] = {}
    for chunk in rows:
        meta = _meta_of(chunk)
        role = str(meta.get("role") or "")
        by_role[role or "?"] += 1
        text = str(getattr(chunk, "text", "") or "").strip()
        text_groups[text].append(chunk)
        source = str(getattr(chunk, "source", "") or "")
        source_of[chunk.id] = source
        parts = source.split("/")
        session = _session_of(chunk)
        if role == "user":
            per_session_questions[session][text] += 1

    dup_groups = {t: items for t, items in text_groups.items() if len(items) > 1}
    dup_records = sum(len(items) - 1 for items in dup_groups.values())
    assistant = by_role.get("assistant", 0)
    user = by_role.get("user", 0)

    print(f"  按角色：user={user} assistant={assistant} 其它={len(rows) - user - assistant}")
    print(f"  完全重复（正文逐字相同）：{len(dup_groups)} 组、多出 {dup_records} 条"
          f"（占全部 {dup_records / len(rows):.1%}）")
    repeated_questions = {s: {q: n for q, n in d.items() if n > 1}
                          for s, d in per_session_questions.items()}
    repeated_questions = {s: d for s, d in repeated_questions.items() if d}
    print(f"  同一会话里被重复问到的用户问题：{len(repeated_questions)} 组"
          f"（例：{list(repeated_questions.items())[:1]}）")

    # 助手记录按"近似"统计（只看长度+前若干字，避免引入额外依赖）：同会话同长度段
    approx = collections.Counter()
    for chunk in rows:
        meta = _meta_of(chunk)
        if str(meta.get("role") or "") != "assistant":
            continue
        text = str(getattr(chunk, "text", "") or "").strip()
        approx[(len(text) // 200, text[:40])] += 1
    approx_dupes = sum(n - 1 for n in approx.values() if n > 1)
    print(f"  助手记录里\"长度段+开头40字\"相同的近似重复：多出 {approx_dupes} 条")

    if args.examples and dup_groups:
        print("  完全重复示例：")
        for text, items in list(dup_groups.items())[: args.examples]:
            print(f"    x{len(items)}  {text[:60]!r}")
            for chunk in items[:2]:
                print(f"        id={chunk.id}  source={source_of.get(chunk.id)}")

    if args.dump:
        print("\n  逐条（按会话 + 轮次排序）：")
        rows_sorted = sorted(
            rows, key=lambda c: (str(getattr(c, "source", "")),
                                 int(_meta_of(c).get("turn_index") or 0)))
        for chunk in rows_sorted:
            meta = _meta_of(chunk)
            source = str(getattr(chunk, "source", "") or "")
            parts = source.split("/")
            session = _session_of(chunk)
            text = str(getattr(chunk, "text", "") or "").replace("\n", " ")[:40]
            print(f"    {session:<28} turn={str(meta.get('turn_index')):<4} "
                  f"role={str(meta.get('role')):<9} mid={str(meta.get('message_id'))[:10]:<11} {text}")

    # 会话 -> 轮次 -> 角色，用来核对"每轮应该正好 2 条"
    per_turn: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for chunk in rows:
        meta = _meta_of(chunk)
        source = str(getattr(chunk, "source", "") or "")
        parts = source.split("/")
        session = _session_of(chunk)
        per_turn[(session, int(meta.get("turn_index") or 0))].append(str(meta.get("role") or ""))
    print("\n  按「会话+轮次」统计（每轮正常应是 user+assistant 各一条）：")
    for (session, turn), roles in sorted(per_turn.items()):
        flag = "" if sorted(roles) == ["assistant", "user"] else "   <== 不完整/重复"
        print(f"    {session:<28} turn={turn:<4} {sorted(roles)}{flag}")

    payload = {
        "user": args.user, "store": getattr(store, "name", "?"), "records": len(rows),
        "by_role": dict(by_role),
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_extra_records": dup_records,
        "exact_duplicate_ratio": round(dup_records / len(rows), 4),
        "repeated_questions": {s: d for s, d in repeated_questions.items()},
        "approx_assistant_duplicates": approx_dupes,
        # ⚠️ **把记忆原文也落盘**：`scripts/check_memory_echo.py` 要用它做"回音检查"
        #    （召回的旧话有没有被模型逐字照抄进答案）。没有这一项，
        #    那边就只能靠人工翻日志，检查做不成自动化。
        "texts": [str(getattr(chunk, "text", "") or "") for chunk in rows],
        "records_detail": [
            {"id": chunk.id, "role": _meta_of(chunk).get("role"),
             "message_id": _meta_of(chunk).get("message_id"),
             "pair_id": _meta_of(chunk).get("pair_id"),
             "text": str(getattr(chunk, "text", "") or "")}
            for chunk in rows
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"  证据已写：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
