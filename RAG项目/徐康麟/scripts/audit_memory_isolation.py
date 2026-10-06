"""长期记忆（B-9）**跨用户隔离审计** —— 启用 `LONGTERM_ENABLED=true` 前必做（BACKLOG 硬要求）。

为什么单独一个脚本：`handoff/BACKLOG.md` B-9 写明「记忆进 Milvus = **新增一个隔离面** ⇒ 必须重做一次
独立的跨用户隔离审计」。记忆与知识库**生命周期不同**、又落在**另一个 collection**
（`legal_rag_memory`），所以鉴权面自检（`scripts/verify_delivery.py`）覆盖不到它。

两阶段（**必须分两次进程**，见下面"为什么"）：

* **phase=http**（API 在跑）：黑盒视角 —— A 用带特征词的对话把消息挤出滑窗（触发沉淀），
  断言 `/health` 里记忆 chunk 数**增长**；随后 B 问同样的问题，断言 B 的回答里**不含 A 的特征词**。
* **phase=recall**（API 在跑）：黑盒视角 —— **召回质量**。A 埋特征词 → 补足轮数挤滑窗 →
  **开新会话**（短期记忆为空）问「我姓什么？」。答得出 ⇒ 长期记忆真的回到了答案里；
  答不出 ⇒ **只写不召回**（`LongTermMemory.recall()` 没接进 `/chat` 的取数链路）。
  这是与"隔离"**不同的**一个判定面，`phase=http` 证不了它。
* **phase=storage**（API 必须**已停**）：白盒视角 —— 直接读 `legal_rag_memory` collection，
  逐条核对 `source`/`doc_id` 是否都带用户维度、A 的记录在、B 的记录为 0、没有空用户维度；
  并**额外直调 `recall()`** 验证召回原语是否可用（把"检索层坏了"与"没接线"分开归因）。
  ⚠️ **为什么必须停 API**：Milvus Lite 是**单进程文件库**（`DataDirLockedError`），
  两个进程同时开会拿不到锁 —— 更糟的是 `MILVUS_FALLBACK_TO_MEMORY=True` 时会**静默回落内存库**，
  于是审计看到"空 collection"给出**假阴性**。这是本轮踩过的坑，写死在用法里。

用法（云端）::

    # 1) 起服务（审计用的开关：SESSION_WINDOW / LONGTERM_THRESHOLD 调小，便于快速触发）
    #    见 .pytmp/cloud/env_memory_audit.sh
    python scripts/audit_memory_isolation.py --phase http    --out eval/results/memory-audit-http.json
    python scripts/audit_memory_isolation.py --phase recall  --out eval/results/memory-recall-probe.json
    # 2) 停 API（保留 Redis/向量库文件即可）
    python scripts/audit_memory_isolation.py --phase storage --out eval/results/memory-audit-storage.json

退出码：0 = 全通过；1 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_delivery import Client, expect, record  # noqa: E402

PASSWORD = "MemAudit#2026"
#: 特征词：出现在 A 的对话里，用来判断"B 能不能看到 A 的记忆"
MARKER_CITY = "兰州"
MARKER_SURNAME = "欧阳"
FILLERS = [
    "劳动争议申请仲裁的时效是多久",
    "民间借贷利率的司法保护上限是多少",
    "醉酒驾驶机动车要承担什么法律后果",
    "工伤是怎么认定的",
    "离婚有没有冷静期",
]
PROBE_QUESTION = "我姓什么？我住在哪个城市？"


def _results_path(tag: str) -> Path:
    return ROOT / "eval" / "results" / f"memory-audit-state-{tag}.json"


# --------------------------------------------------------------------------- HTTP 阶段

def phase_http(args: argparse.Namespace) -> int:
    tag = secrets.token_hex(3)
    client_a = Client(args.base_url, timeout=300.0)
    client_b = Client(args.base_url, timeout=300.0)
    name_a, name_b = f"mema-a-{tag}", f"mema-b-{tag}"

    status_a, _ = client_a.call("POST", "/auth/register",
                                {"username": name_a, "password": PASSWORD})
    status_b, _ = client_b.call("POST", "/auth/register",
                                {"username": name_b, "password": PASSWORD})
    expect(status_a == 201 and status_b == 201, "memory_register",
           f"两个账号已建（{status_a}/{status_b}）", f"注册失败（{status_a}/{status_b}）")
    if not (status_a == 201 and status_b == 201):
        return 1
    _, me_a = client_a.call("GET", "/auth/me")
    _, me_b = client_b.call("GET", "/auth/me")
    user_a, user_b = me_a.get("user_id"), me_b.get("user_id")

    status, health = client_a.call("GET", "/health")
    longterm = (health or {}).get("longterm") if isinstance(health, dict) else None
    record("memory_health_visible", "PASS" if isinstance(longterm, dict) else "FAIL",
           f"/health.longterm = {json.dumps(longterm, ensure_ascii=False)[:160]}",
           )
    if not isinstance(longterm, dict) or not longterm.get("enabled"):
        record("memory_enabled", "FAIL",
               f"**记忆未启用**（LONGTERM_ENABLED 没打开）：{json.dumps(longterm, ensure_ascii=False)[:200]}")
        return 1
    record("memory_enabled", "PASS",
           f"记忆已启用：collection={longterm.get('collection')} 现有 chunks={longterm.get('chunks')}")
    before_chunks = int(longterm.get("chunks") or 0)

    # A：第一轮带特征词，其余为填充；轮数由 SESSION_WINDOW/LONGTERM_THRESHOLD 决定
    session_a = f"{user_a}-lawyer-mema-{tag}"
    turns = [(f"我叫{MARKER_SURNAME}，住在{MARKER_CITY}，我的案子是劳动仲裁")] + FILLERS
    a_statuses: list[int] = []
    for message in turns:
        status, _body = client_a.call("POST", "/chat", {
            "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
            "message": message, "stream": False})
        a_statuses.append(status)
    expect(all(status == 200 for status in a_statuses), "memory_a_turns_ok",
           f"A 完成 {len(turns)} 轮问答（触发沉淀）", f"A 有轮次失败：{a_statuses}",
           statuses=a_statuses)

    _, health_after = client_a.call("GET", "/health")
    after_longterm = (health_after or {}).get("longterm") or {}
    after_chunks = int(after_longterm.get("chunks") or 0)
    expect(after_chunks > before_chunks, "memory_written_after_window",
           f"记忆 chunk 数增加：{before_chunks} -> {after_chunks}（滚出窗口的消息已沉淀）",
           f"记忆没有增长（{before_chunks} -> {after_chunks}）：滑窗/阈值没触发，或写入失败",
           before=before_chunks, after=after_chunks)

    # B：问同样的特征问题，**绝不能**带出 A 的特征词
    session_b = f"{user_b}-lawyer-mema-{tag}"
    status_b_chat, body_b = client_b.call("POST", "/chat", {
        "user_id": user_b, "role_id": "lawyer", "session_id": session_b,
        "message": PROBE_QUESTION, "stream": False})
    answer_b = str((body_b or {}).get("answer") or "")
    leaked = [token for token in (MARKER_CITY, MARKER_SURNAME) if token in answer_b]
    expect(status_b_chat == 200 and not leaked, "memory_no_cross_user_leak",
           "B 问同样的问题：回答里没有 A 的特征词（无跨用户泄露）",
           f"**跨用户泄露**：B 的回答里出现 {leaked}", answer=str(answer_b)[:300])

    # A 自己再问一次：仅记录（弱模型不一定会用记忆，不作硬判据）
    status_a_probe, body_a_probe = client_a.call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
        "message": PROBE_QUESTION, "stream": False})
    answer_a = str((body_a_probe or {}).get("answer") or "")
    record("memory_a_self_recall_informational",
           "PASS" if any(token in answer_a for token in (MARKER_CITY, MARKER_SURNAME)) else "SKIP",
           f"A 自问「{PROBE_QUESTION}」→ {'答出特征词' if any(t in answer_a for t in (MARKER_CITY, MARKER_SURNAME)) else '没答出（弱模型可能不用记忆，不作失败）'}："
           f"{answer_a[:120]}")

    # 把本次审计用到的标识落盘，供 storage 阶段对账
    _results_path(tag).write_text(json.dumps(
        {"tag": tag, "user_a": user_a, "user_b": user_b, "session_a": session_a,
         "session_b": session_b, "turns": len(turns), "before_chunks": before_chunks,
         "after_chunks": after_chunks}, ensure_ascii=False, indent=2), encoding="utf-8")
    record("memory_audit_state_saved", "PASS", f"审计标识已落盘：{_results_path(tag).name}")
    print(f"\n（storage 阶段用法：先停 API，再跑 --phase storage --state {_results_path(tag)}）")
    return 0


# --------------------------------------------------------------------------- 召回质量阶段

#: 把第 1 轮（特征词）挤出滑窗所需的后续轮次。窗口(组)默认 5 ⇒ 6 轮即可滚出，
#: 这里取 8 轮留余量（窗口被调大也不会变成"假阳性"：轮数不足时下面的断言会失败）。
RECALL_FILLERS = [
    "劳动争议申请仲裁的时效是多久",
    "民间借贷利率的司法保护上限是多少",
    "醉酒驾驶机动车要承担什么法律后果",
    "工伤是怎么认定的",
    "离婚有没有冷静期",
    "房东不退押金可以怎么维权",
    "公司拖欠工资我应该找谁",
]


def phase_recall(args: argparse.Namespace) -> int:
    """**召回质量**探针（HTTP 黑盒）：记忆写进去了，**到底能不能回到答案里**？

    为什么单独一个阶段：`phase_http` 只证明"写对了 + 不串户"，**证不了能召回**；
    而 `recall()` 有没有被接进 `/chat` 是另一回事 —— 本阶段就是判定这件事的。

    判定方式（**可判定**，不靠"看起来像回忆"）：

    1. 第 1 轮埋特征词（姓名/城市），后面补足 ``--turns`` 轮把它**挤出滑窗**；
    2. 断言 Milvus 记忆条数确实增长（否则是"没沉淀"，不是"没召回"，要分开报）；
    3. **开一个新会话**再问「我姓什么？」—— 新会话的短期记忆是**空的**，
       此时能答出来只可能来自长期记忆；答不出来就是**没接线**。
    4. 顺带在**原会话**再问一次，覆盖"滚出窗口后同会话还能不能想起"。
    """
    tag = secrets.token_hex(3)
    client = Client(args.base_url, timeout=300.0)
    name_a = f"memr-a-{tag}"

    status, _ = client.call("POST", "/auth/register",
                            {"username": name_a, "password": PASSWORD})
    expect(status == 201, "recall_register", f"账号已建（{status}）", f"注册失败（{status}）")
    if status != 201:
        return 1
    _, me = client.call("GET", "/auth/me")
    user_a = str(me.get("user_id") or "")

    _, health = client.call("GET", "/health")
    longterm = (health or {}).get("longterm") if isinstance(health, dict) else None
    if not isinstance(longterm, dict) or not longterm.get("enabled"):
        record("recall_memory_enabled", "FAIL",
               f"记忆未启用，无法测召回：{json.dumps(longterm, ensure_ascii=False)[:200]}")
        return 1
    before_chunks = int(longterm.get("chunks") or 0)
    record("recall_memory_enabled", "PASS",
           f"记忆已启用（collection={longterm.get('collection')}，测试前 chunks={before_chunks}）")

    session_a = f"{user_a}-lawyer-memr-{tag}"
    turns = [f"我叫{MARKER_SURNAME}，住在{MARKER_CITY}，我的案子是劳动仲裁"]
    turns += RECALL_FILLERS[: max(int(args.turns) - 1, 1)]
    statuses: list[int] = []
    for index, message in enumerate(turns):
        status, _body = client.call("POST", "/chat", {
            "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
            "message": message, "stream": False})
        statuses.append(status)
        if status != 200:
            print(f"    !! 第 {index + 1} 轮失败：{status}")
    expect(all(status == 200 for status in statuses), "recall_write_turns_ok",
           f"A 完成 {len(turns)} 轮（第 1 轮埋特征词，其余为填充）",
           f"有轮次失败：{statuses}", statuses=statuses)

    _, health_after = client.call("GET", "/health")
    after_chunks = int(((health_after or {}).get("longterm") or {}).get("chunks") or 0)
    expect(after_chunks > before_chunks, "recall_memory_written",
           f"记忆条数增长：{before_chunks} -> {after_chunks}（特征词那轮已沉淀）",
           f"记忆没增长（{before_chunks} -> {after_chunks}）⇒ 未沉淀，"
           f"后面的'答不出'不能归因于'没接召回'，请先查写入")

    # ③ 新会话（短期记忆为空）问同一个问题
    session_new = f"{user_a}-lawyer-memr-new-{tag}"
    status_new, body_new = client.call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": session_new,
        "message": PROBE_QUESTION, "stream": False})
    answer_new = str((body_new or {}).get("answer") or "")
    hit_new = [token for token in (MARKER_CITY, MARKER_SURNAME) if token in answer_new]
    expect(status_new == 200 and bool(hit_new), "memory_cross_session_recall",
           f"**新会话**里答出了特征词 {hit_new} ⇒ 长期记忆确实回到了答案里",
           f"新会话（短期记忆为空）**答不出**长期记忆里的特征词 "
           f"（{MARKER_CITY}/{MARKER_SURNAME}）⇒ 记忆只写不召回，"
           f"`LongTermMemory.recall()` 没有接进 /chat 的取数链路",
           answer=answer_new[:300], http_status=status_new)

    # ④ 原会话再问一次（这一轮的特征词已滚出滑窗，窗口里没有）
    status_same, body_same = client.call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
        "message": PROBE_QUESTION, "stream": False})
    answer_same = str((body_same or {}).get("answer") or "")
    hit_same = [token for token in (MARKER_CITY, MARKER_SURNAME) if token in answer_same]
    expect(status_same == 200 and bool(hit_same), "memory_same_session_recall",
           f"原会话（特征词已滚出滑窗）也答出了 {hit_same}",
           f"原会话同样答不出特征词（{MARKER_CITY}/{MARKER_SURNAME}）"
           f"⇒ 滚出滑窗的内容没有回到答案里",
           answer=answer_same[:300], http_status=status_same)

    print(f"\n新会话回答：{answer_new[:200]}")
    print(f"原会话回答：{answer_same[:200]}")

    # 落盘标识：storage 阶段直调 recall() 时要用同一个 user_a
    _results_path(tag).write_text(json.dumps(
        {"tag": tag, "user_a": user_a, "phase": "recall", "session_a": session_a,
         "session_new": session_new, "turns": len(turns),
         "before_chunks": before_chunks, "after_chunks": after_chunks,
         "answer_new": answer_new[:300], "answer_same": answer_same[:300]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    record("recall_state_saved", "PASS", f"召回探针标识已落盘：{_results_path(tag).name}")
    return 0


# --------------------------------------------------------------------------- 存储阶段

def phase_storage(args: argparse.Namespace) -> int:
    from legal_rag.config import RagConfig
    from legal_rag.embedding.base import build_embedder
    from legal_rag.memory.longterm import (
        DEFAULT_MEMORY_COLLECTION, MEMORY_DOC_PREFIX, MEMORY_SOURCE_PREFIX)
    from legal_rag.store.base import build_store

    state = json.loads(Path(args.state).read_text(encoding="utf-8")) if args.state else {}
    user_a, user_b = state.get("user_a", ""), state.get("user_b", "")
    config = RagConfig.from_env()
    if str(config.vector_store).strip().lower() == "memory":
        # 真机踩到：**忘了 source 云端 env** 时 `VECTOR_STORE` 落到默认 `memory`，
        # 于是"打开的是一个空内存库"——结论全是假的。这里直接拦下来。
        record("memory_store_not_degraded", "FAIL",
               "VECTOR_STORE=memory（没 source 云端 env：LEGAL_RAG_MILVUS_URI / VECTOR_STORE）"
               "⇒ 看到的是空内存库，结论不可信")
        return 1
    # ⚠️ **必须给 embedder**：`build_store` 靠它拿到"实测维度"，否则与既有 collection 的 1024 维
    # 冲突 ⇒ **静默降级成内存库**（这条坑写在 `legal_rag/engine.py` 的注释里，本轮又踩了一次）。
    embedder = build_embedder(config.embedding_provider, config.embedding_model,
                              config.embedding_dim, config=config)
    store = build_store(config.vector_store, persist_dir=config.index_dir,
                        collection=DEFAULT_MEMORY_COLLECTION, config=config, embedder=embedder)
    record("memory_store_open", "PASS",
           f"已打开记忆 collection：{DEFAULT_MEMORY_COLLECTION}（store={store.name}）")
    if getattr(store, "name", "") == "memory":
        record("memory_store_not_degraded", "FAIL",
               "**记忆库回落到内存**（多半是 API 没停、Milvus Lite 抢锁）⇒ 看到的是空库，结论不可信")
        return 1

    try:
        chunks = list(store.all_chunks())
    except Exception as exc:  # noqa: BLE001
        record("memory_store_read", "FAIL", f"读记忆库失败：{type(exc).__name__}: {exc}")
        return 1
    record("memory_store_read", "PASS", f"读出 {len(chunks)} 条记忆记录")

    by_user: dict[str, int] = {}
    malformed: list[dict] = []
    for chunk in chunks:
        source = str(getattr(chunk, "source", "") or "")
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        owner = ""
        if source.startswith(MEMORY_SOURCE_PREFIX):
            owner = source[len(MEMORY_SOURCE_PREFIX):].split("/", 1)[0]
        ok = bool(owner) and doc_id == f"{MEMORY_DOC_PREFIX}{owner}"
        by_user[owner] = by_user.get(owner, 0) + 1
        if not ok:
            malformed.append({"id": chunk.id, "source": source, "doc_id": doc_id})

    expect(not malformed, "memory_records_wellformed",
           f"{len(chunks)} 条记录的 source/doc_id **都带用户维度且一致**",
           f"有 {len(malformed)} 条记录格式不对（空用户/前缀不符）：{malformed[:3]}",
           malformed=len(malformed))
    expect(not any(owner == "" for owner in by_user), "memory_no_orphan_records",
           "没有用户维度为空的孤儿记录", f"存在空用户维度记录：{by_user.get('', 0)} 条")

    count_a, count_b = by_user.get(user_a, 0), by_user.get(user_b, 0)
    expect(count_a > 0, "memory_a_records_present",
           f"A（{user_a}）有 {count_a} 条记忆", f"A 没有记忆记录（沉淀没落库）")
    expect(count_b == 0, "memory_b_has_none",
           "B 没有任何记忆记录（B 只问了一轮，本就不该沉淀）",
           f"B 名下出现了 {count_b} 条记录")
    record("memory_users_overview", "PASS",
           f"库内用户维度分布：{json.dumps({k: v for k, v in list(by_user.items())[:6]}, ensure_ascii=False)}"
           f"（共 {len(by_user)} 个用户）")

    # ---------- 召回**原语**是否可用（白盒） ----------
    # 目的：把"检索这一层坏没坏"与"有没有接线"分开归因。
    # 若这里 PASS 而 HTTP 黑盒的跨会话召回 FAIL ⇒ 结论是"**没接线**"，不是"检索/向量库坏了"。
    if not user_a:
        record("memory_recall_primitive", "SKIP", "没有 user_a（--state 里缺标识），跳过召回原语检查")
        return 0
    from legal_rag.memory.longterm import LongTermMemory

    memory = LongTermMemory(store, embedder, enabled=True,
                            threshold=int(getattr(config, "longterm_threshold", 20) or 20),
                            collection=DEFAULT_MEMORY_COLLECTION, config=config)
    hits: list[Any] = []
    used_role = ""
    for role_id in ("lawyer", ""):
        try:
            hits = memory.recall(PROBE_QUESTION, user_id=user_a, role_id=role_id, top_k=5)
        except Exception as exc:  # noqa: BLE001
            record("memory_recall_primitive", "FAIL",
                   f"召回原语**抛异常**（role_id={role_id!r}）：{type(exc).__name__}: {exc}")
            return 0
        if hits:
            used_role = role_id
            break
    texts = [str(getattr(hit.chunk, "text", "") or "") for hit in hits]
    found = [token for token in (MARKER_CITY, MARKER_SURNAME)
             if any(token in text for text in texts)]
    expect(bool(found), "memory_recall_primitive",
           f"召回原语可用：`recall()` 命中 {len(hits)} 条，其中含特征词 {found}"
           f"（role_id={used_role!r}）⇒ 向量库/嵌入/过滤链路是通的",
           f"召回原语**取不到**特征词（命中 {len(hits)} 条，role_id={used_role!r}）"
           f"⇒ 问题在检索层，不只是接线")
    record("memory_recall_primitive_hits", "PASS" if hits else "FAIL",
           f"召回命中：{json.dumps([t[:60] for t in texts[:3]], ensure_ascii=False)}",
           hits=len(hits))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="长期记忆跨用户隔离审计 / 召回探针")
    parser.add_argument("--phase", choices=("http", "recall", "storage"), required=True,
                        help="http=写入与隔离；recall=**召回质量**（能否真的想起）；storage=白盒逐条核对")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--state", default="", help="storage 阶段：http/recall 阶段落盘的标识文件")
    parser.add_argument("--turns", type=int, default=8,
                        help="recall 阶段：会话总轮数（要够多，把第 1 轮特征词挤出滑窗）")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.phase == "http":
        code = phase_http(args)
    elif args.phase == "recall":
        code = phase_recall(args)
    else:
        if not args.state:
            candidates = sorted((ROOT / "eval" / "results").glob("memory-audit-state-*.json"))
            if not candidates:
                print("!! storage 阶段需要 --state（先跑 http 阶段）", file=sys.stderr)
                return 1
            args.state = str(candidates[-1])
            print(f"（未指定 --state，自动取最新：{Path(args.state).name}）")
        code = phase_storage(args)

    from scripts.verify_delivery import CHECKS, _flush

    _flush(args.out)
    failures = [c for c in CHECKS if c["status"] == "FAIL"]
    return 1 if (code or failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
