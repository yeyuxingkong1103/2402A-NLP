#!/usr/bin/env python3
"""把「短期记忆 / 长期记忆到底存了什么」**用真代码跑一遍打出来**（不是手写示意）。

为什么要有这个脚本
==================
"记忆存了什么"这类问题，靠嘴说和靠文档写都容易与实现漂移（字段改名、主键口径调整、
幂等口径变化……）。所以这里**直接调用应用层的同一批函数**，跑一段真实的多轮对话，
再把两边的实际载荷打印出来：

* **短期记忆**：Redis 里每一条消息的 JSON（字段与应用里 `_dump_dict` 逐字一致）；
* **长期记忆**：Milvus 集合 `legal_rag_memory` 里每条记录的字段
  （`id / text / source / doc_id / role_id / summary / 向量维度`）；
* **召回演示**：再问一次"我姓什么"，看**只可能来自长期记忆**的命中。

用法::

    python scripts/demo_memory_payloads.py                     # 打到屏幕（自动切 UTF-8）
    python scripts/demo_memory_payloads.py --out docs/xxx.md    # 同时写文件
    python scripts/demo_memory_payloads.py --window 3 --turns 6 # 换窗口/轮数

对照的文档：`docs/PROJECT-OVERVIEW.md` §三层记忆。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.embedding.offline import OfflineEmbedder  # noqa: E402
from legal_rag.logging_setup import setup_utf8_stdout  # noqa: E402
from legal_rag.memory.longterm import LongTermMemory  # noqa: E402
from legal_rag.memory.session import (InMemorySessionStore, message_id_for,  # noqa: E402
                                      session_key, split_window, with_pair_ids)
from legal_rag.schemas import Message  # noqa: E402
from legal_rag.store.memory_store import MemoryVectorStore  # noqa: E402

USER, ROLE, SESSION = "u-zhang", "lawyer", "s-20260927"

#: 7 轮对话：第 1 轮埋"特征词"（姓名/城市），后面是填充。
#: 窗口默认只留最近 5 组 ⇒ 前 2 组会滚出去，正好演示"沉淀 + 之后还能想起来"。
TURNS: tuple[tuple[str, str], ...] = (
    ("我叫欧阳，住在兰州，我的案子是劳动仲裁", "好的，我记下了：欧阳先生，兰州，劳动仲裁。"),
    ("公司拖欠我两个月工资", "拖欠工资可以申请劳动仲裁，要求支付并加付赔偿金。"),
    ("我的合同没写清工资构成", "合同未明确工资构成的，可按实际发放与同岗位标准认定。"),
    ("我手里有打卡记录和微信聊天记录", "这两类属于有效证据，建议按时间线整理。"),
    ("我老婆也在同一家公司", "她可以单独申请仲裁，主张自己的权利。"),
    ("公司说要对我提反诉", "劳动仲裁中用人单位一般不能对你提反诉。"),
    ("一般多久能有结果", "仲裁一般自受理之日起 45 日内结案，复杂可延长 15 日。"),
)


def run(window: int, turns: int, out: Path | None) -> int:
    """跑一段对话并打印两边的实际载荷；返回 0。"""
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    store = InMemorySessionStore(window=20, turns=window)
    # 嵌入用**零依赖**的离线实现（本机没有 Ollama 也能复现）；线上是 bge-m3（同 1024 维）
    memory = LongTermMemory(MemoryVectorStore(persist_dir=None), OfflineEmbedder(dim=1024),
                            enabled=True, threshold=20, collection="legal_rag_memory")

    emit(f"Redis 键（服务端按 user+role+session 派生）：{session_key(USER, ROLE, SESSION)!r}")
    emit(f"窗口口径：最近 {window} 组问答 = {window * 2} 条消息")
    emit()

    for index, (question, answer) in enumerate(TURNS[:turns], 1):
        seq = index * 2 - 1                    # 与业务库 business.next_seq 同口径（1,3,5...）
        store.append(USER, ROLE, SESSION, Message(role="user", content=question), seq=seq)
        store.append(USER, ROLE, SESSION, Message(role="assistant", content=answer), seq=seq + 1)
        keep, rolled = split_window(
            with_pair_ids(store.raw_messages(USER, ROLE, SESSION),
                          user_id=USER, session_id=SESSION),
            window)
        if rolled:
            result = memory.remember_messages(USER, ROLE, SESSION, rolled)
            emit(f"[第 {index} 轮后] 滚出 {len(rolled)} 条 -> 写长期记忆 "
                 f"written={result['written']} ok={result['ok']}")
        # 与应用 `_remember_turn` 一致：**写成功之后才**裁剪 Redis（顺序不可交换）
        store.trim_window(USER, ROLE, SESSION)

    emit()
    emit("=" * 78)
    emit("【短期记忆】Redis 里现在实际存的内容（逐条）")
    emit("=" * 78)
    for order, item in enumerate(store.raw_messages(USER, ROLE, SESSION), 1):
        payload = json.dumps({k: item.get(k) for k in
                              ("message_id", "role", "content", "created_at", "citations")},
                             ensure_ascii=False)
        emit(f"  {order:>2}. {payload[:150]}")

    emit()
    emit("=" * 78)
    emit("【长期记忆】Milvus 集合 legal_rag_memory 里的记录（逐条）")
    emit("=" * 78)
    for chunk in memory.store.all_chunks():
        emit(f"  id       = {chunk.id}")
        emit(f"  text     = {chunk.text[:60]}")
        emit(f"  source   = {chunk.source}")
        emit(f"  doc_id   = {chunk.doc_id}   （服务端强制过滤用的用户维度）")
        emit(f"  role_id  = {chunk.role_id}")
        emit(f"  summary  = {chunk.summary}")
        emit(f"  向量维度 = {len(chunk.vector)}   条数={memory.count()}")
        emit("  " + "-" * 70)

    emit()
    emit("=" * 78)
    emit("【召回演示】用户在新会话里问「我姓什么？我住在哪个城市？」")
    emit("=" * 78)
    hits = memory.recall("我姓什么？我住在哪个城市？", user_id=USER, role_id=ROLE, top_k=3)
    for rank, hit in enumerate(hits, 1):
        emit(f"  top{rank} score={hit.score:.4f}  {hit.chunk.text[:50]}")
    emit()
    emit("（窗口里已经没有第 1 轮了 —— 上面命中的只可能来自长期记忆）")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n已写：{out}")
    return 0


def main() -> int:
    setup_utf8_stdout()          # 把 stdout/stderr 切成 UTF-8，避免 GBK 控制台丢中文
    parser = argparse.ArgumentParser(description="打印短期/长期记忆的实际载荷")
    parser.add_argument("--window", type=int, default=5, help="窗口组数（与 SESSION_WINDOW 对齐）")
    parser.add_argument("--turns", type=int, default=len(TURNS), help="跑几轮对话")
    parser.add_argument("--out", default="", help="把输出写到文件（UTF-8）")
    args = parser.parse_args()
    return run(max(args.window, 1), max(args.turns, 1),
               Path(args.out) if args.out else None)


if __name__ == "__main__":
    raise SystemExit(main())
