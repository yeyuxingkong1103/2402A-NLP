# -*- coding: utf-8 -*-
"""证据脚本：判定生成器级配对该用 FIFO 还是 LIFO（本机实测两者结果相同，说明日志事件严格交替）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 脚本（证据脚本 · 生成器级配对校验；**只读**，不改交付物）
用法：``pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_generator_pairing.py``

依据：抽取式生成的答案应来自它自己记录的 primary_chunk_id 对应的 chunk 正文。
若某种配对方式的 (enter.question, exit.answer, result.primary_chunk_id) 三者自洽率高，
则该配对方式正确。
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
REPRO = W2 / "优化" / "基线" / "repro"

PUNCT = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def norm(t: str) -> str:
    return "".join(ch for ch in (t or "") if ch not in PUNCT)


def open_obj_question(obj) -> str:
    args = obj.get("args") or []
    return args[0] if args and isinstance(args[0], str) else ""


chunks = {
    json.loads(l)["chunk_id"]: json.loads(l)["content"]
    for l in (REPRO / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}
chunks_norm = {k: norm(v) for k, v in chunks.items()}

for trace in (W1 / "logs" / "rag_trace.jsonl",):
    events = []
    for line in trace.open(encoding="utf-8", errors="replace"):
        if "Generator.generate_extractive" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("function") != "Generator.generate_extractive":
            continue
        if obj.get("event") == "enter":
            events.append(("enter", obj))
        elif obj.get("event") == "exit":
            events.append(("exit", obj))

    print(f"--- {trace.name}: enter={sum(1 for e,_ in events if e=='enter')} exit={sum(1 for e,_ in events if e=='exit')}")

    for mode in ("FIFO", "LIFO"):
        pending = []
        consistent = 0
        inconsistent = 0
        no_primary = 0
        samples = []
        for kind, obj in events:
            if kind == "enter":
                pending.append(obj)
            else:
                if not pending:
                    continue
                opened = pending.pop(0) if mode == "FIFO" else pending.pop()
                res = obj.get("result") or {}
                if not isinstance(res, dict):
                    continue
                ans = norm(res.get("answer") or "")
                primary = res.get("primary_chunk_id")
                if not primary or str(primary) == "None" or primary not in chunks_norm:
                    no_primary += 1
                    continue
                if ans and ans and ans in chunks_norm[primary]:
                    consistent += 1
                else:
                    inconsistent += 1
                    if len(samples) < 3:
                        samples.append((open_obj_question(opened), str(res.get("answer"))[:50], primary))
        total = consistent + inconsistent
        print(
            f"    {mode}: 自洽 {consistent}/{total} = {consistent/total*100:.2f}%  （无 primary 记录 {no_primary}）"
        )
        for s in samples:
            print(f"        反例: q={s[0][:30]} ans={s[1]} primary={s[2]}")
