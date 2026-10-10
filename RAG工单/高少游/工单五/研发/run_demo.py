# -*- coding: utf-8 -*-
"""多轮对话演示入口（命令行）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

按工单给定的 5 轮问答脚本顺序提问，打印每轮的：
    改写后的独立问句 / 命中文档 / 答案 / 证据 / 耗时
并把完整对话与结果落盘到 output/multiturn_demo.json。
"""
from __future__ import annotations

import json
import sys
import time

from src import config
from src.conversation import Conversation
from src.qa_engine import MultiTurnQA


def main() -> int:
    script = json.loads(config.MULTITURN_PATH.read_text(encoding="utf-8"))
    turns = script["turns"]

    print("=" * 76)
    print("招股说明书多轮检索问答演示  |  工单编号: 人工智能 NLP-RAG-Query 理解优化任务")
    print("=" * 76)

    qa = MultiTurnQA()
    qa.warmup()          # 预热，避免首问冷启动超时
    conv = Conversation()
    records = []

    for item in turns:
        q = item["question"]
        print(f"\n【第 {item['turn']} 轮】现象：{item.get('phenomenon','')}")
        print(f"Q: {q}")
        res = qa.ask(q, conv)
        if res.rewritten and res.rewritten != q:
            print(f"  ↳ 改写: {res.rewritten}")
            for r in res.rewrite_reasons:
                print(f"     · {r}")
        print(f"A: {res.answer}")
        print(f"  [文档={res.doc or '未定位'} | 类型={res.answer_type} | 耗时={res.latency}s]")
        if res.evidence:
            e = res.evidence[0]
            print(f"  [证据] {e['source']} p.{e['page']} ({e['kind']})  {e['snippet'][:60]}...")
        records.append({
            "turn": item["turn"],
            "phenomenon": item.get("phenomenon", ""),
            "question": q,
            "rewritten": res.rewritten,
            "rewrite_reasons": res.rewrite_reasons,
            "answer": res.answer,
            "answer_type": res.answer_type,
            "doc": res.doc,
            "latency": res.latency,
            "evidence": res.evidence,
            "reference": item.get("reference", ""),
            "key_tokens": item.get("key_tokens", []),
        })

    out = config.OUTPUT_DIR / "multiturn_demo.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 76)
    print(f"对话记录已保存 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())