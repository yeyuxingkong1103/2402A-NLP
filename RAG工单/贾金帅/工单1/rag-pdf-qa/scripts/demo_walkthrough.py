"""
演示台本（供录制演示视频使用）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
    python scripts/demo_walkthrough.py                # 跑默认 3 个示范问题
    python scripts/demo_walkthrough.py --id 795       # 只跑指定题号
    python scripts/demo_walkthrough.py --ask "你的问题"

设计说明：**纯文本顺序输出，不弹窗、不发声、不做终端动画**。
终端动画（逐字打印、进度条）在 Windows 上一来有 15ms 的 sleep 粒度、
二来会干扰录屏节奏；演示要的是"看得清每一步在做什么"，不是炫技。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm, rag  # noqa: E402
from src.embedder import embed_query  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.query_understanding import understand  # noqa: E402
from src.rag import retrieve_for  # noqa: E402

LINE = "=" * 78


def hr(title: str = "") -> None:
    if title:
        print(f"\n{LINE}\n{title}\n{LINE}")
    else:
        print(LINE)


def step(n: int, text: str) -> None:
    print(f"\n【步骤 {n}】{text}")


def demo_one(question: str, show_llm_only: bool = True) -> None:
    hr()
    print(f"问题：{question}")
    hr()

    # ---- 1 归一化
    step(1, "Query 归一化（切掉输出格式指令、去句末标点、口语转书面）")
    print(f"  原问题   : {question}")
    print(f"  归一化后 : {normalize_query(question)}")

    # ---- 2 理解
    step(2, "Query 理解（意图识别 / 改写检索式 / 子问题拆解）")
    t = time.perf_counter()
    qu = understand(question)
    print(f"  意图      : {qu.intent}")
    print(f"  检索式    : {qu.rewritten}")
    print(f"  子问题    : {qu.sub_questions}")
    print(f"  关键词    : {qu.keywords}")
    print(f"  耗时      : {(time.perf_counter()-t)*1000:.0f} ms"
          f"{'（规则降级）' if qu.degraded else ''}")

    # ---- 3 检索
    step(3, "混合检索（余弦 + BM25 加权融合，阈值闸门筛掉无依据片段）")
    t = time.perf_counter()
    items, info = retrieve_for(qu)
    ms = (time.perf_counter() - t) * 1000
    kb = KnowledgeBase.get()
    print(f"  库内分块  : {len(kb.chunks)} 条")
    print(f"  实际检索式: {info.get('used_query')}")
    if info.get("trace", {}).get("effective_query") and \
       info["trace"]["effective_query"] != info.get("used_query"):
        print(f"  剔除无区分度词后: {info['trace']['effective_query']}")
    print(f"  命中      : {len(items)} 条，耗时 {ms:.0f} ms，闸门拦空={info.get('gated')}")
    for it in items[:5]:
        print(f"    · 依据分 {it.evidence:.4f} / 融合分 {it.score:.4f} "
              f"| 第{it.page}页 | {it.type} | {it.section[:38]}")

    if not items:
        print("\n  >>> 无依据，系统将如实回答『资料中没有相关内容』")
        return

    # ---- 4 生成
    step(4, "生成回答（只依据上述片段，逐句标注来源页码）")
    ans = rag.answer(question)
    print(f"  {ans.answer}")
    print(f"\n  耗时拆分：理解 {ans.timing.get('query_understanding_ms')} ms | "
          f"检索 {ans.timing.get('retrieval_ms')} ms | "
          f"生成 {ans.timing.get('generation_ms')} ms | "
          f"合计 {ans.timing.get('total_ms')} ms")

    # ---- 5 对比
    if show_llm_only:
        step(5, "对照组：同一个问题只问大模型、不给文档（纯 LLM）")
        b = rag.answer_llm_only(question)
        print(f"  {b.answer}")
        print(f"\n  耗时 {b.timing.get('total_ms')} ms")
        print("  ↑ 与上面 RAG 的回答逐项对照，可以看出数据来源的差别。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0, help="只跑指定工单题号")
    ap.add_argument("--ask", default="", help="自定义问题")
    ap.add_argument("--no-llm-only", action="store_true", help="跳过纯 LLM 对照组（省时间/额度）")
    args = ap.parse_args()

    hr()
    print("招股说明书 RAG 问答系统 · 演示")
    print(f"工单编号：{config.WORK_ORDER_NO}")
    hr()

    print("\n【准备】加载索引并预热模型（真实服务在启动时完成这一步）")
    t = time.perf_counter()
    kb = KnowledgeBase.get()
    _ = kb.bm25
    embed_query("预热")
    try:
        llm.chat([{"role": "user", "content": "ping"}], max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        print(f"  大模型预热失败：{exc}")
    print(f"  {kb.meta.get('source_file')} · {len(kb.chunks)} 块 "
          f"（表格块 {kb.meta.get('num_table_chunks')}）· "
          f"向量 {kb.meta.get('embedding_dim')} 维 · 耗时 {time.perf_counter()-t:.1f}s")

    qs = json.loads((config.EVAL_DIR / "ticket_questions.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q["question"] for q in qs}

    if args.ask:
        targets = [args.ask]
    elif args.id:
        if args.id not in by_id:
            print(f"题号 {args.id} 不在工单问题列表里")
            return 2
        targets = [by_id[args.id]]
    else:
        # 默认三题：点状事实 / 多个期间的数据 / 需要带上限定条件的陷阱题
        targets = [by_id[543], by_id[33], by_id[795]]

    for q in targets:
        demo_one(q, show_llm_only=not args.no_llm_only)

    hr()
    print("演示结束。交互式界面：python main.py → http://127.0.0.1:8010/ui")
    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
