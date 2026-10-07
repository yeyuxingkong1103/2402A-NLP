"""
工单5 演示：针对工单原文的 5 轮对话「检索并显示答案」
工单编号：人工智能NLP-RAG-Query理解优化任务

工单产出物原文要求：
    「演示：(a) 完整演示视频　(b) **针对上述 5 轮检索并显示答案**」

本脚本就是 (b) 的可复现版本 —— 把工单原文那 5 句话**按顺序丢进同一个会话**，
每一轮把「系统看到了什么、理解成什么、去哪一页找、答了什么」全部摊开打印。
演示视频直接录这个脚本的终端输出即可（不需要打开浏览器，避免演示环境差异）。

5 轮原文（一字不改）：
    1. 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
    2. 他参与的哪个工程荣获了国家科技进步一等奖？
    3. 这个公司的法定代表人是谁？
    4. 那武汉力源信息技术股份有限公司呢？
    5. 武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？

用法：
    python scripts/demo_dialogue.py                     # 交付配置 rule+llm
    python scripts/demo_dialogue.py --mode off          # 对照：完全不做多轮（第2~4轮必崩）
    python scripts/demo_dialogue.py --no-warmup         # 不做预热（第1轮会含模型加载耗时）
    python scripts/demo_dialogue.py --save              # 额外落一份 Markdown 台本
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, rag  # noqa: E402
from src.session import Session  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_QU

# 工单原文 5 轮，一字不改
QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

# 每轮「这题在考什么」，演示时打在屏幕上，方便评审一眼看懂
WHAT_IT_TESTS = [
    ("单轮直通", "问题自包含，不需要上下文；多轮层必须**不改动**它（改了就是回归）"),
    ("指代消解", "「他」在原文语境里指公司 → 必须替换成上一轮的「武汉兴图新科电子股份有限公司」"),
    ("指代消解", "「这个公司」→ 同样指向兴图新科；且答案不能串成力源信息的赵马克"),
    ("省略补全", "「那X呢？」只有主体、没有谓词 → 继承上一轮的话题（法定代表人），换到新主体"),
    ("单轮直通", "长篇自包含问题；答案在组织结构图里（工单4 的图像块能力在此兑现）"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=config.MULTITURN_MODE,
                    choices=["off", "concat", "rule", "rule+llm"])
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL_FINAL_TOP_K)
    ap.add_argument("--no-warmup", action="store_true", help="跳过预热（第1轮会含模型加载耗时）")
    ap.add_argument("--save", action="store_true", help="再落一份 Markdown 台本")
    args = ap.parse_args()

    config.ensure_dirs()

    if not args.no_warmup:
        print("… 预热中（加载嵌入 / 重排 / 生成模型）")
        t = time.perf_counter()
        rag.answer("武汉兴图新科电子股份有限公司的注册资本是多少？", top_k=args.top_k)
        print(f"… 预热完成，{time.perf_counter() - t:.1f}s\n")

    sid = f"demo-wot5-{int(time.time())}"
    session = Session(session_id=sid)

    print("=" * 78)
    print(f"工单5 演示：多轮对话（{len(QUESTIONS)} 轮）")
    print(f"工单编号：{WORK_ORDER_NO}")
    print(f"消解模式：{args.mode}　　会话：{sid}")
    print("=" * 78)

    rows: list[dict] = []
    for i, (q, (kind, why)) in enumerate(zip(QUESTIONS, WHAT_IT_TESTS), 1):
        print(f"\n{'─' * 78}")
        print(f"第 {i} 轮　【{kind}】{why}")
        print(f"{'─' * 78}")
        print(f"👤 用户：{q}")

        a = rag.answer(q, top_k=args.top_k, session=session)
        qu = a.understanding
        d = getattr(qu, "dialogue", None)

        print(f"\n🔧 系统理解：")
        if d and d.coref:
            print(f"     · 检测到**指代**（代词：{d.inherited_subject or d.subject} 之前出现过）")
        if d and d.ellipsis:
            print(f"     · 检测到**省略式追问**（只有主体、没有谓词）")
        if d and d.subject:
            print(f"     · 本轮主体：{d.subject}　→ 文档：{d.doc_key}")
        if d and d.topic:
            print(f"     · 本轮话题：{d.topic}" +
                  (f"（继承自上一轮）" if d.inherited_topic else ""))
        if d and d.reason:
            print(f"     · 判定说明：{d.reason}")
        if qu and getattr(qu, "rewrite_rejected", False):
            print("     · ⚠️ 改写被判定「把答案猜进了检索式」，已丢弃并退回原文")
        print(f"     · **实际检索式**：{qu.retrieval_query if qu else q}")

        cites = a.citations or []
        pages = " / ".join(f"{c.get('doc_key')} p{c.get('page')}" for c in cites[:5]) or "（无）"
        print(f"\n🔎 命中来源：{pages}")
        print(f"\n🤖 回答：\n{a.answer}")
        print(f"\n⏱  耗时 {a.timing.get('total_ms', 0):.0f} ms"
              f"（理解 {a.timing.get('query_understanding_ms', 0):.0f} / "
              f"检索 {a.timing.get('retrieval_ms', 0):.0f} / "
              f"生成 {a.timing.get('generation_ms', 0):.0f}）")

        rows.append({
            "index": i, "kind": kind, "q": q,
            "resolved": qu.retrieval_query if qu else q,
            "subject": d.subject if d else "", "topic": d.topic if d else "",
            "coref": bool(d.coref) if d else False,
            "ellipsis": bool(d.ellipsis) if d else False,
            "answer": a.answer,
            "citations": [{"doc_key": c.get("doc_key"), "page": c.get("page"),
                           "evidence": c.get("evidence")} for c in cites],
            "total_ms": a.timing.get("total_ms", 0.0),
        })

        # 轮次回写由 rag.answer() 内部完成（见 src/rag.py::_record_turn），
        # 演示脚本不再手动 add —— 真实交互里也不该由调用方负责这件事。

    print(f"\n{'=' * 78}")
    print("演示结束。会话状态：")
    print(json.dumps(session.to_dict(include_answers=False), ensure_ascii=False, indent=1))
    print("=" * 78)

    if args.save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        md = config.EVAL_DIR / f"工单5_演示台本_{args.mode}_{stamp}.md"
        lines = [f"# 工单5 演示台本（5 轮对话）", "",
                 f"- **工单编号**：{WORK_ORDER_NO}",
                 f"- **消解模式**：`{args.mode}`",
                 f"- **会话**：`{sid}`", ""]
        for r in rows:
            src = " / ".join(f"{c['doc_key']} p{c['page']}" for c in r["citations"][:5]) or "（无）"
            lines += [
                f"## 第 {r['index']} 轮　【{r['kind']}】", "",
                f"- **用户**：{r['q']}",
                f"- **检索式**：`{r['resolved']}`",
                f"- **主体**：{r['subject'] or '（未识别）'}　**话题**：{r['topic'] or '（未识别）'}",
                f"- **命中来源**：{src}",
                f"- **耗时**：{r['total_ms']:.0f} ms", "",
                "**回答**：", "", "> " + (r["answer"] or "").replace("\n", "\n> "), "",
            ]
        md.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[台本] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
