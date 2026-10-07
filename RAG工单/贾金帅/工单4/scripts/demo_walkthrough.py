"""
演示台本（供录制演示视频使用）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化（工单2）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化（工单3）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    python scripts/demo_walkthrough.py                # 跑默认 3 个示范问题
    python scripts/demo_walkthrough.py --id 2         # 只跑指定题号
    python scripts/demo_walkthrough.py --ask "你的问题"
    python scripts/demo_walkthrough.py --compare       # 优化前后对比（工单2 主线演示）
    python scripts/demo_walkthrough.py --compare-image # 图像解析前后对比（工单4 主线演示）
    python scripts/demo_walkthrough.py --compare-table # 表格解析前后对比（工单3 主线演示）
    python scripts/demo_walkthrough.py --compare-table --id 4

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


def fact_report(contexts: list[str], must_have: list[str]) -> str:
    """关键事实命中情况（与验收指标同一口径）。"""
    if not must_have:
        return "（非验收题，无标注关键事实）"
    joined = "\n".join(contexts)
    hit = [k for k in must_have if k in joined]
    miss = [k for k in must_have if k not in joined]
    out = f"命中 {len(hit)}/{len(must_have)}"
    if hit:
        out += f"  ✓ {'、'.join(hit)}"
    if miss:
        out += f"\n       ✗ 漏掉：{'、'.join(miss)}"
    return out


def demo_compare(question: str, must_have: list[str]) -> None:
    """工单2 主线演示：同一个问题，优化前 vs 优化后，看检索到的原文差别。"""
    from src.baseline import answer_naive, get_naive_index

    hr(f"优化前后对比：{question}")

    ni = get_naive_index()
    kb = KnowledgeBase.get()
    print(f"  优化前索引：定长 {config.NAIVE_CHUNK_SIZE} 字滑窗 · {len(ni.chunks)} 块 · 纯向量检索")
    print(f"  优化后索引：标题感知分块 + 表格独立成块 · {len(kb.chunks)} 块 · 混合检索 + DF过滤 + 闸门")

    # ---- 优化前
    step(1, "优化前（朴素 RAG）：定长切分 + 纯向量余弦 top-k")
    t = time.perf_counter()
    n = answer_naive(question)
    print(f"  耗时 {n['timing']['total_ms']} ms")
    print(f"  检索到的原文页码：{[c['page'] for c in n['citations']]}")
    print(f"  关键事实：{fact_report(n['contexts'], must_have)}")
    print(f"\n  回答：{n['answer']}")

    # ---- 优化后
    step(2, "优化后（本系统）：混合检索 + 无区分度词过滤 + 阈值闸门")
    a = rag.answer(question)
    qu = a.understanding
    if qu and qu.bypassed:
        print(f"  Query 理解：高置信直通（依据分 {qu.bypass_evidence}，跳过 LLM 往返）"
              f" 耗时 {a.timing.get('query_understanding_ms')} ms")
    else:
        print(f"  Query 理解：LLM 改写  耗时 {a.timing.get('query_understanding_ms')} ms")
    print(f"  检索耗时 {a.timing.get('retrieval_ms')} ms")
    used = (a.retrieval or {}).get("trace", {}).get("effective_query")
    if used:
        print(f"  实际检索式（已剔除无区分度成分）：{used}")
    print(f"  检索到的原文页码：{[c['page'] for c in a.citations]}")
    print(f"  关键事实：{fact_report([c['text'] for c in a.citations], must_have)}")
    print(f"\n  回答：{a.answer}")
    print(f"\n  总耗时 {a.timing.get('total_ms')} ms"
          f"（{'✓ 达标' if (a.timing.get('total_ms') or 0) <= 3000 else '超 3 秒'}）")

    # ---- 纯 LLM
    step(3, "参照组：不给文档，直接问大模型")
    b = rag.answer_llm_only(question)
    print(f"  {b.answer}")
    print("  ↑ 与上面两栏对照：这一步没有任何检索，数据全靠模型记忆。")


def demo_table_compare(question: str, must_have: list[str]) -> None:
    """
    工单3 主线演示：**表格不结构化 vs 结构化**。

    跑法刻意做成「同一套检索算法换一份分块」，让差别只能来自表格解析：
      ① 两份索引各自召回 top-k，打印页码 / 类型 / 依据分
      ② 两份上下文各自喂给同一个模型、同一段 prompt
      ③ 对齐打印关键事实命中情况 —— 一眼看出「哪一条事实因为表格没结构化而漏掉」
    """
    from src.retriever import build_context, retrieve

    hr()
    print(f"问题：{question}")
    hr()

    kb_tbl = KnowledgeBase.get()
    try:
        kb_not = KnowledgeBase.get_notable()
    except FileNotFoundError as exc:
        print(f"对照索引不可用：{exc}")
        print("请先执行：python scripts/build_index.py --no-tables --out notable")
        return
    _ = kb_not.bm25

    results = {}
    for label, kb in (("优化前 · 表格不结构化", kb_not), ("优化后 · 表格结构化", kb_tbl)):
        step(1 if kb is kb_not else 2, label)
        t = time.perf_counter()
        r = retrieve(question, kb=kb)
        ms = (time.perf_counter() - t) * 1000
        items = list(r.items)
        contexts = [it.text for it in items]
        print(f"  库内分块 {len(kb.chunks)} 条；召回 {len(items)} 条，检索耗时 {ms:.0f} ms，"
              f"闸门拦空={r.gated}")
        for it in items[:6]:
            print(f"    · 依据分 {it.evidence:.4f} | {it.doc_key} 第{it.page}页 | "
                  f"{it.type:<5} | {it.section[:34]}")
        if must_have:
            print(f"  关键事实检查：{fact_report(contexts, must_have)}")
        t = time.perf_counter()
        ans = llm.chat([
            {"role": "system", "content": rag._RAG_SYSTEM},
            {"role": "user",
             "content": f"【资料片段】\n{build_context(items, max_chars=4800)}\n\n"
                        f"【问题】\n{question}\n\n请依据上述资料片段回答。"},
        ])
        print(f"\n  回答（生成 {time.perf_counter()-t:.1f}s）：\n  {ans.strip()}")
        results[label] = contexts

    if must_have:
        step(3, "结论：两条链路的关键事实命中率对比")
        for label, ctxs in results.items():
            hit = [k for k in must_have if k in "\n".join(ctxs)]
            print(f"  {label:<22} {len(hit)}/{len(must_have)}"
                  f"{'   ← 漏掉：' + '、'.join(k for k in must_have if k not in hit) if len(hit) < len(must_have) else '   ← 全中'}")


def demo_image_compare(question: str, must_have: list[str]) -> None:
    """
    工单4 主线演示：**图像不解析 vs 图像多模态解析**（可选再叠加 CLIP 跨模态召回）。

    三条链路跑同一个问题：
      ① 对照：图内散字与图题留在正文（工单1/2/3 的做法）
      ② 本工单：图区域经 qwen-vl-plus 解析成结构化描述独立成块
      ③ 本工单 + CLIP：再打开以文搜图通道
    每条都打印召回页码 / 类型 / 依据分（含 CLIP 相似度），并**打印命中的裁剪图路径**
    —— 录演示视频时把那张图打开，就能直接看到"答案确实出自这张图"。
    """
    from src.retriever import build_context, retrieve

    hr()
    print(f"问题：{question}")
    hr()

    kb_img = KnowledgeBase.get()
    try:
        kb_no = KnowledgeBase.get_noimage()
    except FileNotFoundError as exc:
        print(f"对照索引不可用：{exc}")
        print("请先执行：python scripts/build_index.py --no-images --out noimage")
        return
    _ = kb_no.bm25

    arms = [
        ("优化前 · 图像不解析", kb_no, False),
        ("优化后 · 图像多模态解析", kb_img, False),
        ("优化后 · 图像解析 + CLIP 跨模态召回", kb_img, True),
    ]

    results = {}
    for i, (label, kb, use_clip) in enumerate(arms, 1):
        step(i, label)
        t = time.perf_counter()
        r = retrieve(question, kb=kb, use_clip=use_clip)
        ms = (time.perf_counter() - t) * 1000
        items = list(r.items)
        contexts = [it.text for it in items]
        n_img = sum(1 for it in items if it.type == "image")
        print(f"  库内分块 {len(kb.chunks)} 条；召回 {len(items)} 条（图像块 {n_img}），"
              f"检索耗时 {ms:.0f} ms，闸门拦空={r.gated}")
        for it in items[:6]:
            clip = f" | CLIP {it.clip_sim:.3f}" if it.clip_sim else ""
            print(f"    · 依据分 {it.evidence:.4f}{clip} | {it.doc_key} 第{it.page}页 | "
                  f"{it.type:<5} | {it.section[:34]}")
        figs = [it for it in items if it.type == "image" and it.image]
        for it in figs[:3]:
            print(f"    ▸ 命中图：{it.image}（{it.fig_type}；"
                  f"图题 {it.caption or '无'}）")
        if must_have:
            print(f"  关键事实检查：{fact_report(contexts, must_have)}")
        t = time.perf_counter()
        ans = llm.chat([
            {"role": "system", "content": rag._RAG_SYSTEM},
            {"role": "user",
             "content": f"【资料片段】\n{build_context(items, max_chars=4800)}\n\n"
                        f"【问题】\n{question}\n\n请依据上述资料片段回答。"},
        ])
        print(f"\n  回答（生成 {time.perf_counter()-t:.1f}s）：\n  {ans.strip()}")
        results[label] = contexts

    if must_have:
        step(4, "结论：三条链路的关键事实命中率对比")
        for label, ctxs in results.items():
            joined = "".join("".join(s.split()) for s in ctxs)
            hit = [k for k in must_have if "".join(k.split()) in joined]
            miss = [k for k in must_have if k not in hit]
            print(f"  {label:<30} {len(hit)}/{len(must_have)}"
                  f"{'   ← 漏掉：' + '、'.join(miss) if miss else '   ← 全中'}")


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
    if qu.bypassed:
        print(f"  策略      : ★ 高置信直通（首条依据分 {qu.bypass_evidence} ≥ "
              f"{config.QU_BYPASS_MIN_EVIDENCE}，跳过 LLM 往返）")
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
        print(f"  剔除无区分度成分后: {info['trace']['effective_query']}")
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
    ap.add_argument("--compare", action="store_true", help="优化前后对比模式（工单2 主线演示）")
    ap.add_argument("--compare-table", action="store_true",
                    help="表格解析前后对比模式（工单3 主线演示）")
    ap.add_argument("--compare-image", action="store_true",
                    help="图像解析前后对比模式（工单4 主线演示）")
    args = ap.parse_args()

    hr()
    print("招股说明书 RAG 问答系统 · 演示")
    print(f"工单编号：{config.WORK_ORDER_NO_IMAGE}（{config.WORK_ORDER_SHORT}）")
    hr()

    print("\n【准备】加载索引并预热模型（真实服务在启动时完成这一步）")
    t = time.perf_counter()
    kb = KnowledgeBase.get()
    _ = kb.bm25
    _ = kb.ubiquitous_phrases
    embed_query("预热")
    if args.compare:
        try:
            from src.baseline import get_naive_index

            get_naive_index()
        except Exception as exc:  # noqa: BLE001
            print(f"  朴素基线索引不可用：{exc}")
            print("  请先执行：python scripts/build_baseline_index.py")
            return 2
    try:
        llm.chat([{"role": "user", "content": "ping"}], max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        print(f"  大模型预热失败：{exc}")
    print(f"  {kb.meta.get('source_file')} · {len(kb.chunks)} 块 "
          f"（表格块 {kb.meta.get('num_table_chunks')}）· "
          f"向量 {kb.meta.get('embedding_dim')} 维 · 耗时 {time.perf_counter()-t:.1f}s")

    qs = json.loads((config.EVAL_DIR / config.EVAL_DATASET).read_text(encoding="utf-8"))
    by_id = {q["id"]: q["question"] for q in qs}
    must_by_id = {q["id"]: (q.get("must_have") or []) for q in qs}

    if args.ask:
        targets = [(args.ask, [])]
    elif args.id:
        if args.id not in by_id:
            print(f"题号 {args.id} 不在工单问题列表里")
            return 2
        targets = [(by_id[args.id], must_by_id.get(args.id, []))]
    elif args.compare_image:
        # 工单4 演示默认挑两题：一题答案只在图的**连接线拓扑**里（组织结构图），
        # 一题答案只在图的**柱状数值**里（IC 市场应用结构与增长图）——
        # 正好覆盖"层级归属"和"图表数值"两类图，正文里都没有。
        targets = [(by_id[5], must_by_id[5]), (by_id[6], must_by_id[6])]
    elif args.compare_table:
        # 工单3 演示默认挑两题：一题答案在「募投项目表」里，一题答案在「关联方表」里
        targets = [(by_id[2], must_by_id[2]), (by_id[3], must_by_id[3])]
    elif args.compare:
        # 对比演示默认挑两题：一题看「多个期间的数据」，一题看「限定条件的陷阱」
        targets = [(by_id[260], must_by_id[260]), (by_id[795], must_by_id[795])]
    else:
        # 默认三题：新文档表格题 / 多个期间的数据 / 需要带上限定条件的陷阱题
        targets = [(by_id[1], must_by_id[1]), (by_id[33], must_by_id[33]),
                   (by_id[795], must_by_id[795])]

    for q, must in targets:
        if args.compare_image:
            demo_image_compare(q, must)
        elif args.compare_table:
            demo_table_compare(q, must)
        elif args.compare:
            demo_compare(q, must)
        else:
            demo_one(q, show_llm_only=not args.no_llm_only)

    hr()
    print("演示结束。交互式界面：python main.py → http://127.0.0.1:8010/ui")
    print("界面上「对比模式 → 表格解析前后（工单3）」可逐个问题做实链路演示。")
    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
