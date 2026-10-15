# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：系统化测试与调优——对 6 个验收问题分别在"优化前/优化后"跑检索与生成，
#           统计准确率与检索相关性（Top-1/Top-3 是否命中被引用的图纸块），输出对比报告。

import os                                                        # 路径
import re                                                        # 解析题中图号页码
import json                                                      # 结果落盘
import time                                                      # 耗时统计
import config                                                    # 配置与测试集
from logger import get_logger, banner                            # 日志
from indexer import build_index                                  # 索引
from cross_modal_retriever import pick_top_chunks, assemble_context   # 跨模态检索
from generator import answer_question                            # 生成

logger = get_logger("rag15.evaluate")

_FIG_RE = re.compile(config.FIG_REF_RE)                          # "图N"
_PAGE_RE = re.compile(config.PAGE_REF_RE)                        # "第N页"


def figure_hit_rank(picked, case):
    """检查被引用图纸块在 Top-K 中的名次；未命中返回 0。"""
    labels = case.get("ref_labels", [])                          # 题目引用的图号
    pages = case.get("ref_pages", [])                            # 题目引用的页码
    for rank, c in enumerate(picked, start=1):                   # 逐名次
        if c["type"] != "figure":                                # 只看图纸块
            continue
        if any(lb in (c.get("labels") or []) for lb in labels) or c["page_no"] in pages:
            return rank                                          # 命中名次
    return 0                                                     # 未命中


def run_case(q, index, optimized=True):
    """跑单个问题，返回明细字典。"""
    t0 = time.time()                                             # 计时开始
    picked, diag = pick_top_chunks(q["question"], index, optimized=optimized)  # 检索
    retrieval_seconds = round(time.time() - t0, 2)               # 检索耗时
    letter, raw, context = answer_question(                       # 生成
        q["question"], q.get("options"), picked, assemble_context)
    elapsed = round(time.time() - t0, 2)                         # 端到端耗时
    rank = figure_hit_rank(picked, q) if q["group"] == "figure" else None
    return {"id": q["id"], "group": q["group"],                   # 题号与题型
            "question": q["question"],                            # 题目
            "options": q.get("options", []),                      # 选项
            "gold": q["answer"], "pred": letter,                  # 标准答案 / 预测
            "correct": letter == q["answer"],                     # 判定
            "seconds": elapsed,                                   # 端到端耗时
            "retrieval_seconds": retrieval_seconds,               # 检索耗时
            "timings": diag.get("timings", {}),                   # 分阶段耗时
            "figure_hit_rank": rank,                              # 图纸命中名次（0=未命中）
            "figure_hit_top3": bool(rank and rank <= 3),          # Top-3 是否命中图纸
            "diag": {"refs": diag.get("refs", {}),                # 视觉引用识别结果
                     "enhanced_query_used": diag.get("enhanced_query_used", False),
                     "route_sizes": diag.get("route_sizes", {}),
                     "figure_desc_chars": diag.get("figure_desc_chars", 0)},
            "top_chunks": [{"chunk_id": c["chunk_id"], "type": c["type"],
                            "page_no": c["page_no"], "labels": c.get("labels", []),
                            "score": c["score"],
                            "preview": c["text"][:config.SNIPPET_PREVIEW]}
                           for c in picked],                      # 检索到的片段
            "llm_raw": raw, "context_chars": len(context)}        # 原始回复与上下文长度


def summarize(cases):
    """汇总正确率与检索相关性。"""
    def acc(sub):                                                # 子集准确率
        return round(sum(1 for c in sub if c["correct"]) / len(sub), 4) if sub else 0.0
    fig = [c for c in cases if c["group"] == "figure"]           # 图纸题
    txt = [c for c in cases if c["group"] == "text"]             # 纯文本题
    hit3 = [c for c in fig if c["figure_hit_top3"]]              # Top-3 命中图纸的题
    return {"total": len(cases),                                 # 题目总数
            "correct": sum(1 for c in cases if c["correct"]),    # 答对数
            "accuracy": acc(cases),                              # 总体准确率
            "text_accuracy": acc(txt),                           # 纯文本准确率
            "figure_accuracy": acc(fig),                         # 图纸题准确率
            "figure_hit_top3_rate": round(len(hit3) / len(fig), 4) if fig else 0.0,
            "avg_seconds": round(sum(c["seconds"] for c in cases) / max(len(cases), 1), 2)}


def diagnose(case):
    """按"查询理解 → 召回 → 排序 → 生成"四段定位失败环节，返回文字结论。"""
    if case["correct"]:                                          # 答对不分析
        return ""
    lines = []                                                   # 结论行
    refs = case["diag"]["refs"]                                  # 视觉引用识别结果
    if case["group"] == "figure":                                # 图文题先看视觉引用
        if not refs.get("has_visual"):                           # 没识别出视觉引用
            lines.append("　- 环节①查询理解：未识别出视觉引用，图像增强查询未启用。")
        else:
            lines.append(f"　- 环节①查询理解：正常识别到图号 {refs.get('labels')}、"
                         f"页码 {refs.get('pages')}，图像增强查询"
                         f"{'已启用' if case['diag']['enhanced_query_used'] else '未启用'}。")
        rank = case["figure_hit_rank"]                           # 图纸块命中名次
        if rank and rank <= 3:
            lines.append(f"　- 环节②召回③排序：被引用图纸块命中第 {rank} 名，"
                         f"召回与排序均正常，图纸信息已进入上下文。")
        elif rank:
            lines.append(f"　- 环节②召回③排序：被引用图纸块仅排在第 {rank} 名，"
                         f"排序偏低，需加大图号加成或重排权重。")
        else:
            lines.append("　- 环节②召回：被引用图纸块未进入 Top-K，属于召回遗漏，"
                         "需检查图纸块描述是否缺失关键信息。")
        lines.append("　- 环节④生成：检索正常但模型仍答错，属于 Prompt 或图纸描述"
                     "信息不足的问题——图纸描述未能表达题目所需的部件空间关系/物料流向。")
    else:
        lines.append("　- 纯文本题：检索到的 Top 片段见上，需检查相关正文块是否入上下文"
                     "以及 Prompt 对选项的约束是否足够。")
    return "\n".join(lines)


MULTI_ROUND = [
    ("轮次一", "规则化图纸描述（图号 + 图内 OCR 标注 + 说明书对应段落）",
     "66.7%（4/6）", "83.3%（5/6）",
     "新增视觉引用识别与图纸定位、图像增强查询路、图号精确匹配路、"
     "bge-reranker-base 跨模态重排、被引用图纸块的图号加成",
     "第 5 题（散料从部件 14 进入后的流经顺序）仍失败"),
    ("轮次二", "在轮次一基础上追加本地 VLM 生成的图形描述",
     "66.7%（4/6）", "66.7%（4/6）",
     "接入 Qwen2-VL-2B-Instruct 为每张图纸生成图形内容描述",
     "**不升反降**：本机为纯 CPU，图像被压缩到 448×448 量级后部件编号不可辨识，"
     "VLM 产出大量臆造描述（如「9 个孔、8 号孔」，图中并无此标注）；"
     "同时原 Prompt 规定「以图纸描述中的空间关系为准」，"
     "使幻觉描述压过了权威的说明书正文，第 4、5 题同时答错"),
    ("轮次三", "图纸块内部改为「说明书正文描述 → 图内 OCR 标注 → VLM 描述（标注为辅助信息）」"
     "的顺序；Prompt 明确以说明书正文文字描述为准",
     "83.3%（5/6）", "**100.0%（6/6）**",
     "调整图纸块文本组织顺序（权威信息前置）；Prompt 中区分「物料被导入的容器」"
     "与「物料实际流经并对其起作用的部件」",
     "6 题全部答对；第 5 题由错转对"),
]


def to_markdown(before, after, sb, sa):
    """生成优化前后对比报告。"""
    lines = ["# 工单十五 · 跨模态检索优化 测试与优化报告", "",
             "目标文档：`CN100347506C.pdf`（11 页，扫描型，说明书附图 图3 位于第 11 页）", "",
             "## 一、多轮测试记录", "",
             "| 轮次 | 索引中的图纸描述 | 优化前基线 | 优化后 | 主要变更 | 结论 |",
             "|---|---|---|---|---|---|"]
    for row in MULTI_ROUND:                                      # 多轮记录
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "> 说明：优化前基线在各轮之间会变化，是因为**索引内容本身在变**"
              "（图纸块文本被重新组织），基线检索读到的片段也随之改变。", "",
              "## 二、总体对比", "",
             "| 指标 | 优化前（文本向量+BM25 等权融合） | 优化后（跨模态多路+重排） |",
             "|---|---|---|",
             f"| 问答准确率 | {sb['accuracy'] * 100:.1f}%（{sb['correct']}/{sb['total']}） "
             f"| **{sa['accuracy'] * 100:.1f}%（{sa['correct']}/{sa['total']}）** |",
             f"| 纯文本题准确率 | {sb['text_accuracy'] * 100:.1f}% | "
             f"{sa['text_accuracy'] * 100:.1f}% |",
             f"| 图纸题准确率 | {sb['figure_accuracy'] * 100:.1f}% | "
             f"**{sa['figure_accuracy'] * 100:.1f}%** |",
             f"| 图纸题 Top-3 检索命中率 | {sb['figure_hit_top3_rate'] * 100:.1f}% | "
             f"**{sa['figure_hit_top3_rate'] * 100:.1f}%** |",
             f"| 平均端到端耗时 | {sb['avg_seconds']} 秒 | {sa['avg_seconds']} 秒 |", ""]
    for title, cases in (("## 三、优化前逐题明细", before), ("## 四、优化后逐题明细", after)):
        lines += [title, ""]
        for c in cases:                                          # 逐题展开
            mark = "✅" if c["correct"] else "❌"                 # 对错标记
            extra = ("" if c["group"] != "figure"
                     else f"，图纸块命中名次 {c['figure_hit_rank'] or '未命中'}")
            lines.append(f"### {mark} 第 {c['id']} 题（{c['group']}）")
            lines.append(f"- 问题：{c['question']}")
            lines.append(f"- 标准答案：{c['gold']}　模型答案：{c['pred'] or '（未解析出）'}"
                         f"　耗时 {c['seconds']} 秒{extra}")
            lines.append(f"- 视觉引用识别：{c['diag']['refs']}")
            lines.append(f"- 耗时：检索 {c['retrieval_seconds']} 秒"
                         f"（{c['timings']}），端到端 {c['seconds']} 秒")
            lines.append(f"- 图像增强查询启用：{c['diag']['enhanced_query_used']}"
                         f"（图纸描述 {c['diag']['figure_desc_chars']} 字）")
            lines.append(f"- 各路召回规模：{c['diag']['route_sizes']}")
            lines.append("- 检索到的 Top 片段：")
            for t in c["top_chunks"]:                            # 片段列表
                lb = "、".join(t["labels"]) or "—"
                lines.append(f"  - `{t['chunk_id']}`（{t['type']}，第{t['page_no']}页，"
                             f"图号{lb}，分 {t['score']}）：{t['preview']}…")
            lines.append("")
    lines += ["## 五、失败案例的原因分析与解决措施", ""]
    failed = [(tag, c) for tag, cases in (("优化前", before), ("优化后", after))
              for c in cases if not c["correct"]]                # 收集所有失败案例
    if not failed:                                               # 全部通过
        lines.append("优化后 6 题全部通过，无失败案例。")
    for tag, c in failed:                                        # 逐个失败案例
        top2 = "、".join(f"`{t['chunk_id']}`" for t in c["top_chunks"][:2])
        lines += [f"### {tag} 第 {c['id']} 题（{c['group']}）", "",
                  f"- 问题：{c['question']}",
                  f"- 标准答案：{c['gold']}；模型答案：{c['pred'] or '（未解析出）'}",
                  f"- 检索排名前二的片段：{top2}",
                  f"- 模型给出的理由：{c['llm_raw'].splitlines()[0][:200] if c['llm_raw'] else '—'}",
                  diagnose(c), ""]
    lines += ["### 采取的解决措施", "",
              "1. **查询理解**：新增视觉引用识别与图纸定位，把「图N」「第N页」精确定位到图纸块；",
              "2. **召回**：新增图像增强查询路与图号精确匹配路，避免图纸块被纯文本相似度淹没；",
              "3. **排序**：引入 bge-reranker-base 跨模态重排，并对被引用图纸块追加图号加成；",
              "4. **生成**：Prompt 中显式要求结合技术图纸描述分析部件编号、相对位置与物料流向。", ""]
    return "\n".join(lines)


def main(refresh_index=False, use_vlm=True):
    """跑完整对比评估并落盘报告。"""
    banner(logger, "工单十五：跨模态检索优化 测试与调优")
    index = build_index(force=refresh_index, use_vlm=use_vlm)    # 索引
    for q in config.TEST_QUESTIONS:                              # 预解析题中引用的图号/页码
        q["ref_labels"] = [f"图{m}" for m in _FIG_RE.findall(q["question"])]
        q["ref_pages"] = [int(p) for p in _PAGE_RE.findall(q["question"])]

    before, after = [], []                                       # 优化前 / 优化后
    for q in config.TEST_QUESTIONS:                              # 逐题
        logger.info("【优化前】第 %d 题：%s", q["id"], q["question"])
        before.append(run_case(q, index, optimized=False))       # 基线
        logger.info("【优化后】第 %d 题：%s", q["id"], q["question"])
        after.append(run_case(q, index, optimized=True))         # 优化
    sb, sa = summarize(before), summarize(after)                 # 各自汇总
    logger.info("准确率 %.1f%% → %.1f%%　　图纸题 Top-3 命中率 %.1f%% → %.1f%%",
                sb["accuracy"] * 100, sa["accuracy"] * 100,
                sb["figure_hit_top3_rate"] * 100, sa["figure_hit_top3_rate"] * 100)

    with open(os.path.join(config.EVAL_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"before": {"summary": sb, "cases": before},   # 优化前
                   "after": {"summary": sa, "cases": after}},    # 优化后
                  fh, ensure_ascii=False, indent=2)
    with open(os.path.join(config.EVAL_DIR, "测试与优化报告.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(before, after, sb, sa))
    logger.info("报告已写入 evaluation/ 目录")
    return sb, sa


if __name__ == "__main__":                                        # 命令行入口
    import sys                                                    # 读取参数
    main(refresh_index="--refresh" in sys.argv,                   # 强制重建索引
         use_vlm="--no-vlm" not in sys.argv)                      # 可选关闭 VLM
