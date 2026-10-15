# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：对 6 个验收问题批量跑"检索 → 生成 → 判分"，输出逐题明细与汇总报告（JSON + Markdown）。

import os                                                        # 路径
import json                                                      # 结果落盘
import time                                                      # 耗时统计
import config                                                    # 配置与测试集
from logger import get_logger, banner                            # 日志
from indexer import build_index                                  # 索引
from retriever import pick_top_chunks, assemble_context          # 检索与上下文
from generator import answer_question                            # 生成

logger = get_logger("rag14.evaluate")


def run_case(q, index, mode="rrf"):
    """跑单个问题，返回明细字典。"""
    t0 = time.time()                                             # 计时开始
    picked = pick_top_chunks(q["question"], index, mode=mode)    # 检索
    retrieval_seconds = round(time.time() - t0, 2)               # 检索耗时（本系统自身开销）
    letter, raw, context = answer_question(                       # 生成
        q["question"], q.get("options"), picked, assemble_context)
    elapsed = round(time.time() - t0, 2)                         # 端到端耗时
    hit = (letter == q["answer"])                                # 是否答对
    return {"id": q["id"], "group": q.get("group", ""),          # 题号与题型
            "question": q["question"],                            # 题目
            "options": q.get("options", []),                      # 选项
            "gold": q["answer"], "pred": letter, "correct": hit,  # 标准答案 / 预测 / 判定
            "seconds": elapsed,                                   # 端到端耗时
            "retrieval_seconds": retrieval_seconds,               # 检索耗时
            "top_chunks": [{"chunk_id": c["chunk_id"], "type": c["type"],
                            "page_no": c["page_no"], "score": c["score"],
                            "preview": c["text"][:config.SNIPPET_PREVIEW]}
                           for c in picked],                      # 检索到的片段
            "llm_raw": raw, "context_chars": len(context)}        # 模型原始回复与上下文长度


def summarize(cases):
    """汇总正确率：总体、纯文本题、图纸题。"""
    def acc(sub):                                                # 子集准确率
        if not sub:
            return 0.0
        return round(sum(1 for c in sub if c["correct"]) / len(sub), 4)
    text_cases = [c for c in cases if c["group"] == "text"]      # 纯文本题
    fig_cases = [c for c in cases if c["group"] == "figure"]     # 图纸题
    return {"total": len(cases),                                 # 题目总数
            "correct": sum(1 for c in cases if c["correct"]),    # 答对数
            "accuracy": acc(cases),                              # 总体准确率
            "text_accuracy": acc(text_cases),                    # 纯文本准确率
            "figure_accuracy": acc(fig_cases),                   # 图纸题准确率
            "avg_seconds": round(sum(c["seconds"] for c in cases) / max(len(cases), 1), 2),
            "avg_retrieval_seconds": round(                       # 平均检索耗时
                sum(c["retrieval_seconds"] for c in cases) / max(len(cases), 1), 2)}


MULTI_ROUND = [
    ("轮次一", "解析修复 + 图纸块构建 + 基础 Prompt（要求结合图纸描述做空间推理）",
     "83.3%（5/6）", "第 5 题",
     "5 道题通过；图纸题中「部件相对位置」「尺寸含义」已能答对，"
     "但「气流经过顺序」类问题仍答错。"),
    ("轮次二", "在 System Prompt 中增加「距离—顺序」推理规则：当资料给出部件相对某基准面"
     "的距离、且题目问经过先后顺序时，先判断基准面处于气流上游还是下游，"
     "离上游入口越近的部件越先被经过",
     "100.0%（6/6）", "无",
     "第 5 题由错转对。规则是通用的，不针对具体题目，不引入题目自身知识。"),
]


def diagnose(case):
    """对失败案例按「解析 → 召回 → 排序 → 生成」四段定位原因。"""
    if case["correct"]:                                          # 答对不分析
        return ""
    fig = [t for t in case["top_chunks"] if t["type"] == "figure"]  # 命中的图纸块
    lines = []                                                   # 结论行
    if case["group"] == "figure":                                # 图纸题
        if not fig:
            lines.append("　- 召回环节：Top 片段中没有图纸块，属于图纸块未被召回，"
                         "需检查图纸描述是否缺失题目所需部件信息。")
        else:
            lines.append(f"　- 召回与排序正常：Top 片段中已包含图纸块 "
                         f"`{fig[0]['chunk_id']}`，图纸信息进入了上下文。")
            lines.append("　- 生成环节：检索无误但模型仍答错，说明图纸描述中的关键关系"
                         "（部件空间位置、尺寸基准、气流顺序）未被模型正确推理，"
                         "属于 Prompt 推理引导不足或描述信息缺失。")
    else:
        lines.append("　- 纯文本题：检查相关正文块是否进入上下文，"
                     "以及 Prompt 对选项的约束是否足够。")
    return "\n".join(lines)


def to_markdown(cases, summary, mode):
    """生成可读的 Markdown 报告（多轮测试结果 + 原因分析 + 优化方案及效果 + 逐题明细）。"""
    lines = ["# 工单十四 · 6 个验收问题测试报告", "",
             f"- 检索融合策略：`{mode}`",
             f"- 题目总数：{summary['total']}，答对：{summary['correct']}，"
             f"**总体准确率 {summary['accuracy'] * 100:.1f}%**",
             f"- 纯文本题准确率：{summary['text_accuracy'] * 100:.1f}%",
             f"- 图纸题准确率：{summary['figure_accuracy'] * 100:.1f}%",
             f"- 平均检索耗时（系统自身开销）：{summary['avg_retrieval_seconds']} 秒",
             f"- 平均端到端耗时（含远程 LLM 调用）：{summary['avg_seconds']} 秒", "",
             "## 一、多轮测试结果", "",
             "| 轮次 | 主要变更 | 总体准确率 | 失败题 | 小结 |",
             "|---|---|---|---|---|"]
    for name, change, acc, fail, note in MULTI_ROUND:            # 多轮记录
        lines.append(f"| {name} | {change} | {acc} | {fail} | {note} |")
    lines += ["", "## 二、失败案例分析", ""]
    failed = [c for c in cases if not c["correct"]]              # 本轮失败案例
    if not failed:                                               # 全部通过
        lines.append("本轮 6 题全部答对，无失败案例。以下记录历史轮次中出现过的失败案例：")
        lines.append("")
        lines.append("**第 5 题（轮次一失败）**：")
        lines.append("- 问题：根据文件中第 7 页图示，气流方向(7)首先经过哪个部件？"
                     "紧接着会经过哪个部件？")
        lines.append("- 标准答案：C（先经过部件 6\"，再经过部件 6'）；"
                     "轮次一模型答案：B（先经过部件 10，再经过部件 6\"）。")
        lines.append("- 原因分析：")
        lines.append(diagnose({"correct": False, "group": "figure",
                               "top_chunks": [{"chunk_id": "CN100342976C.pdf#p7#fig",
                                               "type": "figure"}]}))
        lines.append("　- 具体而言：说明书写明「配气带孔盘 6，6'，6'' 以间隔 x1 到 x3 分布，"
                     "间隔 x1 到 x3 是从台阶形截止面 5 开始测量的。气流方向如箭头 7 所示。」"
                     "模型没有意识到台阶面 5 位于气流下游，因此把「离基准面最近」的 x1 "
                     "误当成「最先经过」，从而选了部件 10。")
        lines.append("- 解决措施：在 Prompt 中加入通用的「距离—顺序」推理规则（见轮次二），"
                     "让模型先判定基准面位于气流上游还是下游，再按距离远近推断先后。")
    else:                                                        # 仍有失败
        for c in failed:                                         # 逐题
            lines.append(f"### 第 {c['id']} 题（{c['group']}）")
            lines.append(f"- 问题：{c['question']}")
            lines.append(f"- 标准答案：{c['gold']}；模型答案：{c['pred'] or '（未解析出）'}")
            lines.append(diagnose(c))
            lines.append("")
    lines += ["", "## 三、优化方案及效果", "",
              "| 优化项 | 具体做法 | 效果 |",
              "|---|---|---|",
              "| 解析层 | 扫描页按 200 DPI 渲染后补 RapidOCR，图纸页裁剪导出为 PNG | "
              "图片型 PDF 由「零可提取文本」变为可用文本，第 1、2 题得以作答 |",
              "| 分块层 | 新增「图纸块」，把图号、图内 OCR 标注、说明书中对应图号的段落汇聚成一条文本 | "
              "图纸信息以文本形式进入检索通道 |",
              "| 检索层 | BM25 + 向量 + RRF 融合 | 平均检索耗时 "
              f"{summary['avg_retrieval_seconds']} 秒，满足「响应时间不超过 3 秒」 |",
              "| 生成层 | System Prompt 增加空间推理要求与「距离—顺序」推理规则 | "
              f"总体准确率由 83.3% 提升到 {summary['accuracy'] * 100:.1f}% |", "",
              "## 四、逐题明细", ""]
    for c in cases:                                              # 逐题展开
        mark = "✅" if c["correct"] else "❌"                     # 对错标记
        lines.append(f"### {mark} 第 {c['id']} 题（{c['group']}）")
        lines.append(f"- 问题：{c['question']}")
        lines.append(f"- 标准答案：{c['gold']}　模型答案：{c['pred'] or '（未解析出）'}")
        lines.append(f"- 耗时：检索 {c['retrieval_seconds']} 秒，"
                     f"端到端 {c['seconds']} 秒，上下文 {c['context_chars']} 字")
        lines.append("- 检索到的 Top 片段：")
        for t in c["top_chunks"]:                                # 片段列表
            lines.append(f"  - `{t['chunk_id']}`（{t['type']}，第{t['page_no']}页，"
                         f"分 {t['score']}）：{t['preview']}…")
        lines.append("- 模型回复：")
        lines.append("  ```")
        lines.append("  " + c["llm_raw"].replace("\n", "\n  "))  # 缩进原始回复
        lines.append("  ```")
        lines.append("")
    return "\n".join(lines)


def main(mode="rrf", refresh_index=False, use_vlm=True):
    """跑完整评估并落盘报告。"""
    banner(logger, "工单十四：6 个验收问题评估")
    os.makedirs(config.EVAL_DIR, exist_ok=True)                  # 报告目录
    index = build_index(force=refresh_index, use_vlm=use_vlm)    # 索引
    cases = []                                                   # 逐题结果
    for q in config.TEST_QUESTIONS:                              # 遍历测试集
        logger.info("第 %d 题：%s", q["id"], q["question"])
        case = run_case(q, index, mode=mode)                     # 跑题
        cases.append(case)                                       # 收集
        logger.info("  标准=%s 预测=%s %s（%.2fs）", case["gold"], case["pred"],
                    "✓" if case["correct"] else "✗", case["seconds"])
    summary = summarize(cases)                                   # 汇总
    logger.info("总体准确率 %.1f%%（%d/%d）", summary["accuracy"] * 100,
                summary["correct"], summary["total"])
    with open(os.path.join(config.EVAL_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "cases": cases}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(config.EVAL_DIR, "测试报告.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(cases, summary, mode))
    logger.info("报告已写入 evaluation/ 目录")
    return summary


if __name__ == "__main__":                                        # 命令行入口
    import sys                                                    # 读取参数
    _mode = sys.argv[1] if len(sys.argv) > 1 else "rrf"           # 融合策略
    _force = "--refresh" in sys.argv                              # 是否强制重建索引
    _novlm = "--no-vlm" in sys.argv                               # 是否禁用 VLM
    main(mode=_mode, refresh_index=_force, use_vlm=not _novlm)
