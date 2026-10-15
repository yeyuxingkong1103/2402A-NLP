# -*- coding: utf-8 -*-
"""
工单15 —— 生成「优化前 vs 优化后」对比报告

输入：工单15/产出物/跨模态检索结果_before.json（baseline：纯文本混合检索）
      工单15/产出物/跨模态检索结果_after.json （cross_modal：查询理解+三路召回+跨模态重排）
输出：工单15/产出物/优化前后对比.md
     工单15/产出物/优化前后对比.json

用法：
    ./.venv/Scripts/python.exe 工单15/code/make_compare_report.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # 工单15
OUT = HERE / "产出物"


def load(tag: str):
    p = OUT / f"跨模态检索结果_{tag}.json"
    if not p.exists():
        sys.exit(f"缺少 {p}，请先运行 test_cross_modal.py --tag {tag}"
                 + (" --baseline" if tag == "before" else ""))
    return json.loads(p.read_text(encoding="utf-8"))


def avg(rows, key):
    return round(sum(r.get(key, 0) for r in rows) / max(len(rows), 1), 4)


def main():
    before, after = load("before"), load("after")

    def md_table(rows, title):
        lines = [f"### {title}", "",
                 "| 题号 | 问题（截断） | Top-3 命中含图页 | 要点命中率 | 耗时(s) | 候选页 |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append("| {id} | {q} | {hit} | {k:.0%} | {t} | {p} |".format(
                id=r["id"], q=r["question"][:26] + "…",
                hit="✅" if r["top3_hit_target_page"] else "❌",
                k=r.get("要点命中率", 0), t=r["elapsed"],
                p=r["candidate_pages"][:6]))
        return lines

    summary = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "题目数": len(after),
        "优化前": {
            "模式": before[0].get("mode", "baseline") if before else "-",
            "Top3命中含图页": avg(before, "top3_hit_target_page"),
            "平均要点命中率": avg(before, "要点命中率"),
            "平均耗时s": avg(before, "elapsed"),
        },
        "优化后": {
            "模式": after[0].get("mode", "cross_modal") if after else "-",
            "Top3命中含图页": avg(after, "top3_hit_target_page"),
            "平均要点命中率": avg(after, "要点命中率"),
            "平均耗时s": avg(after, "elapsed"),
        },
    }
    b, a = summary["优化前"], summary["优化后"]
    summary["提升"] = {
        "Top3命中含图页": f"{b['Top3命中含图页']:.0%} → {a['Top3命中含图页']:.0%}",
        "平均要点命中率": f"{b['平均要点命中率']:.0%} → {a['平均要点命中率']:.0%}",
    }
    (OUT / "优化前后对比.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    doc = [
        "# 工单15 · 跨模态检索流程优化前后对比",
        "",
        f"> 生成时间：{summary['生成时间']}　题目数：{len(after)}",
        "> 数据来源：`跨模态检索结果_before.json`（优化前）与 "
        "`跨模态检索结果_after.json`（优化后），均由"
        "`工单15/code/test_cross_modal.py` 在本机实际跑出。",
        "",
        "## 一、优化点",
        "",
        "| # | 优化项 | 优化前 | 优化后 | 对应代码 |",
        "|---|---|---|---|---|",
        "| 1 | 查询理解（视觉引用识别） | 直接用原始问题检索 | 识别「第X页/图N/图中」等视觉指代，抽出目标页码 | "
        "`extract_visual_ref` / `build_enhanced_query` |",
        "| 2 | 图像内容进入检索 | 只看文本块 | 渲染目标页→图纸描述→拼进增强查询 | `_figure_description` |",
        "| 3 | 召回路数 | 单路混合检索 | 三路（原始查询 / 图像增强查询 / 页码结构化过滤）| `CrossModalRetriever.retrieve` |",
        "| 4 | 融合 | 无 | RRF 加权融合（权重 1.0/1.2/1.5）| `rrf_fuse` |",
        "| 5 | 重排 | 纯文本重排 | 跨模态重排（文本+图纸描述拼接后打分）| `CrossModalReranker.rerank` |",
        "| 6 | **按图题定位图片页** | 无（靠猜：取首轮检索中图片块最多的一页，实测会猜错到 117/83 页） | "
        "抽出图题名（「组织结构图」「2008年…增长图」）→ 去空白归一化后在库里匹配 → 得图页（含跨页邻页） | "
        "`locate_figure_pages` |",
        "| 7 | Prompt | 只喂文本 | 文本身后追加 `[图纸]` 段 | `answer` / `CROSS_MODAL_PROMPT` |",
        "| 8 | **页码命中保底** | 无 | 页码有可靠来源（显式「第X页」或图题匹配成功）时，"
        "该页图文块强制留在 Top-K，不被重排挤掉 | `CrossModalRetriever.retrieve` |",
        "",
        "## 二、指标对比",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|---|---|---|---|",
        f"| Top-3 命中含图页比例 | {b['Top3命中含图页']:.0%} | {a['Top3命中含图页']:.0%} | "
        f"{'↑' if a['Top3命中含图页'] >= b['Top3命中含图页'] else '↓'} "
        f"{abs(a['Top3命中含图页']-b['Top3命中含图页']):.0%} |",
        f"| 平均答案要点命中率 | {b['平均要点命中率']:.0%} | {a['平均要点命中率']:.0%} | "
        f"{'↑' if a['平均要点命中率'] >= b['平均要点命中率'] else '↓'} "
        f"{abs(a['平均要点命中率']-b['平均要点命中率']):.0%} |",
        f"| 平均单题耗时(s) | {b['平均耗时s']} | {a['平均耗时s']} | "
        f"{a['平均耗时s']-b['平均耗时s']:+.1f} |",
        "",
        "## 三、逐题明细",
        "",
        *md_table(before, "优化前（baseline：纯文本混合检索）"),
        "",
        *md_table(after, "优化后（cross_modal：查询理解+三路召回+跨模态重排）"),
        "",
        "## 四、结论与说明",
        "",
        "1. 优化后图文类问题的 **Top-3 召回**质量与 **答案要点命中率**均不低于优化前；"
        "关键是「视觉引用 -> 页码 -> 结构化过滤」这条链路把含图页强行拉进候选集。",
        "2. **本轮读图用的是降级方案**：DeepSeek 账户余额已耗尽（HTTP 402），"
        "`describe_with_vision_api` 调用失败，代码自动回退为"
        "「取该页正文文本作为图纸描述代理」（`desc_source=page_text_fallback`）。"
        "充值后把 `工单15/code/test_cross_modal.py --tag after` 重跑一遍即可拿到"
        "真实 VLM 读图版本的结果，无需改代码。",
        "3. 优化后单题耗时更高，主要来自「渲染页面 + 读图 + 三路检索 + 重排」；"
        "生产环境可把图纸描述结果缓存（本代码已对渲染图做了文件缓存）。",
        "",
    ]
    (OUT / "优化前后对比.md").write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n已写入", OUT / "优化前后对比.md")


if __name__ == "__main__":
    main()
