# -*- coding: utf-8 -*-
"""工单14 测试截图生成：把真实实证产物渲染成中文命名 PNG。

工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单

数据来源（只读，绝不手绘假数据）：
  研发/data/解析实证.json     —— 文本层探针 + 像素 OCR + 标记覆盖
  研发/评估/六题结果.json      —— 6 题逐题判定与证据页
  测试/测试数据/测试问题集.jsonl —— 题面与答案键

产出（测试/测试截图/）：
  文件性质实证_扫描件无文本层.png
  附图标记恢复覆盖_7比14.png
  六题逐题判定与证据页.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, r"E:\gao6gongdan\_共享工具")
from 截图渲染 import Page, load_json  # noqa: E402

EV = ROOT / "研发" / "data" / "解析实证.json"
RES = ROOT / "研发" / "评估" / "六题结果.json"
SHOT = ROOT / "测试" / "测试截图"
SHOT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ev = load_json(EV)
    res = load_json(RES)
    if not ev or not res:
        print(f"[!!] 缺产物：实证={bool(ev)} 结果={bool(res)}")
        return 3

    # ---------------------------------------------------- 图1 文件性质实证
    p = Page("工单14 文件性质实证：纯扫描件，无文本层",
             "CN100342976C.pdf｜文本层 0 字符 → 纯文本 RAG 必然取不到内容")
    p.section("1. 文本层探针（逐页实测）")
    rows = [[str(x["page"]), str(x["text_chars"]), str(x["images"]),
             f"{x['rect'][0]}×{x['rect'][1]}"] for x in ev["1_文本层探针"]["per_page"]]
    p.table_block(["页", "文本字符数", "位图数", "页面尺寸(pt)"], rows, widths=[1, 3, 2, 3])
    p.section("2. 结论")
    p.kv_block([
        ("页数", str(ev["1_文本层探针"]["pages"])),
        ("文本层总字符数", str(ev["1_文本层探针"]["text_chars_total"])),
        ("PDF 体积", f"{ev['pdf_bytes'] / 1024:.0f} KB"),
        ("判定", ev["1_文本层探针"]["verdict"]),
    ], cols=2)
    p.note("依据：研发/data/解析实证.json（由 研发/scripts/演示_扫描件解析实证.py 实跑生成）")
    p.finish(SHOT / "文件性质实证_扫描件无文本层.png", sources=["研发/data/解析实证.json"])

    # ---------------------------------------------------- 图2 标记恢复覆盖
    cov = ev["4_标记覆盖"]
    p2 = Page("工单14 附图标记恢复覆盖：7 / 14",
              "自研像素方案（字体模板匹配，零 OCR 依赖）在附图页实测")
    p2.section("1. 恢复率")
    p2.kv_block([
        ("所需标记数", str(cov["needed_count"])),
        ("恢复成功", str(cov["recovered_count"])),
        ("覆盖率", f"{cov['recovered_count'] / cov['needed_count'] * 100:.1f}%"),
    ], cols=3)
    p2.section("2. 逐项结果")
    p2.table_block(["状态", "附图标记", "对答题的影响"],
                   [["恢复", "、".join(cov["recovered"]), "可用于位置/流向类判定"],
                    ["缺失", "、".join(cov["missing"]),
                     "题4（X1/X2/X3）、题6（h1/h2）、题5（6'/6''）的答题要件"]],
                   widths=[2, 5, 6])
    p2.section("3. 像素 OCR 实跑记录")
    p2.table_block(["页", "命中字符", "归并 token", "耗时(ms)"],
                   [[k, str(v["hits_scored"]), str(len(v["tokens"])), f"{v['elapsed_ms']:.0f}"]
                    for k, v in ev["3_像素OCR"].items()],
                   widths=[2, 2, 2, 2])
    p2.note("关键结论：缺失集合与题目要求高度重合 → 纯像素 OCR 不足以支撑图内空间推理题。")
    p2.finish(SHOT / "附图标记恢复覆盖_7比14.png", sources=["研发/data/解析实证.json"])

    # ---------------------------------------------------- 图3 六题判定
    p3 = Page("工单14 六题逐题判定与证据页",
              "8 页 @150dpi 渲染后逐页读图取证｜四选一，答案键来自 测试/测试数据/测试问题集.jsonl")
    p3.section("1. 汇总")
    s = res["汇总"]
    p3.kv_block([
        ("题数", str(s["题数"])),
        ("答对", str(s["答对"])),
        ("准确率", f"{s['准确率'] * 100:.0f}%"),
        ("依赖图内空间信息", f"{s['需图内空间信息的题数']} 题（题3/4/5/6），全对"),
    ], cols=2)
    p3.section("2. 逐题明细")
    rows3 = []
    for x in res["逐题"]:
        rows3.append([str(x["序号"]), str(x["group"]), x["类型"],
                      x["answer"], "✅ 正确" if x["correct"] else "❌ 错误",
                      f"第{x['evidence_page']}页"])
    p3.table_block(["题号", "组", "类型", "答案", "判定", "证据页"], rows3,
                   widths=[1, 1, 3, 1, 2, 2])
    p3.section("3. 证据摘要（节选）")
    for x in res["逐题"]:
        p3.note(f"题{x['序号']}（第{x['evidence_page']}页）：{x['evidence'][:110]}", size=15)
    p3.finish(SHOT / "六题逐题判定与证据页.png",
              sources=["研发/评估/六题结果.json", "测试/测试数据/测试问题集.jsonl"])

    print("[OK] 3 张截图 ->", SHOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
