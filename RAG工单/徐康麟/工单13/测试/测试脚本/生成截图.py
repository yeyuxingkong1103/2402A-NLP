# -*- coding: utf-8 -*-
"""工单13 测试截图生成：把真实产物渲染成中文命名 PNG。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

数据来源（**只读真实产物，绝不手绘**）：
  优化/评估结果/优化前_汇总.json、优化前.jsonl
  优化/评估结果/优化后_汇总.json、优化后.jsonl

产出（测试/测试截图/）：
  优化前后延迟分位对比.png       —— P50/P90/P95/最大/均值 与 SLA 达成率
  五阶段耗时对比.png             —— 各阶段均值（优化前 vs 优化后）与占比
  逐题耗时明细.png               —— 逐题 total/wall 对照
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT
sys.path.insert(0, str(Path(r"E:\gao6gongdan\_共享工具")))
from 截图渲染 import Page, load_json, load_jsonl  # noqa: E402

EVAL = WORK / "优化" / "评估结果"
SHOT = WORK / "测试" / "测试截图"
SHOT.mkdir(parents=True, exist_ok=True)


def stat_of(summary: dict | None, key: str, field: str = "均值") -> float | None:
    """从汇总 json 的『统计』段安全取值（键名可能是中文子字典）。"""
    if not summary:
        return None
    st = summary.get("统计") or {}
    node = st.get(key)
    if isinstance(node, dict):
        v = node.get(field)
        if isinstance(v, (int, float)):
            return float(v)
    if isinstance(node, (int, float)):
        return float(node)
    return None


def num(v, digits: int = 1, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}{suffix}"


def main() -> int:
    before_s = load_json(EVAL / "优化前_汇总.json")
    after_s = load_json(EVAL / "优化后_汇总.json")
    before_q = load_jsonl(EVAL / "优化前.jsonl")
    after_q = load_jsonl(EVAL / "优化后.jsonl")

    if not before_s or not after_s:
        print(f"[!!] 缺汇总文件：前={bool(before_s)} 后={bool(after_s)}")
        return 3

    # ---------------------------------------------------------- 图1 延迟分位对比
    p = Page("工单13 优化前后延迟分位对比", "口径：use_cache=False（只写不读）、丢弃预热轮、17 题测试集")
    p.section("1. 端到端延迟分位（ms）")
    rows = []
    for field in ("均值", "P50", "P90", "P95", "最大"):
        b = stat_of(before_s, "总耗时", field)
        a = stat_of(after_s, "总耗时", field)
        delta = ""
        if b and a is not None:
            delta = f"{(a - b) / b * 100:+.1f}%"
        rows.append([field, num(b), num(a), delta])
    p.table_block(["指标", "优化前", "优化后", "变化"], rows, widths=[3, 3, 3, 2])

    p.section("2. 验收口径（SLA ≤ 3000 ms）")
    rows2 = []
    for label, s in (("优化前", before_s), ("优化后", after_s)):
        p95 = stat_of(s, "总耗时", "P95")
        rate = stat_of(s, "SLA", "达标率") or stat_of(s, "SLA", "通过率")
        rows2.append([label, num(p95), num(rate, 2, "%"),
                      "达标" if (p95 is not None and p95 <= 3000) else "未达标"])
    p.table_block(["档位", "P95 (ms)", "SLA 达标率", "结论"], rows2, widths=[2, 3, 3, 2])
    p.note("说明：本机为 CPU 推理（qwen2.5:3b），生成阶段占端到端耗时大头；数字全部取自 优化/评估结果/*.json。")
    p.finish(SHOT / "优化前后延迟分位对比.png",
             sources=["优化/评估结果/优化前_汇总.json", "优化/评估结果/优化后_汇总.json"])

    # ---------------------------------------------------------- 图2 五阶段耗时
    stage_keys = ["查询处理与增强", "检索", "上下文组装与提示工程", "LLM 生成", "后处理与响应格式化"]
    p2 = Page("工单13 五阶段耗时对比", "阶段均值来自 分段计时剖析.py 的引擎内部分段统计")
    p2.section("1. 各阶段均值（ms）")
    rows3 = []
    for k in stage_keys:
        b = stat_of(before_s, k)
        a = stat_of(after_s, k)
        d = f"{(a - b) / b * 100:+.1f}%" if (b and a is not None) else "—"
        rows3.append([k, num(b), num(a), d])
    # 兼容键名带序号（如「2_检索」）
    if all(r[1] == "—" and r[2] == "—" for r in rows3):
        rows3 = []
        for k in stage_keys:
            b = a = None
            for name, s in (("b", before_s), ("a", after_s)):
                st = s.get("统计", {}).get("阶段均值", {}) or {}
                for kk, vv in st.items():
                    if k in str(kk) and isinstance(vv, (int, float)):
                        if name == "b":
                            b = float(vv)
                        else:
                            a = float(vv)
            d = f"{(a - b) / b * 100:+.1f}%" if (b and a is not None) else "—"
            rows3.append([k, num(b), num(a), d])
    p2.table_block(["阶段", "优化前", "优化后", "变化"], rows3, widths=[4, 3, 3, 2])
    p2.section("2. 段占比（优化后）")
    share = (after_s.get("统计") or {}).get("阶段占比") or {}
    rows4 = [[str(k), f"{float(v) * 100:.1f}%"] for k, v in share.items() if isinstance(v, (int, float))]
    if rows4:
        p2.table_block(["阶段", "占比"], rows4, widths=[5, 2])
    else:
        p2.note("（汇总里未提供阶段占比字段）")
    p2.finish(SHOT / "五阶段耗时对比.png",
              sources=["优化/评估结果/优化前_汇总.json", "优化/评估结果/优化后_汇总.json"])

    # ---------------------------------------------------------- 图3 逐题明细
    def per_q(rows_in: list[dict]) -> list[list[str]]:
        out = []
        for r in rows_in[:20]:
            qid = r.get("id") or r.get("题号") or ""
            total = r.get("total_ms")
            wall = r.get("wall_ms")
            if total is None:
                total = (r.get("阶段耗时") or {})
                total = sum(float(v) for v in total.values() if isinstance(v, (int, float))) or None
            out.append([str(qid), num(total), num(wall)])
        return out

    p3 = Page("工单13 逐题耗时明细", "每题 total_ms（引擎内部）与 wall_ms（脚本墙钟）")
    p3.section("优化后（前 20 题）")
    p3.table_block(["题号", "total_ms", "wall_ms"], per_q(after_q), widths=[2, 3, 3])
    p3.section("优化前（前 20 题）")
    p3.table_block(["题号", "total_ms", "wall_ms"], per_q(before_q), widths=[2, 3, 3])
    p3.note("说明：total_ms 为引擎内部统计，wall_ms 为脚本墙钟，两者差值即进程内调度与序列化开销。")
    p3.finish(SHOT / "逐题耗时明细.png",
              sources=["优化/评估结果/优化前.jsonl", "优化/评估结果/优化后.jsonl"])

    print("[OK] 3 张截图已生成 ->", SHOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
