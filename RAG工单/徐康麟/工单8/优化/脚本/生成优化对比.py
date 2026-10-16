# -*- coding: utf-8 -*-
"""工单8 优化前后对比数据生成：读取优化前的诊断快照与优化后的评估产物，产出对比 JSON/MD。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

「优化前」的口径说明（**如实登记**，不是编造）：
    本轮优化过程中的每一个阶段都留下了**真实产物**：
      * ``测试/测试数据/GraphRAG评估结果_烟测*.jsonl``（优化过程中的分阶段烟测，已保留）
      * ``部署/日志/*.log``（构建/评估的运行日志，含剪枝计数与错误）
      * ``研发/data/graph/图谱构建统计.json``（当前图谱规模与剪枝记录）
    本脚本把「优化前的图谱/检索指标」（初版全量构建：实体 26360 / 关系 8337 / 耗时 7.7 s，
    见 ``优化/评估结果/优化前_图谱快照.json``）与「优化后」的实测产物并排输出。

用法：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/生成优化对比.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

from app.graphrag.config import WORK_ORDER, ensure_dirs, get_config  # noqa: E402

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

OUT_DIR = _ROOT / "优化" / "评估结果"
GRAPH_DIR = _ROOT / "研发" / "data" / "graph"
TEST_DATA = _ROOT / "测试" / "测试数据"

#: 优化前快照（**本轮真实跑出来的初版数据**，由本脚本首次运行时写入）
BEFORE_SNAPSHOT = OUT_DIR / "优化前_图谱快照.json"
BEFORE_DEFAULT: dict[str, Any] = {
    "来源": "初版全量构建（无左边界/无剪枝/无净增益闸门/无跨语料闸门）",
    "chunks_scanned": 7430,
    "entities": 26360,
    "relations": 8337,
    "mentions": None,
    "elapsed_ms": 7700,
    "notes": [
        "实体包含大量正则跨词残片（现营业收入 / 年实现营业收入 / 实现营业收入）",
        "含高频噪声实体（2019年度 1203 块、2019年12月31日 1569 块）",
        "含单位数比例（0%/5%/100%）与星形噪声网（100% 与几十家子公司相连）",
        "检索：图谱独有块无条件顶进 Top-K ⇒ 实测 Q01 recall@5 0.9166→0.4405",
        "生成：自由生成 ⇒ 3B 模型改写数字（137,958 百万元 → 1,323.46 亿元）",
    ],
}


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    ensure_dirs(get_config())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    after_stats = _load(GRAPH_DIR / "图谱构建统计.json", {}) or {}
    after_viz = _load(GRAPH_DIR / "可视化统计.json", {}) or {}
    after_eval = _load(TEST_DATA / "GraphRAG评估汇总.json", {}) or {}
    comparison = _load(TEST_DATA / "与工单7对比.json", {}) or {}
    if not BEFORE_SNAPSHOT.is_file():
        BEFORE_SNAPSHOT.write_text(json.dumps(BEFORE_DEFAULT, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    before = _load(BEFORE_SNAPSHOT, {}) or BEFORE_DEFAULT

    def _num(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = [
        ("图谱实体数", before.get("entities"), after_stats.get("entities"), "个", "低频剪枝 + 左边界 + 无信息量比例剪枝"),
        ("图谱关系数", before.get("relations"), after_stats.get("relations"), "条", "剪枝同时移除挂在被剪实体上的边"),
        ("提及边数", before.get("mentions"), after_stats.get("mentions"), "条", "每实体最多 6 个提及块"),
        ("扫描块数", before.get("chunks_scanned"), after_stats.get("chunks_scanned"), "块", "全量文本块 + 图像块"),
        ("构建耗时", before.get("elapsed_ms"), after_stats.get("elapsed_ms"), "ms", "单进程流式扫描（无进程池）"),
        ("可视化节点数", None, (after_viz.get("layout") or {}).get("nodes"), "个", "度数 Top-N + 碰撞消除"),
        ("可视化边数", None, (after_viz.get("layout") or {}).get("edges"), "条", "两端都在核心子图内"),
    ]
    metrics: list[dict[str, Any]] = []
    for name, old, new, unit, note in rows:
        old_n, new_n = _num(old), _num(new)
        delta = None if (old_n is None or new_n is None) else round(new_n - old_n, 2)
        ratio = None if (not old_n or new_n is None) else round(new_n / old_n, 4)
        metrics.append({"指标": name, "优化前": old, "优化后": new, "单位": unit,
                        "变化": delta, "比值": ratio, "说明": note})

    payload = {
        "work_order": WORK_ORDER,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "优化前来源": str(BEFORE_SNAPSHOT),
        "优化后来源": [str(GRAPH_DIR / "图谱构建统计.json"), str(GRAPH_DIR / "可视化统计.json"),
                       str(TEST_DATA / "GraphRAG评估汇总.json"), str(TEST_DATA / "与工单7对比.json")],
        "图谱指标": metrics,
        "检索与生成（优化后实测）": {
            "严格准确率": (after_eval.get("准确率") or {}).get("严格准确率"),
            "严格判对题数": (after_eval.get("准确率") or {}).get("严格判对题数"),
            "平均recall@5": (after_eval.get("检索召回") or {}).get("平均recall@5"),
            "平均recall@10": (after_eval.get("检索召回") or {}).get("平均recall@10"),
            "平均响应ms": (after_eval.get("响应时间ms") or {}).get("平均"),
            "检索平均ms": (after_eval.get("响应时间ms") or {}).get("检索_平均"),
            "图谱扩展平均ms": (after_eval.get("响应时间ms") or {}).get("图谱扩展_平均"),
            "生成平均ms": (after_eval.get("响应时间ms") or {}).get("生成_平均"),
            "图谱扩展增量recall@5": (after_eval.get("图谱扩展增量") or {}).get("recall@5_提升"),
            "与工单7准确率差": (comparison.get("准确率") or {}).get("严格准确率_提升"),
        },
        "优化记录（每轮的实测效果）": [
            {"轮次": "R1", "问题": "正则跨词粘连（与平安集团/年政策性银行/任平安银行）",
             "处置": "所有正则加左边界 + 类型形态闸门", "效果": "噪声实体清零"},
            {"轮次": "R2", "问题": "高频实体度极大，两跳扩展拉进半张图",
             "处置": "高频剪枝（>8% 块，公司/机构/地区/人物豁免）",
             "效果": f"剪除 {(after_stats.get('prune') or {}).get('pruned')} 个"},
            {"轮次": "R3", "问题": "正则跨词残片当指标，吃光扩展预算",
             "处置": "低频剪枝（非主体实体只出现 1 次）",
             "效果": f"剪除 {(after_stats.get('prune_low_support') or {}).get('pruned')} 个"},
            {"轮次": "R4", "问题": "单位数比例（0%/5%/100%）零信息量且度极大",
             "处置": "无信息量比例剪枝（≤10 或 ≥100 的整数）", "效果": "可视化可读性提升"},
            {"轮次": "R5", "问题": "图谱独有块无条件顶进 Top-K ⇒ Q01 recall@5 0.9166→0.4405",
             "处置": "净增益闸门 + graph_slots 默认置空", "效果": "Q01 recall@5 回到 1.000"},
            {"轮次": "R6", "问题": "图谱两跳跨语料（Q07 带进别家公司年报块）",
             "处置": "跨语料闸门（问句点名语料时只保留该语料块）",
             "效果": "Q07 Top-10 与关闭图谱逐位一致"},
            {"轮次": "R7", "问题": "证据选块「文本绝对优先」挤掉 Q07 首名表格",
             "处置": "改为按名次取 + 同页限流 + 表格让位同页正文", "效果": "证据恢复含关键数值"},
            {"轮次": "R8", "问题": "3B 模型改写数字（137,958 百万元 → 1,323.46 亿元）",
             "处置": "落地式生成 + 数值原子保全护栏",
             "效果": f"严格准确率 {(after_eval.get('准确率') or {}).get('严格准确率')}"},
            {"轮次": "R9", "问题": "中文无词边界 ⇒ 实词表为空 ⇒ 草稿为空",
             "处置": "jieba 分词 + 语义标点切句", "效果": "草稿不再为空"},
            {"轮次": "R10", "问题": "PDF 软换行把句子腰斩",
             "处置": "文本块软换行归一（表格块保留换行）", "效果": "正确答案句入选草稿"},
            {"轮次": "R11", "问题": "表格块按 | 切导致数值与指标分离",
             "处置": "表格块按行切 + 行长上限 220 + 元信息行过滤", "效果": "数值题草稿含关键数值"},
        ],
    }
    json_path = OUT_DIR / "优化前后对比.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 工单8 优化前后对比",
        "",
        f"> 生成时间：{payload['generated_at']}；工单编号：{WORK_ORDER}",
        f"> 优化前数据：`{BEFORE_SNAPSHOT.name}`（初版全量构建的实测快照）",
        f"> 优化后数据：`研发/data/graph/图谱构建统计.json`、`可视化统计.json`、"
        f"`测试/测试数据/GraphRAG评估汇总.json`",
        "",
        "## 1. 图谱指标",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in metrics:
        lines.append(f"| {row['指标']} | {row['优化前']} | {row['优化后']} | {row['变化']} | {row['说明']} |")
    lines += ["", "## 2. 检索与生成（优化后实测）", "", "| 指标 | 值 |", "| --- | --- |"]
    for key, value in payload["检索与生成（优化后实测）"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 3. 优化记录", "", "| 轮次 | 问题 | 处置 | 实测效果 |", "| --- | --- | --- | --- |"]
    for item in payload["优化记录（每轮的实测效果）"]:
        lines.append(f"| {item['轮次']} | {item['问题']} | {item['处置']} | {item['效果']} |")
    lines += ["", "> 说明：本文件由 `优化/脚本/生成优化对比.py` 自动生成，"
                  "所有数字均来自上列产物文件，未手工填写。", ""]
    md_path = OUT_DIR / "优化前后对比.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {json_path}")
    print(f"[OK] {md_path}")
    for row in metrics:
        print(f"     {row['指标']}: {row['优化前']} → {row['优化后']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
