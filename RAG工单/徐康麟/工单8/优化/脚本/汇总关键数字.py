# -*- coding: utf-8 -*-
"""工单8 临时汇总脚本：打印最终评估关键数字（供测试报告/截图引用核对）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "测试" / "测试数据"
GR = ROOT / "研发" / "data" / "graph"


def main() -> int:
    s = json.loads((TD / "GraphRAG评估汇总.json").read_text(encoding="utf-8"))
    c = json.loads((TD / "与工单7对比.json").read_text(encoding="utf-8"))
    g = json.loads((GR / "图谱构建统计.json").read_text(encoding="utf-8"))
    v = json.loads((GR / "可视化统计.json").read_text(encoding="utf-8"))
    acc, rec, lat, sla, ab = s["准确率"], s["检索召回"], s["响应时间ms"], s["SLA"], s["图谱扩展增量"]
    print("=== 图谱 ===")
    print(f"扫描 {g['chunks_scanned']} 块 | 实体 {g['entities']} | 关系 {g['relations']} | "
          f"提及边 {g['mentions']} | 耗时 {g['elapsed_ms']:.0f} ms")
    print(f"实体分布 {g['entities_by_type']}")
    print(f"关系分布 {g['relations_by_type']}")
    print(f"剪枝：高频 {g['prune']['pruned']} / 低频 {g['prune_low_support']['pruned']}")
    print(f"可视化：{v['layout']['nodes']} 节点 / {v['layout']['edges']} 边 / "
          f"布局 {v['layout']['layout_ms']} ms / PNG {v['png']['bytes'] // 1024} KB")
    print("\n=== 评估（工单8 Graph RAG，17 题）===")
    print(f"严格准确率 {acc['严格准确率']}（{acc['严格判对题数']}/{s['有效题数']}）判对 {acc['判对题号']}")
    print(f"部分正确 {acc['部分正确题号']} | 平均覆盖率 {acc['平均关键词覆盖率']}")
    print(f"recall@5 {rec['平均recall@5']} | recall@10 {rec['平均recall@10']}")
    print(f"响应 平均 {lat['平均']} ms（检索 {lat['检索_平均']} / 图谱 {lat['图谱扩展_平均']} / "
          f"生成 {lat['生成_平均']}）P90 {lat['P90']} 最大 {lat['最大']}")
    print(f"SLA：达标 {sla['达标题数']}/{s['有效题数']} = {sla['达标率']}，超时 {sla['超时题号']}")
    print(f"页命中率 {s['页命中']['命中率']} | 拒答 {s['拒答']['拒答题数']} | 跨语料泄漏 {s['跨语料泄漏']['题数']}")
    print(f"图谱扩展增量：{ab['基础检索_recall@5']} → {ab['图谱增强_recall@5']}（{ab['recall@5_提升']}）"
          f"，图谱独有块进 Top5 题数 {ab['图谱独有块进入Top5的题数']}")
    print("\n=== 与工单7 对比 ===")
    print(f"严格准确率 {c['准确率']['工单7_严格准确率']} → {c['准确率']['工单8_严格准确率']} "
          f"（{c['准确率']['严格准确率_提升']:+}）")
    print(f"recall@5 {c['检索召回']['工单7_平均recall@5']} → {c['检索召回']['工单8_平均recall@5']} "
          f"（{c['检索召回']['recall@5_提升']:+}）")
    print(f"recall@10 {c['检索召回']['工单7_平均recall@10']} → {c['检索召回']['工单8_平均recall@10']} "
          f"（{c['检索召回']['recall@10_提升']:+}）")
    print(f"响应 {c['响应时间']['工单7_平均ms']} ms → {c['响应时间']['工单8_平均ms']} ms "
          f"（{c['响应时间']['变化ms']:+} ms）")
    print(f"平均覆盖率 {c['准确率']['工单7_平均关键词覆盖率']} → {c['准确率']['工单8_平均关键词覆盖率']} "
          f"（{c['准确率']['平均覆盖率_提升']:+}）")
    print(f"宽松判对 {c['准确率']['工单7_宽松判对题数']} → {c['准确率']['工单8_宽松判对题数']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
