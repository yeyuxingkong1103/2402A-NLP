# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""生成测试截图（全部为**真实产物数字**，禁止手绘假数据）。

截图清单（中文命名，文件名即测试内容）：
  数据集规模与构成.png         ← 研发/data/dataset/数据集统计.json
  训练损失下降曲线.png         ← 研发/data/train_log.jsonl（PIL 自绘折线图）
  微调前后检索指标对比.png     ← 研发/评估/评估结果_{微调前,微调后}.json
  正负例相似度间隔对比.png     ← 同上（相似度指标）+ 柱状图
  逐query检索对照.png          ← 优化/评估结果/逐query检索对照.csv

用法：pwsh -NoProfile -File run_py.ps1 测试/生成截图.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "研发" / "src"))
sys.path.insert(0, r"E:\gao6gongdan\_共享工具")

from 截图渲染 import Page, clip  # noqa: E402
from plot import bar_chart, line_chart  # noqa: E402

OUT = ROOT / "测试" / "测试截图"
EVAL = ROOT / "研发" / "评估"
DATA = ROOT / "研发" / "data"
CMP = ROOT / "优化" / "评估结果"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------- 图1 数据集
def 图_数据集规模与构成() -> None:
    st = load_json(DATA / "dataset" / "数据集统计.json")
    if not st:
        print("[跳过] 数据集统计缺失")
        return
    p = Page("工单11 数据集规模与构成（真实产物）",
             f"来源语料：工单7 8847 分块（9 份 A 股年报）｜随机种子 {st['随机种子']}｜生成耗时 {st['生成耗时秒']}s")
    y = p.section("一、语料与留出")
    p.kv_block([
        ("语料分块总数", st["语料分块总数"]),
        ("参与生成分块数", st["参与生成分块数"]),
        ("语料损坏行数", st["语料损坏行数"]),
        ("锚点总数", st["锚点总数"]),
        ("覆盖指标数", st["指标覆盖数"]),
        ("留出页数 / 总页数", f"{st['页级留出']['留出页数']} / {st['页级留出']['总页数']}"),
        ("留出比例", st["页级留出"]["留出比例配置"]),
        ("留出检索测试集", st["留出检索测试集条数"]),
    ])
    y = p.section("二、各 split 记录数与格式分布（原文点名的三种格式）")
    rows = []
    for split in ("train", "dev", "test"):
        fm = st["各split格式分布"][split]
        rows.append([split, st["各split记录数"][split], fm.get("三元组", 0),
                     fm.get("正例对", 0), fm.get("相似度分数对", 0)])
    p.table_block(["split", "记录总数", "三元组", "正例对", "相似度分数对"], rows,
                  widths=[120, 160, 160, 160, 200])
    y = p.section("三、锚点来源与负例类型")
    p.table_block(["锚点来源", "条数"], [[k, v] for k, v in st["锚点来源分布"].items()],
                  widths=[300, 200])
    p.table_block(["负例类型", "条数"], [[k, v] for k, v in st["负例类型分布"].items()],
                  widths=[300, 200])
    p.note(f"文本长度：锚点中位 {st['文本长度']['anchor']['中位']} 字，"
           f"正例段中位 {st['文本长度']['positive']['中位']} 字（p90 {st['文本长度']['positive']['p90']}）")
    p.finish(OUT / "数据集规模与构成.png",
             sources=["研发/data/dataset/数据集统计.json", "研发/data/dataset/{train,dev,test}.jsonl"])


# --------------------------------------------------------------- 图2 损失曲线
def 图_训练损失下降曲线() -> None:
    log = load_jsonl(DATA / "train_log.jsonl")
    if not log:
        print("[跳过] 训练日志缺失")
        return
    summary = load_json(DATA / "finetuned" / "训练摘要.json") or {}
    steps = [r["step"] for r in log]
    # 降采样画点，避免 100+ 点糊在一起（保留全部趋势）
    keep = list(range(len(log))) if len(log) <= 120 else list(range(0, len(log), max(1, len(log) // 110)))
    line_chart(DATA / "loss_curve.png", title="工单11 Embedding 微调训练损失曲线",
               subtitle=f"{summary.get('损失函数','')}｜{summary.get('实际步数','?')} 步 = "
                        f"{summary.get('实际轮次','?')} 轮｜批大小 {summary.get('批大小','?')}｜CPU {summary.get('线程数','?')} 线程",
               series=[("训练损失(逐步)", [steps[i] for i in keep], [log[i]["loss"] for i in keep]),
                       ("训练损失(滑动均值)", [steps[i] for i in keep], [log[i]["loss_滑动均值"] for i in keep])],
               xlabel="训练步数", ylabel="损失值")
    p = Page("工单11 训练损失下降曲线（真实训练日志）",
             f"底座 {Path(str(summary.get('底座模型',''))).name}｜损失函数 {summary.get('损失函数','')}")
    p.section("一、训练概况（研发/data/finetuned/训练摘要.json）")
    p.kv_block([
        ("实际步数", summary.get("实际步数")),
        ("实际轮次", summary.get("实际轮次")),
        ("训练三元组数", summary.get("训练三元组数")),
        ("批大小 / 学习率", f"{summary.get('批大小')} / {summary.get('学习率')}"),
        ("可训练参数", f"{summary.get('参数统计',{}).get('可训练参数')} / {summary.get('参数统计',{}).get('总参数')}"),
        ("冻结策略", summary.get("冻结策略")),
        ("总耗时(秒)", summary.get("总耗时秒")),
        ("每步平均(秒)", summary.get("每步平均秒")),
        ("初始损失", summary.get("初始损失")),
        ("末段平均损失", summary.get("末段平均损失")),
        ("损失下降", summary.get("损失下降")),
        ("下降比例", summary.get("损失下降比例")),
        ("停止原因", summary.get("停止原因")),
    ])
    p.section("二、损失采样点（每 10% 取一点，全部来自 train_log.jsonl）")
    n = len(log)
    idxs = sorted({0, n - 1} | {int(n * q / 10) for q in range(1, 10)})
    p.table_block(["step", "epoch", "loss", "滑动均值", "学习率", "正例相似度", "负例相似度", "相似度间隔"],
                  [[log[i]["step"], log[i]["epoch"], log[i]["loss"], log[i]["loss_滑动均值"],
                    log[i]["lr"], log[i].get("正例相似度"), log[i].get("负例相似度"),
                    log[i].get("相似度间隔")] for i in idxs if i < n],
                  widths=[80, 80, 130, 130, 130, 150, 150, 150])
    p.note(f"损失曲线图同时落盘：{DATA / 'loss_curve.png'}（PIL 自绘，本机无 matplotlib）")
    p.finish(OUT / "训练损失下降曲线.png",
             sources=["研发/data/train_log.jsonl", "研发/data/finetuned/训练摘要.json"],
             extra_note=f"共 {n} 个训练步的完整日志")


# --------------------------------------------------- 图3 微调前后检索指标对比
def 图_检索指标对比() -> None:
    b = load_json(EVAL / "评估结果_微调前.json")
    a = load_json(EVAL / "评估结果_微调后.json")
    rows_csv = load_csv(CMP / "对比表.csv")
    if not (b and a):
        print("[跳过] 评估结果缺失")
        return
    p = Page("工单11 微调前后检索指标对比（真实评估数字）",
             f"留出测试集 {b.get('留出测试集',{}).get('查询数')} 条查询｜真实问题 17 道｜检索池 2500 块（含全部金标准）")

    def g(d, *ks):
        cur = d
        for k in ks:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    p.section("一、留出测试集（页级留出，训练中从未出现）")
    rows = []
    for label, key in (("Recall@1", "Recall@1"), ("Recall@5", "Recall@5"), ("Recall@10", "Recall@10"),
                       ("MRR", "MRR"), ("nDCG@10", "nDCG@10"), ("平均命中排名", "平均命中排名")):
        rows.append([label, g(b, "留出测试集", key), g(a, "留出测试集", key)])
    p.table_block(["指标", "微调前", "微调后"], rows, widths=[260, 200, 200])
    ci1 = g(b, "留出测试集", "Recall@1 自助法95%CI")
    ci5 = g(a, "留出测试集", "Recall@1 自助法95%CI")
    p.note(f"Recall@1 自助法 95% 置信区间：微调前 {ci1} → 微调后 {ci5}")

    p.section("二、端到端真实检索（工单7 真实索引分块 + 17 道真题，关键词反推金标准）")
    rows = []
    for label, key in (("命中@1", "Recall@1"), ("命中@3", "Recall@3"), ("命中@5", "Recall@5"),
                       ("命中@10", "Recall@10"), ("MRR", "MRR"), ("关键词覆盖@10", "关键词覆盖@10")):
        rows.append([label, g(b, "真实问题", key), g(a, "真实问题", key)])
    p.table_block(["指标", "微调前", "微调后"], rows, widths=[260, 200, 200])

    if rows_csv:
        p.section("三、对比表（优化/评估结果/对比表.csv 原样摘录）")
        p.table_block(["指标组", "指标", "微调前", "微调后", "变化", "变化率", "结论"],
                      [[r["指标组"], r["指标"], r["微调前"], r["微调后"], r["变化"], r["变化率"], r["结论"]]
                       for r in rows_csv], widths=[150, 150, 120, 120, 120, 120, 100])
    p.finish(OUT / "微调前后检索指标对比.png",
             sources=["研发/评估/评估结果_微调前.json", "研发/评估/评估结果_微调后.json",
                      "优化/评估结果/对比表.csv"])


# ------------------------------------------------- 图4 相似度间隔对比（含柱状图）
def 图_相似度间隔对比() -> None:
    b = load_json(EVAL / "评估结果_微调前.json")
    a = load_json(EVAL / "评估结果_微调后.json")
    if not (b and a):
        print("[跳过] 评估结果缺失")
        return
    sb, sa = b.get("相似度指标", {}), a.get("相似度指标", {})
    bar_chart(CMP / "相似度间隔对比.png",
              title="工单11 正例对 / 负例对平均余弦相似度对比",
              subtitle=f"测试集三元组 {sb.get('三元组条数')} 条｜间隔越大越好",
              labels=["正例对平均相似度", "负例对平均相似度", "相似度间隔"],
              groups=[("微调前", [sb.get("正例对平均相似度") or 0, sb.get("负例对平均相似度") or 0,
                                  sb.get("相似度间隔") or 0]),
                      ("微调后", [sa.get("正例对平均相似度") or 0, sa.get("负例对平均相似度") or 0,
                                  sa.get("相似度间隔") or 0])],
              value_fmt="{:.4f}")
    p = Page("工单11 正负例相似度间隔对比（嵌入空间是否被拉开）",
             f"数据：测试集三元组 {sb.get('三元组条数')} 条（锚点-正例-负例，微调前用于训练的样本不参与）")
    p.section("一、相似度指标（研发/评估/评估结果_*.json → 相似度指标）")
    keys = ["正例对平均相似度", "负例对平均相似度", "相似度间隔", "正负例判别准确率",
            "正例相似度标准差", "负例相似度标准差", "正例相似度最小", "负例相似度最大", "三元组条数"]
    p.table_block(["指标", "微调前", "微调后"], [[k, sb.get(k), sa.get(k)] for k in keys],
                  widths=[300, 200, 200])
    p.section("二、负例类型分布")
    p.table_block(["负例类型", "条数"], [[k, v] for k, v in (sb.get("负例类型分布") or {}).items()],
                  widths=[300, 200])
    p.note("判别准确率 = 随机取一对（正例, 负例），正例相似度严格更高的概率（采样 2000×2000，越接近 1 越好）")
    p.finish(OUT / "正负例相似度间隔对比.png",
             sources=["研发/评估/评估结果_微调前.json", "研发/评估/评估结果_微调后.json",
                      "优化/评估结果/相似度间隔对比.png"])


# --------------------------------------------------------------- 图5 逐query
def 图_逐query检索对照() -> None:
    rows = load_csv(CMP / "逐query检索对照.csv")
    if not rows:
        print("[跳过] 逐query对照缺失")
        return
    real = [r for r in rows if str(r["题号"]).startswith("Q")]
    hold = [r for r in rows if str(r["题号"]).startswith("留出")]

    def 排名(r, k):
        v = r.get(k, "")
        return "未命中" if v in ("", "None", None) else v

    def 改善(r):
        v = r.get("排名变化(正=改善)", "")
        if v in ("", "None", None):
            return "—"
        try:
            n = int(float(v))
        except ValueError:
            return "—"
        return f"+{n}" if n > 0 else str(n)

    imp = sum(1 for r in rows if 改善(r).startswith("+"))
    wor = sum(1 for r in rows if 改善(r).startswith("-"))
    p = Page("工单11 逐 query 检索对照（微调前 vs 微调后）",
             f"共 {len(rows)} 条查询：真实问题 {len(real)} 道 + 留出测试集 {len(hold)} 条｜排名变化为正=改善")
    p.section("一、17 道真实问题（工单7 测试问题集，判分关键词反推金标准）")
    p.table_block(["题号", "查询", "金标准块数", "微调前排名", "微调后排名", "排名变化", "前命中@10", "后命中@10"],
                  [[r["题号"], clip(r["查询"], 26), r["金标准分块数"], 排名(r, "微调前首个命中排名"),
                    排名(r, "微调后首个命中排名"), 改善(r), r["微调前命中@10"], r["微调后命中@10"]]
                   for r in real], widths=[70, 330, 100, 120, 120, 100, 100, 100], max_rows=20)
    p.section("二、留出测试集前 24 条（页级留出，训练未见过）")
    p.table_block(["题号", "查询", "微调前排名", "微调后排名", "排名变化", "前命中@10", "后命中@10"],
                  [[r["题号"], clip(r["查询"], 34), 排名(r, "微调前首个命中排名"),
                    排名(r, "微调后首个命中排名"), 改善(r), r["微调前命中@10"], r["微调后命中@10"]]
                   for r in hold[:24]], widths=[110, 400, 130, 130, 110, 110, 110], max_rows=24)
    p.note(f"全量明细见 优化/评估结果/逐query检索对照.csv（共 {len(rows)} 行）："
           f"排名改善 {imp} 条、下降 {wor} 条")
    p.finish(OUT / "逐query检索对照.png",
             sources=["优化/评估结果/逐query检索对照.csv", "研发/评估/评估结果_微调前.json",
                      "研发/评估/评估结果_微调后.json"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CMP.mkdir(parents=True, exist_ok=True)
    图_数据集规模与构成()
    图_训练损失下降曲线()
    图_检索指标对比()
    图_相似度间隔对比()
    图_逐query检索对照()
    print(f"\n截图输出目录：{OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
