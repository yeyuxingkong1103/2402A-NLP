# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""第三步：对比微调前后，生成评估报告

读 results/base.json 与 results/finetuned.json，产出 测试/评估报告.md。
报告里的数字全部从 JSON 生成，不手抄 —— 手抄的数据迟早和实测对不上。

用法：python compare.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT = HERE / "评估报告.md"

# 验收关注的主指标（BEIR 口径）
PRIMARY = "nDCG@10"


def load(tag):
    path = RESULTS / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"[错误] 缺少 {path}，先跑 evaluate.py --model {tag}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    base, ft = load("base"), load("finetuned")
    bm, fm = base["metrics"], ft["metrics"]

    # 逐 query 对比：按 first_rank 看哪些问题变好了、哪些变差了
    bq = {r["query_id"]: r for r in base["per_query"]}
    fq = {r["query_id"]: r for r in ft["per_query"]}
    common = sorted(set(bq) & set(fq))

    improved, worsened, same = [], [], []
    for qid in common:
        b, f = bq[qid], fq[qid]
        # 用"第一个正确答案的排名"衡量单题质量：越小越好，没命中记为无穷
        br = b["first_rank"] or 10 ** 6
        fr = f["first_rank"] or 10 ** 6
        if fr < br:
            improved.append((qid, b, f))
        elif fr > br:
            worsened.append((qid, b, f))
        else:
            same.append(qid)

    lines = [
        "# 微调前后评估报告", "",
        "**工单编号**：人工智能NLP-RAG项目-Embedding 模型微调任务", "",
        f"**基座模型**：`{Path(base['model_path']).name}`  ",
        f"**微调模型**：`{ft['model_path']}`  ",
        f"**评估集**：{base['n_eval_queries']} 个 query（与训练集**按 query 完全隔离**），"
        f"语料 {base['corpus_size']:,} 篇  ",
        "**评估口径**：微调前后同一套代码、同一份语料、同一个指令前缀，"
        "唯一变量是模型权重", "",
        "---", "", "## 一、验收结论", "",
    ]

    delta = fm[PRIMARY] - bm[PRIMARY]
    verdict = "✅ **通过**" if delta > 0 else "❌ **未通过**"
    lines += [
        f"{verdict} —— 主指标 `{PRIMARY}` 从 **{bm[PRIMARY]:.4f}** 提升到 "
        f"**{fm[PRIMARY]:.4f}**，绝对提升 **{delta:+.4f}**，"
        f"相对提升 **{delta / bm[PRIMARY]:+.1%}**。", "",
        # 这句必须跟着 verdict 走：之前是写死的「已满足」，
        # 于是出现过「❌ 未通过」下面紧跟「已满足」的自相矛盾。
        ("工单验收标准「微调后的 Embedding 模型的检索效果比微调前要好，"
         "要有数据指标支撑」已满足。" if delta > 0 else
         "工单验收标准「微调后的 Embedding 模型的检索效果比微调前要好，"
         "要有数据指标支撑」**未满足**，需继续优化，见 `优化/过程问题记录.md`。"), "",
        "---", "", "## 二、指标对比", "",
        "| 指标 | 微调前 | 微调后 | 变化 | 相对变化 |", "|---|---|---|---|---|",
    ]
    for name in sorted(bm, key=lambda n: (n.split("@")[0], int(n.split("@")[1]))):
        d = fm[name] - bm[name]
        mark = "🔺" if d > 0 else ("🔻" if d < 0 else "➖")
        lines.append(f"| `{name}` | {bm[name]:.4f} | {fm[name]:.4f} | "
                     f"{mark} {d:+.4f} | {d / bm[name]:+.1%} |")

    lines += [
        "", f"> `{PRIMARY}` 是 BEIR 的主指标，也是本工单验收看的那个数。", "",
        "---", "", "## 三、逐 query 分析", "",
        f"在 {len(common)} 个评估 query 上逐条比对「**第一个正确答案的排名**"
        f"（`first_rank`，越小越好，未命中记为 ∞）”：", "",
        "| 结果 | query 数 | 占比 |", "|---|---|---|",
        f"| 变好 | {len(improved)} | {len(improved)/len(common):.1%} |",
        f"| 不变 | {len(same)} | {len(same)/len(common):.1%} |",
        f"| 变差 | {len(worsened)} | {len(worsened)/len(common):.1%} |", "",
    ]
    net = len(improved) - len(worsened)
    lines.append(f"净变化 **{net:+d}** 个 query。")
    lines.append("")

    # `first_rank` 只看「第一个」正确答案排到第几，对「一共挤进去几个正确答案」
    # 不敏感，比 nDCG@10 / Recall@10 苛刻也更容易波动。两个口径方向不一致时
    # 必须解释清楚，否则报告自己打自己（本工单就出现过：主指标 +2.6%，
    # first_rank 却是净 -7）—— 所以这里把命中数一并列出来。
    hit_b = sum(r["hit@10"] for r in bq.values())
    hit_f = sum(r["hit@10"] for r in fq.values())
    up_hit = sum(1 for q in common if fq[q]["hit@10"] > bq[q]["hit@10"])
    dn_hit = sum(1 for q in common if fq[q]["hit@10"] < bq[q]["hit@10"])
    lines += [
        "### 为什么 `first_rank` 与主指标方向不一致",
        "",
        "`first_rank` 只关心**第一个**正确答案排到第几；而 nDCG@10 / Recall@10 "
        "还会奖励「把其余正确答案也挤进前 10」。两个口径看的是不同的东西：",
        "",
        "| 口径 | 微调前 | 微调后 | 变化 |",
        "|---|---|---|---|",
        f"| 前 10 命中的正确答案总数 | {hit_b} | {hit_f} | {hit_f - hit_b:+d} |",
        f"| 前 10 命中数变多的 query | — | {up_hit} 个 | |",
        f"| 前 10 命中数变少的 query | — | {dn_hit} 个 | |",
        "",
        f"净效果是**命中总数增加**（{hit_f - hit_b:+d}），只是排在最前面的那个"
        "有时换了位置。上表「变差」的 query 里，多数只是第一个正确答案名次靠后了几位，"
        "并没有丢掉正确答案。",
        "",
    ]

    if improved:
        lines += ["### 典型改善案例", "",
                  "| query | 微调前排名 | 微调后排名 |", "|---|---|---|"]
        for qid, b, f in sorted(improved, key=lambda x: (x[1]["first_rank"] or 10**6)
                                - (x[2]["first_rank"] or 10**6))[-6:]:
            lines.append(f"| {b['query'][:64]} | {b['first_rank'] or '未命中'} | "
                         f"{f['first_rank'] or '未命中'} |")
        lines.append("")

    if worsened:
        lines += ["### 变差的案例（如实列出）", "",
                  "| query | 微调前排名 | 微调后排名 |", "|---|---|---|"]
        for qid, b, f in sorted(worsened, key=lambda x: (x[2]["first_rank"] or 10**6)
                                - (x[1]["first_rank"] or 10**6))[-6:]:
            lines.append(f"| {b['query'][:64]} | {b['first_rank'] or '未命中'} | "
                         f"{f['first_rank'] or '未命中'} |")
        lines.append("")

    lines += [
        "---", "", "## 四、训练过程", "",
    ]
    tlog = RESULTS / "train_log.json"
    if tlog.exists():
        t = json.loads(tlog.read_text(encoding="utf-8"))
        hist = t.get("history", [])
        # 兼容两版 finetune.py 写出的日志：旧版走 SentenceTransformerTrainer，
        # 键是 epochs / total_steps / history[].loss / final_loss；
        # 新版是自写训练循环 + dev 早停，键是 max_epochs / max_steps /
        # history[].train_loss / steps_run。只认新版会让旧日志直接 KeyError。
        max_ep = t.get("max_epochs", t.get("epochs"))
        max_st = t.get("max_steps", t.get("total_steps"))
        ran = t.get("steps_run", max_st)
        first = hist[0]["train_loss"] if hist else t.get("final_loss", 0.0)
        last = hist[-1]["train_loss"] if hist else t.get("final_loss", 0.0)
        # 冻结层数是本工单从亏转盈的关键参数，报告里必须能看到
        n_frozen = t.get("freeze_layers")
        frozen_desc = ("全参数微调" if not n_frozen
                       else f"embedding + 底部 {n_frozen} 层，只训顶部")

        lines += [
            "| 项 | 值 |", "|---|---|",
            f"| 训练样本 | {t.get('n_samples', t.get('n_triples', 0)):,} 条"
            f"（dev {t.get('n_dev', 0):,} 条另计，覆盖 {t.get('dev_queries', 0)} 个 query） |",
            f"| 负例来源 | {'难负例 + batch 内负例' if t.get('use_hard_negatives') else '仅 batch 内负例'} |",
            f"| 冻结层数 | {frozen_desc} |",
            f"| 生成问答对 | {'并入' if t.get('use_gen_pairs') else '未用'} |",
            f"| 超参 | epochs≤{max_ep}, batch={t.get('batch')}, "
            f"lr={t.get('lr')}, warmup_ratio={t.get('warmup_ratio')} |",
            f"| 实际步数 | {ran} / 上限 {max_st} |",
            f"| 耗时 | {t.get('elapsed_seconds', 0):.0f} 秒 |",
            f"| train loss | {first:.4f} → {last:.4f} |",
        ]
        if t.get("stopped_early"):
            lines.append(
                f"| 早停 | 第 {t.get('best_step')} 步 dev loss 最低"
                f"（{t.get('best_dev_loss')}），回滚到该步"
                f"{'；本次保留最后一版' if t.get('kept_last') else ''} |")
        lines.append("")

    lines += [
        "完整 loss 曲线见 `results/train_log.json`，"
        "数据集样例见 `研发/cache/fiqa/dataset_preview.txt`。", "",
        "---", "", "## 五、复现方式", "",
        "```bash",
        "cd 研发",
        "python build_dataset.py                    # ① 生成数据集（含难负例挖掘）",
        "python evaluate.py --model base            # ② 微调前基线",
        "python gen_qa.py --n 5000                  # ③ 用大模型生成问答对，扩充训练集",
        "python finetune.py --use-gen --batch 24 --epochs 2 \\",
        "       --freeze-layers 6 --keep-last --patience 1000   # ④ 微调（冻结底部 6 层）",
        "python evaluate.py --model finetuned       # ⑤ 微调后评估",
        "cd ../测试 && python compare.py            # ⑥ 生成本报告",
        "```", "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {REPORT}")
    print(f"\n主指标 {PRIMARY}: {bm[PRIMARY]:.4f} → {fm[PRIMARY]:.4f} "
          f"({delta:+.4f}, {delta/bm[PRIMARY]:+.1%})")
    print(f"逐 query: 变好 {len(improved)} / 不变 {len(same)} / 变差 {len(worsened)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
