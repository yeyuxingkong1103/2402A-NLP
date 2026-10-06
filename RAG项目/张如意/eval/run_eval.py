# -*- coding: utf-8 -*-
"""评估跑器：三层指标 + 分层统计 + 验收判定。

## 三层指标

**① 检索层**（只在正例上算）
  - `recall@k`：期望卡有多少落进 top-k
  - `MRR`：第一张命中卡的倒数排名

**② 兜底层**
  - **误兜底率**：该答的答了「暂无」← **更严重**，验收 < 5%
  - **漏兜底率**：该说「暂无」的却给了结果 ← 会误导农户用药，验收 0%

**③ 生成层**（仅高危，逐字比对）
  - 每个药剂的 `dose` / `pre_harvest_interval` / `max_uses_per_season`
    是否**逐字**出现在答案里
  - **数值列完整性**：三列缺一即该条失败（用户 2026-09-20 补充的要求）
  - ⚠️ **校验对象＝真正被渲染的卡**（2026-09-29 变更 A 的连带项）：
    `compose()` 只渲染 top-1 所在那一组高危卡，口径取自 `gen.top1_high_group`；
    其余卡片的数值**正确地不出现**在答案里，不该再判失败（详见该处注释）

## 验收标准（用户定稿）
  | 分层 | 标准 |
  | --- | --- |
  | 高危问题 | **100%** |
  | 普通问题 | ≥ 90% |
  | 误兜底率 | < 5% |
  | 该放宽却兜底 | **0**（2026-09-29 变更 B 进的门线：没达标就别说过关） |

## 用法
    python eval/run_eval.py                  # 全量
    python eval/run_eval.py --limit 30       # 快速回归
    python eval/run_eval.py --no-gen         # 跳过生成层（省时间）
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# 项目根目录（本文件位于 eval/ 下，根目录是其上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 把检索层、生成层源码目录加入 sys.path，便于 import search / answer
sys.path.insert(0, os.path.join(ROOT, "src", "retrieve"))
sys.path.insert(0, os.path.join(ROOT, "src", "generate"))

# 默认评估数据集（--dataset 未指定时加载 eval/datasets/ 下全部）
DATASET = os.path.join(ROOT, "eval", "datasets", "auto.jsonl")
# 机器可读评估报告输出路径
REPORT = os.path.join(ROOT, "eval", "report.json")
# 检索层评估截断位置：只看 top-5 返回
K = 5


# 参与评估的数据集文件列表（auto=语料反推，collect=真实采集）
DATASETS = ["auto.jsonl", "collect.jsonl"]


def source_group(src):
    """auto 系列是**从语料反推**的（自问自答）；collected 是真实问法。"""
    return "collected" if src == "collected" else "auto"


def load_dataset(limit=0, path=None):
    """默认加载 eval/datasets/ 下全部数据集，并给每条打 source_group。

    分层看 auto / collected 是**这一版评估的重点**：
    auto 测「能不能召回自己」，collected 测真实表述差异，两者差距就是乐观偏差。
    """
    d = os.path.join(ROOT, "eval", "datasets")
    # --dataset 指定路径时只加载该文件；否则加载 DATASETS 列表里的全部数据集
    files = [path] if path else [os.path.join(d, f) for f in DATASETS]
    rows = []
    for f in files:
        if not os.path.exists(f):  # 某个数据集文件不存在时跳过，不报错
            continue
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                r["_file"] = os.path.basename(f)  # 记录来源文件，便于排查
                r["source_group"] = source_group(r.get("source"))  # 打上 auto/collected 分层标签
                rows.append(r)
    # --limit > 0 时只取前 limit 条（快速回归用）
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- 生成层检查
def check_numbers(ans, card):
    """高危字段保真 + 数值列完整性。

    返回 (失败原因列表, 检查项数)。空列表 = 通过。
    """
    fails = []
    # 三个必须逐字保真的数值列：用量 / 安全间隔期 / 每季最多使用次数
    fields = ("dose", "pre_harvest_interval", "max_uses_per_season")
    for i, ch in enumerate(card.get("chemicals", [])):
        for f in fields:
            v = ch.get(f)
            # 数值列非空且没有**逐字**出现在答案文本里 → 记为失真
            if v and str(v) not in ans:
                fails.append(f"chemicals[{i}].{f}={v!r} 未逐字出现在答案里")
    return fails


def completeness(card):
    """数值列完整性：该卡有几种数值列（用于分母）。"""
    n = 0
    for ch in card.get("chemicals", []):
        # 统计每个药剂条目里非空的数值列个数，累加即全卡数值列总数
        n += sum(1 for f in ("dose", "pre_harvest_interval", "max_uses_per_season")
                 if ch.get(f))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=None,
                    help="不指定则加载 eval/datasets/ 下全部（auto + collect）")
    ap.add_argument("--no-gen", action="store_true")
    a = ap.parse_args()

    cases = load_dataset(a.limit, a.dataset)
    if not cases:
        print(f"评估集为空：{a.dataset}\n先跑 python eval/build_dataset.py")
        return 2

    from search import Searcher, search  # 检索层：向量化检索 + 状态判定
    import answer as gen                 # 生成层：组装最终答案（dry_run 不调 LLM）

    print(f"载入 {len(cases)} 条用例…")
    s = Searcher()

    # 分层桶：按 (source_group, 高危/普通) 分组累计各指标
    buckets = defaultdict(lambda: {"n": 0, "recall": 0.0, "mrr": 0.0, "n_pos": 0,
                                   "false_fb_positive": 0, "false_fb_noplan": 0,
                                   "missed_fb": 0, "n_fb": 0,
                                   "num_ok": 0, "num_tot": 0, "num_fail": 0,
                                   "num_norender": 0, "relaxed": 0})
    failures = []  # 失败明细：(分层key, 失败类型, 问句, 原因)
    _tier = lambda k: k

    for i, c in enumerate(cases, 1):
        tier = "高危" if c["is_high_risk"] else "普通"  # 分层：是否问用药
        key = (c["source_group"], tier)  # 分桶 key：数据来源 × 高危层级
        b = buckets[key]
        b["n"] += 1

        # 调检索层：返回 (状态位, 归一化问句, 意图, 检索结果, 判定原因)
        st, q2, intent, res, why = search(s, c["query"], crop=c["crop"], topk=K)
        got = [r["chunk_id"] for r in res]  # top-K 返回的卡号列表
        fellback = st.startswith("fallback")  # 是否硬兜底（fallback_* 开头）

        if c["expect_fallback"]:
            # ---- 负例分支：期望系统说「暂无」----
            b["n_fb"] += 1
            # 「漏兜底」的判定口径（2026-09-20 修正）：
            #   系统对无方案的问句**允许两种正确行为**：
            #     ① 硬兜底（fallback_*）
            #     ② 放宽到正文 + 「暂无具体用药方案」提示（ok_noplan）——这是定稿口径
            #   **只有返回了附录B 的具体用药方案才算漏兜底**——那等于把别的病虫害的
            #   农药方案给了农户，是唯一真正危险的情况。
            # 附录B 卡号形如 GBZxxxx-p20-c06：末段数字结尾 + 含 -p2（附录B 页码段）+ 含 -c
            gave_plan = any(cid.rsplit("-", 1)[0].endswith(tuple("0123456789"))
                            and "-p2" in cid and "-c" in cid for cid in got)
            if not fellback and st != "ok_noplan" and gave_plan:
                b["missed_fb"] += 1  # 漏兜底：该说暂无却给了具体用药方案（最危险）
                failures.append((key, "漏兜底", c["query"],
                                 f"无方案却返回了附录B 用药方案 {got[:2]}"))
            elif st == "ok_noplan":
                b["relaxed"] = b.get("relaxed", 0) + 1  # 走了「放宽+提示」，合格
        elif c.get("expect_noplan"):
            # ---- 期望放宽分支：语料只在正文提到，无用药方案 ----
            # 期望：返回正文内容 + 「暂无用药方案」提示（ok_noplan），
            # 而不是硬兜底——这是 2026-09-20 定稿的口径
            b["n_noplan"] = b.get("n_noplan", 0) + 1
            if st != "ok_noplan":
                b["false_fb_noplan"] += 1  # 该放宽却硬兜底了
                failures.append((key, "该放宽却兜底", c["query"], f"{st}: {why}"))
        else:
            # ---- 正例分支：期望召回指定卡片 ----
            b["n_pos"] += 1
            if fellback:
                b["false_fb_positive"] += 1  # 误兜底：该答的答了「暂无」
                failures.append((key, "误兜底", c["query"], f"{st}: {why}"))
            else:
                exp = set(c["expect_cards"])
                hit = [cid for cid in got if cid in exp]  # top-K 中命中的期望卡
                # recall 贡献 = 命中期望卡数 / 期望卡总数
                b["recall"] += len(hit) / len(exp) if exp else 0
                if hit:
                    # MRR 贡献 = 1 / 第一张命中卡的排名（排名从 1 起）
                    b["mrr"] += 1 / (got.index(hit[0]) + 1)
                else:
                    failures.append((key, "未召回", c["query"], f"期望 {sorted(exp)[:2]}，实际 {got[:2]}"))

            # ---- 生成层：高危字段保真（仅高危正例）----
            if not a.no_gen and c["is_high_risk"] and not fellback:
                # dry_run=True：不调生成 LLM，按模板拼答案，保证评测可重复
                # nq 传归一化问句：症状判据与检索层 coverage_check 同输入（否则口语
                # 病名问法会被误判成症状问法，答案形态跟着变）
                ans = gen.compose(c["query"], c["crop"], res, dry_run=True, nq=q2)
                # 【指标口径变更（2026-09-29，变更 A 的连带项）】**只校验真正被渲染的那些卡**
                # = `compose()` 渲染的那一组高危卡（top-1 所在组），口径与渲染同取自
                # `gen.top1_high_group`（一处定义，不是各写一遍）。
                # 原口径遍历 top-5 里**每张** needs_verification_hint 卡，要求它的
                # dose/间隔期逐字出现在答案里——变更 A 之后，别的病虫害的方案
                # **正确地不再出现**在答案里（农户不该看到别家药方），旧口径等于
                # 用「不该出现的东西没出现」判失败。**这不是放宽**：被渲染的卡
                # 仍然逐列逐字比对，一列不落；只是把分母收窄到「渲染过的」。
                cards = gen.top1_high_group(res)
                if not cards:
                    # top-1 是普通卡 → **一张高危卡都不渲染**，没有可校验的数值列。
                    # 单独计数并在报告里显式打印——不能让「校验对象为空」悄悄混进
                    # 「通过」，那才是真正的放宽（数字要能被读者复核）。
                    b["num_norender"] += 1
                else:
                    fails = []
                    for card in cards:
                        fails += check_numbers(ans, card)  # 数值逐字保真检查
                        b["num_tot"] += completeness(card)  # 累计数值列总数（分母）
                    if fails:
                        b["num_fail"] += 1
                        failures.append((key, "数值失真", c["query"], fails[0]))
                    else:
                        b["num_ok"] += 1

        if i % 25 == 0:  # 进度提示
            print(f"  …{i}/{len(cases)}", flush=True)

    # ---------------------------------------------------------------- 报告
    lines = ["=" * 84, "评估报告", "=" * 84,
             f"数据集: {a.dataset or 'eval/datasets/ 全部'}", f"用例数: {len(cases)}",
             f"分层: " + "、".join(f"{k} {v['n']}" for k, v in sorted(buckets.items())), ""]

    ok_all = True  # 总体验收标志：任一分层未达标即置 False
    for grp in ("collected", "auto"):
        g = {k: v for k, v in buckets.items() if k[0] == grp}  # 该来源下所有分桶
        if not g:
            continue
        # 分组说明：collected 标注已经用户逐条终审（2026-09-27），auto 是自问自答偏乐观
        tagline = ("真实问法（12316 采集，**标注经用户逐条终审 2026-09-27**）"
                   if grp == "collected" else "从语料反推（自问自答，结果偏乐观）")
        lines.append("")
        lines.append(f"████ {grp} —— {tagline}")
    for key, b in sorted(buckets.items()):
        grp, tier = key
        lines.append(f"── [{grp}] {tier}（{b['n']} 条）" + "─" * 40)
        if b["n_pos"]:
            # 检索层指标：recall@K = recall 累计 / 正例数；MRR = mrr 累计 / 正例数
            r = b["recall"] / b["n_pos"]
            m = b["mrr"] / b["n_pos"]
            lines.append(f"   检索  recall@{K} {r:6.1%}   MRR {m:.3f}   （正例 {b['n_pos']} 条）")
            # 验收线：高危 100%，普通 ≥90%（用户定稿）
            # collected 已于 2026-09-27 逐条终审，不再豁免，与 auto 同一把尺子
            target = 1.00 if tier == "高危" else 0.90
            mark = "✅" if r >= target else "❌"
            lines.append(f"   验收  {mark} 召回率 ≥ {target:.0%}")
            if r < target:
                ok_all = False
        if b["n_fb"]:
            # 漏兜底率：漏兜底数 / 负例总数，验收要求 0%
            miss = b["missed_fb"] / b["n_fb"]
            lines.append(f"   漏兜底 {b['missed_fb']}/{b['n_fb']} = {miss:.1%}"
                         f"  {'✅' if miss == 0 else '❌ 应 0%'}"
                         f"（其中 {b.get('relaxed',0)} 条走「放宽+提示」，合格）")
            if miss > 0:
                ok_all = False
        if b["n_pos"]:
            # 误兜底率：**只读正例分支的计数器**（false_fb_positive）。
            # 原实现一个 false_fb 被两处复用，noplan 分支一旦触发会污染这一行。
            ff = b["false_fb_positive"] / b["n_pos"]
            lines.append(f"   误兜底 {b['false_fb_positive']}/{b['n_pos']} = {ff:.1%}"
                         f"  {'✅' if ff < 0.05 else '❌ 应 <5%'}")
            if ff >= 0.05:
                ok_all = False
        if b.get("n_noplan"):
            # 【2026-09-29 用户裁决·变更 B】这一行**进验收门线**。
            # 原先它只打印一行、不碰 ok_all：collected #18 明明挂在失败明细里，
            # 总评却仍是「✅ 全部达标」——门线的意义就是「没达标就别说过关」。
            ok_np = b["false_fb_noplan"] == 0
            lines.append(f"   该放宽却兜底 {b['false_fb_noplan']}/{b['n_noplan']}"
                         f"（应返回正文+提示，不是硬兜底）"
                         f"  {'✅' if ok_np else '❌ 应 0'}")
            if not ok_np:
                ok_all = False
        if b["num_tot"] or b.get("num_norender"):
            # 生成层数值保真：通过的卡数 / 总卡数（num_tot 是校验过的数值列数）
            # num_norender：top-1 是普通卡、一张高危卡都没渲染的条数——分母收窄到
            # 「渲染过的」，这一项必须显式打印，读者才能核出有多少条退出了校验。
            sk = (f"；另 {b['num_norender']} 条 top-1 是普通卡、未渲染高危卡（不计入）"
                  if b.get("num_norender") else "")
            lines.append(f"   数值保真 {b['num_ok']}/{b['num_ok']+b['num_fail']} 条卡通过"
                         f"（共校验 {b['num_tot']} 个数值列）"
                         f"  {'✅' if b['num_fail'] == 0 else '❌'}{sk}")
            if b["num_fail"]:
                ok_all = False
        lines.append("")

    if failures:
        # 失败明细最多打印前 20 条，完整明细存 report.json
        lines.append(f"── 失败明细（共 {len(failures)} 条，列前 20）" + "─" * 30)
        for tier, kind, q, why in failures[:20]:
            lines.append(f"   [{tier}/{kind}] {q}")
            lines.append(f"        {why}")
        lines.append("")

    lines.append("=" * 84)
    lines.append("总体：" + ("✅ 全部达标" if ok_all else "❌ 有未达标项"))
    lines.append("")
    lines.append("⚠️ 本评估集是从语料反推的（自问自答），结果偏乐观。")
    lines.append("   真实农户问法请以 source=collected 追加后重跑。")

    txt = "\n".join(lines)
    print("\n" + txt)
    # 写出机器可读的 JSON 报告（分桶指标 + 完整失败明细 + 总体结论）
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"buckets": {f"{k[0]}|{k[1]}": dict(v) for k, v in buckets.items()},
                   "failures": failures, "ok": ok_all}, f, ensure_ascii=False, indent=1)
    print(f"\n→ 机器可读报告: {REPORT}")
    # 退出码：全部达标返回 0，否则返回 1（便于 CI 判定）
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
