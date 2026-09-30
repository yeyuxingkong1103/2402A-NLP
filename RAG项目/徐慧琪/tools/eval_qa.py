"""评估集打分：把「感觉还行」变成可复现的数字。

存在的理由：技术方案 12.2 定的指标里，Recall@20/@50、精排 top3、条款号
直问准确率都能自动算——前提是有一套冻结的题与一个固定的口径。没有它，
调参与回归都只能凭感觉。

**口径自曝**：本集的题面由 AI 照着法条起草（逐条标记 source=ai_draft），
属「看着答案出题」，只能证明链路通、只能当回归基线，**不得作为召回率依据**。
报告首段固定写出这句话，免得后人把这批数字当质量结论引用。

用法：cd D:/xinzg6/fl && python tools/eval_qa.py [--limit N]
      cd D:/xinzg6/fl && python tools/eval_qa.py --retrieval-only   # 判据 8 的检索段计时
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

# 脚本在 tools/ 下而 app 包在 backend/ 下，故把 backend 挂进搜寻路径。
# app.* 的导入一律推迟到 main()：本模块的指标函数是纯函数，单测导入它时
# 不该被迫拉起 torch / langchain / MySQL
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# 检索段延迟记录的渲染与历史解析已拆到 tools/latency_report.py：
# 本文件贴着「单文件 ≤ 300 非空行」的硬闸门，再塞代码就只能靠压注释了
from latency_report import (  # noqa: E402
    latency_history, render_latency_report, retrieval_seconds)

EVAL_PATH = ROOT / "data" / "eval" / "qa_eval_v0.jsonl"
REPORT_PATH = ROOT / "data" / "eval" / "eval_report.md"
# 检索段延迟（判据 8）单独落一份：主报告的延迟含生成，对"检索 P95 < 2s"只能作上界。
# 落成文件而不是只打印，是为了让复核者能重跑复现，而不是只能信任报告里的转述
LATENCY_PATH = ROOT / "data" / "eval" / "retrieval_latency.md"

# Recall@k 的 k 取自技术方案 12.2。定义在此而不是散在 compute_metrics 里：
# 报数口径只能有一处来源，改 k 时不必满文件找
RECALL_KS = (20, 50)


def load_items(path: pathlib.Path = EVAL_PATH) -> list[dict]:
    """读冻结的评估集。每行一个 JSON，不允许多行对象。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eval_md5(path: pathlib.Path = EVAL_PATH) -> str:
    """评估集指纹。报告里记它，才能证明这批数字是哪一版题算出来的。"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _article_no(block) -> int:
    """单个块 → 条号。两种形状都认：块 dict，或已经是条号的裸 int。

    行的载荷在两条路上形状不同——实跑由 _evaluate_one 塞进的是完整块（dict），
    而最小可测的行只给条号；指标函数不该逼调用方先做一次搬运。
    只认 article_no 这一个键：父块与精确块的键集不对称（精确块 14 键、
    向量块 12 键），换别的键会在其中一条路上对不上号（同 merge_blocks 的教训）。
    """
    return int(block["article_no"]) if isinstance(block, dict) else int(block)


def _article_nos(blocks) -> list[int]:
    """把块序列压成条号序列，保持原顺序——recall/topk 都靠位置说话。"""
    return [_article_no(block) for block in blocks]


def recall_at(blocks, gold: list[int], k: int) -> bool:
    """gold 是否**全部**落在前 k 个父块里。

    「全部」是刻意的：多标准条款的题只召回一半时，答案必然缺一半，
    算命中会让召回率虚高。
    """
    got = set(_article_nos(blocks)[:k])
    return all(no in got for no in gold)


def topk_hit(blocks, gold: list[int], k: int) -> bool:
    """精排 top k 是否命中。与 recall_at 分开写：这里只问「最靠前的几条对不对」，
    是 AC-4 的口径；recall 问的是「召回里有没有」。

    与 recall_at 的两点差别都在名字里：只认前缀（前 k 条以外不算），
    且「任一命中」即可——精排把正确的条推到最前就算干活了，
    不必像召回的「全部」那样要求一网打尽
    """
    got = set(_article_nos(blocks)[:k])
    return any(no in got for no in gold)


def _rate(numerator: int, denominator: int) -> float:
    """分母为 0 时返回 0.0：空跑或空子集的诚实读数是「无从谈起」，不是崩溃。"""
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """最近秩分位。样本少时不插值，取真实观测到的那个值更老实。

    100 题时 p50 落在第 50 个观测值（下标 49）、p95 落在第 95 个，
    由单测钉死；本函数的秩定义若改动（换成插值或 ceil），那条测试必须先红。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def compute_metrics(rows: list[dict]) -> dict:
    """由每题的实测结果算总指标。rows 为空时返回 0 值而不是抛错。

    两条除零通路（空 rows、空「条款号直问」子集）都走 _rate 的 0.0 出口：
    一次跑空或全语义题的评测若在这里抛异常，整轮产物就没了。

    直问率只统计 `条款号直问` 子集：混入语义题会把 AC-2 的 100% 门槛稀释成
    一个看着还行的数字，而错误恰恰出在直问这一路。
    """
    n = len(rows)
    direct = [r for r in rows if r["category"] == "条款号直问"]
    latencies = [r["latency"] for r in rows]
    metrics = {
        "n": n,
        "top3_rate": _rate(sum(1 for r in rows if topk_hit(r["top3"], r["gold"], 3)), n),
        "direct_n": len(direct),
        "direct_rate": _rate(sum(1 for r in direct if r["exact_top1"]), len(direct)),
        "cite_rate": _rate(sum(1 for r in rows if r["cite_ok"]), n),
        "abstain_rate": _rate(sum(1 for r in rows if r["status"] == "abstain"), n),
        "error_rate": _rate(sum(1 for r in rows if r["status"] == "error"), n),
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
    }
    # 两个 k 共用同一份「粗排+回填」结果（_evaluate_one 把它记在 recalled_20/50
    # 两处），截断由 recall_at 的 [:k] 做；k 值只在 RECALL_KS 一处定义，
    # 免得出现「表头写 20、代码算 30」的口径漂移
    for k in RECALL_KS:
        metrics[f"recall{k}"] = _rate(
            sum(1 for r in rows if recall_at(r[f"recalled_{k}"], r["gold"], k)), n)
    return metrics


CAVEAT = ("**本评估集的题面由 AI 照着法条起草（逐条标记 source=ai_draft），"
          "属「看着答案出题」，只能作为链路回归基线，不得作为召回率或质量的依据。**"
          "需求文档 6.3 要求的律师真实问题 + 执业律师抽检仍是待办。")


def render_report(metrics: dict, meta: dict) -> str:
    """渲染 markdown 报告。首段固定自曝评估集局限。"""
    lines = ["# ③a 在线问答链路 评测报告", "",
             CAVEAT, ""]
    # 小样本跑的警告紧贴自曝声明：正典文件名配上满屏 100.0%，读者唯一能
    # 识破的线索就是样本量，藏在表格里等于没写（meta 无此键则整行不出现）
    if meta.get("sample_warning"):
        lines += [meta["sample_warning"], ""]
    lines += [f"- 评估集：`{EVAL_PATH.name}`，MD5 `{meta['md5']}`，共 {meta['n']} 题",
              f"- 生成后端：{meta['endpoint']}", ""]
    lines += ["## 指标", "",
              "| 指标 | 值 | 口径 |", "|---|---|---|",
              f"| Recall@20 | {metrics['recall20']:.1%} | gold 全部落在粗排+回填前 20 |",
              f"| Recall@50 | {metrics['recall50']:.1%} | 同上，前 50 |",
              f"| 精排 top3 命中率 | {metrics['top3_rate']:.1%} | AC-4 判据，门槛 85% |",
              f"| 条款号直问准确率 | {metrics['direct_rate']:.1%}（{metrics['direct_n']} 题）"
              f" | AC-2 判据，门槛 100% |",
              # 用户 2026-09-28 裁决：此行只能叫「四关通过率」，口径说明须写进产物（否则重跑即被洗掉）
              f"| 四关通过率 | {metrics['cite_rate']:.1%} | 四关全过的题占比；"
              f"**不是** 12.2 的「输出引用与标准条款一致的比例」（见下） |",
              f"| 降级率 | {metrics['abstain_rate']:.1%} | 校验/解析两次未过 → 未找到相关依据 |",
              f"| 错误率 | {metrics['error_rate']:.1%} | LLM 或检索不可用 |",
              f"| 端到端延迟 P50 / P95 | {metrics['p50']:.2f}s / {metrics['p95']:.2f}s"
              f" | **含生成**，故对 AC-12（只管检索段，门槛 P95 < 2s）只能作上界参考 |", ""]
    lines += ["## 怎么读这张表", "",
              "- **幻觉率 0 是「拦截后」口径**：达标靠四关校验，不靠模型。"
              "所以真正要盯的是**降级率**——它高了说明模型产出大半被拦下，指标好看而系统不可用。",
              "- 条款号直问准确率只统计 `条款号直问` 类题目；混入语义题会把 AC-2 的门槛稀释。",
              "- **「四关通过率」不是「引用准确率」**：它只证明引用**没编造、没错引原文**，"
              "**不证明引对了条**——「引错法条但四关全过」不算它的失败（实例见 Task 13 冒烟第 3 题）。"
              "技术方案 12.2 定义的「输出引用与标准条款一致的比例」本期**未测**："
              "那需要 gold 对齐指标（拿引用与评估集 `gold_articles` 比对），属后续迭代，不在 ③a 范围。", ""]
    return "\n".join(lines)


def _evaluate_one(answerer, item: dict, log) -> dict:
    """跑一道题，采集四类证据：召回、精排、直问置顶、生成与校验结果。

    不再单独跑一次检索：Answerer 把 RetrievalResult 放在 QAResult 里带出来了
    （见 answer.py 的 QAResult.retrieval），重跑一次要多烧一次精排与编码。

    两个块来源必须分清（Task 5 曾对调过）：recalled_blocks 是**父块**，
    gold 标的正是条号，父块才是一条一条的；chunks 是子块，同一条会重复出现
    且未经回填，拿它算召回会虚高。
    """
    start = time.perf_counter()
    qa = answerer.answer(item["query"], item["side"])
    latency = time.perf_counter() - start
    retrieved = qa.retrieval
    blocks = retrieved.blocks if retrieved else []
    recalled = retrieved.recalled_blocks if retrieved else []
    # 引用准确率看终态：必须 ok、有引用、且没有任何一关的失败记录。
    # 首轮失败又被重生成救回来的（attempts=2）仍算过——四关是最终出口
    cite_ok = qa.status == "ok" and bool(qa.citations) and not qa.failures
    exact_top1 = bool(blocks) and blocks[0]["article_no"] in item["gold_articles"]
    log(f"{item['id']} {item['query'][:20]}… {qa.status} {latency:.2f}s")
    # top3 字段存的是精排全量（RERANK_OUTPUT_TOPK=5 条）：截到 3 由 topk_hit 做，
    # 将来 AC-4 若要改看 top5 不必重跑评测
    return {"id": item["id"], "category": item["category"], "gold": item["gold_articles"],
            "recalled_20": recalled, "recalled_50": recalled,
            "top3": blocks, "status": qa.status, "cite_ok": cite_ok,
            "exact_top1": exact_top1, "failures": qa.failures, "latency": latency}


def main() -> int:
    parser = argparse.ArgumentParser(description="在冻结评估集上跑一遍问答链路并出报告")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只量检索段延迟（判据 8），不跑生成")
    args = parser.parse_args()
    items = load_items()
    if args.limit:
        # 截题不改 MD5：报告里的指纹仍指向整份冻结集，n 只说明这轮跑了几题
        items = items[:args.limit]
    # 延迟导入：只有真跑才需要连接、模型与 LLM；纯函数单测导入本模块时不碰这些。
    # 装配整段走共享工厂（core/factory.py）：连接参数、模型目录与「模型文件真在」
    # 的自检从此只有 core/config.py 一处口径 —— 手写 connect()/load_model() 时
    # FL_* 覆盖与 validate() 对评测**全部失效**，会出现「换了模型目录、评测却用
    # 另一套模型出数」而报告里看不出来的漂移。with_extras=False：评测跑的是法条侧
    # 问答，附加区块要额外调 DeepSeek（花钱）且不属于本报告的任何一条指标
    from app.core.factory import build_services
    services = build_services(with_extras=False)

    if args.retrieval_only:
        # 本模式不跑生成（模型只用于检索）：判据 8 量的是检索段，掺进生成会污染读数
        from app.retrieval.pipeline import retrieve

        # 连接与模型由工厂各建一次：每题重载 bge-m3 会把加载耗时算进计时，
        # 量出来的就不是延迟了
        measure = lambda question: retrieve(question, conn=services.conn,
                                            client=services.client,
                                            encoder=services.encoder,
                                            reranker=services.reranker)
        seconds = retrieval_seconds(items, measure)
        # 日期与运行条件必须进产物：没有它们，一份延迟记录只是串无法归因的数字；
        # 宿主上的并发负载采集不到，如实写明"未采集"而不是替读者猜原因
        try:
            import torch
            conditions = (f"GPU {torch.cuda.get_device_name(0)}；编码/精排按 app 默认走 cuda；宿主并发负载未采集")
        except Exception:
            conditions = "CUDA 不可用/未识别；宿主上的并发负载未采集"
        meta = {"name": EVAL_PATH.name, "md5": eval_md5(), "n": len(items),
                "started": time.strftime("%Y-%m-%d %H:%M"), "conditions": conditions,
                "command": "python tools/eval_qa.py --retrieval-only"}
        if args.limit:
            # 截题跑同样要自曝：否则一份 n=5 的记录会顶着判据 8 的名义被引用。
            # 备注一栏也要标：历史行默认写"自动记录"，部分运行会混进累计与不达标计数
            meta["warning"] = f"⚠️ **本次只量了前 {len(items)} 题（`--limit`），不是判据 8 的全量读数。**"
            meta["note"] = f"**部分运行**：只量了前 {len(items)} 题（`--limit`）"
        # 历史行从旧产物读回再原样回写：跨次重跑累积，离群跑不会被干净跑洗掉
        old = LATENCY_PATH.read_text(encoding="utf-8") if LATENCY_PATH.exists() else ""
        LATENCY_PATH.write_text(render_latency_report(seconds, meta, latency_history(old)),
                                encoding="utf-8")
        print(f"检索段延迟：P50 {_percentile(seconds, 50):.3f}s / "
              f"P95 {_percentile(seconds, 95):.3f}s → {LATENCY_PATH}")
        return 0

    # 模型只加载一次（工厂里已加载）：每题重载 bge-m3 会让 100 题跑上一小时。
    # Answerer 由工厂接上**同一批**重资源（conn / client / encoder / reranker），
    # 不是为评测另装一套
    answerer = services.answerer
    rows = [_evaluate_one(answerer, item, print) for item in items]
    metrics = compute_metrics(rows)
    meta = {"md5": eval_md5(), "n": len(items),
            "endpoint": "律师侧 ollama/qwen2.5:3b；公众侧 deepseek/deepseek-chat"}
    if args.limit:
        # 截题跑必须自曝样本量：报告落在正典路径上，而"生成后端"那行是静态的，
        # 说明不了这轮实际跑了谁——不写清楚就会被当成 100 题结论引用
        sides = "、".join(sorted({item["side"] for item in items}))
        meta["sample_warning"] = (
            f"⚠️ **本次为小样本调试跑（n={len(items)}，`--limit`），"
            f"实际覆盖侧别：{sides}，不是全量评测结论。**")
    REPORT_PATH.write_text(render_report(metrics, meta), encoding="utf-8")
    print(f"\n报告已写入 {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
