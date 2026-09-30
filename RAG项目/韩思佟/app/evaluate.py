"""评估优化：测试题→重排前后检索→计算指标→可选RAGAS→保存报告。

retrieval只检查“资料找得好不好”，不会调用DeepSeek；ragas检查最终回答，
默认也只校验本地题目，只有明确加--run才调用在线模型并产生API费用。
"""
from __future__ import annotations

import argparse  # 读取retrieval/ragas及其命令行参数。
import json  # 捕获评测集JSON格式错误。
import math  # 检查RAGAS分数是否为正常有限数字。
import sys  # 读取终端参数，并把错误写到stderr。
from pathlib import Path  # 统一Windows与Ubuntu路径写法。

from app.internal import evaluation_engine as backend  # 复杂报告与RAGAS适配放在内部。

BASE = Path(__file__).resolve().parents[1]  # 当前app目录的上一级是项目根目录。
RETRIEVAL_METRICS = ("hit_at_k", "recall_at_k", "precision_at_k", "mrr")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
EXPECTED_RAGAS = backend.EXPECTED_RAGAS  # 固定版本，避免不同RAGAS算法混着比较。
SetupError = backend.SetupError

# 八个指标都按“越接近1越好”理解：前四个测检索，后四个测最终回答。
# hit_at_k：前K条里只要出现一条正确资料就是1，一条都没有就是0。
# recall_at_k（召回率）：找回的正确资料数÷应该找回的正确资料总数，检查“漏没漏”。
# precision_at_k（准确率）：前K个位置中的正确资料数÷K，检查“混入多少无关资料”。
# mrr：第一条正确资料名次的倒数；第1名得1，第2名得0.5，第4名得0.25。
# faithfulness（忠实度）：回答中的说法能否由检索资料支持，主要检查模型有没有编造。
# answer_relevancy（回答相关性）：回答是否真正回应用户问题，避免答非所问。
# context_precision（资料准确性）：排在前面的检索资料是否与人工参考答案相关。
# context_recall（资料召回率）：人工参考答案需要的信息是否已被检索资料覆盖。

# retrieved是系统实际返回的块号；relevant是人工提前标好的正确块号。
# 例：retrieved=[9, 21, 19, 3]、relevant=[19, 21]、K=4：命中两个正确块，
# 所以hit=1、recall=2/2=1、precision=2/4=0.5，首个正确块排第2，所以MRR=1/2=0.5。
# row表示“某一道题在某一种方案下的一条结果”；同一道题做前后对比会有两条row。
# variant=hybrid表示混合检索，variant=hybrid_rerank表示混合检索后再用BGE精排。
# 检索row的status为ok/failed；RAGAS row会依次经历pending、generated、evaluated，
# 指标部分失败为partial_failure，检索或回答失败为generation_failed，裁判失败为evaluation_failed。


# ---------- 1. 检索指标：用人工标注检查找回的知识块 ----------
def retrieval_score_case(retrieved, relevant, top_k=4):
    """输入系统块号、正确块号和K；处理匹配关系；输出一道题的四项指标。"""
    if top_k < 1:  # K表示只看前几个结果，必须至少为1。
        raise ValueError("top_k必须为正整数")
    ranked, seen = [], set()  # ranked保持名次；seen用于去掉重复知识块。
    for chunk_id in retrieved:
        if chunk_id not in seen:
            ranked.append(chunk_id)
            seen.add(chunk_id)
        if len(ranked) == top_k:  # 收够前K个唯一结果后停止。
            break

    expected = set(relevant)  # 集合适合做交集，也会去掉重复标注。
    if not expected:  # 空正确答案表示知识库范围外的问题。
        scores = {name: None for name in RETRIEVAL_METRICS}
        scores.update(out_of_scope_correct=float(not ranked), matched=[])
        return scores  # 越界题不硬算普通指标；没有返回资料才算正确。
    matched = set(ranked) & expected  # 交集就是系统真正找回的正确块。
    first_rank = None
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in expected:
            first_rank = rank
            break
    return {
        "hit_at_k": float(bool(matched)),  # 前K个至少命中一个就是1。
        "recall_at_k": len(matched) / len(expected),  # 找回正确数÷应找回数。
        "precision_at_k": len(matched) / top_k,  # 找回正确数÷K个位置。
        "mrr": 1 / first_rank if first_rank else 0.0,  # 首个正确名次的倒数。
        "out_of_scope_correct": None,
        "matched": sorted(matched),
    }


def retrieval_summarize(results):
    """输入所有逐题结果；按方案只平均成功题；输出重排前后汇总。"""
    summary, variants = {}, set()
    for row in results:
        variants.add(row["variant"])  # variant是hybrid或hybrid_rerank。
    for variant in sorted(variants):
        rows, completed = [], []
        for row in results:
            if row["variant"] == variant:
                rows.append(row)
                if row["status"] == "ok":
                    completed.append(row)
        item = {"total": len(rows), "completed": len(completed),
                "failed": len(rows) - len(completed)}
        for metric_name in (*RETRIEVAL_METRICS, "out_of_scope_correct"):
            values = []
            for row in completed:
                value = row["scores"][metric_name]
                if value is not None:  # None是不适用，不能冒充0分。
                    values.append(value)
            item[metric_name] = sum(values) / len(values) if values else None
            item[metric_name + "_count"] = len(values)  # 记录均值用了几题。
        times = [row["elapsed_ms"] for row in completed]
        item["mean_retrieval_ms"] = sum(times) / len(times) if times else None
        summary[variant] = item
    return summary


# ---------- 2. RAGAS指标：检查最终回答和资料的质量 ----------
def metric_result(value=None, status="ok", reason=None):
    """输入原始分数和状态；排除NaN/无穷；输出统一的分数结果字典。"""
    if status == "ok":
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {"value": None, "status": "undefined", "reason": "non_finite_score"}
        value = float(value)
    else:
        value = None  # 失败、跳过和未运行都不能进入平均分。
    return {"value": value, "status": status, "reason": reason}


def aggregate(results):
    """输入逐题RAGAS结果；只平均成功且有效的分数；输出各方案汇总。"""
    output, variants = {}, set()
    for row in results:
        variants.add(row["variant"])
    for variant in sorted(variants):
        rows = [row for row in results if row["variant"] == variant]
        output[variant] = {}
        for metric_name in RAGAS_METRICS:
            values = []
            for row in rows:
                item = row["metrics"][metric_name]
                if item["status"] == "ok" and item["value"] is not None:
                    values.append(item["value"])
            output[variant][metric_name] = {
                "mean": sum(values) / len(values) if values else None,
                "valid": len(values), "total": len(rows),
                "excluded": len(rows) - len(values),
            }
    return output


def paired_deltas(results):
    """同题计算“精排后-精排前”；正数表示提升，负数表示下降，0表示不变。"""
    pairs = {}  # 结构是：题号→方案名→该题结果。
    for row in results:
        pairs.setdefault(row["id"], {})[row["variant"]] = row
    output = {}
    for metric_name in RAGAS_METRICS:
        differences = []
        for pair in pairs.values():
            before = pair.get("hybrid")
            after = pair.get("hybrid_rerank")
            if not before or not after:  # 缺任一边就不能公平对比。
                continue
            before_score = before["metrics"][metric_name]
            after_score = after["metrics"][metric_name]
            # 只比较前后都成功的同一道题，避免失败样本被当成0分拉低平均值。
            if before_score["status"] == after_score["status"] == "ok":
                differences.append(after_score["value"] - before_score["value"])
        output[metric_name] = {
            "mean_delta": sum(differences) / len(differences) if differences else None,
            "paired_valid": len(differences),
        }
    return output


# ---------- 3. 真实流程：internal处理文件、模型适配和原子保存 ----------
# 这些公开名字供旧脚本和测试继续调用；答辩只需知道它们的中文职责。
load_retrieval_cases = backend.load_retrieval_cases  # 读取并校验检索题。
retrieval_markdown = backend.retrieval_markdown  # 把检索结果排成Markdown。
report_prefix = backend.report_prefix  # 给运行中/失败报告加后缀，保护成功报告。
build_retrieval_engine = backend.build_retrieval_engine  # 加载真实RAG检索器。
load_ragas_cases = backend.load_ragas_cases  # 读取并校验生成评测题。
validate_references = backend.validate_references  # 核对参考摘录来自真实知识块。
safe_error = backend.safe_error  # 隐藏错误中可能出现的密钥和隐私。
package_versions = backend.package_versions  # 记录RAGAS等依赖版本。
ragas_markdown = backend.ragas_markdown  # 把RAGAS结果排成Markdown。
build_metrics = backend.build_metrics  # 创建四个真实RAGAS指标。
score_row = backend.score_row  # 给一条回答计算适用指标。
new_row = backend.new_row  # 新建一条尚未执行的评测记录。


def save_retrieval_report(report, output):
    """输入检索报告和路径；用本文件汇总规则保存JSON与Markdown。"""
    return backend.save_retrieval_report(report, output, retrieval_summarize)


def save_ragas_report(report, output):
    """输入RAGAS报告和路径；先用本文件规则汇总，再原子保存两个文件。"""
    report["summary"] = aggregate(report["results"])
    report["paired_deltas"] = paired_deltas(report["results"])
    return backend.save_ragas_report(report, output)


def run_retrieval(args):
    """真实流程：读题→建检索器→逐题跑重排前后→算指标→保存报告。"""
    # 把本文件中容易讲解的指标函数交给internal的真实逐题循环。
    return backend.run_retrieval(
        args, retrieval_score_case, retrieval_summarize, build_retrieval_engine
    )


def run_ragas(args):
    """校验题目；仅args.run为真时生成回答并调用RAGAS裁判，可能产生费用。"""
    # 真实顺序：读题与核对出处→检查--run→加载固定版RAGAS和RAG引擎→逐方案检索→
    # DeepSeek生成回答→RAGAS计算适用指标→保存每题状态、汇总均值和精排前后差值。
    # 不带--run会在核对本地题目后停止，不加载问答模型和裁判，也不会产生API费用。
    # 带--run会产生“回答生成+自动裁判”调用；--compare-rerank会给每题跑两种方案，
    # 因而通常比单方案调用更多。可先用--limit 1小样本试跑，再决定是否扩大评测。
    # 版本读取和指标创建通过普通参数传入，测试可替换，运行时不会改全局状态。
    return backend.run_ragas(args, EXPECTED_RAGAS, package_versions, build_metrics)


# ---------- 4. 命令行：选择免费检索评估或可选付费RAGAS ----------
def parse_retrieval_args(argv=None):
    """把检索命令文字转换成参数；默认只读本地题目，不调用付费模型。"""
    return backend.parse_retrieval_args(argv, base=BASE)


def parse_ragas_args(argv=None):
    """把RAGAS命令文字转换成参数；不带--run时只校验，不产生费用。"""
    return backend.parse_ragas_args(argv, base=BASE)


def retrieval_main(argv=None):
    """检索入口：输入错误返回2；运行是否成功由run_retrieval返回。"""
    try:
        return run_retrieval(parse_retrieval_args(argv))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("评测输入错误：" + str(error), file=sys.stderr)
        return 2


def ragas_main(argv=None):
    """RAGAS入口：默认只校验；输入错误返回2。"""
    try:
        return run_ragas(parse_ragas_args(argv))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("输入/文件错误：" + str(error), file=sys.stderr)
        return 2


def main(argv=None):
    """选择retrieval或ragas；返回0成功、1逐题运行有失败、2输入或初始化错误。"""
    # 0也包括“ragas未带--run，只完成免费校验”；1表示流程启动后至少一题失败；
    # 2表示参数/文件输入错误，或付费评测所需的固定版本、模型等初始化失败。
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("retrieval", "ragas"),
                        help="retrieval测检索；ragas测最终回答")
    if not arguments:
        parser.print_help()
        return 2
    command = parser.parse_args(arguments[:1]).command
    remaining = arguments[1:]
    return retrieval_main(remaining) if command == "retrieval" else ragas_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())  # 把0/1/2交给终端，方便部署脚本判断结果。
