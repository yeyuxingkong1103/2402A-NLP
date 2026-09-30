"""JMeter完整RAG压力测试：读JTL→判成功→分档统计→总报告→保存。

``docs/verification/rag-chat-load.jmx``负责并发请求/api/chat并写出JTL；
本文件只统计已有JTL，不会再次请求DeepSeek，也不会产生新的API费用。
"""
import argparse  # 读取--jtl、--label和--output三个命令行参数。
import csv  # JTL使用CSV格式，本模块用它按表头读取每条采样记录。
import json  # 把最终报告保存为便于程序再次读取的JSON文件。
import math  # ceil用于计算P50、P95等分位数所在的样本名次。
from collections import Counter  # 统计各HTTP状态码和失败原因分别出现几次。
from datetime import datetime, timezone  # 给报告记录带时区的生成时间。
from pathlib import Path  # 兼容Windows和Ubuntu的文件路径写法。

BASE = Path(__file__).resolve().parents[1]  # 当前文件向上两级得到项目根目录。
JTL_COLUMNS = {  # 这些列共同描述开始时间、耗时、并发、HTTP结果和断言结果。
    "timeStamp", "elapsed", "label", "responseCode",
    "success", "failureMessage", "grpThreads", "URL",
}


def percentile(values, ratio):
    """输入延迟列表和比例（0.95就是P95）；用最近名次法输出分位延迟或None。"""
    if not values:  # 空列表没有P95，明确返回None，不编造0毫秒。
        return None
    ordered = sorted(values)  # 新建从快到慢的列表，不修改调用者的原列表。
    rank = math.ceil(ratio * len(ordered))  # 例如20条的P95名次=ceil(0.95×20)=19。
    index = max(0, rank - 1)  # 人的名次从1开始，Python下标从0开始，所以减1。
    return round(ordered[index], 3)  # 输出毫秒并保留三位小数。


def read_jtl(path, label="RAG_CHAT"):
    """输入：JTL路径和标签；处理：校验表头并筛选；输出：完整RAG采样记录。"""
    path = Path(path)  # 字符串路径也统一转成Path，方便Windows和Ubuntu共用。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)  # 首行表头自动变成后续每个字典的键。
        missing = JTL_COLUMNS - set(reader.fieldnames or [])  # 计算必需列与实际列的差集。
        if missing:  # 缺列就无法可靠计算指标，因此立即停止并列出列名。
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path}缺少JTL列：{names}")
        rows = []  # 只收集RAG_CHAT，注册和清历史等准备请求不计入性能。
        for row in reader:
            if row["label"] == label:
                rows.append(row)
    if not rows:  # 文件可能存在，但标签写错或测试没有真正发出问答请求。
        raise ValueError(f"{path}没有标签为{label}的样本")
    return rows


def sample_passed(row):
    """输入：一条JTL记录；处理：同时检查HTTP与JMeter断言；输出：成功布尔值。"""
    http_ok = row["responseCode"] == "200"  # 后端必须返回HTTP 200。
    assertions_ok = row["success"].lower() == "true"  # answer、sources等断言也要通过。
    return http_ok and assertions_ok  # 200但回答字段为空仍算失败，不能虚报成功。


def summarize_jtl(path, label="RAG_CHAT"):
    """输入：一档并发JTL；处理：判成功、算窗口/QPS/延迟；输出：本档指标字典。"""
    rows = read_jtl(path, label)  # 第一步：读取并筛出完整RAG问答。
    starts, latencies, passed = [], [], []  # 三个列表按同一顺序保存每条采样数据。
    concurrency = 0  # 最终取grpThreads最大值作为本档实际最大并发。
    try:
        for row in rows:  # 逐行转换数值，避免复杂推导式让初学者难跟踪。
            starts.append(int(row["timeStamp"]))  # 请求开始时间，单位毫秒。
            latencies.append(float(row["elapsed"]))  # 完整问答耗时，单位毫秒。
            concurrency = max(concurrency, int(row["grpThreads"]))
            passed.append(sample_passed(row))  # HTTP和业务断言都通过才记True。
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}包含无效数值字段") from error

    successes = sum(passed)  # Python中True等于1，因此求和就是成功数量。
    failures = len(rows) - successes  # 总样本减去成功样本得到失败数。
    end_times = []  # 每个请求的结束时刻等于开始时刻加本次耗时。
    for index in range(len(rows)):
        end_times.append(starts[index] + latencies[index])
    window_seconds = (max(end_times) - min(starts)) / 1000  # 从最早开始到最晚结束，毫秒转秒。
    qps = len(rows) / window_seconds if window_seconds else None  # QPS=问答总数÷观测秒数，不是1000÷平均延迟。

    failure_counts = Counter()  # 失败时优先统计JMeter给出的业务断言消息。
    status_counts = Counter()  # 无论成功失败，都统计HTTP状态码便于排查。
    for index, row in enumerate(rows):
        status_counts[row["responseCode"]] += 1
        if not passed[index]:
            reason = row["failureMessage"].strip() or f"HTTP {row['responseCode']}"
            failure_counts[reason] += 1

    return {  # 输出字段会写入JSON，也会被markdown函数转成答辩表格。
        "source": str(path), "target": rows[0]["URL"], "label": label,
        "concurrency": concurrency, "samples": len(rows),
        "successes": successes, "failures": failures,
        "error_rate": round(failures / len(rows), 6),
        "success_rate": round(successes / len(rows), 6),
        "window_seconds": round(window_seconds, 3),
        "throughput_qps": round(qps, 6) if qps is not None else None,  # 所有请求吞吐。
        # 只有零失败时才叫“成功问答QPS”，否则不拿总请求吞吐掩盖错误。
        "successful_qps": round(qps, 6) if qps is not None and failures == 0 else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 3),
        "min_latency_ms": round(min(latencies), 3),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p90_latency_ms": percentile(latencies, 0.90),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "max_latency_ms": round(max(latencies), 3),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def jmeter_report(paths, label="RAG_CHAT"):
    """输入：多档JTL路径；处理：每档独立统计再求总数；输出：完整RAG总报告。"""
    if not paths:  # 没有原始证据时拒绝生成空报告。
        raise ValueError("至少需要一个JTL文件")
    runs = []
    for path in paths:  # 各并发档必须分别算QPS，不能把不同时间窗口混在一起。
        runs.append(summarize_jtl(path, label))
    samples, successes = 0, 0
    for run in runs:  # 总报告只合计样本和成功数，仍保留每档自己的性能指标。
        samples += run["samples"]
        successes += run["successes"]
    failures = samples - successes
    return {
        "kind": "jmeter_full_rag", "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "runs": runs, "samples": samples,
        "successes": successes, "failures": failures,
        "error_rate": round(failures / samples, 6),
        "coverage": "Redis读取→BGE向量化→Milvus+BM25→RRF→BGE精排→DeepSeek→后处理→Redis写回",
        "scope": "每轮先清空历史，测试完整首轮RAG；不含多轮Query改写；低成本小样本基线",
    }


def report_paths(output):
    """输入：文件前缀、JSON/MD路径或目录；处理：统一后缀；输出：两个报告路径。"""
    output = Path(output)
    if output.suffix.lower() in {".json", ".md"}:  # 传入某种报告后缀时先去掉它。
        stem = output.with_suffix("")
    elif output.exists() and output.is_dir():  # 传入现有目录时使用固定报告名。
        stem = output / "pressure_report"
    else:
        stem = output  # 无后缀路径直接当作两个报告共用的文件名前缀。
    return stem.with_suffix(".json"), stem.with_suffix(".md")


def markdown(report):
    """把总报告排成Markdown表格；这里只管展示格式，答辩不用逐行讲。"""
    header = ("| 并发 | 样本 | 成功/失败 | 错误率 | 成功问答QPS | 平均/P50/P90/P95/P99延迟(ms) | Min/Max(ms) |\n"
              "|---:|---:|---:|---:|---:|---:|---:|")
    lines = []
    latency_keys = ["average_latency_ms", "p50_latency_ms", "p90_latency_ms",
                    "p95_latency_ms", "p99_latency_ms"]
    for run in report["runs"]:
        qps = "不可用" if run["successful_qps"] is None else f"{run['successful_qps']:.3f}"
        latency_values = []
        for key in latency_keys:
            latency_values.append(f"{run[key]:.3f}")
        latency = "/".join(latency_values)
        lines.append(f"| {run['concurrency']} | {run['samples']} | "
                     f"{run['successes']} / {run['failures']} | {run['error_rate'] * 100:.2f}% | "
                     f"{qps} | {latency} | {run['min_latency_ms']:.3f} / {run['max_latency_ms']:.3f} |")
    return ("# JMeter完整RAG问答压力测试报告\n\n"
            f"生成时间：{report['created_at']}\n\n覆盖链路：{report['coverage']}。\n\n"
            + header + "\n" + "\n".join(lines) + "\n\n## 统计边界\n\n"
            + report["scope"] + "。只有错误率为0%的档位才展示成功问答QPS。"
            "高分位来自少量样本，不代表长期容量上限。\n")


def save_report(report, output):
    """输入：总报告和输出位置；处理：序列化；输出：已写入的JSON与Markdown路径。"""
    json_path, md_path = report_paths(output)
    json_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在时自动创建。
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    return json_path, md_path


def build_parser():
    """输入：无；处理：声明三个参数；输出：只允许JTL汇总的命令行解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jtl", type=Path, nargs="+", required=True,
                        help="一个或多个JMeter CSV格式JTL；只读取文件，不发请求")
    parser.add_argument("--label", default="RAG_CHAT", help="JTL中完整问答采样器的标签")
    parser.add_argument("--output", type=Path, help="报告文件名前缀；省略时写入outputs/jmeter")
    return parser


def main(argv=None):
    """输入：可选命令行参数；处理：汇总并保存；输出：成功0、可预期失败2。"""
    args = build_parser().parse_args(argv)
    try:
        report = jmeter_report(args.jtl, args.label)
        output = args.output or BASE / "outputs" / "jmeter" / "rag_chat_report"
        json_path, md_path = save_report(report, output)
    except (ValueError, OSError) as error:
        print(f"压力测试报告失败：{error}")
        return 2
    print(f"完成：{report['successes']}/{report['samples']}成功，错误率={report['error_rate'] * 100:.2f}%")
    print(f"JSON报告：{json_path}\nMarkdown报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # 把0或2交给终端，方便部署脚本判断执行结果。
