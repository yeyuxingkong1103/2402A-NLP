#!/usr/bin/env python3
"""汇总 Jmeter 的 .jtl 结果：吞吐、延迟分位、以及经 nginx 时的 worker 分发。

为什么要有它：Jmeter 的 HTML 报告（`-e -o`）里 QPS 是**它自己按采样窗口算的**，跟
我们口头说的「整机吞吐」口径不一定一致；而并发下用 `N / sum(各请求耗时)` 会得出
「吞吐随并发反降」的假象（sum(ts) ≈ N × 墙钟，等于把吞吐少算 N 倍）。
这里统一按 **N / 实际墙钟** 算，和 HTML 报告互为交叉验证。

用法：
    .venv/bin/python deploy/jmeter/summarize.py deploy/jmeter/result-nginx-llm.jtl

前置条件与副作用：
    - 只读那个 .jtl 文件：不联网、不连库、不改文件，随便重复跑。
    - 输入必须是 **CSV 格式**的 jtl（Jmeter 默认），且带 elapsed / timeStamp / label /
      success 列。列名缺失会直接 KeyError 崩掉，不做兜底——列是由 jmeter 的
      save.saveservice.* 开关决定的，静默跳过某个缺失列会让统计"少了一部分"却毫无提示。
    - 退出码：0 正常、1 文件里没有采样记录、2 参数个数不对（会把本文件 docstring 打出来）。

若 jtl 里带 responseHeaders 列，本脚本会顺带统计各 worker 的分发计数。
**但别指望能轻易拿到这一列**：本机实测 JMeter 5.6.3 下，
`-Jjmeter.save.saveservice.responseHeaders=true`、`-q` 附加属性文件、
乃至在 .jmx 里给 ResultCollector 直接写 `<responseHeaders>true</responseHeaders>`，
三种方式都不落盘（同一次 `-Jjmeter.save.saveservice.print_field_names=false` 却生效，
说明属性机制本身没坏，是这一项的问题）。想看分发请用 `scripts/verify_lb.sh`。
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import Counter

# X-Upstream 是 scripts/setup_nginx.sh 往 rag.conf 里加的自定义响应头，值就是实际处理
# 该请求的 127.0.0.1:800X。只有经 nginx 的请求才有这个头，所以它同时也是「有没有走
# 负载均衡」的判据。\S+ 而不是 \S+$：头后面还跟着 \r\n，行尾锚定会匹配失败。
UPSTREAM_RE = re.compile(r"X-Upstream:\s*(\S+)", re.I)


def percentile(values: list[float], pct: float) -> float:
    """线性插值分位（与 Jmeter 的 P90 口径接近，不追求逐位一致）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(path: str) -> int:
    """读一个 jtl 并打印吞吐/延迟/分发汇总，返回退出码（0 正常，1 没有采样记录）。"""
    # errors="replace"：jtl 里混进半个多字节字符（请求被截断时会出现）不该让整个汇总挂掉，
    # 替换成 U+FFFD 继续算——反正它只出现在标签文本里，不影响数值列
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"❌ {path} 里没有采样记录")
        return 1

    # jtl 的 elapsed / timeStamp 都是毫秒（timeStamp 是 epoch 毫秒），统一除 1000 变秒
    elapsed = [float(r["elapsed"]) / 1000.0 for r in rows]
    # 用 .get 而不是 r["success"]：列被关掉时不崩，但要注意默认值坑——漏了这一列时
    # 每行都取到 ""，不等于 "true"，于是会统计成「全部失败」。看到成功率 0% 先看
    # jmeter 那边 save.saveservice.successful 开着没有，别急着怀疑被测服务。
    ok = sum(1 for r in rows if r.get("success", "").lower() == "true")
    # 墙钟：首条开始 → 末条结束。这是吞吐的正确分母（不是 sum(elapsed)）。
    start = min(float(r["timeStamp"]) for r in rows)
    end = max(float(r["timeStamp"]) + float(r["elapsed"]) for r in rows)
    wall = (end - start) / 1000.0

    n = len(rows)
    print(f"文件        {path}")
    print(f"请求数      {n}（成功 {ok}，失败 {n - ok}，成功率 {ok / n:.2%}）")
    print(f"墙钟        {wall:.1f}s（首条开始→末条结束）")
    print(f"吞吐        {n / wall:.3f} req/s  = {n} / {wall:.1f}s   ← 正确的并发吞吐口径")
    print(f"朴素口径    {n / sum(elapsed):.3f} req/s  = N / sum(各请求耗时)  ← 反例，仅作对照")
    print(
        f"延迟        均值 {statistics.fmean(elapsed):.1f}s  "
        f"P50 {percentile(elapsed, 0.50):.1f}s  P90 {percentile(elapsed, 0.90):.1f}s  "
        f"最大 {max(elapsed):.1f}s"
    )

    # 按 label 分组（label 就是 jmeter 里的 Sampler 名，本仓按接口命名）。
    # 整体均值会被最慢的那个接口主导，分接口看才分得清是 /chat 慢还是压测脚本本身慢。
    by_label: dict[str, list[tuple[float, bool]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(
            (float(r["elapsed"]) / 1000.0, r.get("success", "").lower() == "true")
        )
    print("按接口：")
    for label, pairs in sorted(by_label.items()):
        vals = [v for v, _ in pairs]
        fails = sum(1 for _, s in pairs if not s)
        print(
            f"  {label:<16} n={len(vals):<4} 均值 {statistics.fmean(vals):.1f}s  "
            f"P90 {percentile(vals, 0.90):.1f}s  失败 {fails}"
        )

    ups = Counter(
        m.group(1) for r in rows if (m := UPSTREAM_RE.search(r.get("responseHeaders") or ""))
    )
    if ups:
        print("worker 分发（X-Upstream，经 nginx 时才有）：")
        for name, cnt in sorted(ups.items()):
            # 40 是条形长度的上限（终端一行放得下），按最大值等比缩放；max(1, ...) 保证
            # 计数最少的那个也留一格，否则"0 条"和"很少"在输出里长得一样
            print(f"  {name:<20} {cnt:>4}  {'#' * max(1, round(cnt * 40 / max(ups.values())))}")
        # 「只有 1 个」只是启发式提醒，**不能当判据**：样本少时全落一个 worker 完全正常
        # （每个 worker 各有自己的轮询指针，起点不一样）。要判定负载均衡真的生效，
        # 用 .env/nginx 里配的 upstream zone，或跑 scripts/verify_lb.sh。
        print(f"  → 落到 {len(ups)} 个 worker 上{'' if len(ups) >= 2 else '（只有 1 个：没走负载均衡？）'}")
    elif any("responseHeaders" in r for r in rows):
        print("worker 分发：jtl 里有 responseHeaders 列但没匹配到 X-Upstream —— 可能没经 nginx")
    else:
        print("worker 分发：无 responseHeaders 列 —— 想看分发用 bash scripts/verify_lb.sh（见本文件顶部说明）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
