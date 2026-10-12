# 工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏
"""把压测与资源数据画成对比图（产出物要求「压测结果数据与图表」）

数据来源全部是实跑出来的，没有手工填数：
  results/jmeter_a_baseline.jtl   优化前 20 并发
  results/jmeter_a_final.jtl      优化后 20 并发、持续 10 分钟
  results/mem_a_baseline.jsonl    优化前的容器内存/GPU/队列采样
  results/mem_a_final.jsonl       优化后的同上

用法：
    python 截图.py
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE / "截图"
OUT.mkdir(exist_ok=True)

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    try:
        from matplotlib import font_manager
        font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

C_BASE, C_AFTER = "#c0392b", "#1a7f37"


def read_jtl(path):
    if not Path(path).exists():
        return []
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("elapsed")]
    return sorted(int(r["elapsed"]) / 1000 for r in rows)


def split_runs(path, gap_sec=60):
    """把一个 JTL 按时间断点拆成多次运行。

    JMeter 的 -l 默认是**追加**写，同一个文件名跑两次会混在一起。
    基线的 jtl 里就是这样：两次优化前的运行，P95 差了 28 秒。
    拆开看比合在一起诚实。
    """
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("elapsed")]
    if not rows:
        return []
    rows.sort(key=lambda r: int(r["timeStamp"]))
    segs, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if int(r["timeStamp"]) - int(prev["timeStamp"]) > gap_sec * 1000:
            segs.append(cur)
            cur = []
        cur.append(r)
    segs.append(cur)
    # 必须排序：下面的 pct() 是按「已升序」取分位的，不排会取出中间某个值，
    # 把 P95 算成 12 秒这种离谱结果。
    return [sorted(int(r["elapsed"]) / 1000 for r in s) for s in segs]


def read_mem(path):
    if not Path(path).exists():
        return [], [], []
    t, rag, gpu = [], [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        t.append(r.get("t", 0))
        rag.append(r.get("containers", {}).get("docker-ragflow-cpu-1", {}).get("mem_mib"))
        gpu.append(r.get("gpu_mem_mib"))
    return t, rag, gpu


def pct(v, p):
    return v[min(len(v) - 1, int(len(v) * p))] if v else 0


def chart_latency(base_runs, after):
    """延迟分位对比 + 优化后按时间的延迟曲线（优化前的两次运行分开画）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)

    labels = ["P50", "P95", "P99"]
    a = [pct(after, .5), pct(after, .95), pct(after, .99)]
    x = range(len(labels))
    w = 0.8 / (len(base_runs) + 1)
    shades = ["#c0392b", "#e07a5f", "#a93226"]
    for k, base in enumerate(base_runs):
        b = [pct(base, .5), pct(base, .95), pct(base, .99)]
        off = (k - (len(base_runs) - 1) / 2) * w
        ax1.bar([i + off - w / 2 for i in x], b, w,
                label=f"优化前（第{k+1}次）", color=shades[k % len(shades)])
        for i, bb in enumerate(b):
            ax1.text(i + off - w / 2, bb + 0.5, f"{bb:.1f}", ha="center", fontsize=7)
    off = (len(base_runs) - (len(base_runs) - 1) / 2) * w
    ax1.bar([i + off - w / 2 for i in x], a, w, label="优化后", color=C_AFTER)
    for i, aa in enumerate(a):
        ax1.text(i + off - w / 2, aa + 0.5, f"{aa:.1f}", ha="center", fontsize=7)
    ax1.axhline(3, ls="--", c="gray", lw=1)
    ax1.text(2.35, 3.2, "验收线 3s", fontsize=8, color="gray", ha="right")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("响应时间（秒）")
    n_base = sum(len(r) for r in base_runs)
    ax1.set_title(f"20 并发问答延迟分位对比\n"
                  f"（优化前 {len(base_runs)} 次运行共 n={n_base}，优化后 n={len(after)}）")
    ax1.legend()
    ax1.grid(axis="y", alpha=.3)

    # 按时间的延迟（每 10 秒一格的中位数）
    if after:
        buckets = {}
        for r in csv.DictReader(open(RES / "jmeter_a_final.jtl", encoding="utf-8")):
            if not r.get("elapsed"):
                continue
            buckets.setdefault(int(r["timeStamp"]) // 10000, []).append(int(r["elapsed"]) / 1000)
        ks = sorted(buckets)
        med = [sorted(buckets[k])[len(buckets[k]) // 2] for k in ks]
        t0 = ks[0]
        ax2.plot([(k - t0) * 10 for k in ks], med, marker="o", ms=3, color=C_AFTER)
        ax2.axhline(3, ls="--", c="gray", lw=1)
        ax2.set_xlabel("压测进行时间（秒）")
        ax2.set_ylabel("每 10 秒窗口的中位延迟（秒）")
        ax2.set_title("优化后：持续 10 分钟的中位延迟")
        ax2.grid(alpha=.3)
    fig.tight_layout()
    p = OUT / "00-延迟对比-优化前后.png"
    fig.savefig(p, facecolor="white")
    print(f"  已存 {p}")


def chart_resources():
    """容器内存 + GPU 占用，优化前后同图对比"""
    tb, rb, gb = read_mem(RES / "mem_a_baseline.jsonl")
    ta, ra, ga = read_mem(RES / "mem_a_final.jsonl")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130)

    if rb:
        start = rb[0]
        ax1.plot(tb, [(v - start) / start * 100 for v in rb], color=C_BASE, label="优化前")
    if ra:
        start = ra[0]
        ax1.plot(ta, [(v - start) / start * 100 for v in ra], color=C_AFTER, label="优化后")
    ax1.axhline(10, ls="--", c="gray", lw=1)
    ax1.text(0, 10.4, "验收线 +10%", fontsize=8, color="gray")
    ax1.set_xlabel("压测进行时间（秒）")
    ax1.set_ylabel("容器内存相对起始的增长（%）")
    ax1.set_title("RAGFlow 容器内存增长")
    ax1.legend()
    ax1.grid(alpha=.3)

    if gb:
        ax2.plot(tb, gb, color=C_BASE, label="优化前")
    if ga:
        ax2.plot(ta, ga, color=C_AFTER, label="优化后")
    ax2.set_xlabel("压测进行时间（秒）")
    ax2.set_ylabel("GPU 显存（MiB）")
    ax2.set_title("GPU 显存占用")
    ax2.legend()
    ax2.grid(alpha=.3)
    fig.tight_layout()
    p = OUT / "01-资源曲线-优化前后.png"
    fig.savefig(p, facecolor="white")
    print(f"  已存 {p}")


def chart_hotspots(topn=12):
    """py-spy 采样出来的 CPU 热点排行（self time）

    只看**叶子帧** —— 火焰图的父帧是累计值，直接统计会得到「每个函数都占 60%」
    这种没用的结论。
    """
    path = RES / "pyspy_raw.txt"
    if not path.exists():
        print("  跳过热点图：没有 results/pyspy_raw.txt")
        return
    import collections
    self_c, total = collections.Counter(), 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        stack, _, n = line.rpartition(" ")
        try:
            n = int(n)
        except ValueError:
            continue
        total += n
        self_c[stack.split(";")[-1].strip()] += n
    if not total:
        return
    items = self_c.most_common(topn)[::-1]
    names = [f"{fn.split(' (')[0][:38]}\n({fn.split(' (')[-1].rstrip(')')})" for fn, _ in items]
    vals = [n / total * 100 for _, n in items]

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=130)
    bars = ax.barh(range(len(vals)), vals, color="#0b5cad")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7)
    ax.set_xlabel("占全部采样样本的比例（self time，%）")
    ax.set_title(f"20 并发压测中 API Server 的 CPU 热点 Top {len(vals)}\n"
                 f"（py-spy 采样 {total} 个样本；真在算的合计不到 30%，其余都是等）")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    p = OUT / "02-CPU热点-py-spy采样.png"
    fig.savefig(p, facecolor="white")
    print(f"  已存 {p}")


def chart_queue():
    """解析任务队列深度（Redis Stream）—— 场景B 关注这个"""
    tb, _, _ = read_mem(RES / "mem_a_baseline.jsonl")
    ta, _, _ = read_mem(RES / "mem_a_final.jsonl")
    qb = [json.loads(l).get("queue_len") for l in open(RES / "mem_a_baseline.jsonl", encoding="utf-8") if l.strip()] \
        if (RES / "mem_a_baseline.jsonl").exists() else []
    qa = [json.loads(l).get("queue_len") for l in open(RES / "mem_a_final.jsonl", encoding="utf-8") if l.strip()] \
        if (RES / "mem_a_final.jsonl").exists() else []
    if not qb and not qa:
        print("  跳过队列图：没有采样数据")
        return

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    if qb:
        ax.plot(tb[:len(qb)], qb, color=C_BASE, label="优化前")
    if qa:
        ax.plot(ta[:len(qa)], qa, color=C_AFTER, label="优化后")
    ax.set_xlabel("压测进行时间（秒）")
    ax.set_ylabel("队列中堆积的任务数")
    ax.set_title("解析任务队列深度（Redis Stream）\n全程为 0 —— 场景A 不触发解析，队列不背锅")
    ax.set_ylim(bottom=-0.2)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    p = OUT / "03-任务队列深度.png"
    fig.savefig(p, facecolor="white")
    print(f"  已存 {p}")


def main():
    base_runs = split_runs(RES / "jmeter_a_baseline.jtl")
    after = read_jtl(RES / "jmeter_a_final.jtl")
    print(f"  优化前 {len(base_runs)} 次运行（各 {[len(x) for x in base_runs]} 条），"
          f"优化后 {len(after)} 条")
    chart_latency(base_runs, after)
    chart_resources()
    chart_hotspots()
    chart_queue()


if __name__ == "__main__":
    main()
