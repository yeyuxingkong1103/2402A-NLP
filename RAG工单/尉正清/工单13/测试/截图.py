# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""生成性能分析的可视化截图（格式要求「测试：图片 一定要多截图」）

四类图：
  1. 延迟对比柱状图（优化前后，串行均值/中位/P95）—— 本工单的核心结论
  2. 分阶段耗时对比（优化前后各阶段各占多少）
  3. 并发吞吐/延迟曲线（找陡降点）
  4. py-spy 火焰图（SVG 渲染后截图）
  5. snakeviz 的 cProfile 火焰图（若 snakeviz 服务在跑）

用法：
    D:/Anaconda/envs/rag_gd/python.exe 截图.py
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHOTS = HERE / "截图"

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

# 中文字体，否则图上全是方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_BEFORE, C_AFTER = "#b45309", "#15803d"      # 优化前=琥珀，优化后=绿


def load_bench(tag):
    p = RESULTS / f"bench_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_jtl(tag, users):
    p = RESULTS / f"jmeter_{tag}_u{users}.jtl"
    if not p.exists():
        return None
    lat, ts = [], []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                lat.append(float(row["elapsed"]))
                ts.append(int(row["timeStamp"]))
            except (KeyError, ValueError):
                continue
    if not lat:
        return None
    lat.sort()
    n = len(lat)
    span = (max(ts) - min(ts)) / 1000.0
    return {"n": n, "avg": sum(lat) / n, "p95": lat[int(n * 0.95)],
            "max": lat[-1], "tps": n / span if span > 0 else 0}


def fig_latency():
    """图 1：优化前后延迟对比 —— 结论直接画出来。"""
    b, a = load_bench("before"), load_bench("after")
    if not b or not a:
        return None
    labels = ["均值", "中位", "P95", "最大"]
    bv = [b["wall"]["mean"], b["wall"]["median"], b["wall"]["p95"], b["wall"]["max"]]
    av = [a["wall"]["mean"], a["wall"]["median"], a["wall"]["p95"], a["wall"]["max"]]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(labels))
    w = 0.36
    r1 = ax.bar([i - w / 2 for i in x], bv, w, label="优化前", color=C_BEFORE)
    r2 = ax.bar([i + w / 2 for i in x], av, w, label="优化后", color=C_AFTER)
    ax.axhline(3, color="#dc2626", ls="--", lw=1.4)
    ax.text(len(labels) - 0.5, 3.15, "验收线 3s", color="#dc2626", fontsize=10, ha="right")
    for rects in (r1, r2):
        for r in rects:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15,
                    f"{r.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("端到端延迟（秒）")
    ax.set_title("RAG 检索延迟：优化前后对比（20 次串行，10 道轮换问题）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(bv + av) * 1.18)
    fig.tight_layout()
    out = SHOTS / "00-延迟对比-优化前后.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_stages():
    """图 2：分阶段耗时对比。"""
    b, a = load_bench("before"), load_bench("after")
    if not b or not a:
        return None
    keys = sorted(set(b["stages"]) | set(a["stages"]))
    keys = [k for k in keys if b["stages"].get(k, {}).get("mean", 0) > 0.001
            or a["stages"].get(k, {}).get("mean", 0) > 0.001]
    bv = [b["stages"].get(k, {}).get("mean", 0) for k in keys]
    av = [a["stages"].get(k, {}).get("mean", 0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(keys))
    w = 0.36
    r1 = ax.barh([i - w / 2 for i in x], bv, w, label="优化前", color=C_BEFORE)
    r2 = ax.barh([i + w / 2 for i in x], av, w, label="优化后", color=C_AFTER)
    for rects in (r1, r2):
        for r in rects:
            ax.text(r.get_width() + 0.04, r.get_y() + r.get_height() / 2,
                    f"{r.get_width():.2f}s", va="center", fontsize=9)
    ax.set_yticks(list(x))
    ax.set_yticklabels(keys)
    ax.set_xlabel("平均耗时（秒）")
    ax.set_title("分阶段耗时：优化前 LLM 生成与检索各占多少")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(bv + av) * 1.25)
    fig.tight_layout()
    out = SHOTS / "01-分阶段耗时对比.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_concurrency():
    """图 3：并发曲线 —— 吞吐与延迟随并发怎么变，陡降点在哪。"""
    users = [1, 4, 8, 16]
    rows = [(u, load_jtl("before", u), load_jtl("after", u)) for u in users]
    rows = [(u, b, a) for u, b, a in rows if b and a]
    if not rows:
        return None
    us = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(us, [r[1]["tps"] for r in rows], "o-", color=C_BEFORE, label="优化前")
    ax1.plot(us, [r[2]["tps"] for r in rows], "s-", color=C_AFTER, label="优化后")
    ax1.set_xlabel("并发线程数")
    ax1.set_ylabel("吞吐（请求/秒）")
    ax1.set_title("吞吐随并发的走势")
    ax1.set_xticks(us)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(us, [r[1]["avg"] / 1000 for r in rows], "o-", color=C_BEFORE, label="优化前")
    ax2.plot(us, [r[2]["avg"] / 1000 for r in rows], "s-", color=C_AFTER, label="优化后")
    ax2.axhline(3, color="#dc2626", ls="--", lw=1.4)
    ax2.text(us[-1], 3.2, "验收线 3s", color="#dc2626", fontsize=9, ha="right")
    ax2.set_xlabel("并发线程数")
    ax2.set_ylabel("平均延迟（秒）")
    ax2.set_title("延迟随并发的劣化")
    ax2.set_xticks(us)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = SHOTS / "02-并发吞吐与延迟曲线.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_pyspy():
    """图 4：py-spy 火焰图（SVG 用浏览器渲染后截图）。"""
    svg = RESULTS / "pyspy_flamegraph.svg"
    if not svg.exists():
        return None
    from playwright.sync_api import sync_playwright
    out = SHOTS / "03-py-spy火焰图-并发压测中.png"
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport={"width": 1680, "height": 950})
        pg.goto(svg.as_uri(), wait_until="load")
        pg.wait_for_timeout(2500)
        # py-spy 的 SVG 很高，整页截图会超时；只截视口，
        # 火焰图的顶层已经能说明「时间花在哪个调用栈上」
        pg.screenshot(path=str(out), full_page=False, timeout=60000)
        br.close()
    return out


def fig_snakeviz():
    """图 5：snakeviz 火焰图（需要 snakeviz 服务在 8080 上跑着）。"""
    pstats = RESULTS / "profile.pstats"
    if not pstats.exists():
        return None
    import urllib.parse
    import urllib.request
    url = "http://127.0.0.1:8080/snakeviz/" + urllib.parse.quote(
        str(pstats).replace("\\", "/"), safe="/:")
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/snakeviz/", timeout=3)
    except Exception:                                    # noqa: BLE001
        print("  [跳过] snakeviz 服务未启动（python -m snakeviz -s -p 8080 profile.pstats）")
        return None
    from playwright.sync_api import sync_playwright
    out = SHOTS / "04-snakeviz-cProfile火焰图.png"
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport={"width": 1680, "height": 1050})
        pg.goto(url, wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(4000)
        pg.screenshot(path=str(out))
        br.close()
    return out


def fig_report():
    """图 6：性能报告页 —— 交付时一眼能看到全部结论。"""
    md = HERE / "性能报告.md"
    if not md.exists():
        return None
    import html as H
    import re
    text = md.read_text(encoding="utf-8")
    body, in_code = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            body.append("<pre>" if not in_code else "</pre>")
            in_code = not in_code
            continue
        if in_code:
            body.append(H.escape(line))
            continue
        if line.startswith("# "):
            body.append(f"<h1>{H.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{H.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{H.escape(line[4:])}</h3>")
        elif line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "th" if "指标" in line or "阶段" in line or "并发" in line else "td"
            body.append("<tr>" + "".join(f"<{tag}>{H.escape(c)}</{tag}>" for c in cells) + "</tr>")
        elif line.strip():
            body.append(f"<p>{H.escape(line)}</p>")

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1100px;margin:0 auto;
     padding:36px 44px;color:#1a1a1a;line-height:1.75;background:#fafafa}}
h1{{font-size:27px;border-bottom:3px solid #15803d;padding-bottom:10px}}
h2{{font-size:20px;margin-top:32px;border-left:4px solid #c2410c;padding-left:10px}}
h3{{font-size:16px;margin-top:22px;color:#374151}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:12px 0;font-size:14px;
      box-shadow:0 1px 3px rgba(0,0,0,.08)}}
th,td{{border:1px solid #e5e7eb;padding:7px 11px;text-align:left}}
th{{background:#f3f4f6}} pre{{background:#1f2937;color:#e5e7eb;padding:12px 16px;
      border-radius:6px;font-size:13px;overflow-x:auto}} p{{margin:8px 0}}
</style></head><body>{''.join(body)}</body></html>"""
    tmp = RESULTS / "_report.html"
    tmp.write_text(html_doc, encoding="utf-8")
    from playwright.sync_api import sync_playwright
    out = SHOTS / "05-性能报告.png"
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport={"width": 1240, "height": 1000})
        pg.goto(tmp.as_uri(), wait_until="load")
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(out), full_page=True)
        br.close()
    tmp.unlink(missing_ok=True)
    return out


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    for fn in (fig_latency, fig_stages, fig_concurrency, fig_pyspy,
               fig_snakeviz, fig_report):
        try:
            out = fn()
            print(f"  {'✓ ' + out.name if out else '· 跳过 ' + fn.__name__}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  ✗ {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n截图目录 {SHOTS}")
    for g in sorted(SHOTS.glob("*.png")):
        print(f"  {g.name}  ({g.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
