# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""演示视频用的出图脚本 —— 把报告里的关键结论画成图

工单产出物五要求演示视频展示「质检报告的核心内容，**重点演示如何从报告中定位
『扫描型PDF』、『重复文件』和『敏感信息待审项』**」。

报告本身有 HTML 版（浏览器里直接看最直观），这个脚本负责出**结论图**：
一屏之内把「该看哪三张列表、分别有多少条、路由怎么分」讲清楚，
录屏时当结论页用，不用在浏览器里翻来翻去。

## 用法

    python 演示质检.py                      # 跑演示样本，出图
    python 演示质检.py <报告.json>           # 用已有的报告出图
    python 演示质检.py <报告.json> --no-run  # 同上，语义更明确

图出到 `截图/演示-质检结论.png`。

## 和其他图的分工

- **本脚本**出的图：从**报告 JSON** 算出来，每个数字都能追溯到报告文件
- **录屏时手截**的图：浏览器里的 HTML 报告、终端里的 API 调用 —— 呈现「过程」

两类都要。脚本图负责「结论可追溯」，手截图负责「界面真实感」。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SKILL = HERE.parent / "研发" / "document_quality_assessment"
SAMPLE = HERE / "演示样本"
OUT = HERE / "截图" / "演示-质检结论.png"

sys.path.insert(0, str(SKILL / "scripts"))
import common   # noqa: E402

common.utf8_console()


def run_assessment():
    """跑一遍演示样本，返回报告 dict。"""
    py = sys.executable
    rep = HERE / "results" / "demo" / "quality_report.json"
    subprocess.run(
        [py, str(SKILL / "scripts" / "assess.py"), str(SAMPLE),
         "-o", str(HERE / "results" / "demo"), "--quiet"],
        check=True, cwd=str(SKILL))
    return json.loads(rep.read_text(encoding="utf-8"))


def setup_font():
    """matplotlib 中文字体。照抄工单14 的做法：逐个试，能用的第一个就用。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for name in ("Microsoft YaHei", "SimHei", "DengXian", "simsun"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def draw(rep):
    plt = setup_font()
    s, d = rep["summary"], rep["details"]

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle("文档质量评估 —— 报告里该看什么", fontsize=19, fontweight="bold", y=0.975)

    # ---- 左上：主路由（工作流按这个分流）----
    ax = fig.add_axes([0.05, 0.655, 0.42, 0.265])
    ax.axis("off")
    ax.set_title("① 主路由：每份文档去哪", fontsize=13, fontweight="bold", loc="left")
    routes = sorted(s["by_primary_route"].items(), key=lambda x: -x[1])
    label = {"ocr": "OCR 解析器（没有文字层）", "text": "直接抽文字", "hybrid": "逐页判断",
             "review": "人工复核", "chunk_aggressive": "超长，切块要更碎",
             "table": "表格解析"}
    y = 0.80
    for name, n in routes:
        ax.text(0.02, y, f"● {name}", fontsize=11, fontweight="bold",
                color="#0b5cad", transform=ax.transAxes)
        ax.text(0.30, y, f"{n} 份", fontsize=11, transform=ax.transAxes)
        ax.text(0.46, y, label.get(name, ""), fontsize=9.5, color="#555",
                transform=ax.transAxes)
        y -= 0.145
    ax.text(0.02, y - 0.04, f"（共 {s['total_files']} 份）", fontsize=9,
            color="#777", transform=ax.transAxes)

    # ---- 右上：三个待办列表的条数 ----
    ax2 = fig.add_axes([0.55, 0.655, 0.42, 0.265])
    ax2.axis("off")
    ax2.set_title("② 三张待办列表：有多少要人工看", fontsize=13, fontweight="bold", loc="left")
    todos = [
        ("扫描型 PDF", sum(1 for _, _, rt in d["tags"]["docs"] if rt == "ocr"),
         "报告第 2 节 / 主路由 ocr", "#b42318"),
        ("完全重复（MD5）", len(d["dup"]["md5_groups"]),
         "报告第 4 节", "#b42318"),
        ("待确认版本冲突", len(d["dup"]["sim_pairs"]),
         "报告第 6 节", "#b42318"),
        ("待审核敏感信息", len(d["sensitive"]["hits"]),
         "报告第 7 节", "#b42318"),
        ("解析失败 / 打不开", len(d["errors"]),
         "报告第 10 节", "#1a7f37"),
    ]
    y = 0.82
    for name, n, where, color in todos:
        # 不用 ✓（U+2713）—— Microsoft YaHei 没有这个字形，会画成豆腐块
        mark = "无" if n == 0 else f"{n} 条"
        ax2.text(0.02, y, name, fontsize=11, transform=ax2.transAxes)
        ax2.text(0.42, y, mark, fontsize=11, fontweight="bold",
                 color="#1a7f37" if n == 0 else color, transform=ax2.transAxes)
        ax2.text(0.58, y, where, fontsize=9, color="#777", transform=ax2.transAxes)
        y -= 0.155

    # ---- 下半：三张列表的样例 ----
    def block(y0, title, lines, note):
        ax = fig.add_axes([0.05, y0, 0.90, 0.135])
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=8)
        ax.text(0.995, 1.0, note, fontsize=9, color="#777", ha="right", va="bottom",
                transform=ax.transAxes)
        if not lines:
            ax.text(0.01, 0.58, "无", fontsize=10.5, color="#1a7f37",
                    transform=ax.transAxes)
        for i, ln in enumerate(lines[:4]):
            ax.text(0.01, 0.62 - i * 0.27, ln, fontsize=10, transform=ax.transAxes)

    docname = lambda p: Path(p).name  # noqa: E731

    scanned = [docname(p) for p, tags, rt in d["tags"]["docs"] if rt == "ocr"]
    block(0.465, "③ 扫描型 PDF（这些没有文字层，必须走 OCR）",
          [f"• {n}" for n in scanned] or [], "列表里每份都带完整路径")

    # 用 ←→ 而不是 ↔（U+2194）—— YaHei 没有后者，会画成豆腐块
    pairs = [f"• {docname(a)}  ←→  {docname(b)}    相似度 {sim}，汉明距离 {dist}"
             for a, b, sim, dist in d["dup"]["sim_pairs"]]
    block(0.30, "④ 待确认的版本冲突（SimHash 近似，每条都要人工确认）", pairs,
          "带路径 + 相似度 + 距离")

    hits = [f"• [{t}] {v}    ← {docname(src)}" for src, t, v, _ in d["sensitive"]["hits"][:3]]
    block(0.135, "⑤ 待审核的敏感信息（每条都带上下文，方便判误报）", hits,
          "带路径 + 命中内容 + 前后 40 字上下文")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"结论文已出图：{OUT}")
    return OUT


def main():
    ap = argparse.ArgumentParser(description="演示视频出图：把质检报告的关键结论画成一屏")
    ap.add_argument("report", nargs="?", help="已有的报告 JSON；不给就先跑一遍演示样本")
    ap.add_argument("--no-run", action="store_true", help="只用给定报告，不重跑质检")
    a = ap.parse_args()

    if a.report:
        rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    else:
        print("跑一遍演示样本…")
        rep = run_assessment()
    draw(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
