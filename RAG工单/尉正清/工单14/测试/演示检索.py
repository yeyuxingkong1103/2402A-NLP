# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""演示脚本：对 6 个问题做检索 + 问答，打印「检索到的片段 → 相似度 → 最终答案」

对应工单产出物三：演示里「显示检索到的答案及检索精确度」。

用法：
    D:/Anaconda/envs/rag_gd/python.exe 演示检索.py            # 打到屏幕
    D:/Anaconda/envs/rag_gd/python.exe 演示检索.py --save     # 另存成图片
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow
from run_round import build_messages, parse_letter

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))
W = 78


def brief(text, n=90):
    return re.sub(r"\s+", " ", text).strip()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="另存为图片")
    ap.add_argument("--top", type=int, default=3, help="每题展示几条召回")
    args = ap.parse_args()

    qs = json.loads((HERE / "测试问题.json").read_text(encoding="utf-8"))
    c = RAGFlow()
    c.load_token()

    rows = []
    for n, q in enumerate(qs, 1):
        code, j = c._req("POST", "/api/v1/retrieval", headers=c._h(), json={
            "question": q["question"], "dataset_ids": [IDS["ds_id"]],
            "similarity_threshold": 0.05, "vector_similarity_weight": 0.3,
            "top_k": 1024, "page_size": args.top,
        }, timeout=180)
        chunks = (j.get("data") or {}).get("chunks", [])
        ans, sec = c.ask(IDS["chat_id"], build_messages(q))
        rows.append({"n": n, "q": q, "chunks": chunks, "ans": ans, "sec": sec,
                     "ok": parse_letter(ans) == q["answer"]})

        print("=" * W)
        print(f"第 {n} 题  {q['question']}")
        for o in q["options"]:
            print(f"    {o}")
        print(f"  期望答案：{q['answer']}   {q['options'][ord(q['answer']) - 65][3:]}")
        print("-" * W)
        print("  检索召回：")
        if not chunks:
            print("    （无 —— 全部低于阈值）")
        for i, ch in enumerate(chunks, 1):
            print(f"    {i}. 相似度 {ch.get('similarity', 0):.3f}"
                  f"（向量 {ch.get('vector_similarity', 0):.3f} /"
                  f" 关键词 {ch.get('term_similarity', 0):.3f}）")
            print(f"       {brief(ch.get('content', ''))}")
        print("-" * W)
        got = parse_letter(ans)
        print(f"  作答：{got or '(空)'}  {'✓ 正确' if rows[-1]['ok'] else '✗ 错误'}"
              f"   响应 {sec:.1f}s")
        print(f"  原文：{brief(ans, 120)}")
        print()

    hit = sum(r["ok"] for r in rows)
    print("=" * W)
    print(f"合计：{hit}/{len(rows)} 正确，平均响应 {sum(r['sec'] for r in rows)/len(rows):.1f}s")

    if args.save:
        save_image(rows)


def save_image(rows):
    """把结果画成一张图，当作演示截图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 找一个能显示中文的字体。等宽字体也要一起换掉，
    # 否则召回片段那几行用的是 DejaVu Sans Mono，中文会全变方框
    for name in ("Microsoft YaHei", "SimHei", "DengXian", "simsun"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.4 * n), dpi=130)
    for ax, r in zip(axes, rows):
        ax.axis("off")
        color = "#1a7f37" if r["ok"] else "#b42318"
        ax.text(0.005, 0.94, f"第{r['n']}题  {r['q']['question']}",
                fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.005, 0.72, "期望 " + r["q"]["answer"] + "：" +
                r["q"]["options"][ord(r["q"]["answer"]) - 65][3:],
                fontsize=9, va="top", color="#444", transform=ax.transAxes)
        y = 0.52
        for i, ch in enumerate(r["chunks"][:3], 1):
            ax.text(0.02, y, f"召回{i}  相似度 {ch.get('similarity', 0):.3f}   "
                             f"{brief(ch.get('content', ''), 62)}",
                    fontsize=8, va="top", color="#0b5cad", transform=ax.transAxes)
            y -= 0.16
        ax.text(0.02, y, f"作答 {(parse_letter(r['ans']) or '(空)')}   "
                         f"响应 {r['sec']:.1f}s", fontsize=10, va="top",
                color=color, fontweight="bold", transform=ax.transAxes)
        ax.plot([0, 1], [0.02, 0.02], color="#ddd", lw=1, transform=ax.transAxes)

    hit = sum(r["ok"] for r in rows)
    avg = sum(r["sec"] for r in rows) / len(rows)
    fig.suptitle(f"CN100342976C.pdf 六题检索与问答演示        "
                 f"准确率 {hit}/{len(rows)}   平均响应 {avg:.1f}s",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = HERE / "截图" / "20-检索与问答演示.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()
