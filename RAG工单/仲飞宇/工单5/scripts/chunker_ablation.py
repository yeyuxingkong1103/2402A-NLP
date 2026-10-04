# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
"""
工单02 · 分块层对照实验（**离线**：不需要 Milvus、不需要嵌入、不建第二个库）。

  python scripts/chunker_ablation.py            # 跑对照并打印表
  python scripts/chunker_ablation.py --dump-gold  # 顺便打印推导出的金句，人工核对

【为什么要做这个对照】
工单02 要求从「pdf解析处理、分块优化、检索优化」三个层面优化。检索层的对比
可以靠三剖面在同一向量库上跑出来；但**分块层的对比需要两个不同的向量库**
（固定长度切分要重新嵌入 1100+ 块），而嵌入路径没有重试，演示前冒不起这个险。

所以分块层改用**离线指标**：金句完整性。
把每题的金句拿出来，看它在「句边界切分」和「固定 N 字切分」下，
能不能完整落在一个 chunk 里。

【必须一起说清的边界，不许夸大】
1. n=10，差距就是 2 道题。只能说「10 题里有 2 题的金句会被固定 500 字切块切断」，
   **不能写成百分比提升**。
2. 「句边界 10/10」是**构造性成立**的 —— 它压根不在句子中间切。这个对照证明的是
   "这个设计约束是必要的"，不是"它最优"，更不能说它解释了准确率。
3. 金句完整性 ≠ 可召回性。金句被切断，两半仍可能各自被召回；
   完整是**必要不充分**条件。
4. 金句是从 questions.json 的 rule_keywords + evidence_page **推导**出来的
   （取证据页里命中关键词最多、且最长的那句），不是人工写的真值。
   用 --dump-gold 逐条人工核对。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, settings                          # noqa: E402
from app.core.chunker import _split_sentences, chunk_pages             # noqa: E402
from app.core.pdf_parser import parse_pdf_full                         # noqa: E402

PDF = settings.data_path / "raw" / "招股说明书1.pdf"
OUT = settings.data_path / "eval" / "chunker-ablation.json"
_PAGE_RE = re.compile(r"1-1-\d+")

# 固定长度切分的对照档位。500 是教科书默认值；300 是工单01 曾经用过的窗口。
FIXED_SIZES = (300, 500, 800)


# ----------------------------------------------------------------------
def _nows(s: str) -> str:
    """去掉所有空白。

    【为什么必须归一化再比较】金句是从「按行拼出来的页文本」里取的，
    而 chunk 里块与块之间的分隔符与页文本不同 —— 直接做子串比较会
    因为换行位置不同而全部判失败，**这是个假象**。
    实测踩过：不归一化时句边界切法会误报 0/10（实际是 10/10）。
    """
    return "".join(s.split())


def page_texts(pages) -> dict[str, str]:
    """把每页的正文 + 表格 Markdown 拼成 {页标签: 文本}。

    【为什么必须带上表格】id=207/531/543 的答案在表格里，不在正文块里。
    只取 page.texts 会让这三题的金句推导落空（实测 id=207 推不出来）。
    """
    out: dict[str, str] = {}
    for pc in pages:
        parts = [t.text for t in pc.texts]
        parts += [tb.to_markdown() for tb in pc.tables]
        out[pc.page_label] = "\n".join(p for p in parts if p)
    return out


def derive_gold_sentence(q: dict, by_label: dict[str, str]) -> str | None:
    """从证据页里推出「金句」。

    【散文题】取命中 rule_keywords 最多、且最长的那**一句**。

    【表格题为什么按行取】实测过：表格题若也按句取，会退化成 850 字的整张表
    （id=531/543），"完整性"就失去意义了。表格的真正原子单位是**行**，
    所以改为在表格 Markdown 里取命中关键词最多、且最长的那一行。
    """
    kws = q.get("rule_keywords") or []
    is_table = str(q.get("category", "")).startswith("表格")
    best, best_key = None, (-1, -1)
    for label in _PAGE_RE.findall(q.get("evidence_page", "")):
        text = by_label.get(label, "")
        units = text.splitlines() if is_table else _split_sentences(text)
        for u in units:
            u = u.strip()
            n = sum(1 for k in kws if k in u)
            if n and (n, len(u)) > best_key:
                best, best_key = u, (n, len(u))
    return best


def fixed_size_chunks(doc_text: str, size: int) -> list[str]:
    """固定长度切分（对照用）。切成大小恰好 size 的连续块。"""
    return [doc_text[i:i + size] for i in range(0, len(doc_text), size)]


def evaluate(chunks: list[str], gold: str, doc_ws: str) -> dict:
    """金句是否完整落在一个 chunk 里 + 是否被切断（均在去空白后比较）。"""
    g = _nows(gold)
    if not g or g not in doc_ws:
        return {"found": False, "intact": None, "straddle": None}
    intact = any(g in _nows(c) for c in chunks)
    # 被切断 ⟺ 金句确实在文档里、但没有任何一个 chunk 完整包含它
    return {"found": True, "intact": intact, "straddle": not intact}


def main() -> int:
    ap = argparse.ArgumentParser(description="工单02 · 分块层离线对照")
    ap.add_argument("--dump-gold", action="store_true",
                    help="打印推导出的金句（人工核对用）")
    args = ap.parse_args()

    if not PDF.exists():
        print(f"找不到 {PDF}", file=sys.stderr)
        return 2

    print("解析 PDF（约 30 秒）…", flush=True)
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=None)
    by_label = page_texts(pages)
    # 阅读顺序拼接成整篇文本，供"固定长度切分"使用
    doc_text = "".join(by_label[k] for k in sorted(by_label))
    doc_ws = _nows(doc_text)

    questions = json.loads(
        (PROJECT_ROOT / "eval" / "questions.json").read_text(encoding="utf-8")
    )["questions"]

    # 句边界切分 = 项目真实使用的分块器（复用同一次解析结果）
    sentence_chunks = [c.content for c in chunk_pages(pages)]
    strategies: dict[str, list[str]] = {"sentence-boundary": sentence_chunks}
    for size in FIXED_SIZES:
        strategies[f"fixed-{size}"] = fixed_size_chunks(doc_text, size)

    rows, golds = [], {}
    for q in questions:
        gold = derive_gold_sentence(q, by_label)
        golds[q["id"]] = gold
        row = {"id": q["id"], "category": q.get("category", ""),
               "gold_sentence": gold, "gold_chars": len(gold) if gold else 0}
        for name, chunks in strategies.items():
            row[name] = evaluate(chunks, gold or "", doc_ws)
        rows.append(row)

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("  工单02 · 分块层对照：金句完整性（离线，未使用向量库）")
    print("=" * 78)
    head = f"{'切法':<20}{'金句完整':>10}{'被切断':>8}{'未找到金句':>12}{'块数':>8}"
    print(head)
    print("-" * len(head))
    summary = {}
    for name, chunks in strategies.items():
        intact = sum(1 for r in rows if r[name]["intact"] is True)
        straddle = sum(1 for r in rows if r[name]["straddle"] is True)
        missing = sum(1 for r in rows if r[name]["found"] is False)
        n_ok = len(rows) - missing
        summary[name] = {"intact": intact, "straddle": straddle,
                         "missing": missing, "n_scored": n_ok,
                         "n_chunks": len(chunks)}
        print(f"{name:<20}{f'{intact}/{n_ok}':>10}{straddle:>8}{missing:>12}"
              f"{len(chunks):>8}")

    # ---------- 逐题明细 ----------
    print("\n  被切断的题（说明切点落在金句内部）：")
    for name in strategies:
        if name == "sentence-boundary":
            continue
        bad = [r["id"] for r in rows if r[name]["straddle"]]
        if bad:
            print(f"    {name:<12} {bad}")

    # 告诫里的数字**从结果算出来**，不写死 —— 否则改了切法或题集后口径就对不上了
    base = summary["sentence-boundary"]
    diffs = {name: base["intact"] - s["intact"]
             for name, s in summary.items() if name != "sentence-boundary"}
    worst = max(diffs, key=lambda k: diffs[k]) if diffs else ""
    print("\n" + "!" * 78)
    print("  口径告诫（写进优化方案文档时必须一并写明）：")
    print(f"  1. n={len(rows)}，量级很小：与句边界相比，{worst} 的相关题数差 "
          f"{diffs.get(worst, 0)} 道。")
    print("     只能表述为「N 题中有 M 题的金句会被固定长度切块切断」，"
          "**不得写成百分比提升**。")
    print("  2. 句边界 10/10 是**构造性成立**的（它不在句子中间切）——"
          "该对照证明的是「这个设计约束是必要的」，")
    print("     不是「它最优」，也不能说它解释了准确率。")
    print("  3. 金句完整 ≠ 可召回：被切断的两半仍可能各自被召回，完整是必要不充分条件。")
    print("  4. 金句由 rule_keywords + evidence_page 推导，非人工真值；"
          "用 --dump-gold 逐条核对。")
    print("  5. 本对照**只用离线指标**，没有建第二个向量库 —— 它度量的是「切点是否落在"
          "金句内部」，")
    print("     不是端到端检索效果；线上检索效果看 eval.py 的三剖面表。")
    print("!" * 78)

    if args.dump_gold:
        print("\n  推导出的金句：")
        for r in rows:
            g = r["gold_sentence"] or "(未推出)"
            print(f"    id={r['id']:<4} {r['gold_chars']:>4}字  {g[:88]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"summary": summary, "strategies": list(strategies), "items": rows,
         "caveats": [
             f"n={len(rows)}，量级很小，不得表述为百分比提升；"
             f"{worst} 与句边界相比相关题数差 {diffs.get(worst, 0)} 道",
             "句边界完整是构造性成立，只证明该约束必要，不证明其最优",
             "金句完整是可召回的充分不必要条件（切断的两半仍可能各自被召回）",
             "金句由 rule_keywords + evidence_page 推导，非人工真值",
             "本对照只用离线指标、未建第二个向量库；端到端效果见 eval.py 三剖面表",
         ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
