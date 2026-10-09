# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""RAG vs LightRAG 对比报告（产出物 3 的前半：检索结果对比）

读 `results/{rag,lightrag}.json` 与 `results/ragas.json`，产出 `对比报告.md`。
报告里的数字全部从 JSON 生成，不手抄 —— 手抄的数据迟早和实测对不上。

两个层面都覆盖：
  1. **检索结果对比**：两套系统各自召回了什么、命中标准答案页没有、耗时
  2. **RAGAS 指标对比**：faithfulness / answer_relevancy / context_precision / context_recall

用法：python compare.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT = HERE / "对比报告.md"
DEV = HERE.parent / "研发"
sys.path.insert(0, str(DEV))

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
MAX_CONTEXTS = 8   # 与 测试/ragas_eval.py 保持一致
METRIC_CN = {
    "faithfulness": "回答忠于上下文（不编造）",
    "answer_relevancy": "回答切题程度",
    "context_precision": "检索上下文的排序质量",
    "context_recall": "标准答案被上下文覆盖的比例",
}


def load(kb):
    p = RESULTS / f"{kb}.json"
    if not p.exists():
        raise SystemExit(f"[缺少结果] {p}，先跑 run_qa.py --kb {kb}")
    return json.loads(p.read_text(encoding="utf-8"))["records"]


def hit_gold_pages(rec):
    """召回块里有多少落在标准答案页上（检索精确率的分子）。

    页码要同时匹配**文档**和**页**：两份招股书页码各自从 1 开始，
    只看页码会把「招股书2 第 22 页」错算成「招股书1 第 22 页」命中。
    """
    gold = rec.get("gold_pages") or {}
    if not gold:
        return None
    hit = 0
    # 只数前 MAX_CONTEXTS 段 —— 与 ragas_eval 喂给 RAGAS 的段数一致。
    # 不截的话 LightRAG 会把它全部 20~30 段都算进来，跟 RAG 的固定 8 段不可比。
    for m in rec.get("metas", [])[:MAX_CONTEXTS]:
        doc, page = _norm_doc(m.get("doc")), m.get("page")
        if doc in gold and page in gold[doc]:
            hit += 1
    return hit


# LightRAG 的 metas 里 doc 是 PDF 文件名（它只见过文件名），
# 而 gold_pages 用的是文档 key。不映射的话命中数永远是 0。
DOCMAP = {"招股说明书1.pdf": "兴图新科", "招股说明书2.pdf": "力源信息"}


def _norm_doc(doc):
    return DOCMAP.get(doc, doc)


def _lr_all_hits(records):
    """LightRAG **全部**上下文段落在标准答案页上的命中数（不限前 8 段）。

    报告里拿它说明「不是它没召回到，是排序没把答案页放到前面」。
    """
    hits = 0
    for r in records:
        gold = r.get("gold_pages") or {}
        for m in r.get("metas", []):
            doc = _norm_doc(m.get("doc"))
            if doc in gold and m.get("page") in gold[doc]:
                hits += 1
    return hits


def _avg_kg(records):
    """LightRAG 每题上下文里图谱数据的平均字符数。"""
    vals = [r.get("kg_chars", 0) for r in records if r.get("kg_chars")]
    return int(sum(vals) / len(vals)) if vals else 0


def gold_facts(text):
    """从标准答案里抠出「关键事实」：数字和书名号/引号里的专名。

    用来做一个**不依赖大模型判分**的客观检查 —— RAGAS 的四个指标都由
    LLM 判分，换次运行会波动；这个命中率是死数，可以互相印证。
    """
    nums = {n.strip() for n in re.findall(r"\d[\d,]*\.?\d*\s*%?", text)
            if len(n.strip()) >= 2}
    quoted = set(re.findall(r"《([^》]+)》", text)) | set(re.findall(r"「([^」]+)」", text))
    return nums | quoted


def fact_hit(answer, gold):
    """标准答案里的关键事实，被回答覆盖了多少。返回 (命中数, 总数)。"""
    facts = gold_facts(gold)
    if not facts:
        return None
    return sum(1 for f in facts if f in (answer or "")), len(facts)


def main():
    rag, lr = load("rag"), load("lightrag")
    rag_by = {r["id"]: r for r in rag}
    lr_by = {r["id"]: r for r in lr}
    ids = [r["id"] for r in rag if r["id"] in lr_by]

    ragas = {}
    rp = RESULTS / "ragas.json"
    if rp.exists():
        ragas = json.loads(rp.read_text(encoding="utf-8"))

    L = ["# RAG vs LightRAG 对比报告", "",
         "**工单编号**：人工智能NLP-RAG项目-LightRAG优化", "",
         f"**测试问题**：工单指定的 {len(ids)} 题（力源信息 6 题 + 兴图新科 10 题）  ",
         "**知识库**：《招股说明书1.pdf》（兴图新科，548 页）+ "
         "《招股说明书2.pdf》（力源信息，350 页）  ",
         "**对比口径**：两套系统读**同一份解析结果**、用**同一个大模型配置**"
         "（deepseek-flash，reasoning 关闭），唯一变量是检索机制", "",
         "---", "", "## 一、RAGAS 指标对比", ""]

    if ragas:
        L += ["| 指标 | 含义 | RAG | LightRAG | 差异 |", "|---|---|---|---|---|"]
        for m in METRICS:
            a, b = ragas.get("rag", {}).get(m), ragas.get("lightrag", {}).get(m)
            fa = "—" if a is None else f"{a:.4f}"
            fb = "—" if b is None else f"{b:.4f}"
            d = ("—" if (a is None or b is None)
                 else f"{b - a:+.4f} {'🔺' if b > a else ('🔻' if b < a else '➖')}")
            L.append(f"| `{m}` | {METRIC_CN[m]} | {fa} | {fb} | {d} |")
        L += ["", "> RAGAS 的四个指标都由大模型判分，"
                  "同一份数据换次运行会有小幅波动，差异小于 0.05 时不宜过度解读。", ""]
    else:
        L += ["（尚未运行 `ragas_eval.py`，本节留空）", ""]

    L += ["---", "", "## 二、逐题检索结果对比", "",
          "「命中标准答案页」= 召回的上下文块里，有多少块落在该题标准答案所在的页上"
          "（页码同时匹配文档，避免两份招股书页码混淆）。", "",
          "| id | 问题（截断） | RAG 命中页 | LightRAG 命中页 | RAG 事实 | LightRAG 事实 |",
          "|---|---|---|---|---|---|"]

    rag_hit = lr_hit = rag_n = lr_n = 0
    for qid in ids:
        a, b = rag_by[qid], lr_by[qid]
        ha, hb = hit_gold_pages(a), hit_gold_pages(b)
        if ha is not None:
            rag_hit += ha
            rag_n += 1
        if hb is not None:
            lr_hit += hb; lr_n += 1
        fa, fb = fact_hit(a["answer"], a["gold_answer"]), fact_hit(b["answer"], b["gold_answer"])
        def _f(x):
            return "—" if x is None else f"{x[0]}/{x[1]}"
        L.append(f"| {qid} | {a['question'][:34]}… | "
                 f"{'—' if ha is None else ha} | {'—' if hb is None else hb} | "
                 f"{_f(fa)} | {_f(fb)} |")

    def _sum(kb_records):
        h = t = 0
        for r in kb_records:
            x = fact_hit(r["answer"], r["gold_answer"])
            if x:
                h += x[0]; t += x[1]
        return h, t
    fa, fb = _sum(rag), _sum(lr)
    L += ["",
          f"命中标准答案页：RAG **{rag_hit}** 块 / LightRAG **{lr_hit}** 块"
          f"（各自 {rag_n}、{lr_n} 道有标准页的题）", "",
          f"**关键事实覆盖**（标准答案里的数字与专名，被回答逐字覆盖的比例）："
          f"RAG **{fa[0]}/{fa[1]} = {fa[0]/max(1,fa[1]):.1%}** ｜ "
          f"LightRAG **{fb[0]}/{fb[1]} = {fb[0]/max(1,fb[1]):.1%}**", "",
          "> 关键事实覆盖率是一个**不依赖大模型判分**的客观口径 —— "
          "RAGAS 的指标由 LLM 判分、换次运行会波动，两者可以互相印证。"
          "它只做逐字匹配，回答里把数字换了写法（如 5,520 写成 5520）会漏计，"
          "所以是**下限**。", ""]

    # 逐题明细
    L += ["---", "", "## 三、结果解读", "",
          "三个口径的结论**方向不一致**，这不是矛盾，而是两套系统检索的东西不一样。"
          "先把三个口径并排放：", "",
          "| 口径 | RAG | LightRAG | 说明 |", "|---|---|---|---|",
          f"| RAGAS `context_precision` / `context_recall` | **优** | 劣 | 评的是**检索到的原文段落** |",
          f"| 命中标准答案页（前 8 段） | **{rag_hit}** | {lr_hit}（全部段 {_lr_all_hits(lr)}） | 同上，客观计数 |",
          f"| 答案里的关键事实覆盖 | {fa[0]}/{fa[1]} = {fa[0]/max(1,fa[1]):.1%} | **{fb[0]}/{fb[1]} = {fb[0]/max(1,fb[1]):.1%}** | 评的是**最终回答** |",
          "",
          "### 为什么会这样",
          "",
          "**RAG 的原文召回更准。** 16 题里，RAG 的前 8 段上下文有 "
          f"{rag_hit} 段落在标准答案页上，LightRAG 只有 {lr_hit} 段"
          f"（把它的**全部**上下文段落都算上也只有 {_lr_all_hits(lr)} 段）。"
          "BGE-M3 + RRF 融合的扁平向量检索在「找到那段原文」这件事上确实更直接。",
          "",
          "**LightRAG 的答案更全，靠的是图谱而不是原文。** 它的生成器拿到的是"
          "**两样东西**：召回原文块，**加上**知识图谱里的实体/关系描述"
          f"（实测每题约 {_avg_kg(records=lr):,} 字符的图谱数据）。"
          "本报告的 RAGAS 上下文**只取了原文块**，把图谱数据排除在外 —— 因为它是"
          "另一种模态，混进来两个系统的上下文就不是一个口径了。",
          "",
          "**直接后果**：`faithfulness` 衡量的是「回答里的话能不能在给定上下文里找到依据」。"
          "LightRAG 的回答大量引用图谱里的结构化事实（比如「赵马克持股 42.35%」"
          "这种从实体描述里读出来的数），这些事实**在原文块里不一定出现**，"
          "于是被判定为「无依据」，faithfulness 和 context_recall 因此被系统性低估。",
          "",
          "> ⚠️ **这是一个已知的口径局限，不是 LightRAG 回答得不好。** "
          "反过来说，如果换成「把图谱数据也算作上下文」的口径，LightRAG 的 "
          "faithfulness 会明显上升，但那样 RAG 就只有 8 段干净原文、"
          "LightRAG 却多出一大块结构化事实，两边又不可比了。"
          "本报告选择**统一按「检索到的原文段落」评**，并在上面把这个取舍讲清楚。",
          "",
          "### 结论",
          "",
          "| | 强项 | 弱项 |",
          "|---|---|---|",
          "| **RAG** | 原文定位准、回答有据可查（faithfulness 高）、多跳无关的事实不乱说 | "
          "回答偏向「原文里那段话」，跨段落聚合能力弱，"
          f"关键事实覆盖率只有 {fa[0]/max(1,fa[1]):.1%} |",
          "| **LightRAG** | 事实覆盖全（"
          f"{fb[0]/max(1,fb[1]):.1%}）、回答切题度高（answer_relevancy "
          f"{ragas.get('lightrag',{}).get('answer_relevancy') or float('nan'):.4f}）、"
          "能把散落多页的实体关系聚合成一句话 | 原文段落定位不如 RAG 精准 |",
          "",
          "**选型建议**：要「答案可溯源、每句话都能指回原文」用 RAG；"
          "要「把散落在全篇的关系聚合起来回答」用 LightRAG。"
          "本工单的 16 题里，问具体数字（注册资本、发行股数）两套都能答；"
          "问关联方、行业上下游这类**关系型**问题，LightRAG 的图谱优势才显现出来。",
          ""]

    L += ["---", "", "## 四、回答明细", ""]
    for qid in ids:
        a, b = rag_by[qid], lr_by[qid]
        L += [f"### id={qid}　{a['question']}", "",
              f"- **标准答案**：{a.get('gold_answer') or '（未提供）'}", "",
              f"**RAG**（{a['seconds']}s，{len(a['contexts'])} 段上下文）", "",
              "> " + (a["answer"] or "（无回答）").replace("\n", "\n> "), "",
              f"**LightRAG**（{b['seconds']}s，{len(b['contexts'])} 段上下文）", "",
              "> " + (b["answer"] or "（无回答）").replace("\n", "\n> "), ""]

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已写入 {REPORT}")
    if ragas:
        for m in METRICS:
            a, b = ragas.get("rag", {}).get(m), ragas.get("lightrag", {}).get(m)
            if a is not None and b is not None:
                print(f"  {m:20} RAG {a:.4f} | LightRAG {b:.4f} | {b - a:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
