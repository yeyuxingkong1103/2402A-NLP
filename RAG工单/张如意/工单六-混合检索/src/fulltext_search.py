# -*- coding: utf-8 -*-
"""
工单06 全文检索演示（倒排索引 + Okapi BM25 + 多字段加权）
工单编号：人工智能NLP-RAG-混合检索任务

本脚本演示工单「全文检索」的全部功能点：
  1. 普通 BM25 查询          —— 关键词打分排序（Okapi BM25 公式）
  2. 布尔查询                —— AND / OR / NOT 三种集合运算
  3. 短语匹配                —— 精确子串命中（专有名词、标准号、数字）
  4. 模糊匹配                —— 通配符 `军用*` + Levenshtein 编辑距离容错
  5. 多字段加权消融          —— 仅正文(1.0) vs 正文(1.0)+章节路径(1.6)
  6. 技术说明                —— 倒排索引结构 / BM25 公式 / 适用场景

运行：
    python 工单06-混合检索/src/fulltext_search.py
产出：
    工单06-混合检索/results/fulltext_demo.md

说明：中文分词优先用 jieba；未安装时 rag_core 自动降级为「单字 + 二元组」，
      检索仍可用，但布尔 AND/NOT 的精度会下降，建议 pip install jieba。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core.bm25 import BM25Retriever                        # noqa: E402
from rag_core.chunk import Chunk                               # noqa: E402
from wo06_common import RESULTS_DIR, md_table, snip, save_markdown  # noqa: E402
from build_index import BM25_PATH, FIELD_WEIGHTS, ensure_index, get_retriever      # noqa: E402

try:
    import jieba  # noqa: F401
    HAS_JIEBA = True
except ImportError:                                            # pragma: no cover
    HAS_JIEBA = False


# ---------------------------------------------------------------------------
# 展示工具
# ---------------------------------------------------------------------------
def hits_table(docs: list[dict], top: int = 5) -> str:
    """把检索结果格式化为 Markdown 表格。"""
    rows = []
    for i, d in enumerate(docs[:top], 1):
        rows.append([
            i, f"《{d.get('doc', '')}》", d.get("page", ""),
            d.get("type", "text"), snip(d.get("section", ""), 26),
            f"{float(d.get('score', 0)):.4f}", snip(d.get("text", ""), 70),
        ])
    if not rows:
        return "（无命中）"
    return md_table(["排名", "文档", "页码", "类型", "章节", "分数", "片段摘要"], rows)


def show(title: str, query: str, docs: list[dict], top: int = 5) -> str:
    """控制台打印 + 返回 Markdown 片段。"""
    print(f"\n【{title}】query = {query!r}  命中 {len(docs)} 条")
    for i, d in enumerate(docs[:top], 1):
        print(f"  {i}. 《{d.get('doc')}》第{d.get('page')}页 "
              f"({d.get('type')}) score={float(d.get('score', 0)):.4f} | "
              f"{snip(d.get('text', ''), 60)}")
    return f"### {title}\n\n**查询**：`{query}`（命中 {len(docs)} 条）\n\n" \
           + hits_table(docs, top)


def retrieve(retr, query: str, top_k: int = 20) -> list[dict]:
    """统一走 Retriever.fulltext_search，保证与主流程同一入口。"""
    return retr.fulltext_search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 各功能演示
# ---------------------------------------------------------------------------
def demo_plain(retr) -> list[str]:
    """1. 普通 BM25 查询。"""
    lines = ["## 1. 普通 BM25 查询",
             "对 query 分词后，用 Okapi BM25 公式对每个词条累加打分并排序。"]
    for q in ["军用领域收入", "注册资本 法定代表人", "募集资金 补充流动资金"]:
        docs = retrieve(retr, q)
        lines.append(show(f"BM25 普通查询：{q}", q, docs))
    return lines


def demo_boolean(retr) -> list[str]:
    """2. 布尔查询（AND / OR / NOT）。"""
    lines = ["## 2. 布尔查询（AND / OR / NOT）",
             "布尔操作符通过 `boolean` 参数传入，query 只写关键词本身（空格分隔）。",
             "AND = 各词条倒排列表求交；OR = 求并；NOT = 首词倒排列表减去其余词条。"]
    cases = [
        ("AND", "军用 收入", "同时含「军用」和「收入」的片段（精确匹配，宁缺毋滥）"),
        ("OR", "军用 民用", "含「军用」或「民用」的片段（扩大召回）"),
        ("NOT", "收入 政府补助", "含「收入」但不含「政府补助」的片段（排除干扰）"),
    ]
    for op, q, desc in cases:
        docs = retr.fulltext_search(q, top_k=20, boolean=op)
        extra = ""
        if not docs and not HAS_JIEBA:
            extra = ("  \n> 注：当前环境未安装 jieba，中文按单字+二元组降级切分，"
                     "NOT 类查询可能为空；`pip install jieba` 后结果更精确。")
        lines.append(f"### {op} 布尔查询\n\n"
                     f"**语义**：{desc}\n\n**查询**：`{q}`（boolean={op}，"
                     f"命中 {len(docs)} 条）\n\n{hits_table(docs)}{extra}")
        print(f"\n【布尔 {op}】query={q!r} 命中 {len(docs)} 条")
    return lines


def demo_phrase(retr) -> list[str]:
    """3. 短语匹配（精确子串）。"""
    lines = ["## 3. 短语匹配（phrase_search）",
             "对专有名词、技术标准名、精确数字最有效：直接做子串扫描，",
             "命中次数作为分数，可避免分词把「视频指挥系统技术标准」拆散后失真。"]
    for phrase in ["视频指挥系统技术标准", "5,520", "42.35%"]:
        docs = retr.bm25.phrase_search(phrase, top_k=5)
        lines.append(show(f"短语匹配：{phrase}", phrase, docs))
    return lines


def demo_fuzzy(retr) -> list[str]:
    """4. 模糊匹配（通配符 + 编辑距离）。"""
    lines = ["## 4. 模糊匹配（fuzzy_search）",
             "两种模式：① 通配符 `前缀*` 正则展开；② 无通配符时按 Levenshtein",
             "编辑距离 ≤ max_dist 近似匹配，用于容忍错别字/形近字。"]
    lines.append(show("通配符匹配：军用*", "军用*",
                      retr.bm25.fuzzy_search("军用*", top_k=5)))
    for typo in ["收人", "利闰"]:
        docs = retr.bm25.fuzzy_search(typo, top_k=5, max_dist=1)
        extra = ""
        if not docs:
            extra = ("  \n> 无命中：当前分词降级模式下候选词较少，"
                     "安装 jieba 后再试；或把 max_dist 调大。")
        lines.append(f"### 编辑距离匹配（错别字容错）\n\n"
                     f"**查询**：`{typo}`（max_dist=1，命中 {len(docs)} 条）"
                     f"{extra}\n\n{hits_table(docs)}")
        print(f"\n【模糊】query={typo!r} 命中 {len(docs)} 条")
    return lines


def demo_field_weights(retr) -> list[str]:
    """
    5. 多字段加权消融：正文 1.0 vs 正文 1.0 + 章节路径 1.6。
    BM25 索引中的 tf 是「按字段权重加权后的词频」：
        tf_weighted(t, d) = 1.0 × tf_text + 1.6 × tf_section
    章节路径（如「第五节 业务与技术 > 一、主营业务」）里的关键词因此更"值钱"。
    """
    base: BM25Retriever = retr.bm25
    chunks = [
        Chunk(text=d["text"], doc=d["doc"], page=d["page"], type=d.get("type", "text"),
              chunk_id=d["chunk_id"], section=d.get("section", ""))
        for d in base.docs
    ]
    text_only = BM25Retriever(field_weights={"text": 1.0})
    text_only.build(chunks)
    weighted = BM25Retriever(field_weights=FIELD_WEIGHTS)
    weighted.build(chunks)

    query = "军用领域收入"
    a, b = text_only.search(query, top_k=5), weighted.search(query, top_k=5)
    rank_a = {d["chunk_id"]: i for i, d in enumerate(a, 1)}
    rank_b = {d["chunk_id"]: i for i, d in enumerate(b, 1)}

    rows = []
    for cid in list(rank_a) + [c for c in rank_b if c not in rank_a]:
        d = next(x for x in (a + b) if x["chunk_id"] == cid)
        ra, rb = rank_a.get(cid, "—"), rank_b.get(cid, "—")
        arrow = ""
        if isinstance(ra, int) and isinstance(rb, int):
            arrow = "↑" if rb < ra else ("↓" if rb > ra else "=")
        rows.append([snip(cid, 24), f"《{d['doc']}》第{d['page']}页",
                     snip(d.get("section", ""), 30), ra, rb, arrow])
    table = md_table(["chunk_id", "来源", "章节路径", "仅正文排名", "加权后排名", "变化"], rows)

    print(f"\n【多字段加权消融】query={query!r}")
    print(table)
    return ["## 5. 多字段加权消融实验",
            f"字段权重：正文 text = {FIELD_WEIGHTS['text']}，"
            f"章节路径 section = {FIELD_WEIGHTS['section']}。\n\n"
            f"**查询**：`{query}`（对比同一批块在两种索引下的排名）\n\n" + table +
            "\n\n> 章节路径权重提升后，章节标题含「军用领域/主营业务」的块排名上升，"
            "说明标题类字段对定位类问题有正向作用（工单「多字段检索」要求）。"]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_index(verbose=True)
    retr = get_retriever()
    retr.load_bm25()          # 显式装载 BM25 索引，供短语/模糊/多字段演示直接调用

    print("=" * 72)
    print("工单06 全文检索演示 | 倒排索引 + Okapi BM25")
    print(f"分词模式：{'jieba 精确模式' if HAS_JIEBA else '降级模式（单字+二元组）'}")
    print(f"字段权重：{FIELD_WEIGHTS}    索引文件：{BM25_PATH}")
    print(f"索引规模：{len(retr.bm25.doc_ids)} 块，"
          f"{len(retr.bm25.inverted)} 个词条，avgdl={retr.bm25.avgdl:.1f}")
    print("=" * 72)

    blocks: list[str] = [
        "本报告由 `src/fulltext_search.py` 自动生成，"
        "所有结果均为真实建索引 + 真实检索所得。\n\n"
        f"- 索引：`{BM25_PATH}`\n"
        f"- 规模：{len(retr.bm25.doc_ids)} 个文本块，{len(retr.bm25.inverted)} 个倒排词条\n"
        f"- 分词：{'jieba' if HAS_JIEBA else '单字+二元组（降级）'}\n"
        f"- 字段权重：正文 text={FIELD_WEIGHTS['text']}，"
        f"章节路径 section={FIELD_WEIGHTS['section']}\n"
        f"- BM25 参数：k1={retr.bm25.k1}，b={retr.bm25.b}",
    ]
    blocks += demo_plain(retr)
    blocks += demo_boolean(retr)
    blocks += demo_phrase(retr)
    blocks += demo_fuzzy(retr)
    blocks += demo_field_weights(retr)

    # 技术说明（对应演示视频要讲的「全文检索用的技术」）
    blocks.append("""## 6. 技术说明

**倒排索引结构**：`term -> {doc_idx: 加权词频}`，另存每篇文档长度 `doc_len`
与平均长度 `avgdl`；查询时只扫描 query 词条对应的倒排列表，避免全量扫描。

**Okapi BM25 打分公式**：

```
score(q, d) = Σ_{t∈q} IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1-b+b·|d|/avgdl) )
IDF(t)      = ln( 1 + (N - n_t + 0.5) / (n_t + 0.5) )
```

其中 f(t,d) 为多字段加权词频：`f = 1.0×tf_text + 1.6×tf_section`；
k1=1.5 控制词频饱和，b=0.75 控制长度归一化强度。

**适用场景**：精确关键词、专有名词、金额/比例/年份、布尔筛选、错别字容错。""")

    path = save_markdown(RESULTS_DIR / "fulltext_demo.md",
                         "工单06 全文检索演示报告（倒排索引 + BM25）", blocks)
    print(f"\n[完成] 报告已输出 -> {path}")


if __name__ == "__main__":
    main()
