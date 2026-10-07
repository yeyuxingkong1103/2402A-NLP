"""
图像解析消融实验（工单4 核心产出）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

单变量对照：
    T0_no_image  图**不经过多模态模型**。图内散字与图题留在正文里当普通文本行，
                 图区域本身不产生任何检索单元（工单1/2/3 的做法）。
    T1_image     图区域经 qwen-vl-plus 生成结构化描述后独立成块；
                 图内散字与图题从正文剔除。

两份索引的**解析 / 分块 / 检索算法完全一致**，只有「图有没有被多模态解析」这一个
变量不同 —— 这是工单2 血泪教训（口径不一致会算出方向相反的假因果）。

用法：
    python scripts/ablation_image.py                        # top_k=8，闸门关
    python scripts/ablation_image.py --with-gate            # 带阈值闸门（端到端口径）
    python scripts/ablation_image.py --top-k 12             # 敏感性
    python scripts/ablation_image.py --clip                 # 打开 CLIP 跨模态召回通道
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.query_norm import normalize_query  # noqa: E402
from src.retriever import retrieve  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_IMAGE


def _norm(s: str) -> str:
    """关键事实匹配前的归一化：NFKC（全角→半角、「−」→「-」）+ 抹掉所有空白。

    必须做：图内文字来自多模态模型的转写，空格口径与 PDF 文字层不一致
    （「IC卡」vs「IC 卡」、「2008年中国」vs「2008 年中国」），
    按原字符比对会把「明明答对了」的题判成漏，让消融结论偏保守。
    """
    import unicodedata

    return "".join(unicodedata.normalize("NFKC", s or "").split())

CONFIGS = [
    {"key": "T0_no_image", "name": "图像不解析（优化前）", "kb": "no_image",
     "clip": False, "prior": False,
     "desc": "图内散字与图题留在正文；图区域不产生检索单元（工单1/2/3 的做法）"},
    {"key": "T1_image", "name": "图像多模态解析", "kb": "image",
     "clip": False, "prior": False,
     "desc": "qwen-vl-plus 生成结构化描述独立成块；图内散字与图题从正文剔除"},
    {"key": "T2_image_prior", "name": "图像解析 + 低信息图排序惩罚", "kb": "image",
     "clip": False, "prior": True,
     "desc": "在 T1 之上，对「证照照片」这类低信息图给 0.15 的排序惩罚（只改排序，不动依据分）"},
    {"key": "T3_image_clip", "name": "图像解析 + 惩罚 + CLIP 跨模态召回", "kb": "image",
     "clip": True, "prior": True,
     "desc": "在 T2 之上打开 CLIP 通道：问题向量与图向量直接比相似度（以文搜图）"},
]

# 相邻两条臂都是**单变量**对照，报告里逐个解释：
#   T0 → T1  图有没有被多模态解析（工单4 要证明的那一件事）
#   T1 → T2  有没有启用低信息图排序惩罚（修「证照照片抢排名」的实测代价）
#   T2 → T3  有没有打开 CLIP 跨模态召回（工单备注的另一条路线）
CONTRASTS = [
    ("T0_no_image", "T1_image", "图像解析的净效果（工单4 主结论）"),
    ("T1_image", "T2_image_prior", "低信息图排序惩罚的效果（修代价）"),
    ("T2_image_prior", "T3_image_clip", "CLIP 跨模态召回在图像解析之上的增量"),
]


# ------------------------------------------------------------------ 建对照索引
def build_no_image_index(progress=print):
    """取「图像不解析」的对照索引。

    优先读落盘的那一份（`data/index_noimage/`，
    由 `python scripts/build_index.py --no-images --out noimage` 生成）——
    向量化 1800+ 块在 CPU 上要 3 分钟，不该每跑一次消融就重来。
    落盘不存在才现场构建。

    ⚠️ 现场构建必须走 `parse_pdf(..., figure_drops=None, figures=None)`，
    **不能只把 figures 去掉、却仍传 figure_drops**：那会变成
    「图内的字被删了、图的内容也没进索引」——两头都空，
    是把对照组做成了残废组。对照组的正确形态是"图内文字照旧留在正文里"。
    """
    from src.index_store import KnowledgeBase

    try:
        kb = KnowledgeBase.get(config.INDEX_NOIMAGE_DIR)
        n_img = sum(1 for c in kb.chunks if c["type"] == "image")
        progress(f"[准备] 对照索引（落盘）{len(kb.chunks)} 块，其中 image 型 {n_img}")
        return kb
    except FileNotFoundError:
        progress("[准备] 对照索引未落盘，现场构建（约 3 分钟）…")

    from src.chunker import chunk_document
    from src.embedder import embed_texts
    from src.pdf_parser import parse_pdf

    chunks_all = []
    seq = 0
    t0 = time.perf_counter()
    for d in config.DOCS:
        parsed = parse_pdf(
            config.RAW_DIR / d["file"],
            with_tables=True,
            header_res=[re.compile(d["header_re"])],
            doc_name=d["name"],
            figure_drops=None,      # ← 图内文字**不剔除**
            figures=None,           # ← 图区域**不成块**
        )
        chunks = chunk_document(parsed, doc_key=d["key"], seq_start=seq)
        seq += len(chunks)
        chunks_all.extend(chunks)
        progress(f"[准备] {d['key']} 分块 {len(chunks)} 条")
    vecs = embed_texts([c.text for c in chunks_all])
    progress(f"[准备] 对照索引 {len(chunks_all)} 块，向量 {vecs.shape}，"
             f"耗时 {time.perf_counter()-t0:.1f}s")
    from src.index_store import KnowledgeBase as KB

    return KB([c.to_dict() for c in chunks_all], vecs, {})


# ------------------------------------------------------------------ 指标
def score_case(items: list[dict], must_have: list[str], gold: list[dict],
               counts: list[str] | None = None,
               relations: list[str] | None = None) -> dict:
    """
    计算单题指标。

    gold 形如 ``[{"doc": "liyuan", "pages": [22, 24]}]``。
    **两文档语料下必须带文档键**：兴图第 22 页和力源第 22 页都存在，
    只比页码会把「命中另一家公司的第 22 页」误判成命中。

    counts：形如 ["4", "6"] 的"数量型"关键事实（id=5 问的就是"几个"）。
    这类答案在正文里通常以中文数字或阿拉伯数字混写，且"4 个""6 个"这种
    字符串在语料里极常见，单看它命中没有意义 —— 所以**单独统计**，
    与 must_have 的覆盖率分开看，避免用一个不可靠的指标把结论带偏。
    """
    joined = _norm("\n".join(it["text"] for it in items))
    hit = [k for k in must_have if _norm(k) in joined]
    fact_recall = len(hit) / len(must_have) if must_have else None
    count_hit = [c for c in (counts or []) if _norm(c) in joined]
    # 「归属关系」（工单4 图像题的关键事实形态）。
    # id=5 的答案不是"有哪些部门"这种**名词清单**，而是"谁挂在谁下面"这种**关系**：
    # 名词清单在图内散字里也能凑出来（对照组实测能凑到），关系却只存在于连接线拓扑中。
    # 因此单独列一路指标，用**带箭头的完整关系串**做判据 ——
    # 箭头是图像解析独有的产物，正文里不可能出现。
    rel_hit = [r for r in (relations or []) if _norm(r) in joined]

    rank = None
    for i, it in enumerate(items, 1):
        span = set(range(it["page"], it["page_end"] + 1)) if it["page"] else set()
        for g in gold:
            doc = g.get("doc")
            pages = set(g.get("pages") or [])
            if doc and doc != it.get("doc_key"):
                continue
            if span & pages:
                rank = i
                break
        if rank:
            break

    return {
        "fact_recall": fact_recall,
        "all_facts": (len(hit) == len(must_have)) if must_have else None,
        "missed": [k for k in must_have if k not in hit],
        "counts_hit": count_hit,
        "relations_expected": relations or [],
        "relations_hit": rel_hit,
        "relation_recall": (len(rel_hit) / len(relations)) if relations else None,
        "gold_page_hit": rank is not None,
        "mrr": (1.0 / rank) if rank else 0.0,
        "gold_rank": rank,
        "n_items": len(items),
        "top_evidence": items[0]["evidence"] if items else 0.0,
        "top_doc": items[0]["doc_key"] if items else "",
        "top_page": items[0]["page"] if items else 0,
        "top_type": items[0]["type"] if items else "",
        "image_chunks_in_topk": sum(1 for it in items if it["type"] == "image"),
        "table_chunks_in_topk": sum(1 for it in items if it["type"] == "table"),
    }


def _avg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def summarize(cases: list[dict]) -> dict:
    n = len(cases) or 1
    return {
        "fact_recall": _avg([c["fact_recall"] for c in cases]),
        "all_facts_rate": _avg([1.0 if c["all_facts"] else 0.0 for c in cases]),
        "all_facts_count": sum(1 for c in cases if c["all_facts"]),
        "gold_page_hit_rate": _avg([1.0 if c["gold_page_hit"] else 0.0 for c in cases]),
        "mrr": _avg([c["mrr"] for c in cases]),
        "avg_top_evidence": _avg([c["top_evidence"] for c in cases]),
        "avg_image_chunks": _avg([c["image_chunks_in_topk"] for c in cases]),
        "relation_recall": _avg([c["relation_recall"] for c in cases
                                 if c["relation_recall"] is not None]),
        "avg_table_chunks": _avg([c["table_chunks_in_topk"] for c in cases]),
        "gated_cases": sum(1 for c in cases if c["n_items"] == 0),
        "num_cases": n,
    }


# ------------------------------------------------------------------ 主流程
def run(build_no_image=build_no_image_index, progress=print, with_gate: bool = False,
        top_k: int | None = None, dataset: str = "", use_clip: bool | None = None):
    ds_path = config.EVAL_DIR / (dataset or config.EVAL_DATASET)
    questions = json.loads(ds_path.read_text(encoding="utf-8"))
    top_k = top_k or config.RETRIEVAL_FINAL_TOP_K
    if use_clip is None:
        use_clip = config.CLIP_ENABLED

    from src.index_store import KnowledgeBase

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[实验] 图像解析消融，{len(questions)} 题 × {len(CONFIGS)} 配置，top_k={top_k}，"
          f"阈值闸门={'开' if with_gate else '关'}，CLIP 通道={'开' if use_clip else '关'}")
    print("[说明] 两份索引只有「图是否经多模态解析」一个变量不同；"
          "检索式一律用规则归一化的原问题\n")

    kb_image = KnowledgeBase.get()
    _ = kb_image.bm25                       # 预热：BM25 词表 + 无区分度短语表
    progress(f"[准备] 图像解析索引 {len(kb_image.chunks)} 块（图像块 "
             f"{sum(1 for c in kb_image.chunks if c['type']=='image')}）")
    kb_no = build_no_image(progress)

    kbs = {"image": kb_image, "no_image": kb_no}

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        kb = kbs[cfg.get("kb", "image")]
        # 对照臂**强制关掉 CLIP**：对照组不该因为多了一个可选通道而变得"更好"，
        # 否则 T0/T1 的差距里混进了 CLIP 的贡献，就不是单变量了。
        arm_clip = bool(cfg.get("clip")) and bool(use_clip)
        arm_prior = bool(cfg.get("prior"))
        per_case = []
        for q in questions:
            query = normalize_query(q["question"])
            r = retrieve(query, kb=kb, top_k=top_k, use_sparse=True, use_df_filter=True,
                         apply_gate=with_gate, use_clip=arm_clip,
                         use_image_prior=arm_prior)
            items = [
                {"text": x.text, "page": x.page, "page_end": x.page_end,
                 "evidence": x.evidence, "type": x.type, "doc_key": x.doc_key}
                for x in r.items
            ]
            s = score_case(items, q.get("must_have") or [], q.get("gold") or [],
                           q.get("must_have_counts") or [],
                           q.get("must_have_relations") or [])
            s["id"] = q["id"]
            s["source"] = q.get("source", "")
            s["question"] = q["question"]
            s["image_only"] = bool(q.get("image_only"))
            s["clip_hits"] = sum(1 for x in r.items if x.clip_sim > 0)
            per_case.append(s)
        results[cfg["key"]] = {**cfg, "use_clip": arm_clip, "use_image_prior": arm_prior,
                               "summary": summarize(per_case), "cases": per_case}
        s = results[cfg["key"]]["summary"]
        print(f"  [{cfg['key']:<12}] 关键事实 {s['fact_recall']:.3f} | "
              f"全命中 {s['all_facts_count']}/{len(questions)} | "
              f"正确页 {s['gold_page_hit_rate']:.3f} | MRR {s['mrr']:.3f} | "
              f"top-k 内图像块均 {s['avg_image_chunks']:.2f} | "
              f"归属关系 {_f(s['relation_recall'])} | "
              f"CLIP 命中块均 {_avg([c['clip_hits'] for c in per_case]) or 0:.2f}")

    # ---- 分层：图像题（答案只在图里） vs 表格题 vs 正文题
    #
    # 分层是必需的：图像题只有 2 道，混进 14 道表格/正文题里算平均，
    # 优化效果会被稀释到看不出来；反过来只报图像题又像在挑数据。
    # 两个口径都给出，读者自己判断。
    def subset(key: str, pred) -> dict:
        return summarize([c for c in results[key]["cases"] if pred(c)])

    groups = {}
    for label, pred in (
        ("图像题（答案只在图中：id 5 / 6）", lambda c: c["image_only"]),
        ("表格题（答案在表格中：id 1/2/3/4）", lambda c: c["source"].startswith("工单3")),
        ("正文题（答案在正文中：旧 10 题）", lambda c: c["source"] == "工单1/2"),
    ):
        if not any(pred(c) for c in results["T1_image"]["cases"]):
            continue
        groups[label] = {cfg["key"]: subset(cfg["key"], pred) for cfg in CONFIGS}

    # ---- 两组单变量对照
    contrasts = []
    for a, b, why in CONTRASTS:
        sa, sb = results[a]["summary"], results[b]["summary"]
        contrasts.append({
            "from": a, "to": b, "why": why,
            "var": {"T0_no_image": "图有没有被多模态模型解析",
                    "T1_image": "有没有启用低信息图排序惩罚",
                    "T2_image_prior": "有没有打开 CLIP 跨模态召回"}.get(a, ""),
            "fact_recall": _delta(sa["fact_recall"], sb["fact_recall"]),
            "all_facts_count": (sb["all_facts_count"] or 0) - (sa["all_facts_count"] or 0),
            "gold_page_hit_rate": _delta(sa["gold_page_hit_rate"], sb["gold_page_hit_rate"]),
            "mrr": _delta(sa["mrr"], sb["mrr"]),
            "mrr_pct": _pct(sa["mrr"], sb["mrr"]),
        })
    improvement = contrasts[0]     # 主结论（图像解析的净效果）

    # ---- 逐题对照（重点是排名变化）
    per_question = []
    for q in questions:
        row = {"id": q["id"], "image_only": bool(q.get("image_only")),
               "question": q["question"]}
        for cfg in CONFIGS:
            c = next(c for c in results[cfg["key"]]["cases"] if c["id"] == q["id"])
            row[cfg["key"]] = {
                "fact_recall": c["fact_recall"], "gold_rank": c["gold_rank"],
                "mrr": c["mrr"], "top_type": c["top_type"], "top_page": c["top_page"],
                "missed": c["missed"], "clip_hits": c["clip_hits"],
            }
        row["delta_mrr"] = round(row["T1_image"]["mrr"] - row["T0_no_image"]["mrr"], 4)
        row["delta_mrr_prior"] = round(
            row["T2_image_prior"]["mrr"] - row["T1_image"]["mrr"], 4)
        row["delta_mrr_clip"] = round(
            row["T3_image_clip"]["mrr"] - row["T2_image_prior"]["mrr"], 4)
        per_question.append(row)

    return {
        "work_order_no": WORK_ORDER_NO,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ds_path.name,
        "top_k": top_k,
        "with_gate": with_gate,
        "use_clip": use_clip,
        "num_questions": len(questions),
        "corpus": {
            "T1_image": {"chunks": len(kb_image.chunks),
                         "image_chunks": sum(1 for c in kb_image.chunks if c["type"] == "image"),
                         "table_chunks": sum(1 for c in kb_image.chunks if c["type"] == "table")},
            "T0_no_image": {"chunks": len(kb_no.chunks),
                            "image_chunks": sum(1 for c in kb_no.chunks if c["type"] == "image"),
                            "table_chunks": sum(1 for c in kb_no.chunks if c["type"] == "table")},
        },
        "configs": CONFIGS,
        "contrasts": contrasts,
        "results": results,
        "groups": groups,
        "improvement": improvement,
        "per_question": per_question,
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _pct(a, b):
    if not a or b is None:
        return None
    return round((b - a) / a * 100, 1)


# ------------------------------------------------------------------ 报告
def to_markdown(rep: dict) -> str:
    L = []
    A = L.append
    A("# 图像解析消融实验报告（工单4）\n")
    A(f"- 工单编号：{rep['work_order_no']}")
    A(f"- 生成时间：{rep['generated_at']}")
    A(f"- 题目数：{rep['num_questions']}（2 道图像题 + 4 道表格题 + 10 道正文题）")
    A(f"- top_k：{rep['top_k']}")
    A(f"- 阈值闸门：{'开' if rep['with_gate'] else '关（隔离检索本身的精确度）'}")
    A(f"- CLIP 跨模态通道：{'开' if rep['use_clip'] else '关'}")
    A(f"- 评测集：{rep['dataset']}\n")

    A("## 一、实验设计（两条单变量对照）\n")
    A("| # | 从 | 到 | 唯一变化的变量 | 要回答的问题 |")
    A("|---|---|---|---|---|")
    for i, c in enumerate(rep["contrasts"], 1):
        A(f"| {i} | `{c['from']}` | `{c['to']}` | {c.get('var', '')} | {c['why']} |")
    A("")
    A("| 配置 | 说明 |")
    A("|---|---|")
    for cfg in rep["configs"]:
        clip = "（CLIP 通道开）" if cfg.get("clip") else ""
        A(f"| **{cfg['key']}** | {cfg['name']}{clip}：{cfg['desc']} |")

    A("\n## 二、语料规模\n")
    A("| 配置 | 分块总数 | 图像块 | 表格块 |")
    A("|---|---|---|---|")
    for k in ("T1_image", "T0_no_image"):
        c = rep["corpus"][k]
        A(f"| {k} | {c['chunks']} | {c['image_chunks']} | {c['table_chunks']} |")
    A("")
    A("> 两份索引的解析 / 分块 / 检索算法完全一致，只有「图有没有经过多模态解析」不同。")
    A("> 对照组分块数**略少**是正常的：图内散字与图题在对照组里仍留在正文段落中，")
    A("> 而主线把它们剔掉、换成等量的图像块，两种写法下段落边界的切法略有差别。")

    A("\n## 三、总体检索精确度\n")
    A("| 配置 | 关键事实命中率 | 全部命中题数 | 正确页命中率 | MRR | top1 平均依据分 | top-k 内图像块 |")
    A("|---|---|---|---|---|---|---|")
    for cfg in rep["configs"]:
        s = rep["results"][cfg["key"]]["summary"]
        A(f"| **{cfg['key']}** {cfg['name']} | {_f(s['fact_recall'])} | "
          f"{s['all_facts_count']}/{rep['num_questions']} | {_f(s['gold_page_hit_rate'])} | "
          f"**{_f(s['mrr'])}** | {_f(s['avg_top_evidence'], 4)} | {_f(s['avg_image_chunks'])} |")

    A("\n### 提升量\n")
    A("| 对照 | 关键事实命中率 | 全部命中题数 | 正确页命中率 | MRR |")
    A("|---|---|---|---|---|")
    for c in rep["contrasts"]:
        A(f"| `{c['from']}` → `{c['to']}` | {_s(c['fact_recall'])} | "
          f"{c['all_facts_count']:+d} | {_s(c['gold_page_hit_rate'])} | "
          f"{_s(c['mrr'])}（{_sp(c['mrr_pct'])}） |")

    A("\n## 四、分层结果\n")
    for label, g in rep["groups"].items():
        A(f"### {label}\n")
        A("| 配置 | 关键事实命中率 | 全部命中题数 | 正确页命中率 | MRR |")
        A("|---|---|---|---|---|")
        for cfg in rep["configs"]:
            s = g.get(cfg["key"])
            if not s:
                continue
            A(f"| {cfg['key']} | {_f(s['fact_recall'])} | "
              f"{s['all_facts_count']}/{s['num_cases']} | "
              f"{_f(s['gold_page_hit_rate'])} | {_f(s['mrr'])} |")
        A("")

    A("## 五、逐题对照\n")
    A("| id | 图像题 | 问题 | T0 命中 | T0 排名 | T1 命中 | T1 排名 | T2 命中 | T2 排名 | "
      "T3 命中 | T3 排名 | ΔMRR(解析) | ΔMRR(惩罚) | ΔMRR(CLIP) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rep["per_question"]:
        q = r["question"][:32].replace("武汉力源信息技术股份有限公司", "力源").replace(
            "武汉兴图新科电子股份有限公司", "兴图")
        t0, t1 = r["T0_no_image"], r["T1_image"]
        t2, t3 = r["T2_image_prior"], r["T3_image_clip"]
        A(f"| {r['id']} | {'★' if r['image_only'] else ''} | {q} | "
          f"{_f(t0['fact_recall'])} | {_rank(t0['gold_rank'])} | "
          f"{_f(t1['fact_recall'])} | {_rank(t1['gold_rank'])} | "
          f"{_f(t2['fact_recall'])} | {_rank(t2['gold_rank'])} | "
          f"{_f(t3['fact_recall'])} | {_rank(t3['gold_rank'])} | "
          f"{r['delta_mrr']:+.3f} | {r['delta_mrr_prior']:+.3f} | {r['delta_mrr_clip']:+.3f} |")

    A("\n## 六、未命中事实明细\n")
    any_miss = False
    for r in rep["per_question"]:
        for arm in ("T0_no_image", "T2_image_prior"):
            if r[arm]["missed"]:
                any_miss = True
                A(f"- id={r['id']} [{arm}] 未命中：{'、'.join(r[arm]['missed'])}")
    if not any_miss:
        A(f"- 无（{rep['num_questions']} 题的关键事实在两条链路上均全部命中）")
    return "\n".join(L)


def _f(v, nd: int = 3):
    return "-" if v is None else f"{v:.{nd}f}"


def _s(v):
    return "-" if v is None else f"{v:+.3f}"


def _sp(v):
    return "-" if v is None else f"{v:+.1f}%"


def _rank(v):
    return "-" if not v else f"第 {v} 位"


def save_report(rep: dict, tag: str = "") -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = config.EVAL_DIR / f"工单4_图像解析消融{tag}_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    md = base.with_suffix(".md")
    md.write_text(to_markdown(rep), encoding="utf-8")
    return md, base.with_suffix(".json")


def main() -> int:
    ap = argparse.ArgumentParser(description=f"图像解析消融 | 工单：{WORK_ORDER_NO}")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--with-gate", action="store_true")
    ap.add_argument("--clip", action="store_true", help="打开 CLIP 跨模态召回通道")
    ap.add_argument("--dataset", default="")
    args = ap.parse_args()

    rep = run(with_gate=args.with_gate, top_k=args.top_k or None,
              dataset=args.dataset, use_clip=args.clip or config.CLIP_ENABLED)
    tag = ""
    if args.with_gate:
        tag += "_带闸门"
    if args.clip:
        tag += "_CLIP"
    md, js = save_report(rep, tag)
    print(f"\n[报告] {md.relative_to(config.ROOT_DIR)}")
    print(f"[报告] {js.relative_to(config.ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
