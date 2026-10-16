# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""数据集生成：从工单7 真实语料分块构造 Embedding 微调数据集（只读语料）。

覆盖工单原文《1、微调数据集格式与损失函数基础》点名的三种格式：
  1) 正例对   (问题, 证据段)            —— 格式="正例对"
  2) 三元组   (锚点, 正例, 负例)         —— 格式="三元组"
  3) 带相似度分数的句子对 (句子A, 句子B, 分数) —— 格式="相似度分数对"

锚点来源（全部为语料内的弱监督，**绝不使用工单7 的 17 道测试问题**，避免测试集泄漏）：
  - 指标数值：从分块中抽取「财务指标 + 数值」事实，模板化为自然语言问题；
  - 章节标题：分块 section 标题 -> 正文；
  - 表格标题：表格块的表头/表体 -> 表格内容。

数据划分：按 **页** 留出（默认 12% 页面整体不进训练集），保证 dev/test 的正例分块
在训练中从未出现；负例挖掘分「难（同指标异公司）/ 中（同公司异页）/ 随机（异公司）」。

产出：
  研发/data/dataset/{train,dev,test}.jsonl
  研发/data/dataset/数据集统计.json
  研发/data/dataset/留出测试集.jsonl（检索评测用：查询 + 唯一正例分块 id）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402
from text_utils import clean_display, jaccard, normalize_text, parse_doc_name, sentences  # noqa: E402

# --------------------------------------------------------------- 财务指标词表
# 顺序敏感：长指标优先匹配，避免「净利润」抢走「归母净利润」。
METRICS: tuple[str, ...] = (
    "归属于母公司股东的扣除非经常性损益的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "扣除非经常性损益后归属于母公司股东的净利润",
    "归属于母公司股东的净利润", "归属于上市公司股东的净利润", "归属于本行股东的净利润",
    "归属于母公司普通股股东的净利润", "扣除非经常性损益的净利润", "归母净利润",
    "经营活动产生的现金流量净额", "现金及现金等价物净增加额", "手续费及佣金净收入",
    "归属于母公司股东权益", "归属于母公司所有者权益", "综合偿付能力充足率",
    "核心偿付能力充足率", "偿付能力充足率", "核心一级资本充足率", "一级资本充足率",
    "加权平均净资产收益率", "逾期60天以上贷款占比", "逾期90天以上贷款占比",
    "逾期贷款占比", "关注类贷款占比", "不良贷款率", "拨备覆盖率", "拨贷比",
    "净利息收益率", "净利差", "净资产收益率", "总资产收益率", "成本收入比",
    "综合成本率", "退保率", "总投资收益率", "净投资收益率", "资本充足率",
    "营业收入", "营业总收入", "利息净收入", "已赚保费", "保险业务收入",
    "利润总额", "净利润", "总资产", "资产总额", "总负债", "负债总额",
    "股东权益", "每股净资产", "基本每股收益", "稀释每股收益", "每股现金分红",
    "每股现金股息", "每股股息", "信息科技投入", "研发投入", "总股本", "员工人数",
    "营业网点数量", "客户数", "市场份额", "投资收益", "营业利润", "营业支出",
    "业务及管理费", "资产减值损失", "信用减值损失", "商誉",
)

# 兜底：任意「XX率」+ 百分号
FALLBACK_RATE = re.compile(r"([\u4e00-\u9fa5]{2,12}率)\s*(?:为|是|达到|达|约|上升至|下降至)?\s*[:：]?\s*([0-9][0-9,]*\.?[0-9]*)\s*[%％]")
NUM = re.compile(r"([0-9][0-9,]*\.?[0-9]*)\s*(亿元|万元|百万元|千元|万亿|元|%|％|个百分点)?")
BAD_METRIC_CHARS = ("年", "月", "日", "月未", "月份")

SPLIT_NAMES = ("train", "dev", "test")


# ------------------------------------------------------------------ 读取语料
def load_chunks(path: Path, *, min_chars: int, max_chars: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = {"分块总数": 0, "类型分布": Counter(), "过滤后分块数": 0, "损坏行数": 0}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                stats["损坏行数"] += 1
                continue
            stats["分块总数"] += 1
            stats["类型分布"][c.get("type", "?")] += 1
            content = clean_display(c.get("content") or "")
            if c.get("type") not in ("text", "table"):
                continue
            if not (min_chars <= len(content) <= max_chars):
                continue
            doc = parse_doc_name(c.get("file_name", ""))
            c["_content"] = content
            c["_doc"] = doc
            c["_doc_label"] = f"{doc['简称']}{doc['年份']}年度报告" if doc["简称"] else doc["全称"]
            c["_page"] = int(c.get("page") or 0)
            rows.append(c)
            stats["过滤后分块数"] += 1
    stats["类型分布"] = dict(stats["类型分布"])
    return rows, stats


# -------------------------------------------------------------- 事实抽取
def _window_after(text: str, start: int) -> str:
    """取指标名之后的「同格」窗口：优先截断到下一个竖线/换行。"""
    seg = text[start : start + 40]
    cut = len(seg)
    for ch in ("|", "\n"):
        i = seg.find(ch)
        if 0 <= i < cut:
            cut = i
    seg = seg[:cut]
    if len(seg) < 3:  # 表格里指标名后紧跟竖线，换到下一格取值
        seg2 = text[start + cut : start + cut + 40]
        cut2 = len(seg2)
        for ch in ("|", "\n"):
            i = seg2.find(ch)
            if 0 <= i < cut2:
                cut2 = i
        seg = seg2[:cut2]
    return seg


def _pick_number(seg: str) -> str | None:
    for m in NUM.finditer(seg):
        raw, unit = m.group(1), m.group(2)
        if not unit:
            try:
                fv = float(raw.replace(",", ""))
            except ValueError:
                continue
            if 1900 <= fv <= 2100 and "," not in raw:  # 年份等非数值
                continue
            if "." not in raw and "," not in raw and fv < 1000:
                continue
        if unit in ("%", "％"):
            return f"{raw}%"
        return f"{raw}{unit}" if unit else raw
    return None


def extract_facts(chunk: dict, max_facts: int = 3) -> list[dict]:
    """从一个分块抽取最多 ``max_facts`` 条「指标 + 数值」事实。"""
    text = chunk["_content"]
    norm = normalize_text(text)
    facts: list[dict] = []
    seen: set[str] = set()

    def push(metric: str, value: str, raw_metric: str) -> None:
        key = f"{metric}|{value}"
        if key in seen:
            return
        seen.add(key)
        facts.append({"指标": metric, "数值": value, "原文指标": raw_metric})

    for metric in METRICS:
        if len(facts) >= max_facts:
            break
        key = normalize_text(metric)
        if key not in norm:
            continue
        for m in re.finditer(re.escape(metric), text):
            value = _pick_number(_window_after(text, m.end()))
            if value:
                push(metric, value, metric)
                break
    # 兜底：任意 XX率 = 数字%
    if len(facts) < max_facts:
        for m in FALLBACK_RATE.finditer(text):
            metric = m.group(1)
            if any(b in metric for b in BAD_METRIC_CHARS):
                continue
            if any(metric == f["指标"] for f in facts):
                continue
            push(metric, f"{m.group(2)}%", metric)
            if len(facts) >= max_facts:
                break
    return facts


RATIO_SUFFIX = ("率", "比", "占比", "覆盖", "充足")
AMOUNT_SUFFIX = ("收入", "利润", "资产", "负债", "权益", "保费", "投入", "余额", "净额", "损失", "分红", "股息", "股本", "费用", "收益")


def fact_query(doc_label: str, short: str, year: str, metric: str, rng: random.Random) -> str:
    """把「指标+数值」事实模板化为自然语言问题（数值不出现在问题里）。"""
    y = year.replace("年", "")
    is_ratio = metric.endswith(RATIO_SUFFIX)
    is_amount = metric.endswith(AMOUNT_SUFFIX)
    templates = []
    if is_ratio:
        templates = [
            f"{short}{y}年年度报告期末的{metric}是多少？",
            f"{short}{y}年末{metric}为多少？",
            f"{doc_label}中披露的{metric}是多少？",
        ]
    elif is_amount:
        templates = [
            f"{short}{y}年的{metric}是多少？",
            f"{short}{y}年年度报告中，{metric}的具体金额是多少？",
            f"{doc_label}披露的{metric}是多少？",
        ]
    else:
        templates = [
            f"{short}{y}年年度报告中，{metric}是多少？",
            f"{short}{y}年的{metric}为多少？",
            f"{doc_label}中{metric}的数值是多少？",
        ]
    return rng.choice(templates)


def section_query(doc_label: str, section: str, rng: random.Random) -> str:
    section = section.strip()
    templates = [
        f"{doc_label}中「{section}」章节的主要内容是什么？",
        f"{doc_label}关于{section}有哪些说明？",
        f"{doc_label}的{section}部分披露了哪些信息？",
    ]
    return rng.choice(templates)


def table_query(doc_label: str, page: int, head_cells: list[str], rng: random.Random) -> str | None:
    cells = [c for c in head_cells if c and len(c) <= 12][:3]
    if not cells:
        return None
    joined = "、".join(cells)
    templates = [
        f"{doc_label}第{page}页表格中，{joined}的数据是多少？",
        f"{doc_label}中关于{joined}的表格内容是什么？",
    ]
    return rng.choice(templates)


def table_head_cells(content: str) -> list[str]:
    """从 Markdown 表格中取表头单元格。

    注意：工单7 的分块正文把换行压成了空格，因此不能按行解析，
    改为按 ``|`` 切分后过滤分隔行/空串/纯数字。
    """
    m = re.search(r"\[Markdown\](.*)", content, flags=re.S)
    body = m.group(1) if m else content
    out: list[str] = []
    for cell in body.split("|"):
        cell = cell.strip()
        if not cell or set(cell) <= set("-: "):
            continue
        if cell.isdigit() or len(cell) > 12:
            continue
        out.append(cell)
        if len(out) >= 3:
            break
    return out


# ------------------------------------------------------------------ 主流程
def build_dataset(cfg: Config) -> dict[str, Any]:
    t0 = time.time()
    d_cfg = cfg.hyper["数据集"]
    seed = int(d_cfg.get("随机种子", 20260826))
    rng = random.Random(seed)
    min_chars = int(d_cfg.get("最小分块字符数", 24))
    max_chars = int(d_cfg.get("最大分块字符数", 600))
    holdout_ratio = float(d_cfg.get("留出分块比例", 0.12))
    n_neg = int(d_cfg.get("每个锚点负例数", 2))
    caps = {"train": int(d_cfg.get("训练三元组上限", 2400)),
            "dev": int(d_cfg.get("验证三元组上限", 300)),
            "test": int(d_cfg.get("测试三元组上限", 300))}

    src = cfg.path("语料分块")
    chunks, load_stats = load_chunks(src, min_chars=min_chars, max_chars=max_chars)
    print(f"[数据] 读入分块 {load_stats['分块总数']}，可用（text/table 且长度合规）{len(chunks)}")

    # 页级留出：按 (文档, 页) 划分
    pages_by_doc: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        pages_by_doc[c["_doc"]["file_name"]].append(c["_page"])
    holdout_pages: dict[str, set[int]] = {}
    for fn, pages in pages_by_doc.items():
        uniq = sorted(set(pages))
        rng.shuffle(uniq)
        k = max(1, int(round(len(uniq) * holdout_ratio)))
        holdout_pages[fn] = set(uniq[:k])
    n_holdout_pages = sum(len(v) for v in holdout_pages.values())
    print(f"[数据] 页级留出：{n_holdout_pages} 页 / {sum(len(set(v)) for v in pages_by_doc.values())} 页")

    # 指标倒排：指标 -> [(chunk, 数值)]
    metric_index: dict[str, list[tuple[dict, str]]] = defaultdict(list)

    # ---------- 生成锚点 ----------
    anchors: list[dict] = []
    for c in chunks:
        doc = c["_doc"]
        label = c["_doc_label"]
        short, year = doc["简称"], doc["年份"]
        if not short:
            continue
        in_holdout = c["_page"] in holdout_pages.get(doc["file_name"], set())
        facts = extract_facts(c)
        for f in facts:
            metric_index[f["指标"]].append((c, f["数值"]))
        for f in facts[:2]:
            anchors.append({
                "来源": "指标数值",
                "anchor": fact_query(label, short, year, f["指标"], rng),
                "正向分块": c["chunk_id"], "文档": label, "页码": c["_page"],
                "指标": f["指标"], "数值": f["数值"], "留出": in_holdout,
            })
        sec = clean_display(c.get("section") or "", 40)
        if sec and 2 <= len(sec) <= 24:
            anchors.append({
                "来源": "章节标题",
                "anchor": section_query(label, sec, rng),
                "正向分块": c["chunk_id"], "文档": label, "页码": c["_page"],
                "指标": None, "数值": None, "留出": in_holdout,
            })
        if c.get("type") == "table":
            heads = table_head_cells(c["_content"])
            q = table_query(label, c["_page"], heads, rng) if heads else None
            if q:
                anchors.append({
                    "来源": "表格标题", "anchor": q, "正向分块": c["chunk_id"],
                    "文档": label, "页码": c["_page"], "指标": None, "数值": None,
                    "留出": in_holdout,
                })

    # 去重（同一 anchor 文本只保留一条）
    seen_anchor: set[str] = set()
    dedup: list[dict] = []
    dup = 0
    for a in anchors:
        k = normalize_text(a["anchor"]) + "|" + a["正向分块"]
        if k in seen_anchor:
            dup += 1
            continue
        seen_anchor.add(k)
        dedup.append(a)
    anchors = dedup
    print(f"[数据] 锚点 {len(anchors)} 条（去重丢弃 {dup} 条）；覆盖指标 {len(metric_index)} 个")

    by_id = {c["chunk_id"]: c for c in chunks}
    # 每文档锚点数配平：训练组与留出组分别限流，并按来源配额保证配比
    # （指标数值查询最接近真实用户问题，配额最高）
    quota_train = d_cfg.get("每文档配额", {"指标数值": 130, "章节标题": 60, "表格标题": 30})
    quota_hold = d_cfg.get("留出每文档配额", {"指标数值": 25, "章节标题": 13, "表格标题": 5})
    keep: list[dict] = []
    doc_used: Counter = Counter()  # 键 = (文档, 是否留出, 来源)
    for a in sorted(anchors, key=lambda x: (not x["留出"], x["文档"], x["来源"], x["正向分块"])):
        table = quota_hold if a["留出"] else quota_train
        key = (a["文档"], a["留出"], a["来源"])
        if doc_used[key] >= int(table.get(a["来源"], 0)):
            continue
        doc_used[key] += 1
        keep.append(a)
    anchors = keep
    print(f"[数据] 按来源配额后锚点 {len(anchors)} 条；"
          f"留出锚点 {sum(1 for a in anchors if a['留出'])} 条；"
          f"来源分布 {dict(Counter(a['来源'] for a in anchors))}；"
          f"文档分布 {dict(Counter(a['文档'] for a in anchors))}")

    # ---------- 负例挖掘 ----------
    # 关键防泄漏：负例一律只从「非留出页」中抽取，保证留出测试集的分块
    # 在训练数据（正例与负例）中均从未出现。
    def _is_holdout(c: dict) -> bool:
        return c["_page"] in holdout_pages.get(c["_doc"]["file_name"], set())

    neg_pool = [c for c in chunks if not _is_holdout(c)]
    print(f"[数据] 负例候选池（非留出页）{len(neg_pool)} 块")

    def is_same_doc(a: dict, c: dict) -> bool:
        return c["_doc_label"] == a["文档"]

    def find_negative(a: dict, pool: list[dict], prefer: str = "hard") -> tuple[dict | None, str]:
        """负例挖掘：hard=同指标异公司；medium=同公司异页；random=异公司。

        ``prefer`` 决定优先返回哪一种（都取自「非留出页」候选池，防泄漏）。
        """
        pos = by_id.get(a["正向分块"])
        metric = a.get("指标")
        hard = medium = rand = None
        if metric and d_cfg.get("难负例同指标异公司", True):
            cands = [c for c, _v in metric_index.get(metric, []) if not is_same_doc(a, c) and not _is_holdout(c)]
            if cands:
                hard = rng.choice(cands)
        if pos is not None:
            cands = [c for c in pool if is_same_doc(a, c) and c["_page"] != pos["_page"]]
            if cands:
                medium = rng.choice(cands)
        cands = [c for c in pool if not is_same_doc(a, c)]
        if cands:
            rand = rng.choice(cands)
        order = {"hard": ((hard, "难负例同指标异公司"), (medium, "中负例同公司异页"), (rand, "随机负例异公司")),
                 "medium": ((medium, "中负例同公司异页"), (hard, "难负例同指标异公司"), (rand, "随机负例异公司")),
                 "random": ((rand, "随机负例异公司"), (medium, "中负例同公司异页"), (hard, "难负例同指标异公司"))}
        for c, kind in order.get(prefer, order["hard"]):
            if c is not None:
                return c, kind
        return None, "无可用负例"

    # ---------- 组装记录 ----------
    records: dict[str, list[dict]] = {k: [] for k in SPLIT_NAMES}
    holdout_anchors = [a for a in anchors if a["留出"]]
    train_anchors = [a for a in anchors if not a["留出"]]
    rng.shuffle(holdout_anchors)
    rng.shuffle(train_anchors)

    # dev/test 以三元组条数上限为依据反推锚点数
    n_dev_anchor = max(1, caps["dev"] // max(1, n_neg))
    n_test_anchor = max(1, caps["test"] // max(1, n_neg))
    split_anchor = {
        "train": train_anchors[: caps["train"] // max(1, n_neg)],
        "dev": holdout_anchors[:n_dev_anchor],
        "test": holdout_anchors[n_dev_anchor : n_dev_anchor + n_test_anchor],
    }

    counters = {k: Counter() for k in SPLIT_NAMES}
    neg_kind = Counter()
    lengths = defaultdict(list)
    for split, items in split_anchor.items():
        idx = 0
        for a in items:
            pos = by_id.get(a["正向分块"])
            if pos is None:
                continue
            idx += 1
            base = {"split": split, "锚点序号": idx, "来源": a["来源"], "文档": a["文档"], "页码": a["页码"],
                    "指标": a["指标"], "数值": a["数值"], "正向分块": a["正向分块"]}
            # 格式1：正例对
            if d_cfg.get("包含正例对", True):
                rec = dict(base, id=f"{split}-pair-{idx:05d}", **{"格式": "正例对"},
                           anchor=a["anchor"], positive=pos["_content"], negative=None, score=1.0)
                records[split].append(rec)
                counters[split]["正例对"] += 1
            # 格式2/3：三元组 + 相似度分数对
            for j in range(n_neg):
                neg, kind = find_negative(a, neg_pool, prefer="hard" if j == 0 else "medium")
                if neg is None:
                    continue
                neg_kind[kind] += 1
                if d_cfg.get("包含三元组", True):
                    rec = dict(base, id=f"{split}-tri-{idx:05d}-{j}", **{"格式": "三元组"},
                               anchor=a["anchor"], positive=pos["_content"],
                               negative=neg["_content"], score=None,
                               **{"负向分块": neg["chunk_id"], "负例类型": kind,
                                  "负例分数": round(jaccard(a["anchor"], neg["_content"]), 4)})
                    records[split].append(rec)
                    counters[split]["三元组"] += 1
                if d_cfg.get("包含相似度分数对", True):
                    s_pos = 1.0
                    rec = dict(base, id=f"{split}-sim-{idx:05d}-{j}", **{"格式": "相似度分数对"},
                               anchor=a["anchor"], positive=pos["_content"],
                               negative=neg["_content"], score=round(jaccard(a["anchor"], neg["_content"]), 4),
                               **{"负向分块": neg["chunk_id"], "正例分数": s_pos, "负例类型": kind})
                    records[split].append(rec)
                    rec2 = dict(base, id=f"{split}-simpos-{idx:05d}-{j}", **{"格式": "相似度分数对"},
                                anchor=a["anchor"], positive=pos["_content"], negative=None,
                                score=s_pos, **{"正例分数": s_pos, "负例类型": "正例"})
                    records[split].append(rec2)
                    counters[split]["相似度分数对"] += 2
            lengths["anchor"].append(len(a["anchor"]))
            lengths["positive"].append(len(pos["_content"]))

    # 控制各 split 规模到上限（按格式配额裁剪）
    for split in SPLIT_NAMES:
        cap = caps[split]
        if split == "train":
            recs = records[split]
            tri = [r for r in recs if r["格式"] == "三元组"]
            pair = [r for r in recs if r["格式"] == "正例对"]
            sim = [r for r in recs if r["格式"] == "相似度分数对"]
            rng.shuffle(pair); rng.shuffle(sim)
            records[split] = tri[:cap] + pair[: max(1, cap // 2)] + sim[: cap]
            counters[split] = Counter(r["格式"] for r in records[split])
        else:
            recs = records[split]
            tri = [r for r in recs if r["格式"] == "三元组"][: caps[split]]
            keep_idx = {r["锚点序号"] for r in tri}
            others = [r for r in recs if r["格式"] != "三元组" and r["锚点序号"] in keep_idx]
            records[split] = tri + others
            counters[split] = Counter(r["格式"] for r in records[split])

    # ---------- 留出检索测试集 ----------
    retrieval_set = []
    for a in split_anchor["test"]:
        retrieval_set.append({
            "id": f"ret-{len(retrieval_set)+1:04d}", "query": a["anchor"],
            "gold_chunk_id": a["正向分块"], "来源": a["来源"], "文档": a["文档"],
            "页码": a["页码"], "指标": a["指标"],
        })

    # ---------- 写出 ----------
    out_dir = cfg.path("数据集目录")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        p = out_dir / f"{split}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in records[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[数据] {p}  {len(records[split])} 条")

    ret_path = out_dir / "留出测试集.jsonl"
    with ret_path.open("w", encoding="utf-8") as fh:
        for r in retrieval_set:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def q(vals: list[int], p: float) -> int:
        if not vals:
            return 0
        v = sorted(vals)
        return v[min(len(v) - 1, int(len(v) * p))]

    stats = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "生成耗时秒": round(time.time() - t0, 1),
        "随机种子": seed,
        "输入语料": str(src),
        "语料分块总数": load_stats["分块总数"],
        "语料类型分布": load_stats["类型分布"],
        "参与生成分块数": load_stats["过滤后分块数"],
        "语料损坏行数": load_stats["损坏行数"],
        "锚点总数": len(anchors),
        "指标覆盖数": len(metric_index),
        "页级留出": {"留出页数": n_holdout_pages, "总页数": sum(len(set(v)) for v in pages_by_doc.values()),
                     "留出比例配置": holdout_ratio},
        "各split记录数": {k: len(records[k]) for k in SPLIT_NAMES},
        "各split格式分布": {k: dict(counters[k]) for k in SPLIT_NAMES},
        "各split文档分布": {k: dict(Counter(r["文档"] for r in records[k])) for k in SPLIT_NAMES},
        "锚点来源分布": dict(Counter(a["来源"] for a in anchors)),
        "负例类型分布": dict(neg_kind),
        "留出检索测试集条数": len(retrieval_set),
        "文本长度": {
            "anchor": {"中位": q(lengths["anchor"], 0.5), "p90": q(lengths["anchor"], 0.9), "最大": max(lengths["anchor"] or [0])},
            "positive": {"中位": q(lengths["positive"], 0.5), "p90": q(lengths["positive"], 0.9), "最大": max(lengths["positive"] or [0])},
        },
        "配置快照": d_cfg,
        "文件MD5": {},
    }
    for split in SPLIT_NAMES:
        p = out_dir / f"{split}.jsonl"
        stats["文件MD5"][p.name] = hashlib.md5(p.read_bytes()).hexdigest()
    stats["文件MD5"][ret_path.name] = hashlib.md5(ret_path.read_bytes()).hexdigest()

    (out_dir / "数据集统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[数据] 统计 -> {out_dir / '数据集统计.json'}")
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单11 数据集生成")
    ap.add_argument("--统计", action="store_true", help="仅打印统计")
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = Config()
    cfg.ensure_dirs()
    stats = build_dataset(cfg)
    if args.统计:
        print(json.dumps(stats, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
