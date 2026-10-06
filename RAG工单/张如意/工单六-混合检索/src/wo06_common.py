# -*- coding: utf-8 -*-
"""
工单06 公共模块：评测问题集 / 检索指标 / 报告输出
工单编号：人工智能NLP-RAG-混合检索任务

本模块被以下实验脚本复用（不直接运行）：
  · rerank_comparison.py             —— 3 种重排算法 + 级联的对比实验
  · retrieval_strategy_comparison.py —— 5 种检索策略的准确率/召回率对比
  · adaptive_feedback_demo.py        —— 基于用户反馈的自适应重排演示

指标口径（对齐工单「准确率≥90%、召回率≥95%」的验收要求）：
  · 准确率 Accuracy@k —— top-k 片段覆盖该问题【全部关键信息点】的问题占比。
    含义：只有全部关键信息点都进了上下文，生成阶段才可能答对，
    因此它是「问答准确率」在检索层的可自动计算代理指标。
  · 召回率 Recall@k   —— 被 top-k 片段覆盖的【关键信息点】比例（重要信息召回）。
  · 命中率 HitRate@k  —— top-k 中至少命中 1 个相关片段的问题占比。
  · MRR / Precision@k —— 标准检索排序指标。

相关片段判定：片段属于参考答案所在文档，且至少包含一个关键信息点。
所有关键信息点（含金额、比例、人名等）均已对照《招股说明书1/2》原文核实，
可在 demo/演示脚本.md 的「检索精度提升测试问题及检索答案」一节逐条查看。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# 路径与全局参数（与 rag_core.config 保持一致）
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5            # 最终返回条数（送入生成模型的上下文片段数）
RECALL_K = 20        # 召回阶段条数（重排前候选池大小）
ACC_TARGET = 0.90    # 工单准确率目标
REC_RATIO_TARGET = 0.95   # 工单召回率目标


# ---------------------------------------------------------------------------
# 评测问题集
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalQuestion:
    qid: int
    question: str
    doc: str                   # 参考答案所在文档（与建索引时的 doc_name 一致）
    anchors: tuple[str, ...]   # 关键信息点（答案必须包含的锚点词/数值）
    note: str = ""             # 备注（答案出处/口径说明）


WO06_QA_SET: list[EvalQuestion] = [
    # ---- 《招股说明书1》武汉兴图新科（军工） ----
    EvalQuestion(260, "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                 "招股说明书1", ("军用领域", "6,464.51"),
                 "原文 p129：6,464.51/14,414.16/18,780.67/4,627.14 万元（四期）"),
    EvalQuestion(95, "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
                 "招股说明书1", ("视频指挥系统技术标准",),
                 "原文 p27：全军第一个视频指挥系统技术标准《某视频技术规范1.0》"),
    EvalQuestion(33, "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
                 "招股说明书1", ("军用领域", "82.10%"),
                 "原文 p129：82.10%/97.31%/94.84%/94.34%"),
    EvalQuestion(34, "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
                 "招股说明书1", ("上游", "电子元器件"),
                 "原文 p152：电子元器件制造企业、机箱机柜等金属壳体制造企业"),
    EvalQuestion(957, "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
                 "招股说明书1", ("重要供应商", "视频指挥"),
                 "原文 p27：军队视频指挥领域的重要供应商"),
    EvalQuestion(793, "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
                 "招股说明书1", ("下游行业",),
                 "原文 p152：军队、政府机关、能源等行业企业"),
    EvalQuestion(795, "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
                 "招股说明书1", ("国家科技进步一等奖",),
                 "原文 p27：某情报、指挥、控制与通信网络一体化工程（C4ISR）"),
    EvalQuestion(543, "武汉兴图新科电子股份有限公司注册资本是多少？",
                 "招股说明书1", ("注册资本", "5,520"),
                 "原文 p22/p52：5,520.00 万元"),
    EvalQuestion(531, "武汉兴图新科电子股份有限公司法定代表人是谁？",
                 "招股说明书1", ("法定代表人", "程家明"),
                 "原文 p22：法定代表人 程家明"),
    EvalQuestion(207, "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
                 "招股说明书1", ("补充流动资金", "15,000.00"),
                 "原文 p30 募集资金用途表：补充流动资金 15,000.00 万元"),
    # ---- 《招股说明书2》武汉力源信息 ----
    EvalQuestion(1, "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
                 "招股说明书2", ("发行股数", "发行后总股本"),
                 "原文：本次发行概况/股本结构章节"),
    EvalQuestion(2, "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
                 "招股说明书2", ("募集资金投资项目",),
                 "原文：募集资金运用章节"),
    EvalQuestion(3, "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
                 "招股说明书2", ("赵马克", "42.35%"),
                 "原文 p157：赵马克 42.35% 控股股东"),
    EvalQuestion(4, "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
                 "招股说明书2", ("融冰投资", "武汉博润"),
                 "原文 p157：融冰投资/武汉博润/上海博润/听音投资/联众聚源/力源贸易/普芯达等"),
]


def get_question(qid: int) -> EvalQuestion | None:
    """按 id 取问题。"""
    for q in WO06_QA_SET:
        if q.qid == qid:
            return q
    return None


# ---------------------------------------------------------------------------
# 相关性 / 覆盖度计算
# ---------------------------------------------------------------------------
def anchor_hit(doc: dict, anchor: str) -> bool:
    """锚点词是否出现在片段的正文或章节路径中。"""
    text = (doc.get("text") or "") + " " + (doc.get("section") or "")
    return anchor in text


def covered_anchors(docs: list[dict], q: EvalQuestion) -> list[bool]:
    """逐个关键信息点判断是否被 top-k 片段覆盖。"""
    return [any(anchor_hit(d, a) for d in docs) for a in q.anchors]


def is_relevant(doc: dict, q: EvalQuestion) -> bool:
    """相关片段 = 属于参考答案文档 且 至少命中一个关键信息点。"""
    if q.doc and q.doc not in (doc.get("doc") or ""):
        return False
    return any(anchor_hit(doc, a) for a in q.anchors)


def question_passed(docs: list[dict], q: EvalQuestion) -> bool:
    """该问题是否「检索可答对」：top-k 覆盖全部关键信息点。"""
    cov = covered_anchors(docs, q)
    return bool(cov) and all(cov)


# ---------------------------------------------------------------------------
# 指标汇总
# ---------------------------------------------------------------------------
def evaluate(docs_by_qid: dict[int, list[dict]],
             latencies: dict[int, float] | None = None,
             k: int = TOP_K) -> dict:
    """
    汇总一次实验的检索指标。

    Args:
        docs_by_qid: {qid: 检索结果片段列表（按排名）}
        latencies:   {qid: 该问题检索耗时（秒）}
        k:           截断位置（与 TOP_K 一致）
    """
    acc, recall, hit, mrr, prec = [], [], [], [], []
    for q in WO06_QA_SET:
        docs = (docs_by_qid.get(q.qid) or [])[:k]
        cov = covered_anchors(docs, q) if docs else [False] * len(q.anchors)
        recall.append(sum(cov) / len(cov))
        acc.append(1.0 if (docs and all(cov)) else 0.0)

        rel_flags = [is_relevant(d, q) for d in docs]
        rel = [d for d, f in zip(docs, rel_flags) if f]
        hit.append(1.0 if rel else 0.0)
        prec.append(len(rel) / len(docs) if docs else 0.0)
        if rel:
            first = next(i for i, f in enumerate(rel_flags, 1) if f)
            mrr.append(1.0 / first)
        else:
            mrr.append(0.0)

    out = {
        "n": len(WO06_QA_SET),
        "k": k,
        "accuracy": round(mean(acc), 4),
        "recall": round(mean(recall), 4),
        "hit_rate": round(mean(hit), 4),
        "mrr": round(mean(mrr), 4),
        "precision": round(mean(prec), 4),
    }
    if latencies:
        vals = [latencies[q.qid] for q in WO06_QA_SET if q.qid in latencies]
        if vals:
            out["latency_avg"] = round(mean(vals), 3)
            out["latency_max"] = round(max(vals), 3)
    return out


# ---------------------------------------------------------------------------
# 报告输出工具
# ---------------------------------------------------------------------------
def _cell(v) -> str:
    """表格单元格转义：片段里可能来自 Markdown 表格（含 | 与换行）。"""
    return str(v).replace("|", "\\|").replace("\n", " ").strip()


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """生成 Markdown 表格文本（自动转义竖线与换行，避免表格被撑破）。"""
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(out)


def snip(text: str, n: int = 90) -> str:
    """单行片段摘要（去掉换行，便于放进 Markdown 表格）。"""
    s = " ".join((text or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def passage_lines(docs: list[dict], k: int = 3, max_chars: int = 160) -> list[str]:
    """把 top-k 片段格式化为「来源 + 摘要」列表，作为检索答案证据。"""
    lines = []
    for i, d in enumerate(docs[:k], 1):
        score = d.get("final_score", d.get("score", 0))
        sec = f" 章节：{d['section']}" if d.get("section") else ""
        lines.append(
            f"{i}. 《{d.get('doc', '')}》第{d.get('page', '?')}页 "
            f"（{d.get('type', 'text')}）{sec} 分数={float(score):.4f}\n"
            f"   {snip(d.get('text', ''), max_chars)}"
        )
    return lines


def save_json(path: Path, obj: dict) -> Path:
    """落盘 JSON（自动处理 numpy 标量）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _plain(o):
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(v) for v in o]
        if hasattr(o, "item"):          # numpy 标量
            return o.item()
        return o

    path.write_text(json.dumps(_plain(obj), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def save_markdown(path: Path, title: str, blocks: list[str]) -> Path:
    """落盘 Markdown 报告。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n" + "\n\n".join(blocks) + "\n"
    path.write_text(body, encoding="utf-8")
    return path


def print_metrics(tag: str, m: dict) -> None:
    """控制台打印一行指标。"""
    print(f"[{tag}] 准确率={m['accuracy']:.2%} 召回率={m['recall']:.2%} "
          f"HitRate@{m['k']}={m['hit_rate']:.2%} MRR={m['mrr']:.4f} "
          f"P@{m['k']}={m['precision']:.2%} "
          f"平均耗时={m.get('latency_avg', 0):.3f}s")
