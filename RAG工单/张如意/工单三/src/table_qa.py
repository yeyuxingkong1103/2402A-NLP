# -*- coding: utf-8 -*-
"""
表格类问题检索问答（工单03 验收项 2：id 1~4 的检索结果必须准确无误）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本模块针对《招股说明书2.pdf》（武汉力源信息技术股份有限公司）的 4 个
「答案全部藏在表格里」的问题，逐条展示：

    ① 检索到的表格原文（Markdown，保留表头与行列对应）
    ② 生成的答案（LLM；无 API Key 时自动降级为抽取式作答）
    ③ 检索精确度（可量化、可复现的三项指标）

检索精确度口径（全部基于确定性关键词匹配，不依赖 LLM 打分）：
    · 片段精度 precision@k   = 命中了答案要点的片段数 / 返回片段数
    · 要点召回 keypoint_recall = 检索上下文里出现的答案要点数 / 要点总数
    · 检索精确度 = 0.5 × precision@k + 0.5 × keypoint_recall
    · 答案要点命中率 answer_accuracy = 生成答案里出现的要点数 / 要点总数

用法：
    python table_qa.py                 # 跑 id 1~4，输出 json + md
    python table_qa.py --no-llm        # 无 API Key 时走抽取式兜底
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config                      # noqa: E402
import build_index_with_tables as bi             # noqa: E402

RESULTS_DIR = ROOT / "工单03-表格解析及检索优化" / "results"

# ---------------------------------------------------------------------------
# 1. 问题集（工单原文指定，与 rag_core.config.QUESTIONS_LIYUAN 保持一致）
# ---------------------------------------------------------------------------
TABLE_QUESTIONS = config.QUESTIONS_LIYUAN        # id 1~4

# ---------------------------------------------------------------------------
# 2. 答案基准（人工核对《招股说明书2.pdf》后固化，用于自动判分）
#    must  = 答案必须包含的要点（全部命中才算答对）
#    bonus = 加分项（答了更好，不扣分）
# ---------------------------------------------------------------------------
TABLE_GROUND_TRUTH: dict[int, dict] = {
    1: {
        "must": ["1,670万股", "25.04%"],
        "bonus": ["6,670万股", "人民币普通股"],
        "pages": [2, 22, 24],
        "tables": ["本次发行概况（摘要「四、本次发行情况」表）",
                   "第四节「本次发行概况」表"],
        "reference": "本次发行股数为1,670万股，占发行后总股本（6,670万股）的比例为25.04%。",
    },
    2: {
        "must": ["仓储及物流中心", "研发中心", "电子商务平台",
                 "扩充产品种类和数量", "其他与主营业务相关的营运资金"],
        "bonus": ["3,393.40", "1,526.38", "2,492.78", "9,000.00"],
        "pages": [22, 24],
        "tables": ["募集资金运用（「五、募集资金用途」表）"],
        "reference": "本次募集资金拟投资项目包括：仓储及物流中心、研发中心、"
                     "电子商务平台、扩充产品种类和数量、其他与主营业务相关的营运资金。",
    },
    3: {
        "must": ["赵马克", "42.35%", "控股股东"],
        "bonus": ["公司控股股东"],
        "pages": [157],
        "tables": ["关联方及关联交易 -> 1、存在控制关系的关联方"],
        "reference": "与本公司存在控制关系的关联方为赵马克，持股比例42.35%，"
                     "与本公司关系为公司控股股东。",
    },
    4: {
        # 主答案 = 「2、不存在控制关系的关联方」表中的 7 家（全部必须命中）
        "must": ["融冰投资", "武汉博润", "上海博润", "听音投资",
                 "联众聚源", "力源贸易", "普芯达"],
        # 加分项 = 紧随其后的「3、报告期内曾为关联方但目前已不存在关联关系的公司」
        #          2 家（佰力电子、盈硅电子）。它们属于相邻但不同的小节，
        #          表格解析中极易被误并进上一张表，故单列为加分项。
        "bonus": ["佰力电子", "盈硅电子"],
        "pages": [157, 158],
        "tables": ["关联方及关联交易 -> 2、不存在控制关系的关联方",
                   "（相邻）3、报告期内曾为关联方但目前已不存在关联关系的公司"],
        "reference": "不存在控制关系的关联方企业有：融冰投资、武汉博润、上海博润、"
                     "听音投资、联众聚源、力源贸易、普芯达。（另有报告期内曾为关联方、"
                     "目前已不存在关联关系的佰力电子、盈硅电子）",
    },
}

# 该问题应当命中的文档（评估检索命中率用）
REFERENCE_DOC = "招股说明书2"


# ---------------------------------------------------------------------------
# 3. 文本归一化与要点匹配
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """
    归一化后再匹配，避免「同一个数字不同写法」导致误判：
      · 去掉千分位逗号：1,670 -> 1670
      · 全角转半角、去空白
      · 统一百分号、括号
    """
    if not s:
        return ""
    s = str(s)
    s = s.replace("，", ",").replace("％", "%").replace("（", "(").replace("）", ")")
    s = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", s)        # 数字中的千分位逗号/空格
    s = re.sub(r"[\s　]+", "", s)
    return s.lower()


def match_points(text: str, points: list[str]) -> list[str]:
    """返回在 text 中命中的要点列表（归一化匹配）。"""
    t = normalize_text(text)
    return [p for p in points if normalize_text(p) in t]


def unmatch_points(text: str, points: list[str]) -> list[str]:
    """返回未命中的要点列表。"""
    t = normalize_text(text)
    return [p for p in points if normalize_text(p) not in t]


# ---------------------------------------------------------------------------
# 4. 检索精确度计算
# ---------------------------------------------------------------------------
def score_retrieval(question_gt: dict, docs: list[dict]) -> dict:
    """
    计算一条问题的检索侧量化指标。

    返回：
      precision_at_k       命中要点的片段数 / 返回片段数
      keypoint_recall      检索上下文中出现的要点数 / 要点总数
      retrieval_precision  0.5×precision@k + 0.5×keypoint_recall（本工单定义的检索精确度）
      table_hit            前 k 条里是否命中「含要点的表格片段」
      hit_pages            命中片段所在页码
    """
    must = question_gt["must"]
    if not docs:
        return {"precision_at_k": 0.0, "keypoint_recall": 0.0,
                "retrieval_precision": 0.0, "table_hit": False,
                "hit_pages": [], "relevant_chunks": [], "missing": list(must)}

    relevant = [d for d in docs if match_points(d.get("text", ""), must)]
    precision_at_k = len(relevant) / len(docs)

    ctx_all = "\n".join(d.get("text", "") for d in docs)
    hit = match_points(ctx_all, must)
    keypoint_recall = len(hit) / max(len(must), 1)

    table_hit = any(d.get("type") == "table" and match_points(d.get("text", ""), must)
                    for d in docs)

    return {
        "precision_at_k": round(precision_at_k, 4),
        "keypoint_recall": round(keypoint_recall, 4),
        "retrieval_precision": round(0.5 * precision_at_k + 0.5 * keypoint_recall, 4),
        "table_hit": table_hit,
        "hit_pages": [d.get("page") for d in relevant][:8],
        "relevant_chunks": [d.get("chunk_id") for d in relevant][:8],
        "missing": unmatch_points(ctx_all, must),
    }


def score_answer(answer: str, question_gt: dict) -> dict:
    """答案要点命中率（全部 must 命中即判对）。"""
    must, bonus = question_gt["must"], question_gt.get("bonus", [])
    hit = match_points(answer, must)
    return {
        "answer_accuracy": round(len(hit) / max(len(must), 1), 4),
        "correct": len(hit) == len(must),
        "hit_points": hit,
        "missing_points": unmatch_points(answer, must),
        "bonus_hit": match_points(answer, bonus),
    }


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def run_table_qa(collection: str = bi.COLLECTION_WITH_TABLES,
                 top_k: int = 5,
                 use_llm: bool = True,
                 use_table_boost: bool = True,
                 out_dir: Path | None = None,
                 verbose: bool = True) -> dict:
    """对 id 1~4 四个表格问题逐条检索问答并打分，结果落盘。"""
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = bi.load_retriever(collection)

    records, t0 = [], time.perf_counter()
    for q in TABLE_QUESTIONS:
        qid = q["id"]
        gt = TABLE_GROUND_TRUTH[qid]
        if verbose:
            print(f"\n{'=' * 74}\n[id {qid}] {q['question']}")

        res = bi.answer_with_tables(
            q["question"], retriever, top_k=top_k,
            use_table_boost=use_table_boost, use_llm=use_llm)

        rs = score_retrieval(gt, res["docs"])
        ans = score_answer(res["answer"], gt)

        # 检索到的表格原文（只取表格类片段，最多 2 张，避免报告过长）
        table_docs = [d for d in res["docs"] if d.get("type") == "table"]
        top_tables = [{
            "chunk_id": d.get("chunk_id"), "doc": d.get("doc"), "page": d.get("page"),
            "caption": d.get("caption", ""), "rows": d.get("rows"),
            "cols": d.get("cols"), "score": round(float(d.get("final_score", 0)), 4),
            "table_boost": d.get("table_boost"),
            "text": d.get("text", ""),
        } for d in table_docs[:2]]

        rec = {
            "id": qid,
            "question": q["question"],
            "reference_answer": gt["reference"],
            "must_points": gt["must"],
            "bonus_points": gt.get("bonus", []),
            "answer_grounding_pages": gt["pages"],
            "answer_grounding_tables": gt["tables"],
            "answer": res["answer"],
            "answer_mode": res["mode"],
            "retrieved_tables": top_tables,
            "retrieved_pages": [d.get("page") for d in res["docs"]],
            "retrieval": rs,
            "answer_score": ans,
            "timings": res["timings"],
        }
        records.append(rec)

        if verbose:
            print(f"  → 检索精确度 {rs['retrieval_precision']:.3f}"
                  f"（precision@k={rs['precision_at_k']:.2f}, "
                  f"要点召回={rs['keypoint_recall']:.2f}, 表格命中={rs['table_hit']}）"
                  f" | 答案要点命中率 {ans['answer_accuracy']:.2f}"
                  f" | 检索耗时 {res['timings']['retrieve'] * 1000:.0f}ms"
                  f" | 总耗时 {res['timings']['total']:.2f}s")
            if rs["missing"]:
                print(f"    未召回要点：{rs['missing']}")
            if ans["missing_points"]:
                print(f"    答案缺要点：{ans['missing_points']}")

    summary = {
        "collection": collection,
        "n_questions": len(records),
        "top_k": top_k,
        "use_table_boost": use_table_boost,
        "use_llm": use_llm,
        "avg_retrieval_precision": round(
            sum(r["retrieval"]["retrieval_precision"] for r in records) / max(len(records), 1), 4),
        "avg_keypoint_recall": round(
            sum(r["retrieval"]["keypoint_recall"] for r in records) / max(len(records), 1), 4),
        "table_hit_rate": round(
            sum(1 for r in records if r["retrieval"]["table_hit"]) / max(len(records), 1), 4),
        "answer_accuracy": round(
            sum(1 for r in records if r["answer_score"]["correct"]) / max(len(records), 1), 4),
        "avg_answer_accuracy": round(
            sum(r["answer_score"]["answer_accuracy"] for r in records) / max(len(records), 1), 4),
        "avg_retrieve_ms": round(
            sum(r["timings"]["retrieve"] for r in records) / max(len(records), 1) * 1000, 1),
        "max_total_seconds": round(max((r["timings"]["total"] for r in records), default=0), 2),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    payload = {"summary": summary, "records": records}
    (out_dir / "table_qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, out_dir / "table_qa.md")
    if verbose:
        print(f"\n汇总：检索精确度 {summary['avg_retrieval_precision']:.3f}，"
              f"要点召回 {summary['avg_keypoint_recall']:.3f}，"
              f"表格命中率 {summary['table_hit_rate']:.3f}，"
              f"答案准确率 {summary['answer_accuracy']:.3f}")
        print(f"结果已写入 {out_dir / 'table_qa.json'} 与 table_qa.md")
    return payload


def _write_markdown(payload: dict, path: Path) -> None:
    """生成人读报告 table_qa.md（演示视频与验收时直接投屏用）。"""
    s = payload["summary"]
    lines = [
        "# 表格类问题检索问答报告",
        "",
        "> 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化",
        "",
        "## 一、总体结果",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 问题数 | {s['n_questions']} |",
        f"| 索引集合 | `{s['collection']}` |",
        f"| 平均检索精确度 | **{s['avg_retrieval_precision']:.4f}** |",
        f"| 平均要点召回率 | {s['avg_keypoint_recall']:.4f} |",
        f"| 表格命中率 | {s['table_hit_rate']:.4f} |",
        f"| 答案准确率（要点全中） | **{s['answer_accuracy']:.4f}** |",
        f"| 平均检索耗时 | {s['avg_retrieve_ms']} ms |",
        f"| 单问最大总耗时 | {s['max_total_seconds']} s |",
        "",
        "## 二、逐题明细",
        "",
    ]
    for r in payload["records"]:
        rs, ans = r["retrieval"], r["answer_score"]
        lines += [
            f"### id {r['id']}：{r['question']}",
            "",
            f"- **答案定位**：第 {'、'.join(map(str, r['answer_grounding_pages']))} 页，"
            f"表格：{'；'.join(r['answer_grounding_tables'])}",
            f"- **参考答案**：{r['reference_answer']}",
            f"- **检索精确度**：{rs['retrieval_precision']:.4f}"
            f"（precision@k={rs['precision_at_k']:.4f}，要点召回={rs['keypoint_recall']:.4f}，"
            f"表格命中={'是' if rs['table_hit'] else '否'}）",
            f"- **答案要点命中率**：{ans['answer_accuracy']:.4f}"
            f"（{'正确' if ans['correct'] else '未全中'}）",
            f"- **生成模式**：{r['answer_mode']}",
            f"- **耗时**：检索 {r['timings']['retrieve'] * 1000:.0f} ms / "
            f"总计 {r['timings']['total']:.2f} s",
            "",
            "**检索到的表格原文：**",
            "",
        ]
        if r["retrieved_tables"]:
            for t in r["retrieved_tables"]:
                lines += [
                    f"<sub>chunk={t['chunk_id']} 第{t['page']}页 "
                    f"标题「{t['caption']}」 相似度={t['score']}"
                    f"（表格加权 {t.get('table_boost', '-')}）</sub>",
                    "",
                    "```markdown",
                    t["text"].strip(),
                    "```",
                    "",
                ]
        else:
            lines += ["（本次前 k 条结果中没有表格片段）", ""]
        lines += ["**生成的答案：**", "", "```", r["answer"].strip(), "```", ""]
        if rs["missing"]:
            lines.append(f"> 未召回的要点：{'、'.join(rs['missing'])}")
            lines.append("")
        if ans["missing_points"]:
            lines.append(f"> 答案中缺失的要点：{'、'.join(ans['missing_points'])}")
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="工单03 表格类问题（id 1~4）检索问答")
    ap.add_argument("--collection", default=bi.COLLECTION_WITH_TABLES)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="不调用生成模型，走抽取式兜底")
    ap.add_argument("--no-boost", action="store_true", help="关闭表格块优先级提升")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    run_table_qa(collection=args.collection, top_k=args.top_k,
                 use_llm=not args.no_llm,
                 use_table_boost=not args.no_boost,
                 out_dir=Path(args.out))


if __name__ == "__main__":
    main()
