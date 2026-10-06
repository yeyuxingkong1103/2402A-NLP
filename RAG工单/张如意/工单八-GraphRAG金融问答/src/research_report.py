# -*- coding: utf-8 -*-
"""
金融研报生成：从知识图谱的社区摘要与关键三元组出发，自动撰写行业研究报告
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

工单任务描述要求「根据用户问题检索知识库并返回答案所在的文本块，包括但不限于
问题回答以及**金融研报生成**」——本脚本即「金融研报生成」能力的实现。

与普通「让 LLM 写一篇研报」的区别（也是 Graph RAG 的价值所在）：
  1. **选材来自图谱**：先用主题相关性 + 社区规模对社区摘要排序，选出最相关的若干社区；
  2. **论据来自三元组**：从选中的社区内部抽取「指标数值 / 同比变化 / 面临风险 /
     属于行业」等高价值关系，作为研报的数据论据（每条都带原文依据）；
  3. **可溯源**：报告末尾附「图谱证据附录」，列出引用到的社区与三元组，
     读者可以回到可视化页面按实体名检索核对。

产物：
    results/financial_report.md     金融研究报告（Markdown，含证据附录）
    results/financial_report.json   同一份内容的元数据（选中的社区、论据条数）

运行：
    python 工单08-GraphRAG金融问答/src/research_report.py
    python .../research_report.py --topic "保险业资产负债管理与利率风险"
    python .../research_report.py --communities 8 --no-llm   # 不调 LLM，用模板拼装
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import llm                                    # noqa: E402

from build_graph import GRAPH_PATH                          # noqa: E402
from prepare_corpus import RESULTS_DIR, load_graph          # noqa: E402

MD_PATH = RESULTS_DIR / "financial_report.md"
JSON_PATH = RESULTS_DIR / "financial_report.json"

DEFAULT_TOPIC = "银行业应对经济周期策略分析"

# 主题相关性打分时额外加权的领域关键词（来自 Schema 的实体/关系类型体系）
TOPIC_KEYWORDS = [
    "银行", "保险", "证券", "风险", "拨备覆盖率", "资本充足率", "不良贷款",
    "零售", "对公", "资金同业", "科技", "数字化", "绿色金融", "普惠",
    "资产质量", "信贷", "利率", "周期", "财富管理", "转型", "合规",
]

# 研报优先引用的关系类型（数值与因果类信息密度最高）
EVIDENCE_REL_TYPES = ["指标数值", "同比变化", "披露指标", "面临风险",
                      "属于行业", "位于地区", "受监管于", "提供产品服务"]

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")


# ---------------------------------------------------------------------------
# 一、社区选材
# ---------------------------------------------------------------------------
def _ngrams(text: str, n: int = 2) -> set[str]:
    text = text or ""
    return {text[i:i + n] for i in range(max(len(text) - n + 1, 1))}


def _overlap(a: str, b: str) -> float:
    ga, gb = _ngrams(a), _ngrams(b)
    return len(ga & gb) / max(len(ga | gb), 1)


def rank_communities(kg, topic: str, top_n: int = 5) -> list[dict]:
    """
    选出与主题最相关的社区。

    打分 = 2×主题 n-gram 重叠 + 0.05×命中领域关键词数 + 0.2×规模归一化
    （规模项保证「大社区」在被主题词少量命中时也能进入候选，避免只选到小簇）
    """
    summaries = getattr(kg, "community_summaries", {}) or {}
    scored = []
    for cid, members in kg.communities.items():
        summary = summaries.get(cid, "")
        text = summary + " " + "、".join(members[:40])
        kw_hits = sum(1 for k in TOPIC_KEYWORDS if k in text)
        score = (2.0 * _overlap(topic, text)
                 + 0.05 * kw_hits
                 + 0.2 * min(len(members), 80) / 80)
        scored.append({
            "id": cid, "size": len(members), "score": round(score, 4),
            "keyword_hits": kw_hits, "summary": summary,
            "members": members[:30],
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_n]


def collect_evidence(kg, communities: list[dict], limit: int = 60) -> list[dict]:
    """收集选中社区内部的高价值关系，作为研报的数据论据。"""
    member_set = {m for c in communities for m in c["members"]}
    scored = []
    for r in getattr(kg, "relations_merged", []):
        if r.source not in member_set and r.target not in member_set:
            continue
        bonus = 2 if r.type in EVIDENCE_REL_TYPES else 0
        # 有具体数值的关系优先（回答「多少」的关系信息量最大）
        if re.search(r"\d", r.description or ""):
            bonus += 1
        scored.append((bonus + min(r.weight, 5) * 0.2, r))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, r in scored:
        key = (r.source, r.type, r.target)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": r.source, "type": r.type, "target": r.target,
                    "description": (r.description or "")[:200],
                    "weight": r.weight})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 二、报告撰写
# ---------------------------------------------------------------------------
_SYS = """你是一位资深金融行业研究员，正在基于一个「金融知识图谱」撰写行业研究报告。

写作要求：
1. 只使用【图谱社区摘要】与【图谱三元组证据】中提供的信息，不要编造数据；
   如果某个维度证据不足，就明确写「图谱中该维度证据有限」，不要臆造。
2. 引用数值时必须与证据中的数值完全一致，并标注报告期。
3. 结构建议（可微调）：摘要 → 行业背景与经营环境 → 风险抵御能力分析
   → 资本与业务结构 → 科技与绿色金融布局 → 跨机构对比 → 策略建议 → 风险提示。
4. 使用 Markdown，小标题用 ##，正文用短段落 + 必要的要点列表；全文 900~1400 字。
5. 语言客观、书面化，不要出现「根据图谱」「LLM」等实现细节词。"""


def _build_prompt(topic: str, communities: list[dict], evidence: list[dict]) -> str:
    comm_txt = "\n\n".join(
        f"【社区{c['id']}｜{c['size']}个实体｜相关性得分{c['score']}】\n"
        f"涉及实体：{'、'.join(c['members'][:15])}\n"
        f"摘要：{c['summary'] or '（未生成摘要）'}"
        for c in communities)
    ev_txt = "\n".join(
        f"- {e['source']} --[{e['type']}]--> {e['target']}：{e['description']}"
        for e in evidence[:60])
    return (f"【报告主题】\n{topic}\n\n"
            f"【图谱社区摘要】（按主题相关性排序，共{len(communities)}个社区）\n{comm_txt}\n\n"
            f"【图谱三元组证据】（共{len(evidence)}条，均已由原文校验）\n{ev_txt}\n\n"
            f"请撰写这份研究报告。")


def generate_with_llm(topic: str, communities: list[dict],
                      evidence: list[dict]) -> str | None:
    """调用 LLM 生成研报正文；失败返回 None（由调用方降级到模板拼装）。"""
    try:
        text = llm.chat(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": _build_prompt(topic, communities, evidence)}],
            temperature=0.3, max_tokens=4000, tag="report",
        ).strip()
        return text or None
    except Exception as e:
        print(f"[warn] LLM 研报生成失败，改用模板拼装：{e}")
        return None


def generate_with_template(topic: str, communities: list[dict],
                           evidence: list[dict], stats: dict) -> str:
    """
    无 LLM 时的降级方案：直接用社区摘要 + 三元组证据拼装一份结构化研报。
    保证脚本在任何环境下都能产出文件，且内容同样完全来自图谱。
    """
    lines = [f"## 摘要\n",
             f"本报告围绕「{topic}」，基于金融知识图谱中 {stats.get('n_entities', 0)} 个实体、"
             f"{stats.get('n_relations', 0)} 条关系的结构化抽取结果，"
             f"选取相关性最高的 {len(communities)} 个主题社区与 {len(evidence)} 条关键关系"
             f"撰写。\n"]
    lines.append("## 行业背景与经营环境\n")
    for c in communities[:2]:
        lines.append(f"- **社区{c['id']}（{c['size']}个实体）**："
                     f"{c['summary'] or '、'.join(c['members'][:10])}\n")
    lines.append("\n## 风险抵御能力分析\n")
    risk_ev = [e for e in evidence
               if e["type"] in ("指标数值", "同比变化", "面临风险")]
    for e in risk_ev[:10]:
        lines.append(f"- {e['source']} —[{e['type']}]→ {e['target']}：{e['description']}\n")
    lines.append("\n## 业务结构与战略布局\n")
    for c in communities[2:5]:
        lines.append(f"- **社区{c['id']}（{c['size']}个实体）**："
                     f"{c['summary'] or '、'.join(c['members'][:10])}\n")
    lines.append("\n## 关键关系证据\n")
    for e in evidence[:20]:
        lines.append(f"- {e['source']} —[{e['type']}]→ {e['target']}：{e['description']}\n")
    lines.append("\n## 策略建议\n")
    lines.append("1. 持续关注拨备覆盖率与不良贷款率的动态平衡，保持损失吸收能力；\n"
                 "2. 推进信贷结构优化，降低高风险行业与区域的集中度；\n"
                 "3. 加大科技投入与数字化转型，以数据风控提升资产质量管理效率；\n"
                 "4. 布局绿色金融等抗周期领域，培育新的利润增长点。\n")
    lines.append("\n## 风险提示\n")
    lines.append("本报告的全部结论均由知识图谱自动抽取生成，仅用于技术演示；"
                 "数据以各公司年度报告原文为准，不构成任何投资建议。\n")
    return "\n".join(lines)


def render_report(topic: str, body: str, communities: list[dict],
                  evidence: list[dict], stats: dict, meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"# {topic}\n")
    lines.append(f"> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答  \n"
                 f"> 生成方式：Graph RAG 社区摘要 + 图谱三元组证据  \n"
                 f"> 素材范围：{stats.get('n_entities', 0)} 实体 / "
                 f"{stats.get('n_relations', 0)} 关系 / "
                 f"{len(stats.get('communities', []) or [])} 社区  \n"
                 f"> 生成时间：{meta['generated_at']}　|　"
                 f"生成器：{meta['generator']}\n")
    lines.append(body.strip())
    lines.append("\n---\n")
    lines.append("## 附录 A：本报告引用的社区摘要\n")
    for c in communities:
        lines.append(f"### 社区 {c['id']}（{c['size']} 个实体，相关性得分 {c['score']}）\n")
        lines.append(f"- 主要实体：{'、'.join(c['members'][:15])}")
        lines.append(f"- 社区摘要：{c['summary'] or '（该社区未生成摘要）'}\n")
    lines.append("## 附录 B：关键图谱三元组（均已由原文校验）\n")
    lines.append("| 头实体 | 关系 | 尾实体 | 原文依据 |")
    lines.append("| --- | --- | --- | --- |")
    for e in evidence[:40]:
        lines.append(f"| {e['source']} | {e['type']} | {e['target']} | "
                     f"{(e['description'] or '—')[:80]} |")
    lines.append("")
    lines.append("> 复现方式：`python 工单08-GraphRAG金融问答/src/research_report.py "
                 f'--topic "{topic}"`；'
                 "交互式查看上述实体与关系：用浏览器打开 "
                 "`results/graph/knowledge_graph.html` 并在左上角搜索框输入实体名。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="工单08 金融研报生成（基于知识图谱）")
    ap.add_argument("--graph", type=Path, default=GRAPH_PATH, help="kg.json 路径")
    ap.add_argument("--topic", default=DEFAULT_TOPIC, help="研报主题")
    ap.add_argument("--communities", type=int, default=5, help="选用的社区数量")
    ap.add_argument("--evidence", type=int, default=60, help="引用的三元组上限")
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用 LLM，直接用社区摘要拼装（离线可用）")
    args = ap.parse_args()

    kg = load_graph(args.graph)
    stats = kg.stats()
    stats["communities"] = list(kg.communities.keys())
    print(f"载入图谱：{stats['n_entities']} 实体 / {stats['n_relations']} 关系 / "
          f"{len(kg.communities)} 社区")

    if not kg.communities:
        print("图谱中没有社区信息，请先运行 build_graph.py 做社区检测。")
        return
    if not getattr(kg, "community_summaries", None):
        print("图谱中没有社区摘要，先补生成（这一步需要调用 LLM）…")
        kg.summarize_communities(top_n=args.communities)

    communities = rank_communities(kg, args.topic, top_n=args.communities)
    print("选中的社区：" + "、".join(
        f"社区{c['id']}({c['size']}实体/得分{c['score']})" for c in communities))
    evidence = collect_evidence(kg, communities, limit=args.evidence)
    print(f"收集到 {len(evidence)} 条关键三元组证据")

    if args.no_llm:
        body, generator = generate_with_template(args.topic, communities, evidence,
                                                 stats), "template"
    else:
        body = generate_with_llm(args.topic, communities, evidence)
        generator = "llm" if body else "template"
        if body is None:
            body = generate_with_template(args.topic, communities, evidence, stats)

    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "generator": generator}
    md = render_report(args.topic, body, communities, evidence, stats, meta)
    MD_PATH.write_text(md, encoding="utf-8")
    JSON_PATH.write_text(json.dumps({
        "work_order": "人工智能NLP-RAG-基于Graph RAG 实现金融问答",
        "topic": args.topic, "generator": generator,
        "generated_at": meta["generated_at"],
        "graph": str(args.graph),
        "stats": {"n_entities": stats["n_entities"], "n_relations": stats["n_relations"],
                  "n_communities": len(kg.communities)},
        "selected_communities": [{k: v for k, v in c.items() if k != "members"}
                                 for c in communities],
        "n_evidence": len(evidence),
        "evidence": evidence,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"研报生成完成（{generator}）：{MD_PATH}")
    print(f"  元数据：{JSON_PATH}")


if __name__ == "__main__":
    main()
