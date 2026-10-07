"""
工单5 交付报告：多轮对话评测 —— 逐轮消解 + 逐轮作答 + 准确率
工单编号：人工智能NLP-RAG-Query理解优化任务

工单产出物原文要求：
    ① 功能实现代码
    ② 演示：(a) 完整演示视频  (b) **针对上述 5 轮检索并显示答案**
验收标准原文：
    功能 1) 准确性 ≥ 90%  2) 检索答案  3) 交互友好（支持多轮对话与用户反馈）4) 中英文
    性能 1) 响应 ≤ 3 秒   2) 资源消耗合理、高并发稳定

本脚本把「准确性 ≥ 90%」这句验收标准落成可核对的数据：

    对 `data/eval/questions_wot5.json` 里的每场对话，**按人类真实顺序逐轮提问**
    （同一个 Session 贯穿全场），每一轮记四件事：

      1. 消解：原话 → 系统改写出的自包含检索式（指代/省略是否解对）
      2. 检索：命中的块落在哪份文档、哪一页（是否召回到人工标注的答案页）
      3. 作答：模型给的答案 + 引用页
      4. 判定：三项确定性判据 ——
           subject_ok  主体是否等于标注主体（抓「把力源答成兴图」）
           facts_ok    关键事实是否全部出现（must_have 全中）
           clean_ok    是否出现禁止串号词（must_not_have 全不中）

    **一轮判对 = 三项全过**。整场的准确率 = 判对轮次 / 总轮次。

用法：
    python scripts/report_wot5.py                      # 全部对话（含 LLM 生成，约 2~4 分钟）
    python scripts/report_wot5.py --dialogue D1        # 只跑工单原题的 5 轮
    python scripts/report_wot5.py --mode off           # 换消融臂：off/concat/rule/rule+llm
    python scripts/report_wot5.py --no-answer          # 不调大模型，只看消解+检索（零成本）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, rag  # noqa: E402
from src.session import Session, Turn  # noqa: E402
from src.query_understanding import understand  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_QU

EVAL_FILE = config.EVAL_DIR / "questions_wot5.json"
OUT_DIR = config.EVAL_DIR


def _norm(s: str) -> str:
    """归一化：NFKC（全角→半角、『−』→『-』）+ 去空白 + 去逗号 + 去百分号。

    为什么去逗号/百分号：PDF 文字层出「1,670万股」、模型有时写「1670 万股」；
    标注写「25.04」而原文是「25.04%」。这些都是**同一个事实的不同写法**，
    不抹平就会出现「明明答对了被判漏」的假阴性。
    去空白的原因同工单4：图描述与正文层的空格口径不一致（「IC卡」vs「IC 卡」）。
    """
    t = unicodedata.normalize("NFKC", s or "")
    for ch in (" ", "\t", "\n", "\u3000", ",", "，", "%", "％"):
        t = t.replace(ch, "")
    return t


def judge(answer: str, turn: dict, res, doc_hit: bool, strict_clean: bool = True) -> dict:
    """
    对一轮做确定性判定。

    `strict_clean` 区分两个评测层：
      * True（默认，生成答案层）——「串号检查」生效：答案里出现另一家公司的人名/数字就判错。
        这是真正能抓住「把力源答成兴图」的判据。
      * False（零成本检索层）——禁用串号检查。**这不是放水，是判据本身不适用**：
        干扰项「宋丽萍」是深圳证券交易所的法定代表人，就印在力源 p26 上，
        和力源自己的法定代表人赵马克同页；召回到它说明**检索没错**，
        错的是「选谁当答案」这一步 —— 那是生成层的职责。
    """
    a = _norm(answer)
    must = turn.get("must_have") or []
    bad = turn.get("must_not_have") or []
    hit = [k for k in must if _norm(k) in a]
    missed = [k for k in must if k not in hit]
    leaked = [k for k in bad if _norm(k) in a] if strict_clean else []

    exp_subj = turn.get("expect_subject") or ""
    subject_ok = True if not exp_subj else (_norm(exp_subj) == _norm(res.subject))

    facts_ok = len(missed) == 0
    clean_ok = len(leaked) == 0 if strict_clean else None
    return {
        "subject_ok": subject_ok,
        "subject_got": res.subject,
        "subject_expect": exp_subj,
        "facts_ok": facts_ok,
        "must_have_hit": hit,
        "must_have_missed": missed,
        "must_have_rate": round(len(hit) / len(must), 4) if must else None,
        "clean_ok": clean_ok,
        "leaked": leaked,
        "doc_gold_hit": doc_hit,
        "correct": bool(subject_ok and facts_ok and (clean_ok is not False)),
    }


def gold_hit(citations: list[dict], gold_pages: dict) -> bool:
    """引用里有没有落在人工标注的答案页上（工单要求「检索答案」要有出处）。"""
    for c in citations or []:
        pages = (gold_pages or {}).get(c.get("doc_key") or "")
        if not pages:
            continue
        lo = c.get("page") or 0
        hi = c.get("page_end") or lo
        if any(lo <= p <= hi for p in pages):
            return True
    return False


def run_turn(q: str, session: Session, mode: str, with_answer: bool) -> dict:
    """跑一轮：消解（只做记录，真正的消解发生在 answer() 内部）→ 检索 → 作答。"""
    t0 = time.perf_counter()
    if with_answer:
        # ⚠️ `mode` 必须显式传下去：不传的话 `answer()` 会用 config 里的默认模式，
        # 结果消融的每一臂跑的都是同一个配置（实测过：T0 off 臂照样给出消解后的检索式，
        # 四臂全部 100%，实验完全失效）。
        a = rag.answer(q, session=session, mode=mode)
        answer = a.answer or ""
        qu = a.understanding
        citations = a.citations or []
        total_ms = a.timing.get("total_ms", 0.0)
        d = getattr(qu, "dialogue", None)
        retrieval_query = getattr(qu, "retrieval_query", q) if qu else q
        rewrite_rejected = bool(getattr(qu, "rewrite_rejected", False))
        items = []
    else:
        # 零成本路径：不调生成模型，但**消解与检索走完全相同的代码路径**。
        # 不能自己拼 retrieve() —— 那会绕过 rag.retrieve_for() 里的多轮文档硬限定
        # （doc_filter），检索集里混进另一家公司的块，实测会把「串号」判据全判错。
        qu = understand(q, session=session, mode=mode)
        d = qu.dialogue
        items, _rinfo = rag.retrieve_for(qu, top_k=config.RETRIEVAL_FINAL_TOP_K)
        citations = [
            {"doc_key": it.doc_key, "page": it.page, "page_end": it.page_end,
             "type": it.type, "evidence": round(it.evidence, 4), "section": it.section}
            for it in items
        ]
        answer = ""
        total_ms = (time.perf_counter() - t0) * 1000
        retrieval_query = getattr(qu, "retrieval_query", q)
        rewrite_rejected = bool(getattr(qu, "rewrite_rejected", False))

    return {
        "q": q,
        "resolved": (d.resolved if d else q),
        "retrieval_query": retrieval_query,
        "mode": (d.mode if d else mode),
        "coref": bool(d.coref) if d else False,
        "ellipsis": bool(d.ellipsis) if d else False,
        "subject": (d.subject if d else ""),
        "doc_key": (d.doc_key if d else ""),
        "topic": (d.topic if d else ""),
        "inherited_subject": (d.inherited_subject if d else ""),
        "inherited_topic": (d.inherited_topic if d else ""),
        "degraded": bool(d.degraded) if d else False,
        "llm_used": bool(d.llm_used) if d else False,
        "rule_evidence": round(d.rule_evidence, 4) if d else 0.0,
        "reason": (d.reason if d else ""),
        "rewrite_rejected": rewrite_rejected,
        "answer": answer,
        "citations": citations,
        "items": items,
        "total_ms": round(total_ms, 2),
    }


def run_dialogue(dlg: dict, mode: str, with_answer: bool) -> dict:
    sid = f"eval-wot5-{dlg['id']}-{mode}"
    s = Session(session_id=sid)
    turns_out: list[dict] = []
    for i, turn in enumerate(dlg["turns"], 1):
        r = run_turn(turn["q"], s, mode, with_answer)
        r["index"] = i
        r["type"] = turn.get("type")
        r["note"] = turn.get("note", "")
        r["gold_pages"] = turn.get("gold_pages") or {}
        if not with_answer:
            # 无答案时用召回块的 doc/page 兜底判「是否够到答案页」
            r["answer"] = "".join(it.text for it in r["items"])
        r["judge"] = judge(r["answer"], turn, _ResView(r),
                           gold_hit(r["citations"], turn.get("gold_pages")),
                           strict_clean=with_answer)
        r.pop("items", None)

        # 有生成时 `rag.answer()` 已把本轮回写进会话（见 rag._record_turn），
        # 这里**不能再手动 add**，否则每轮重复入库、会话状态翻倍。
        # 零成本路径不走 answer()，必须自己补上。
        if not with_answer:
            s.add(Turn(q=turn["q"], a=r["answer"][:200], resolved=r["resolved"],
                       subject=r["subject"], doc_key=r["doc_key"],
                       topic=r["topic"], intent=r.get("intent", "其他")))
        turns_out.append(r)

    n = len(turns_out)
    n_ok = sum(1 for t in turns_out if t["judge"]["correct"])
    return {
        "id": dlg["id"],
        "title": dlg["title"],
        "source": dlg.get("source", ""),
        "mode": mode,
        "turns": turns_out,
        "n_turns": n,
        "n_correct": n_ok,
        "accuracy": round(n_ok / n, 4) if n else None,
        "subject_acc": round(sum(1 for t in turns_out if t["judge"]["subject_ok"]) / n, 4) if n else None,
        "fact_acc": round(sum(1 for t in turns_out if t["judge"]["facts_ok"]) / n, 4) if n else None,
        "gold_hit_acc": round(sum(1 for t in turns_out if t["judge"]["doc_gold_hit"]) / n, 4) if n else None,
        "avg_ms": round(sum(t["total_ms"] for t in turns_out) / n, 2) if n else None,
    }


class _ResView:
    """把 run_turn 的结果包装成一个只有 .subject 的小对象，喂给 judge()。"""

    def __init__(self, r: dict) -> None:
        self.subject = r.get("subject", "")


def _fmt_turn_md(t: dict) -> list[str]:
    j = t["judge"]
    flag = "✅" if j["correct"] else "❌"
    lines = [
        f"#### 第 {t['index']} 轮 {flag}　`{t['type']}`",
        "",
        f"- **用户原话**：{t['q']}",
        f"- **消解后检索式**：`{t['retrieval_query']}`",
    ]
    if t["coref"]:
        lines.append(f"  - 指代消解：命中代词，主体 ← `{t['subject']}`")
    if t["ellipsis"]:
        lines.append(f"  - 省略补全：继承话题 `{t['inherited_topic'] or t['topic']}`，主体 ← `{t['subject']}`")
    if t["degraded"]:
        lines.append("  - ⚠️ 消解降级：退回原话检索")
    if t["llm_used"]:
        lines.append(f"  - 规则候选依据分 {t['rule_evidence']:.3f} < 闸门，已调 LLM 兜底")
    else:
        lines.append(f"  - 规则候选依据分 {t['rule_evidence']:.3f}，未调 LLM")
    if t["rewrite_rejected"]:
        lines.append("  - ⚠️ 改写被诊断出「把答案猜进检索式」，已丢弃、退回原文")
    lines.append(f"- **主体判定**：{'✅' if j['subject_ok'] else '❌'} 得到 `{j['subject_got']}`"
                 f"{'（期望 ' + j['subject_expect'] + '）' if not j['subject_ok'] else ''}")
    lines.append(f"- **关键事实**：{'✅' if j['facts_ok'] else '❌'} 命中 {len(j['must_have_hit'])}/"
                 f"{len(j['must_have_hit']) + len(j['must_have_missed'])}"
                 + (f"，漏 `{'` `'.join(j['must_have_missed'])}`" if j["must_have_missed"] else ""))
    if j["clean_ok"] is None:
        lines.append("- **串号检查**：—（检索层不适用，见脚本 docstring）")
    elif j["leaked"]:
        lines.append(f"- **串号检查**：❌ 出现禁止词 `{'` `'.join(j['leaked'])}`")
    else:
        lines.append("- **串号检查**：✅ 未出现其他公司/干扰项")
    lines.append(f"- **答案页命中**：{'✅' if j['doc_gold_hit'] else '⚠️'} 标注页 "
                 f"`{json.dumps(t['gold_pages'], ensure_ascii=False)}`")
    lines.append(f"- **耗时**：{t['total_ms']:.0f} ms")
    lines.append(f"- **答案**：{(t['answer'] or '').strip()[:400].replace(chr(10), ' ') or '（未生成）'}")
    cites = [f"{c.get('doc_key')} p{c.get('page')}" for c in (t["citations"] or [])[:6]]
    if cites:
        lines.append(f"- **引用页**：{' / '.join(cites)}")
    lines.append("")
    return lines


def build_md(report: dict) -> str:
    lines = [
        f"# 工单5 多轮对话评测报告",
        "",
        f"- **工单编号**：{WORK_ORDER_NO}",
        f"- **消解模式**：`{report['mode']}`",
        f"- **生成时间**：{report['ts']}",
        f"- **评测集**：`data/eval/questions_wot5.json`（{report['n_dialogues']} 场对话 / {report['n_turns']} 轮）",
        "",
        "## 一、总览",
        "",
        "| 场次 | 标题 | 轮数 | 判对 | 准确率 | 主体正确率 | 关键事实全中率 | 答案页命中率 | 平均耗时 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in report["dialogues"]:
        lines.append(
            f"| {d['id']} | {d['title']} | {d['n_turns']} | {d['n_correct']} | "
            f"{d['accuracy']:.1%} | {d['subject_acc']:.1%} | {d['fact_acc']:.1%} | "
            f"{d['gold_hit_acc']:.1%} | {d['avg_ms']:.0f} ms |"
        )
    lines += [
        f"| **合计** | — | **{report['n_turns']}** | **{report['n_correct']}** | "
        f"**{report['accuracy']:.1%}** | {report['subject_acc']:.1%} | "
        f"{report['fact_acc']:.1%} | {report['gold_hit_acc']:.1%} | "
        f"{report['avg_ms']:.0f} ms |",
        "",
        f"> 验收线：准确性 ≥ 90%　→　本次 **{report['accuracy']:.1%}** "
        f"（{'达标' if report['accuracy'] >= 0.9 else '未达标'}）",
        "",
        "判定口径：一轮判对 = 主体正确 **且** 关键事实全中 **且** 未出现串号词（三项确定性判据，无人工主观分）。",
        "",
    ]

    for d in report["dialogues"]:
        lines += [f"## {d['id']}　{d['title']}", ""]
        if d.get("source"):
            lines.append(f"*来源：{d['source']}*")
            lines.append("")
        lines.append(f"**本场准确率 {d['accuracy']:.1%}（{d['n_correct']}/{d['n_turns']}）**")
        lines.append("")
        for t in d["turns"]:
            lines += _fmt_turn_md(t)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", action="append", default=[], help="只跑指定场次，如 D1")
    ap.add_argument("--mode", default=config.MULTITURN_MODE,
                    choices=["off", "concat", "rule", "rule+llm"])
    ap.add_argument("--no-answer", action="store_true", help="不调大模型，只看消解+检索")
    args = ap.parse_args()

    config.ensure_dirs()
    spec = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    dialogues = spec["dialogues"]
    if args.dialogue:
        want = {x.upper() for x in args.dialogue}
        dialogues = [d for d in dialogues if d["id"].upper() in want]
    if not dialogues:
        print("[FAIL] 没有匹配的对话场次")
        return 2

    print(f"[工单5] 评测集：{EVAL_FILE}")
    print(f"[工单5] 模式：{args.mode}　场次：{[d['id'] for d in dialogues]}　"
          f"生成答案：{'否' if args.no_answer else '是'}")
    outs = []
    for d in dialogues:
        print(f"\n=== {d['id']} {d['title']} ===")
        r = run_dialogue(d, args.mode, not args.no_answer)
        for t in r["turns"]:
            flag = "OK " if t["judge"]["correct"] else "NG "
            print(f"  {flag}第{t['index']}轮 [{t['type']}] {t['q'][:34]}")
            print(f"       → {t['retrieval_query'][:70]}")
            print(f"       主体={t['subject'] or '(未识别)'}  {t['total_ms']:.0f}ms")
            if not t["judge"]["correct"]:
                print(f"       漏={t['judge']['must_have_missed']} 串号={t['judge']['leaked']} "
                      f"主体期望={t['judge']['subject_expect']}")
        print(f"  → 准确率 {r['accuracy']:.1%}（{r['n_correct']}/{r['n_turns']}）")
        outs.append(r)

    n_turns = sum(r["n_turns"] for r in outs)
    n_ok = sum(r["n_correct"] for r in outs)
    report = {
        "work_order_no": WORK_ORDER_NO,
        "mode": args.mode,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_file": str(EVAL_FILE),
        "n_dialogues": len(outs),
        "n_turns": n_turns,
        "n_correct": n_ok,
        "accuracy": round(n_ok / n_turns, 4) if n_turns else None,
        "subject_acc": round(sum(1 for r in outs for t in r["turns"] if t["judge"]["subject_ok"]) / n_turns, 4) if n_turns else None,
        "fact_acc": round(sum(1 for r in outs for t in r["turns"] if t["judge"]["facts_ok"]) / n_turns, 4) if n_turns else None,
        "gold_hit_acc": round(sum(1 for r in outs for t in r["turns"] if t["judge"]["doc_gold_hit"]) / n_turns, 4) if n_turns else None,
        "avg_ms": round(sum(t["total_ms"] for r in outs for t in r["turns"]) / n_turns, 2) if n_turns else None,
        "dialogues": outs,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "noanswer_" if args.no_answer else ""
    json_path = OUT_DIR / f"工单5_多轮对话评测_{tag}{args.mode}_{stamp}.json"
    md_path = OUT_DIR / f"工单5_多轮对话评测_{tag}{args.mode}_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(build_md(report), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"总准确率：{report['accuracy']:.1%}（{n_ok}/{n_turns}）　"
          f"主体 {report['subject_acc']:.1%}　事实 {report['fact_acc']:.1%}　"
          f"答案页 {report['gold_hit_acc']:.1%}　平均 {report['avg_ms']:.0f} ms")
    print(f"[JSON] {json_path}")
    print(f"[MD]   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
