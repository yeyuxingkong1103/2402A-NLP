"""订正评测集的**条号标注**（P8 度量前提；2026-09-23 用户授权）。

背景（证据见 `docs/RERANK-EXPERIMENT.md` 与 `eval/results/article-label-audit.json`）：
`articles` 字段是早期用词面建议（`annotate_expected_articles.py`）自动填的，实测大面积不可靠 ——
① 张冠李戴（L22 期望"商标法第一条"，命中的其实是《商标案件管辖…解释》的**受案范围**）；
② 只被**交叉引用**（N01 的"第三十六条"来自批复正文里对另一部司法解释的引用；N02 的
"第一千一百九十五条"是**民法典**的条，而该题期望来源只有那份司法解释 ⇒ 永远不可能命中）；
③ **版本漂移**（同一条内容在 2013/2019 版是第五十七条，在 2026 公布/2027 施行的新版是第七十二条）。

订正规则（**只用外部可验证的证据，不用检索结果当标准答案**）：
1. ``notes_articles``：题目的 ``notes`` 是**人工写的**，里面点名了条号，且与 ``articles`` 不重叠
   ⇒ 以 notes 为准（notes 在测量之前就存在，不受任何一次实验影响）；
2. ``impossible_article``：某条号在"期望来源匹配的文件"里**根本不是条文标题**
   ⇒ 删掉；全删则 ``articles=[]``（退回"只判来源"）；
3. ``corpus_evidence``：上面两条都不适用、但语料原文证明现标注与问题无关的（人工核实过），
   在 ``MANUAL`` 表里逐条写明依据。

用法::

    python scripts/fix_expected_articles.py                 # 只报告（dry-run，默认）
    python scripts/fix_expected_articles.py --apply         # 写回 qa_set.jsonl
    python scripts/fix_expected_articles.py --out eval/results/label-corrections.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import console_safe  # noqa: E402
from scripts.audit_expected_articles import article_headings, load_files  # noqa: E402

_ARTICLE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+条")

#: 语料原文已核实的个例（规则 1/2 覆盖不到）：{题号: (新条号, 依据)}
MANUAL: dict[str, tuple[list[str], str]] = {
    "L07": (["第一百零一条"],
            "现标注第一百零二条在现行有效版《道路交通安全法》里讲「六个月内发生二次以上特大交通事故的"
            "专业运输单位」，与「逃逸后果」无关；第一百零一条=「造成交通事故后逃逸的…吊销机动车驾驶证，"
            "且终生不得重新取得机动车驾驶证」，与本题 notes「吊销/终身禁驾」完全对应"),
    "L13": (["第四十九条"],
            "现行有效《公司法》（2024-07-01 施行）第四十九条=「股东应当按期足额缴纳…未按期足额缴纳"
            "出资的…还应当对给公司造成的损失承担赔偿责任」，正面回答「出资不实要承担什么责任」；"
            "原第二十八条在现行版里是「决议无效/撤销后的登记撤销」（无关），且原命中来自"
            "《公司法时间效力规定》里的**交叉引用**；原第十五条来自《公司法解释（三）》且讲的是"
            "非货币财产出资贬值，不针对「出资不实」"),
    "L23": (["第六十五条", "第十一条"],
            "现行有效《专利法》第六十五条=「未经专利权人许可，实施其专利，即侵犯其专利权」；"
            "第十一条=专利权效力（未经许可不得实施）——两者才是「怎样判断是否侵犯专利权」的依据；"
            "原第六十八条讲的是**假冒专利**"),
}

#: notes 里点名多条的题（此处 notes 是"从第X条起"的意思，保留原条号并补上 notes 的）
_UNION_IDS = {"L18"}


def propose(items: list[dict], files: dict[str, str]) -> list[dict]:
    """给出订正建议（不改输入）。"""
    changes: list[dict] = []
    for item in items:
        current = [str(x) for x in (item.get("articles") or [])]
        if not current:
            continue
        sources = tuple(str(x) for x in (item.get("sources") or ()))
        item_id = str(item.get("id"))
        noted = _ARTICLE.findall(str(item.get("notes") or ""))
        existing = [name for name in files if any(key in name for key in sources)]

        def is_heading(article: str) -> bool:
            return any(article in article_headings(files[name]) for name in existing)

        proposed: list[str] = []
        reason = ""
        rule = ""

        if item_id in MANUAL:
            proposed, reason = list(MANUAL[item_id][0]), MANUAL[item_id][1]
            rule = "corpus_evidence"
        elif noted and not set(noted) & set(current):
            keep = list(current) + [a for a in noted if a not in current] if item_id in _UNION_IDS \
                else list(dict.fromkeys(noted))
            proposed, rule = keep, "notes_articles"
            reason = (f"notes（人工撰写）点名 {noted}，与 articles {current} 完全不重叠 ⇒ 以 notes 为准"
                      f"（notes 早于任何实验存在，不受测量影响）")
        else:
            survivors = [a for a in current if is_heading(a)]
            if len(survivors) != len(current):
                proposed, rule = survivors, "impossible_article"
                dropped = [a for a in current if a not in survivors]
                reason = (f"条号 {dropped} 在「期望来源匹配的文件」里不是条文标题（多半来自正文里的"
                          f"交叉引用，或属于别的文件）⇒ 删除；"
                          + ("删除后退回「只判来源」" if not survivors else "保留其余条号"))

        if not proposed and not rule:
            continue
        if sorted(proposed) == sorted(current):
            continue
        changes.append({"id": item_id, "rule": rule, "before": current, "after": proposed,
                        "reason": reason, "question": str(item.get("question") or ""),
                        "sources": list(sources)})
    return changes


def apply_changes(path: Path, changes: list[dict]) -> int:
    """按 id 写回 ``articles`` 与 ``notes``（notes 末尾追加一条订正记录）。"""
    by_id = {change["id"]: change for change in changes}
    lines: list[str] = []
    touched = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            lines.append(line)
            continue
        data = json.loads(line)
        change = by_id.get(str(data.get("id")))
        if change is None:
            lines.append(json.dumps(data, ensure_ascii=False))
            continue
        data["articles"] = change["after"]
        stamp = "2026-09-23 订正条号（%s）：%s -> %s；依据：%s" % (
            change["rule"], change["before"], change["after"], change["reason"])
        data["notes"] = (str(data.get("notes") or "").rstrip() + "。 " + stamp).lstrip("。 ")
        lines.append(json.dumps(data, ensure_ascii=False))
        touched += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return touched


def main() -> int:
    parser = argparse.ArgumentParser(description="订正评测集条号标注（默认 dry-run）")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    parser.add_argument("--out", default="eval/results/label-corrections.json")
    parser.add_argument("--apply", action="store_true", help="写回 qa_set.jsonl（默认只报告）")
    args = parser.parse_args()

    path = Path(args.qa_file)
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    files = load_files(Path(args.corpus), skip_copies=False)   # 核对时要看全版本
    changes = propose(items, files)

    counts: dict[str, int] = {}
    for change in changes:
        counts[change["rule"]] = counts.get(change["rule"], 0) + 1
    print(f"共 {len(changes)} 条建议订正：" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for change in changes:
        print(f"\n[{change['rule']}] {change['id']}  {change['before']} -> {change['after']}")
        print(f"    问：{console_safe(change['question'])[:60]}  来源={change['sources']}")
        # ⚠️ `reason` 是**数据**（写进 JSON 报告，里面按项目习惯用 ⇒ 做箭头）；
        #    但把它打到 GBK 控制台会丢整行，所以过一道 console_safe（⇒ 变成 ->）。
        print(f"    依据：{console_safe(change['reason'])[:160]}")

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"changes": changes, "counts": counts},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n建议已写：{target}")

    if args.apply:
        touched = apply_changes(path, changes)
        print(f"已写回 {touched} 条到 {path}")
    else:
        print("（dry-run：加 --apply 才会写回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
