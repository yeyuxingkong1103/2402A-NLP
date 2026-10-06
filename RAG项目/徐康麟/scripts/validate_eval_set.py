#!/usr/bin/env python3
"""题集自校验（**离线也能跑**）：期望来源/期望要点必须真的在语料里。

为什么必须有（真机踩过）：L22「什么行为构成商标侵权」的期望要点写成「未经许可」，
而《商标法》原文是「未经商标注册人的许可」⇒ 全库 grep 0 命中 ⇒ 这条题**无论检索多好
都判不过**，还会被误读成"检索缺陷"。这类出题错误让机器抓，别靠人眼。

用法：
    python scripts/validate_eval_set.py                       # 默认 eval/qa_set.jsonl + knowledge/lawyer
    python scripts/validate_eval_set.py --qa-file x.jsonl --corpus knowledge/lawyer
退出码：0 = 全部通过；1 = 有问题（逐条打印）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.eval_set import load_qa_set, validate_against_corpus  # noqa: E402

_ARTICLE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+条")


def notes_article_conflicts(items) -> list[tuple[str, list[str], list[str]]]:
    """``notes`` 里点名的条号与 ``articles`` **完全不重叠**的题（返回逐条三元组）。

    纯函数、不读语料，便于单测；判据见文件顶部注释与 `eval/LABEL-CHANGES.md`。
    """
    found: list[tuple[str, list[str], list[str]]] = []
    for item in items:
        articles = [str(x) for x in (getattr(item, "articles", ()) or ())]
        if not articles:
            continue
        noted = _ARTICLE.findall(str(getattr(item, "notes", "") or ""))
        if noted and not set(noted) & set(articles):
            found.append((str(item.id), noted, articles))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="评测题集自校验")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    args = parser.parse_args()

    items = load_qa_set(args.qa_file)
    problems = validate_against_corpus(items, args.corpus)

    by_kind: dict[str, int] = {}
    for problem in problems:
        by_kind[problem["kind"]] = by_kind.get(problem["kind"], 0) + 1
    multi = [item.id for item in items if item.turns]

    print(f"题集 {args.qa_file}：{len(items)} 题（多轮题 {len(multi)} 条：{', '.join(multi) or '无'}）")
    print(f"语料 {args.corpus}")
    print(f"问题 {len(problems)} 条：" + (", ".join(f"{k}={v}" for k, v in by_kind.items()) or "无"))
    for problem in problems:
        print(f"  [{problem['kind']}] {problem['id']}: {problem['detail']}")

    expect_only = {"answerable", "must_refuse", "must_route_general"}
    bad_expect = [item.id for item in items if item.expect not in expect_only]
    if bad_expect:
        print(f"  [bad_expect] {', '.join(bad_expect)}")
    dup = {item.id for item in items if [i.id for i in items].count(item.id) > 1}
    if dup:
        print(f"  [duplicate_id] {', '.join(sorted(dup))}")

    # 条号标注 vs 人工 notes 的**冲突检查**（2026-09-23 新增）。
    # 为什么：`articles` 早期是脚本按词面自动建议的，而 `notes` 是人工写的，实测 15 条两者
    # 完全不重叠、且**错的多半是 articles**（见 `eval/LABEL-CHANGES.md`）。这里只**告警**
    # 不判失败 —— 冲突需要人判断，但绝不能让它悄悄存在。
    noted_conflicts = notes_article_conflicts(items)
    if noted_conflicts:
        print(f"  [notes_conflict] {len(noted_conflicts)} 条：notes 点名的条号与 articles 完全不重叠"
              f"（请用 scripts/fix_expected_articles.py 核对）")
        for item_id, noted, articles in noted_conflicts:
            print(f"      {item_id}: notes={noted} articles={articles}")

    if problems or bad_expect or dup:
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
