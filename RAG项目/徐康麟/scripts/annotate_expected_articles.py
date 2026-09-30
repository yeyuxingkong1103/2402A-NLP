#!/usr/bin/env python3
"""给评测题集补 **期望条号**（``articles``）—— 机器从语料里定位，不靠人记。

为什么需要（P8 度量缺陷）：检索侧指标原来只测"期望**文件**有没有进前 k"，而一部法有几十个
条文块 ⇒ 文件级 Recall@5 0.88 完全掩盖了真问题："**那一条**（如《商标法》第五十七条）
没进前 k"。加上条号后，`scripts/eval_retrieval.py` 就能算**条号级** Recall@k/MRR，
这才是 P8 的对症指标。

定位规则（`legal_rag/eval_set.suggest_articles`）：按 ``第X条`` 切块，取包含**全部**要点的条；
一个都没有时退而取包含**任一**要点的条，并把这种"弱匹配"在报告里点出来。

用法：
    python scripts/annotate_expected_articles.py --dry-run     # 先看会写什么
    python scripts/annotate_expected_articles.py               # 写回 qa_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.eval_set import load_qa_set, suggest_articles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="补期望条号（机器定位）")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    corpus = Path(args.corpus)
    items = load_qa_set(qa_path)
    records = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]

    annotated = weak = missing = 0
    for record, item in zip(records, items):
        if item.expect != "answerable" or not item.sources or not item.claims:
            continue
        files: list[Path] = []
        for source in item.sources:
            files.extend(path for path in corpus.rglob("*.md") if source in path.name)
        found: list[str] = []
        strict = False
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            found = suggest_articles(text, item.claims, strict_only=True)
            if found:
                strict = True
                break
            if not found:
                found = suggest_articles(text, item.claims)
            if found:
                break
        if not found:
            missing += 1
            print(f"  [定位失败] {item.id}：{item.sources} 里没有含 {item.claims} 的条文")
            continue
        record["articles"] = found[:2]
        if strict:
            annotated += 1
        else:
            weak += 1
        print(f"  {item.id}: articles={record['articles']}（{'强匹配' if strict else '弱匹配'}）")

    print(f"\n强匹配 {annotated} 条 / 弱匹配 {weak} 条 / 没定位到 {missing} 条")
    if not args.dry_run:
        qa_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                           encoding="utf-8")
        print(f"已写回：{qa_path}")
    else:
        print("（--dry-run：未写盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
