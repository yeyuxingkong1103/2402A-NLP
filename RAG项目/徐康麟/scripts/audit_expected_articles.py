"""核查评测集的**条号标注**是否站得住（P8 的度量前提，2026-09-23 发现标注不可靠后补的工具）。

为什么需要它：P8 一直用"条号级 Recall@k"衡量"法条级精度"，但 2026-09-23 核查发现标注有三类毛病：
1. **张冠李戴**：标注的条号来自**别的文件**（如 L22 期望"第一条"，命中的其实是
   《最高人民法院关于商标法修改决定施行后商标案件管辖和法律适用问题的解释》第一条"受案范围"）；
2. **交叉引用**：条号只在别的条文正文里被**引用**（如 L13 的"第二十八条"出现在
   《…时间效力…规定》里"第二十八条第二款的规定"），而不是那一条本身；
3. **版本漂移**：同一部法有多个版本、**同一条内容在不同版本里条号不同** —— 实测《商标法》
   "有下列行为之一的，均属侵犯注册商标专用权"在 2013/2019 版是**第五十七条**，
   在 2026 公布、2027 施行的新版里是**第七十二条**。只写条号的标注是**版本无关**的。

本工具只做**事实核对**，不改标注：给出每题标注的可疑类别与证据，改与不改由人决定
（`--report` 可传入检索报告，把"模型挑中的条文"一起列出来当旁证）。
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

_ARTICLE = re.compile(r"^第[一二三四五六七八九十百千零〇两]+条")
_META = re.compile(r"^-\s*([^：]+)：(.+)$")


def article_headings(text: str) -> list[str]:
    """按行首判断，取出真正的条文标题（"第X条"单独起行）。"""
    heads: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().replace("\u3000", " ").strip()
        match = _ARTICLE.match(stripped)
        if match:
            heads.append(match.group(0))
    return heads


def metadata(text: str) -> dict[str, str]:
    """语料文件头部的元数据（公布日期/施行日期/时效性…）。"""
    out: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        match = _META.match(line.strip())
        if match:
            out[match.group(1)] = match.group(2).strip()
    return out


def classify(article: str, sources: tuple[str, ...], files: dict[str, str]) -> dict:
    """一条标注 vs 语料：它是**真正的条文标题**还是只在正文里被引用？在哪几个版本里？"""
    headings: list[dict] = []
    references: list[str] = []
    for name, text in files.items():
        if sources and not any(key in name for key in sources):
            continue
        meta = metadata(text)
        in_heading = article in article_headings(text)
        if in_heading:
            headings.append({"file": name, "施行": meta.get("施行日期", ""),
                             "公布": meta.get("公布日期", ""), "时效性": meta.get("时效性", "")})
        elif article in text:
            references.append(name)
    if headings:
        verdict = "heading"
    elif references:
        verdict = "reference_only"
    else:
        verdict = "not_found"
    return {"article": article, "verdict": verdict, "headings": headings,
            "reference_files": references[:3]}


def load_files(corpus: Path, *, skip_copies: bool = True) -> dict[str, str]:
    """读语料：默认跳过 ``__<hash>`` 复制件（版本盘点会把它们当非现行，但核查时要看全版本）。"""
    files: dict[str, str] = {}
    for path in corpus.rglob("*.md"):
        name = path.name
        if skip_copies and re.search(r"__[0-9a-f]{6,}\.", name):
            continue
        try:
            files[name] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return files


def audit(items: list[dict], files: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        articles = tuple(item.get("articles") or ())
        if not articles:
            continue
        sources = tuple(item.get("sources") or ())
        checks = [classify(article, sources, files) for article in articles]
        if all(check["verdict"] == "heading" for check in checks):
            verdict = "ok"
        elif any(check["verdict"] == "heading" for check in checks):
            verdict = "partial"
        elif any(check["verdict"] == "reference_only" for check in checks):
            verdict = "reference_only"
        else:
            verdict = "not_found"
        file_versions = {entry["file"]: entry for check in checks for entry in check["headings"]}
        rows.append({"id": item.get("id"), "sources": list(sources), "articles": list(articles),
                     "verdict": verdict, "checks": checks,
                     "版本数": len({(v["施行"], v["时效性"]) for v in file_versions.values()})})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="核查评测集条号标注（只报事实，不改标注）")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    parser.add_argument("--report", default="", help="检索报告（含 picked_labels）作为旁证")
    parser.add_argument("--out", default="")
    parser.add_argument("--quiet", action="store_true", help="只打印有问题的题")
    args = parser.parse_args()

    items = [json.loads(line) for line in
             Path(args.qa_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    files = load_files(Path(args.corpus))
    rows = audit(items, files)

    picked: dict[str, list[str]] = {}
    if args.report and Path(args.report).is_file():
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        for summary in (report.get("rankers") or {}).values():
            for row in summary.get("detail") or []:
                if row.get("picked_labels"):
                    picked.setdefault(str(row["id"]), [])
                    for label in row["picked_labels"]:
                        if label not in picked[str(row["id"])]:
                            picked[str(row["id"])].append(label)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"题集 {args.qa_file}：{len(rows)} 道有标注的题；语料 {len(files)} 个文件（已跳过 __hash 复制件）")
    print("判定分布：" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for row in rows:
        if args.quiet and row["verdict"] == "ok":
            continue
        print(f"\n[{row['verdict']}] {row['id']}  期望来源={row['sources']}  标注={row['articles']}")
        for check in row["checks"]:
            if check["verdict"] == "heading":
                for entry in check["headings"]:
                    print(f"    {check['article']}: 条文标题 于 {entry['file']} "
                          f"(施行={entry['施行'] or '?'} 时效性={entry['时效性'] or '?'})")
            else:
                print(f"    {check['article']}: {check['verdict']}（只在正文被引用："
                      f"{check['reference_files']}）")
        if picked.get(str(row["id"])):
            print(f"    选择器挑中：{picked[str(row['id'])]}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"counts": counts, "items": rows, "picked": picked},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
