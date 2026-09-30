#!/usr/bin/env python3
"""**记忆回音检查**：召回的旧话有没有被模型逐字照抄进答案？

为什么需要它（`docs/DESIGN-TODO.md` D5 的验收口径 ②）
====================================================
开启长期记忆召回后，提示词里多了一段"用户此前说过的话"。它**不是法律依据**，
但如果模型把这段旧话**逐字抄进答案**，会产生两类问题：

* **看起来像模型在跟用户攀谈**（个性化过头），而不是在回答法律问题；
* 一旦旧话里带条号/结论，就有"**把用户自己的话当成法条**"的风险。

所以 D5 要求：**答案里不许出现与记忆原文逐字重合的长片段**。
判据是**机器筛 + 人工复核**：机器只负责把可疑的挑出来。

⚠️ 2026-09-27：**第一次在真实数据上跑，就发现判据本身会误导人**
--------------------------------------------------------------
第一次跑（107 题 × 两臂 + 真实记忆原文）报了 **13 项可疑**，两臂**一模一样**，
其中最大一项是 126 字的逐字重合：

    「这个问题看起来是法律问题，但我在法规资料里没找到足够贴近的内容，先不硬答、也不给您编条文。」

它**不是记忆泄漏** —— 它是 `legal_rag/engine.py::out_of_scope_reply()` 里
**写死的固定文案**：养记忆时助手就用它回过话（于是进了记忆库），
回答评测题时又用它（于是在答案里）。**同一个代码常量在两边都出现**，与"召回"无关。
两臂完全一致这件事本身就是证据：召回关掉那一臂根本没有记忆进提示词，却报出同一批。

于是本脚本做三件修正（都是"让读数能被人正确解释"）：

1. **模板识别**：从**源码**（`legal_rag/engine.py`、`legal_rag/retrieve/hybrid.py`）
   用 `ast` 取出"用户可见的固定文案"，命中它的重合标为 ``kind="engine_template"``，
   并注明来自哪个文件 —— 单一样本来自源码，**不在这里抄一份**（抄了就会漂移）；
2. **归属**：把每一项重合的片段**回填到具体记忆记录**上，给出 ``matched_role``
   （user / assistant）与 ``message_id``。**只有 user 角色的重合才是"把用户的话抄进答案"**；
   命中 assistant 记录多半是"模型重复了自己过去的措辞"（自回声，另一种现象）；
3. **两臂差分**：一次传两臂（召回关 / 开）时，额外给"**只有某一臂独有**的可疑项"。
   与召回有关的可疑项**必然出现在开的那一臂**；两臂共有的只能来自共同因素（引擎模板等）。

⚠️ **归因只到"片段来自哪条记忆"，不到"因果"**：要断因果必须两臂差分 + 人工看。

退出码
======
* ``0`` 没有**需要人处理**的可疑项（命中引擎固定文案的**不算** —— 那是设计如此，见上）；
* ``1`` 有 ``kind="memory_text"`` 的可疑项 ⇒ **待人复核**（不等于"结论是坏的"：
  同类措辞本来也可能撞车）；
* ``2`` 输入不合法（记忆文件不存在 / 没有 ``texts`` 字段）⇒ **明确报错**，
  **绝不静默通过**（静默通过会给出**假的"没回音"**）。

用法::

    # 1) 先导出一份记忆原文（含 texts / records_detail）
    python scripts/audit_memory_duplicates.py --user eval-recall-ab \\
        --out eval/results/memory-duplicates.json
    # 2) 再对两臂答案做回音检查（一次给两臂，才有差分结论）
    python scripts/check_memory_echo.py --in eval/results/recall-off.jsonl \\
        --in eval/results/recall-on.jsonl --memory eval/results/memory-duplicates.json \\
        --min-chars 30 --out eval/results/memory-echo-recall-ab.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import setup_utf8_stdout  # noqa: E402

#: 「用户可见的固定文案」住在哪几个文件里（要加新的就往这里加）。
#: ⚠️ 用**源码解析**而不是 `import`：这两个模块会拖起 torch/FlagEmbedding 一类重依赖，
#: 而本脚本只想知道"有哪些固定句子"。
DEFAULT_TEMPLATE_FILES = ("legal_rag/engine.py", "legal_rag/retrieve/hybrid.py")

#: 短于这个长度的字符串常量不算"固定文案"（避免把 `"ok"`、`"。"` 之类当成模板）。
MIN_TEMPLATE_CHARS = 12

#: 至少要有这么多个汉字才算"给用户看的文案"。这条筛掉的是**日志格式串**与**字段名**
#: （`"入口RagEngine.prepare(role=%s,...)"`、`"fallback_reason"`）——
#: 它们也是字符串常量，但永远不会出现在答案里，留着只会让"模板"这个口径变浑。
MIN_TEMPLATE_CJK = 4

_CJK = re.compile(r"[\u4e00-\u9fff]")
_FORMAT_SPEC = re.compile(r"%[sdrf]")


def normalize(text: str) -> str:
    """去掉所有空白 —— 中文答案里的换行/空格差异不该影响"是不是同一句话"。"""
    return re.sub(r"\s+", "", text or "")


# ---------------------------------------------------------------- 源码里的固定文案

def _docstring_ids(tree: ast.AST) -> set[int]:
    """收集所有 docstring 节点的 id —— 它们也是字符串常量，但**不是**用户可见文案。"""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            first = body[0] if body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                ids.add(id(first.value))
    return ids


def engine_templates(paths: tuple[str, ...] = DEFAULT_TEMPLATE_FILES) -> list[tuple[str, str]]:
    """从源码取出 ``[(相对路径, 去空白后的文案), ...]``。

    相邻字面量在 AST 里已经被 Python 折成一个常量（``"a" "b"`` ⇒ ``"ab"``），
    所以 `out_of_scope_reply()` 那种拼接能被整段拿到。解析失败**不静默**：打印告警。
    """
    found: list[tuple[str, str]] = []
    for relative in paths:
        path = ROOT / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:                       # pragma: no cover
            print(f"[注意] 读不到/解析不了固定文案来源 {relative}：{exc}（模板识别降级）",
                  file=sys.stderr)
            continue
        skip = _docstring_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue
            text = normalize(node.value)
            if len(text) < MIN_TEMPLATE_CHARS:
                continue
            if len(_CJK.findall(text)) < MIN_TEMPLATE_CJK:      # 日志/字段名，不是给用户看的
                continue
            if _FORMAT_SPEC.search(text):                       # 带 %s/%d 的是日志格式串
                continue
            found.append((relative, text))
    return found


def _template_hit(snippet: str, templates: list[tuple[str, str]]) -> str:
    """命中的模板来自哪个文件；没命中返回空串。"""
    for relative, text in templates:
        if text and (text in snippet or snippet in text):
            return relative
    return ""


# ---------------------------------------------------------------- 逐题判定

def longest_common_substring(answer: str, memory_texts: list[str]) -> tuple[int, str]:
    """返回 ``(最长公共子串长度, 命中的片段)``（片段取自记忆原文那一侧）。

    用标准库 ``difflib.SequenceMatcher.find_longest_match``（不引入新依赖）：
    对每条记忆原文各算一次，取最长的那次。
    """
    target = normalize(answer)
    best_size, best_text = 0, ""
    if not target:
        return 0, ""
    for memory in memory_texts:
        source = normalize(memory)
        if not source:
            continue
        match = SequenceMatcher(None, source, target, autojunk=False).find_longest_match(
            0, len(source), 0, len(target))
        if match.size > best_size:
            best_size = match.size
            best_text = source[match.a:match.a + match.size]
    return best_size, best_text


def _attribute(snippet: str, records: list[dict]) -> tuple[str, str]:
    """把片段回填到具体记忆记录：返回 ``(role, message_id)``；找不到返回 ("", "")。"""
    for record in records:
        if snippet and snippet in normalize(str(record.get("text") or "")):
            return str(record.get("role") or ""), str(record.get("message_id") or "")
    return "", ""


def scan_run(path: Path, memory_texts: list[str], records: list[dict],
             templates: list[tuple[str, str]], min_chars: int) -> dict:
    """评一份答案文件，返回该臂的明细 + 汇总。"""
    records_in_file = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    per_item: list[dict] = []
    skipped = 0
    for record in records_in_file:
        answer = str(record.get("answer") or "")
        if not answer.strip():
            skipped += 1          # 空答案（通常是出错题）不参与判定，但要计数
            continue
        size, snippet = longest_common_substring(answer, memory_texts)
        role, message_id = _attribute(snippet, records)
        template_file = _template_hit(snippet, templates)
        per_item.append({
            "id": record.get("id"), "expect": record.get("expect"),
            "longest_common": size, "snippet": snippet,
            "suspicious": size >= min_chars,
            # 命中源码里的固定文案 ⇒ 与"召回"无关（设计如此）
            "kind": "engine_template" if template_file else "memory_text",
            "template_file": template_file or None,
            "matched_role": role or None, "matched_message_id": message_id or None,
        })

    flagged = [item for item in per_item if item["suspicious"]]
    actionable = [item for item in flagged if item["kind"] == "memory_text"]
    sizes = [item["longest_common"] for item in per_item]
    by_kind: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for item in flagged:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        key = item["matched_role"] or "unknown"
        by_role[key] = by_role.get(key, 0) + 1
    return {
        "input": str(path),
        "items": len(per_item), "skipped_empty": skipped,
        "suspicious_items": [item["id"] for item in flagged],
        "actionable_items": [item["id"] for item in actionable],
        #: ⭐ 真正要看的一格：**逐字抄了"用户自己说过的话"** 的题目
        "user_role_echo_items": [item["id"] for item in actionable
                                 if item["matched_role"] == "user"],
        "by_kind": by_kind, "by_matched_role": by_role,
        "max_longest_common": max(sizes) if sizes else 0,
        "mean_longest_common": round(statistics.fmean(sizes), 1) if sizes else 0.0,
        "p95_longest_common": (sorted(sizes)[int(len(sizes) * 0.95)] if sizes else 0),
        "per_item": per_item,
    }


def _differential(runs: dict[str, dict]) -> dict:
    """两臂差分：**只有某一臂独有**的可疑项才可能与那一臂的开关有关。

    召回开/关只有一处不同（提示词里有没有记忆分区）⇒ 与召回有关的可疑项**必然**只在
    开的那一臂出现。两臂共有的只能来自共同因素（引擎固定文案、模型自身措辞）。
    """
    names = list(runs)
    if len(names) < 2:
        return {}
    sets = {name: set(runs[name]["suspicious_items"]) for name in names}
    shared = set.intersection(*sets.values())
    only_in: dict[str, list] = {}
    for name in names:
        others = set().union(*[sets[other] for other in names if other != name])
        only_in[name] = sorted(sets[name] - others)
    return {
        "runs": names,
        "shared_in_all_runs": sorted(shared),
        "only_in": only_in,
        "note": ("只有某一臂独有的可疑项才可能与那一臂的开关（召回）有关；"
                 "两臂共有的只能来自共同因素 —— 先看 by_kind 里有多少是引擎固定文案"),
    }


def main(argv: list[str] | None = None) -> int:
    setup_utf8_stdout()          # 把 stdout/stderr 切成 UTF-8，避免 GBK 控制台丢中文
    parser = argparse.ArgumentParser(description="记忆回音检查（答案有没有照抄记忆原文）")
    parser.add_argument("--in", dest="inputs", action="append", required=True,
                        help="评测结果 JSONL（可多次；一次给两臂才有差分结论）")
    parser.add_argument("--memory", required=True,
                        help="记忆原文 JSON（来自 audit_memory_duplicates.py --out）")
    parser.add_argument("--min-chars", type=int, default=30,
                        help="判为可疑回音的最长公共子串长度（默认 30 字）")
    parser.add_argument("--out", default="", help="把报告写成 JSON")
    args = parser.parse_args(argv)

    memory_path = Path(args.memory)
    if not memory_path.is_file():
        print(f"!! 记忆原文文件不存在：{memory_path}\n"
              f"   先跑：python scripts/audit_memory_duplicates.py --user <账号> "
              f"--out {memory_path}", file=sys.stderr)
        return 2
    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    memory_texts = [str(t) for t in (payload.get("texts") or []) if str(t).strip()]
    if not memory_texts:
        print(f"!! {memory_path} 里没有 texts 字段（或为空）—— 无法做回音检查。\n"
              f"   该字段由 audit_memory_duplicates.py 导出；若用的是旧文件请重新导出。",
              file=sys.stderr)
        return 2
    records = [r for r in (payload.get("records_detail") or []) if isinstance(r, dict)]
    templates = engine_templates()
    print(f"记忆原文 {len(memory_texts)} 条（含归属信息 {len(records)} 条）；"
          f"判定阈值 = 最长公共子串 >= {args.min_chars} 字")
    print(f"源码固定文案 {len(templates)} 段（来自 {', '.join(DEFAULT_TEMPLATE_FILES)}）")

    report: dict = {"memory_file": str(memory_path), "memory_texts": len(memory_texts),
                    "records_detail": len(records), "templates": len(templates),
                    "min_chars": args.min_chars, "runs": {}}
    for input_path in args.inputs:
        path = Path(input_path)
        summary = scan_run(path, memory_texts, records, templates, args.min_chars)
        report["runs"][path.stem] = summary

        flagged = [item for item in summary["per_item"] if item["suspicious"]]
        print(f"\n=== {path.name}：{summary['items']} 题（跳过空答案 {summary['skipped_empty']}）")
        print(f"    最长公共子串：max={summary['max_longest_common']} "
              f"mean={summary['mean_longest_common']} p95={summary['p95_longest_common']}")
        if not flagged:
            print("    [OK] 没有题目与记忆原文有超阈值的逐字重合")
            continue
        print(f"    可疑 {len(flagged)} 题：其中**引擎固定文案** {summary['by_kind'].get('engine_template', 0)} 题、"
              f"**记忆正文** {summary['by_kind'].get('memory_text', 0)} 题")
        print(f"    归属：{summary['by_matched_role'] or '{}'}")
        if summary["user_role_echo_items"]:
            print(f"    [注意] 逐字抄了**用户自己说过的话**：{summary['user_role_echo_items']}（要人工看）")
        for item in flagged[:10]:
            tag = "引擎文案" if item["kind"] == "engine_template" else f"记忆/{item['matched_role']}"
            print(f"      {item['id']}  重合 {item['longest_common']} 字 [{tag}]："
                  f"『{item['snippet'][:36]}』")

    if len(report["runs"]) > 1:
        differential = _differential(report["runs"])
        report["differential"] = differential
        print("\n=== 两臂差分（与召回有关的可疑项必然只在『开』那一臂出现）")
        print(f"    两臂共有：{len(differential['shared_in_all_runs'])} 题")
        for name, ids in differential["only_in"].items():
            print(f"    只在 {name}：{len(ids)} 题 {ids[:8]}")
        if not any(differential["only_in"].values()):
            print("    [结论] 两臂**完全一致**，所以这些重合与召回**无关**（共同因素："
                  "引擎固定文案、或模型复述自己过去的措辞）；与召回有关的重合必须只在开的那一臂")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n证据已写：{out}")

    # 退出码：只有"记忆正文被照抄"才需要人处理；命中引擎固定文案是设计如此（见文件头）
    actionable = sum(len(run["actionable_items"]) for run in report["runs"].values())
    if actionable:
        print(f"[注意] {actionable} 项**记忆正文**重合待人复核（不是『结论坏了』）")
    return 1 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
