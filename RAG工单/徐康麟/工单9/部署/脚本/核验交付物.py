# -*- coding: utf-8 -*-
"""工单9 交付物核验：逐项检查五个中文目录与硬性产物是否齐备。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

核验内容（与 `_共享约定.md` §1 的交付目录契约 + 工单验收标准对应）：
    * 五个一级目录存在：设计 / 研发 / 测试 / 优化 / 部署；
    * 设计文档 10 份齐备；
    * 研发产物：app/graphrag_opt、评估、scripts、requirements.txt、README.md，
      且**所有源码头部含工单编号**；
    * 测试：pytest 文件、测试数据（逐题 jsonl/csv/汇总 json、稳定性、容错、多语言）、
      **测试截图 ≥5 张且文件名为中文**；
    * 优化：方案文档 + 脚本目录 + 评估结果（对比 csv/md/json）；
    * 部署：脚本 / 配置 / 日志 / README / 本文档；
    * **资源纪律自检**：源码中不得出现 multiprocessing / ProcessPoolExecutor / Pool( / joblib / n_jobs；
    * 只读边界：产物路径不得落在 工单7 / 工单8 下（本工单只写自己目录）。

用法：
    pwsh -NoProfile -File run_py.ps1 部署/脚本/核验交付物.py
    pwsh -NoProfile -File run_py.ps1 部署/脚本/核验交付物.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"
ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DIRS = ["设计", "研发", "测试", "优化", "部署"]
DESIGN_DOCS = ["工单需求原文.md", "需求分析.md", "优化方案设计.md", "系统架构.md",
               "评估指标定义.md", "接口设计.md", "技术选型.md", "环境事实.md",
               "验收标准.md", "README.md"]
DEV_FILES = ["requirements.txt", "README.md"]
DEV_DIRS = ["app/graphrag_opt", "app/ui", "评估", "scripts"]
EVAL_FILES = ["参考答案集.jsonl", "参考答案集核对报告.json", "ragas近似.py", "run_eval_opt.py"]
OPT_FILES = ["优化方案与效果.md"]
OPT_DIRS = ["脚本", "评估结果"]
OPT_RESULT_FILES = ["优化前后对比.json", "优化前后对比.md", "优化前后对比.csv",
                    "优化后_逐题.jsonl", "优化前_逐题.jsonl", "汇总_after.json", "汇总_before.json"]
DEPLOY_FILES = ["README.md", "交付物核验.md"]
DEPLOY_DIRS = ["脚本", "配置", "日志"]
TEST_DATA_FILES = ["稳定性_连续查询.json", "容错降级.json", "多语言问答.jsonl", "多语言汇总.json"]

#: 资源纪律禁用模式（`_共享约定.md` §5.5 规则 2/10）
#: 说明：本文件自身会包含这些字面量（它就是检查器），因此扫描时**跳过自己**；
#: 同时跳过注释行与正则定义行，避免「检查器自证有罪」的假阳性。
FORBIDDEN = re.compile(r"\bmultiprocessing\b|ProcessPoolExecutor|(?<![\w.])Pool\(|(?<![\w.])joblib\b|n_jobs")
SELF_PATH = Path(__file__).resolve()

#: 源码必须带头部工单编号注释的文件类型
SOURCE_SUFFIX = {".py", ".ps1"}

#: **本工单自己新增/修改的**源码范围 —— 这些文件必须含本工单编号；
#: 其余文件是从 工单7/工单8 复用的**上游代码**（只读），
#: 它们保留**各自的**工单编号（工单7/工单8 的编号），本工单不改写上游文件。
OWNED_PREFIXES = ("研发/app/graphrag_opt", "研发/app/ui", "研发/评估", "研发/scripts",
                  "测试", "部署/脚本", "优化/脚本")


def _check(path: Path, *, name: str, kind: str, results: list[dict[str, Any]]) -> bool:
    exists = path.exists()
    results.append({"项": f"{kind}：{name}", "路径": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "存在": exists, "类型": kind})
    return exists


def scan_work_order_comment(results: list[dict[str, Any]]) -> dict[str, Any]:
    """检查源码头部注释是否含工单编号（`_共享约定.md` §4.2 硬性要求）。

    分两类判定（**如实区分「我写的」与「复用上游的」**）：
        * **本工单新增/修改的文件**（``OWNED_PREFIXES``）—— 必须含
          ``人工智能NLP-RAG-Graph RAG 优化任务``；
        * **从 工单7/工单8 复用的上游文件** —— 保留其**原工单编号**即可
          （本工单不修改上游文件，也不该把别人的编号改掉）。
    """
    missing_own: list[str] = []
    upstream_no_header: list[str] = []
    upstream_ok = 0
    owned_checked = 0
    for path in sorted((ROOT / "研发").rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIX:
            continue
        if any(part in ("第三方库", "__pycache__") for part in path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            head = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:40])
        except Exception:  # noqa: BLE001
            missing_own.append(f"{rel}（读取失败）")
            continue
        owned = any(rel.startswith(p) for p in OWNED_PREFIXES)
        if owned:
            owned_checked += 1
            if WORK_ORDER not in head:
                missing_own.append(rel)
            continue
        # 上游复用文件：必须至少带一个「工单编号」头（内容不限）
        if "工单编号" in head:
            upstream_ok += 1
        else:
            upstream_no_header.append(rel)
    return {"本工单文件数": owned_checked, "本工单缺失编号": missing_own,
            "上游复用文件数": upstream_ok, "上游缺工单编号头": upstream_no_header,
            "通过": not missing_own and not upstream_no_header}


def scan_forbidden(results: list[dict[str, Any]]) -> dict[str, Any]:
    """资源纪律自检：源码中不得出现进程池/并行库关键字。

    跳过：本检查器自身、注释行、以及**说明这些禁用的是哪几个词**的字符串定义行
    （否则检查器会「自证有罪」）。
    """
    hits: list[str] = []
    checked = 0
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in ("__pycache__", "第三方库", ".venv") for part in path.parts):
            continue
        try:
            if path.resolve() == SELF_PATH:
                continue
        except OSError:
            continue
        checked += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for idx, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "FORBIDDEN" in stripped:
                continue
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{idx}: {stripped[:90]}")
    return {"检查文件数": checked, "命中": hits, "通过": not hits}


def scan_shots(results: list[dict[str, Any]]) -> dict[str, Any]:
    """截图核验：数量 ≥5，且文件名必须是中文（不得是纯数字/纯英文/拼音）。"""
    shot_dir = ROOT / "测试" / "测试截图"
    pngs = sorted(shot_dir.glob("*.png")) if shot_dir.is_dir() else []
    bad_names: list[str] = []
    for png in pngs:
        stem = png.stem
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stem))
        if not has_cjk:
            bad_names.append(png.name)
    return {"截图数": len(pngs), "文件": [p.name for p in pngs],
            "非中文命名": bad_names, "数量达标(>=5)": len(pngs) >= 5,
            "命名达标": not bad_names,
            "通过": len(pngs) >= 5 and not bad_names}


def scan_boundary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """只读边界：本工单目录下不得出现指向 工单7/工单8 的**写入型**引用。

    这里做「配置里是否写死了别人的路径用于写入」的静态扫描（Junction 与只读读取不算）。
    """
    notes: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for idx, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"工单[78].*(write_text|open\(.*[\"']w|\.mkdir|save_graph|_write_jsonl)", line):
                notes.append(f"{path.relative_to(ROOT)}:{idx}: {line.strip()[:90]}")
    return {"疑似越界写入": notes, "通过": not notes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 交付物核验")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    for name in REQUIRED_DIRS:
        _check(ROOT / name, name=name, kind="一级目录", results=results)
    for name in DESIGN_DOCS:
        _check(ROOT / "设计" / name, name=name, kind="设计文档", results=results)
    for name in DEV_DIRS:
        _check(ROOT / "研发" / name, name=name, kind="研发目录", results=results)
    for name in DEV_FILES:
        _check(ROOT / "研发" / name, name=name, kind="研发文件", results=results)
    for name in EVAL_FILES:
        _check(ROOT / "研发" / "评估" / name, name=name, kind="评估产物", results=results)
    for name in OPT_FILES:
        _check(ROOT / "优化" / name, name=name, kind="优化文档", results=results)
    for name in OPT_DIRS:
        _check(ROOT / "优化" / name, name=name, kind="优化目录", results=results)
    for name in OPT_RESULT_FILES:
        _check(ROOT / "优化" / "评估结果" / name, name=name, kind="评估结果", results=results)
    for name in DEPLOY_FILES:
        _check(ROOT / "部署" / name, name=name, kind="部署文档", results=results)
    for name in DEPLOY_DIRS:
        _check(ROOT / "部署" / name, name=name, kind="部署目录", results=results)
    for name in TEST_DATA_FILES:
        _check(ROOT / "测试" / "测试数据" / name, name=name, kind="测试数据", results=results)
    _check(ROOT / "测试" / "test_工单9优化.py", name="test_工单9优化.py", kind="测试脚本", results=results)
    _check(ROOT / "测试" / "测试报告.md", name="测试报告.md", kind="测试文档", results=results)

    extra = {
        "源码工单编号": scan_work_order_comment(results),
        "资源纪律自检": scan_forbidden(results),
        "截图核验": scan_shots(results),
        "只读边界": scan_boundary(results),
    }

    missing = [r for r in results if not r["存在"]]
    ok = (not missing) and all(v["通过"] for v in extra.values())

    if args.json:
        print(json.dumps({"工单": WORK_ORDER, "通过": ok, "缺失": missing, "附加检查": extra,
                          "明细": results}, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    print("=" * 76)
    print(f"{WORK_ORDER} —— 交付物核验")
    print("=" * 76)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(str(row["类型"]), []).append(row)
    for kind, rows in groups.items():
        good = sum(1 for r in rows if r["存在"])
        mark = "✅" if good == len(rows) else "❌"
        print(f"{mark} {kind}：{good}/{len(rows)}")
        for row in rows:
            if not row["存在"]:
                print(f"     [缺失] {row['路径']}")
    print("-" * 76)
    for name, info in extra.items():
        mark = "✅" if info["通过"] else "❌"
        detail = {k: v for k, v in info.items() if k != "通过"}
        print(f"{mark} {name}：{json.dumps(detail, ensure_ascii=False)[:220]}")
    print("=" * 76)
    print(f"总判定：{'✅ 全部通过' if ok else '❌ 存在缺失项'}"
          f"（缺失 {len(missing)} 项）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
