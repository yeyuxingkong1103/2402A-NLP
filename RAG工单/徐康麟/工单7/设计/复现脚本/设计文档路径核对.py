# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-图像内容解析及检索优化
用途：【设计文档路径存在性核对护栏】
      扫描设计文档里出现的**仓库内文件路径**，逐条核对是否真实存在，
      防止「文档指向不存在的脚本/产物」这类缺陷（T8 曾发现 `研发/scripts/evaluate.py` 一例，实际应为
      `测试/在线/run_16q_eval.py`）。存在性不一致时以**非零退出码**报出。

运行（工作目录 = E:\\gao6gongdan\\工单4）：
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/设计文档路径核对.py
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/设计文档路径核对.py --all-docs   # 连 研发/测试/优化/部署 的 md 一起扫
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/设计文档路径核对.py --json       # 机器可读输出

退出码：0 = 全部路径存在（模板/白名单已显式登记）；1 = 存在失效路径；2 = 用法/IO 错误。

设计约束（工单4 硬性纪律）：
  1. 本文件所有注释含工单编号「人工智能NLP-RAG-图像内容解析及检索优化」，注释一律中文。
  2. 每个函数入口/出口记录结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。
  3. 禁止静默失败：异常必须记日志后传播或显式降级，绝不 pass。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Callable

# 工单编号常量：所有日志与注释均以它标记归属
GONGDAN = "人工智能NLP-RAG-图像内容解析及检索优化"

# 仓库根目录（本文件位于 设计/复现脚本/ 下，向上两级即根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认扫描的文档（设计文档 + 顶层 README）
默认文档 = [
    "设计/环境事实.md",
    "设计/系统架构.md",
    "设计/接口设计.md",
    "设计/验收标准.md",
    "设计/图像解析方案.md",
    "设计/README.md",
    "设计/工单需求原文.md",
    "README.md",
]

# --all-docs 时追加扫描的文档目录（只取 .md，避免把代码里的字符串当文档引用）
附加目录 = ["研发", "测试", "优化", "部署"]

# 五文件夹名——只核对以它们开头的仓库内路径，天然排除 URL 与外部路径
顶层目录 = ("设计", "研发", "测试", "优化", "部署")

# 扩展名（**长扩展名前置于短扩展名前**，否则 .jsonl 会被截成 .json——本脚本第一版就栽在这里）
扩展名 = (
    "safetensors|jsonl|json|yaml|yml|ps1|sh|txt|csv|npy|pkl|py|md|png|whl|bin"
)

# 路径解析：形如 设计/xxx/yyy.py（各段允许中文、字母、数字、下划线、短横、点）
_段 = r"[A-Za-z0-9_\-\.\u4e00-\u9fa5]+"
_路径 = re.compile(r"(?:" + _段 + r"/)+" + _段 + r"\.(?:" + 扩展名 + r")")

# 命名模板/占位模式：这类不是真实路径，显式跳过（跳过也要打印，不静默）
_模板 = re.compile(r"(NN|XX|\*|<|>|…|\?|\{|今天是)")

# 显式白名单：允许「不存在」的路径，每条必须写明理由（不写理由的条目一出现就会打印为待处理）
允许不存在: dict[str, str] = {
    "测试/测试数据/cli_results.jsonl": "CLI `--out` 的示例取值，属运行时生成产物，仓库内不预置",
    "部署/日志/batch.jsonl": "`部署/README.md` 里 `--ask-file ... --out` 的示例取值，属运行时生成产物",
}

_结果: dict[str, Any] = {"存在": [], "失效": [], "模板跳过": [], "白名单放行": []}


def _log(事件: str, **字段: Any) -> None:
    """工单4 结构化日志：单行 JSON，含工单编号，禁止只写「开始/结束」。"""
    payload = {"工单": GONGDAN, "事件": 事件}
    payload.update(字段)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def 步骤(名称: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：记录函数入口/出口的结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。"""

    def 包装(fn: Callable[..., Any]) -> Callable[..., Any]:
        def 内层(*args: Any, **kwargs: Any) -> Any:
            输入摘要 = {
                "args": [str(a)[:120] for a in args],
                "kwargs": {k: str(v)[:120] for k, v in kwargs.items()},
            }
            _log("函数入口", 函数=fn.__qualname__, 步骤=名称, 输入摘要=输入摘要)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:  # 禁止静默失败：记日志后继续传播
                _log(
                    "函数异常",
                    函数=fn.__qualname__,
                    步骤=名称,
                    耗时ms=round((time.perf_counter() - t0) * 1000, 2),
                    异常类型=type(exc).__name__,
                    异常信息=str(exc)[:400],
                    异常堆栈=traceback.format_exc()[:1200],
                )
                raise
            _log(
                "函数出口",
                函数=fn.__qualname__,
                步骤=名称,
                耗时ms=round((time.perf_counter() - t0) * 1000, 2),
                输出摘要=str(out)[:200],
            )
            return out

        return 内层

    return 包装


@步骤("收集文档")
def 收集文档(全部: bool) -> list[str]:
    """收集待扫描的文档相对路径；--all-docs 时把四个交付目录下的 md 也纳入。"""
    文档 = list(默认文档)
    if 全部:
        for d in 附加目录:
            base = os.path.join(ROOT, d)
            for 根, _, 文件们 in os.walk(base):
                if "__pycache__" in 根 or os.sep + "data" + os.sep + "models" in 根:
                    continue
                for f in 文件们:
                    if f.endswith(".md"):
                        文档.append(os.path.relpath(os.path.join(根, f), ROOT).replace("\\", "/"))
    return sorted(set(文档))


@步骤("扫描路径引用")
def 扫描文档(文档列表: list[str]) -> dict[str, list[tuple[str, int]]]:
    """逐行提取仓库内路径引用；返回 {路径: [(文档, 行号), …]}。"""
    引用: dict[str, list[tuple[str, int]]] = {}
    for rel in 文档列表:
        绝对 = os.path.join(ROOT, rel)
        if not os.path.isfile(绝对):
            _log("文档缺失", 文件=rel, 处理="跳过该文档（本身不存在是另一类问题，由白名单/人工判断）")
            continue
        with open(绝对, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                for m in _路径.finditer(line):
                    p = m.group(0).strip("`").rstrip(".,;:)）】")
                    if not p.startswith(顶层目录):
                        continue
                    引用.setdefault(p, []).append((rel, i))
    return 引用


@步骤("核对存在性")
def 核对(引用: dict[str, list[tuple[str, int]]]) -> None:
    """对每条引用做存在性判定，并按「存在 / 模板 / 白名单 / 失效」分流。"""
    for p, 位置 in sorted(引用.items()):
        # 1) 命名模板不是真实路径（如 测试/测试截图/题号NN_关键词_类型.png）
        if _模板.search(p):
            _结果["模板跳过"].append({"路径": p, "位置": ["%s:%d" % x for x in 位置[:3]]})
            continue
        # 2) 真实存在（文件或目录都算通过）
        if os.path.exists(os.path.join(ROOT, p)):
            _结果["存在"].append({"路径": p, "引用处数": len(位置)})
            continue
        # 3) 显式白名单
        if p in 允许不存在:
            _结果["白名单放行"].append(
                {"路径": p, "理由": 允许不存在[p], "位置": ["%s:%d" % x for x in 位置[:3]]}
            )
            continue
        # 4) 失效路径：必须报出，且给出全部引用位置方便逐处修
        _结果["失效"].append({"路径": p, "位置": ["%s:%d" % x for x in 位置]})


@步骤("核对 run_py 命令")
def 核对命令(文档列表: list[str]) -> None:
    """额外核对「run_py.ps1 <路径>」式复现命令里引用的脚本是否存在（照抄会失败的命令）。"""
    漏网: list[dict[str, Any]] = []
    for rel in 文档列表:
        绝对 = os.path.join(ROOT, rel)
        if not os.path.isfile(绝对):
            continue
        for i, line in enumerate(open(绝对, encoding="utf-8"), 1):
            if "run_py.ps1" not in line:
                continue
            for m in re.finditer(r"run_py\.ps1\s+([^\s`\"]+)", line):
                目标 = m.group(1).strip()
                if not 目标.startswith(顶层目录) or ":" in 目标:
                    continue
                if not os.path.exists(os.path.join(ROOT, 目标)):
                    漏网.append({"路径": 目标, "位置": "%s:%d" % (rel, i), "行": line.strip()[:100]})
    _结果["命令引用失效"] = 漏网


def main() -> int:
    """入口：扫描 → 核对 → 打印结论 → 按结果返回退出码。"""
    ap = argparse.ArgumentParser(description="工单4 设计文档路径存在性核对")
    ap.add_argument("--all-docs", action="store_true", help="连 研发/测试/优化/部署 下的 md 一起扫")
    ap.add_argument("--json", action="store_true", help="只输出机器可读 JSON")
    args = ap.parse_args()
    _log("脚本入口", 脚本="设计文档路径核对.py", 参数=vars(args), 仓库根=ROOT)

    try:
        文档列表 = 收集文档(args.all_docs)
        引用 = 扫描文档(文档列表)
        核对(引用)
        核对命令(文档列表)
    except Exception as exc:
        _log("脚本异常", 异常类型=type(exc).__name__, 异常信息=str(exc)[:400], 异常堆栈=traceback.format_exc()[:1200])
        print(f"[错误] 路径核对执行失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    失效数 = len(_结果["失效"]) + len(_结果.get("命令引用失效", []))
    退出码 = 0 if 失效数 == 0 else 1

    if args.json:
        print(json.dumps({**_结果, "失效总数": 失效数}, ensure_ascii=False, indent=2))
        _log("脚本出口", 脚本="设计文档路径核对.py", 存在=len(_结果["存在"]), 失效=失效数, 退出码=退出码)
        return 退出码

    print()
    print("=" * 78)
    print("工单4 设计文档路径存在性核对结论")
    print("=" * 78)
    print(f"扫描文档数     : {len(文档列表)}")
    print(f"路径引用条目   : {len(引用)}")
    print(f"存在           : {len(_结果['存在'])}")
    print(f"命名模板跳过   : {len(_结果['模板跳过'])}")
    print(f"白名单放行     : {len(_结果['白名单放行'])}")
    print(f"命令引用失效   : {len(_结果.get('命令引用失效', []))}")
    print(f"失效路径       : {len(_结果['失效'])}")
    if _结果["失效"]:
        print("\n--- ❌ 失效路径明细（必须修）---")
        for 项 in _结果["失效"]:
            print("  %-56s <- %s" % (项["路径"], " , ".join(项["位置"][:6])))
    if _结果.get("命令引用失效"):
        print("\n--- ❌ run_py 命令引用失效明细（照抄会失败）---")
        for 项 in _结果["命令引用失效"]:
            print("  %-56s <- %s" % (项["路径"], 项["位置"]))
    if _结果["模板跳过"]:
        print("\n--- 命名模板（非真实路径，已跳过）---")
        for 项 in _结果["模板跳过"]:
            print("  %-56s <- %s" % (项["路径"], " , ".join(项["位置"])))
    if _结果["白名单放行"]:
        print("\n--- 白名单放行（理由已登记）---")
        for 项 in _结果["白名单放行"]:
            print("  %-56s 理由：%s" % (项["路径"], 项["理由"]))
    print("\n结论：" + ("✅ 通过（文档引用的路径全部存在）" if 退出码 == 0 else "❌ 不通过（存在失效路径）"))
    _log("脚本出口", 脚本="设计文档路径核对.py", 存在=len(_结果["存在"]), 失效=失效数, 退出码=退出码)
    return 退出码


if __name__ == "__main__":
    sys.exit(main())
