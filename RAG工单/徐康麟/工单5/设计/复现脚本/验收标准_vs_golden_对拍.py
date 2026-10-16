# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-图像内容解析及检索优化
用途：【设计文档 ←→ golden 漂移对拍护栏】
      把 设计/验收标准.md §3 的 16 题逐题页级别，与 研发/data/eval/golden_qa.jsonl
      的 evidence_pages_complete / supporting_pages / related_pages / evidence_pages /
      evidence_pages_union / evidence_pages_primary 逐字段对拍；一旦 golden 变动而设计文档
      未跟随（或反之），立刻以【非零退出码】报出，避免靠人眼发现漂移。

运行（工作目录 = E:\\gao6gongdan\\工单4）：
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/验收标准_vs_golden_对拍.py
    # golden 的 union 字段已知有 1 处待修（id 957，T3 修复中）；修复前可显式白名单放行：
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/验收标准_vs_golden_对拍.py --known-issue union:id957

退出码：0 = 全部通过（含被显式白名单放行的已知问题）；1 = 存在漂移或不变量违反；2 = 用法/ IO 错误。

设计约束（工单4 硬性纪律）：
  1. 本文件所有注释含工单编号「人工智能NLP-RAG-图像内容解析及检索优化」，注释一律中文。
  2. 每个函数入口/出口记录结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。
  3. 禁止静默失败：异常必须记日志后传播或显式降级，绝不 pass。
"""

from __future__ import annotations

import argparse
import hashlib
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
默认文档 = os.path.join(ROOT, "设计", "验收标准.md")
默认golden = os.path.join(ROOT, "研发", "data", "eval", "golden_qa.jsonl")

# golden 中与证据页相关的字段（本脚本对拍范围）
页字段 = (
    "evidence_pages_complete",
    "supporting_pages",
    "related_pages",
    "evidence_pages",
    "evidence_pages_union",
    "evidence_pages_primary",
)

_结果: dict[str, Any] = {"漂移": [], "不变量违反": [], "已知问题放行": [], "已核对题数": 0}


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


@步骤("读取 golden")
def 读golden(路径: str) -> tuple[dict[int, dict[str, Any]], str]:
    """读取 golden jsonl，返回 (按 id 索引的字典, 文件 sha256)。缺字段不补默认值，交由对拍判定。"""
    raw = open(路径, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    表: dict[int, dict[str, Any]] = {}
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception as exc:
            # 显式降级：坏行必须报错定位，不得跳过
            raise ValueError(f"golden 第 {lineno} 行 JSON 解析失败: {exc}") from exc
        表[int(d["id"])] = d
    return 表, sha


@步骤("解析 §3 表格")
def 解析设计表(文档路径: str) -> dict[int, dict[str, Any]]:
    """解析 设计/验收标准.md 的 §3 各表格，抽取每题的证据页单元格并拆成 完整/支撑/相关 三级。

    稳健性要求（本脚本第一版曾因解析不严而误报，故此处严格处理）：
      · 证据页单元格 = 含「完整/支撑/相关」的单元格；否则取「去掉 PDF1/PDF2 后仅剩数字与顿号」的单元格
      · 先剔除括号注释（全角/半角），否则「（仅含「4 部门」）」里的 4 会被误当页码
      · 只认反引号包裹的整数页码，避免把 42.35、15,000.00 之类数值当成页码
    """
    md = open(文档路径, encoding="utf-8").read()
    表: dict[int, dict[str, Any]] = {}
    for line in md.splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        题号 = int(m.group(1))
        单元格 = [c.strip() for c in m.group(2).split("|")]
        # 级别标志必须是「页级别标记」形式（后接反引号页码或 —），
        # 否则「信息系统**相关**的电子元器件…」这类普通中文会被误当级别单元格（第一版就栽在这里）
        证据页格 = next((c for c in 单元格 if re.search(r"(完整|支撑|相关)\s*\*{0,2}\s*(`|—)", c)), None)
        if 证据页格 is None:
            # 扁平行：证据页列 = 去掉文件标记与括号注释后只含数字/顿号/反引号的单元格
            候选 = [
                c
                for c in 单元格
                if re.fullmatch(r"[\s\d、`]+", _去括号(c).replace("PDF1", "").replace("PDF2", ""))
                and re.search(r"\d", c)
            ]
            证据页格 = 候选[0] if 候选 else None
        if 证据页格 is None:
            _log("解析告警", 题号=题号, 原因="未能定位证据页单元格", 行=str(line)[:160])
            continue
        表[题号] = {"单元格": 证据页格, "分级": _拆分级(证据页格), "主证": _取主证(证据页格)}
    return 表


def _去括号(文本: str) -> str:
    """剔除全角/半角括号及其内容，并去掉 Markdown 强调符，避免注释里的数字污染页码集合。"""
    out = re.sub(r"（[^）]*）", "", 文本)
    out = re.sub(r"\([^)]*\)", "", out)
    return out.replace("**", "")


def _拆分级(单元格: str) -> dict[str, list[int]]:
    """把证据页单元格拆成 完整/支撑/相关 三个页码集合；只认反引号包裹的整数。"""

    def 片段(标志: str, 终止符: tuple[str, ...]) -> list[int]:
        if 标志 not in 单元格:
            return []
        剩余 = _去括号(单元格).split(标志, 1)[1]
        for t in 终止符:
            if t in 剩余:
                剩余 = 剩余.split(t, 1)[0]
        return sorted(int(x) for x in re.findall(r"`(\d+)`", 剩余))

    完整 = 片段("完整", ("支撑", "相关"))
    支撑 = 片段("支撑", ("相关",))
    相关 = 片段("相关", ())
    # 扁平行（无任何级别标志）：整格页码视为 evidence_pages 口径；
    # 扁平行既可能写成 `22`、`24`，也可能写成 22、24，两种写法都要认
    if not (完整 or 支撑 or 相关):
        裸 = _去括号(单元格).replace("PDF1", "").replace("PDF2", "")
        扁平 = sorted(set(int(x) for x in re.findall(r"`(\d+)`", 裸)) or set(int(x) for x in re.findall(r"\d+", 裸)))
        return {"扁平": 扁平}
    return {"complete": 完整, "supporting": 支撑, "related": 相关}


def _取主证(单元格: str) -> int | None:
    """从「（主证 `N`）」里取主证页；缺失返回 None（主证只影响引用优先级，不影响命中判定）。"""
    m = re.search(r"主证\s*`?(\d+)`?", 单元格)
    return int(m.group(1)) if m else None


@步骤("对拍校验")
def 对拍(golden表: dict[int, dict[str, Any]], 设计表: dict[int, dict[str, Any]]) -> None:
    """逐题对拍设计文档 §3 与 golden 字段，并校验 golden 自身的两条页集合等式。"""
    # 1) 题目集合必须一致
    g仅 = sorted(set(golden表) - set(设计表))
    设仅 = sorted(set(设计表) - set(golden表))
    if g仅:
        _结果["漂移"].append({"项": "题目缺失", "说明": f"golden 有但设计文档 §3 缺：{g仅}"})
    if 设仅:
        _结果["漂移"].append({"项": "题目多余", "说明": f"设计文档 §3 有但 golden 无：{设仅}"})

    # 2) 逐题逐级对拍
    for 题号 in sorted(set(golden表) & set(设计表)):
        g, s = golden表[题号], 设计表[题号]
        期望 = {
            "complete": sorted(g.get("evidence_pages_complete") or []),
            "supporting": sorted(g.get("supporting_pages") or []),
            "related": sorted(g.get("related_pages") or []),
        }
        实得 = s["分级"]
        if "扁平" in 实得:
            # 扁平行只能与 evidence_pages（=完整+支撑）比较，并额外要求 golden 侧无 related
            期望扁平 = sorted(g.get("evidence_pages") or [])
            if 实得["扁平"] != 期望扁平:
                _结果["漂移"].append(
                    {"题号": 题号, "项": "扁平行页码集合", "文档": 实得["扁平"], "golden": 期望扁平}
                )
            if 期望["related"]:
                _结果["漂移"].append(
                    {
                        "题号": 题号,
                        "项": "扁平行信息不足",
                        "文档": 实得["扁平"],
                        "golden": 期望,
                        "说明": "golden 该题有 related_pages，设计文档必须写成带级别的形式",
                    }
                )
        else:
            for 级 in ("complete", "supporting", "related"):
                if 实得.get(级, []) != 期望[级]:
                    _结果["漂移"].append(
                        {"题号": 题号, "项": f"{级} 页集合", "文档": 实得.get(级, []), "golden": 期望[级]}
                    )
            主证 = s["主证"]
            g主证 = g.get("evidence_pages_primary") or []
            if 主证 is not None and g主证 and 主证 not in g主证:
                _结果["漂移"].append(
                    {"题号": 题号, "项": "主证页", "文档": 主证, "golden": g主证}
                )
        _结果["已核对题数"] += 1

    # 3) golden 自身不变量
    for 题号, g in sorted(golden表.items()):
        c = set(g.get("evidence_pages_complete") or [])
        sp = set(g.get("supporting_pages") or [])
        r = set(g.get("related_pages") or [])
        p = set(g.get("evidence_pages") or [])
        u = set(g.get("evidence_pages_union") or [])
        if (c | sp) != p:
            _结果["不变量违反"].append(
                {"题号": 题号, "等式": "evidence_pages == complete + supporting",
                 "golden实际": sorted(p), "期望": sorted(c | sp)}
            )
        if (c | sp | r) != u:
            _结果["不变量违反"].append(
                {"题号": 题号, "等式": "evidence_pages_union == complete + supporting + related",
                 "golden实际": sorted(u), "期望": sorted(c | sp | r), "缺失": sorted((c | sp | r) - u),
                 "修复归属": "T3（golden 构造方）"}
            )


def main() -> int:
    """入口：对拍 → 打印结论 → 按结果返回退出码。"""
    ap = argparse.ArgumentParser(description="工单4 设计文档 ←→ golden 漂移对拍")
    ap.add_argument("--doc", default=默认文档, help="设计文档路径（默认 设计/验收标准.md）")
    ap.add_argument("--golden", default=默认golden, help="golden jsonl 路径")
    ap.add_argument(
        "--known-issue",
        action="append",
        default=[],
        help="显式白名单放行的已知问题，形如 union:id957；放行会打印 WARNING 但不影响退出码 0",
    )
    args = ap.parse_args()
    _log("脚本入口", 脚本="验收标准_vs_golden_对拍.py", 参数=vars(args), 仓库根=ROOT)

    for 路径 in (args.doc, args.golden):
        if not os.path.isfile(路径):
            # 用法级错误：路径不存在不允许静默继续
            print(f"[错误] 文件不存在: {路径}", file=sys.stderr)
            return 2

    try:
        golden表, sha = 读golden(args.golden)
        设计表 = 解析设计表(args.doc)
        对拍(golden表, 设计表)
    except Exception as exc:
        _log("脚本异常", 异常类型=type(exc).__name__, 异常信息=str(exc)[:400], 异常堆栈=traceback.format_exc()[:1200])
        print(f"[错误] 对拍执行失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 白名单处理：显式放行的已知问题降级为 WARNING，并明确提示何时删掉白名单
    白名单 = set(args.known_issue)
    剩余不变量 = []
    for 项 in _结果["不变量违反"]:
        # 白名单键格式与 --help 一致：union:id<题号> / pages:id<题号>
        键 = "union:id%d" % 项["题号"] if 项["等式"].startswith("evidence_pages_union") else "pages:id%d" % 项["题号"]
        if 键 in 白名单:
            _结果["已知问题放行"].append({**项, "白名单键": 键})
            _log(
                "WARNING_已知问题放行",
                白名单键=键,
                等式=项["等式"],
                缺失=项.get("缺失"),
                提示="该问题已按 --known-issue 显式放行；修复后请移除此白名单，勿长期保留",
            )
        else:
            剩余不变量.append(项)
    _结果["不变量违反"] = 剩余不变量

    漂移数 = len(_结果["漂移"])
    违反数 = len(_结果["不变量违反"])
    放行数 = len(_结果["已知问题放行"])
    退出码 = 0 if (漂移数 == 0 and 违反数 == 0) else 1

    print()
    print("=" * 78)
    print("工单4 设计文档 ←→ golden 对拍结论")
    print("=" * 78)
    print(f"golden 文件      : {args.golden}")
    print(f"golden sha256    : {sha}")
    print(f"设计文档         : {args.doc}")
    print(f"已核对题数       : {_结果['已核对题数']}")
    print(f"设计漂移         : {漂移数}")
    print(f"golden 不变量违反: {违反数}")
    print(f"已知问题放行     : {放行数}")
    if 漂移数:
        print("\n--- 设计文档与 golden 漂移明细 ---")
        for 项 in _结果["漂移"]:
            print("  " + json.dumps(项, ensure_ascii=False))
    if 违反数:
        print("\n--- golden 自身不变量违反明细（需 golden 属主修复）---")
        for 项 in _结果["不变量违反"]:
            print("  " + json.dumps(项, ensure_ascii=False))
    if 放行数:
        print("\n--- 已显式放行的已知问题（修复后请删除白名单）---")
        for 项 in _结果["已知问题放行"]:
            print("  " + json.dumps(项, ensure_ascii=False))
    print("\n结论：" + ("✅ 通过（设计文档与 golden 逐字段一致）" if 退出码 == 0 else "❌ 不通过（存在漂移或不变量违反）"))
    _log("脚本出口", 脚本="验收标准_vs_golden_对拍.py", golden_sha256=sha, 漂移数=漂移数, 不变量违反数=违反数, 放行数=放行数, 退出码=退出码)
    return 退出码


if __name__ == "__main__":
    sys.exit(main())
