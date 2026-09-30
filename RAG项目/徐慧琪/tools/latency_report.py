"""判据 8 的检索段延迟记录：渲染，以及跨次重跑保留的历史行。

独立成模块而不是塞在 eval_qa.py 里：那个文件已贴着「单文件 ≤ 300 非空行」的
硬闸门（Task 13 复审指出），再加东西就得靠压注释了——拆出来两边都不挤。
"""
from __future__ import annotations

import time

# 历史记录块的定界标记：每次重跑都会重写整个文件，没有这个块，「曾有一次不达标」
# 就会被下一次干净跑洗掉——只读本文件的人会得出"无条件达标"
LATENCY_HISTORY_START = "<!-- 历史运行记录：脚本跨次重跑保留 -->"
LATENCY_HISTORY_END = "<!-- 历史记录结束 -->"
LATENCY_HISTORY_HEADER = "| 运行时间 | 题数 | P50 | P95 | ≥2s 条数 | 备注 |"
LATENCY_HISTORY_CELLS = 6
# 记录标题的一个特征子串：用来分辨"这是一份旧延迟记录"（缺标记 = 要报错）
# 与"这是一张白纸"（首次运行 = 允许空表）
LATENCY_TITLE = "检索段延迟记录"


def retrieval_seconds(items: list[dict], retrieve_fn) -> list[float]:
    """逐题量**检索段**耗时（编码→混合检索→回填→精排，不含生成）。

    retrieve_fn 由调用方注入：单测传假函数即可断言，不必拉真模型。
    """
    seconds: list[float] = []
    for item in items:
        start = time.perf_counter()
        retrieve_fn(item["query"])
        seconds.append(time.perf_counter() - start)
    return seconds


def latency_history(text: str) -> list[str]:
    """从旧记录里读回历史数据行；空文件（首次运行）返回空表。

    三条闸门都对着同一个失败类——"记录看着还在，其实没被读出来"：
    ①文件像本记录却没有定界标记 → 报错，静默返回空表就是离群行无声消失的入口；
    ②列数不对的行 → **报错而不是过滤**：这个文件按设计就是手工编辑的（历史里
      4/5 行是手工补录），手工行多一个竖线就该当场喊停，过滤掉等于让它下次消失；
    ③表头分隔行按"每格只有 - 和 :"识别，不按子串含 `---` —— 备注里写 `---` 是
      合法内容，用子串判会把它当成分隔行丢掉（同一种无声丢失）。
    """
    if not text.strip():
        return []
    if LATENCY_HISTORY_START not in text and LATENCY_TITLE in text:
        raise ValueError("延迟记录缺历史定界标记：继续跑会把已记录的离群行洗掉")
    block = text.split(LATENCY_HISTORY_START, 1)[-1].split(LATENCY_HISTORY_END, 1)[0]
    rows: list[str] = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line == LATENCY_HISTORY_HEADER or all(cell and set(cell) <= {"-", ":"}
                                                 for cell in cells):
            continue
        if len(cells) != LATENCY_HISTORY_CELLS:
            raise ValueError(f"历史行列数 {len(cells)} ≠ {LATENCY_HISTORY_CELLS}，"
                             f"请先修好这一行（否则它下次会被悄悄丢掉）：{line[:60]}")
        rows.append(line)
    return rows


def _latency_cell(row: str, index: int) -> str:
    """取历史行的第 index 格；取不到就返回空串（行格式本该先过列数闸门）。"""
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return cells[index] if index < len(cells) else ""


def _latency_p95(row: str) -> float:
    """抠历史行里的 P95 秒数；解析不动返回 0.0（不该让它毁掉整轮产物）。"""
    try:
        return float(_latency_cell(row, 3).rstrip("s"))
    except ValueError:
        return 0.0


def _latency_n(row: str) -> int:
    """抠历史行的题数；解析不动返回 0（0 表示"不算全量"，不参与达标计数）。"""
    try:
        return int(_latency_cell(row, 1))
    except ValueError:
        return 0


def render_latency_report(seconds: list[float], meta: dict, history=None) -> str:
    """渲染检索段延迟记录；历次运行由调用方读回、累积进末尾的历史表。

    首条必须单列（含预热，实测 4.4s vs 常态 0.5s）：混进 P95 会把冷启动说成稳态。
    """
    rows = list(history or [])
    # 备注里的竖线会把历史行撑成 7 列、让按列取值错位：写入时就地换掉
    note = str(meta.get("note", "自动记录")).replace("|", "/")
    lines = ["# ③a 检索段延迟记录（设计文档第十节 判据 8 的证据）", "",
             "判据 8 的门槛**只管检索段**（P95 < 2s），不含生成；主报告的端到端延迟"
             "（含生成）只是上界，不能拿来证成或证伪这一条。", "",
             f"- 评估集：`{meta['name']}`，MD5 `{meta['md5']}`，共 {meta['n']} 题",
             f"- 重跑命令：`{meta['command']}`",
             f"- 运行时间：{meta.get('started', '（未记录）')}；"
             f"运行条件：{meta.get('conditions', '（未记录）')}",
             "- 计量口径：最近秩分位（同 `_percentile`），单位秒", ""]
    if meta.get("warning"):
        # 小样本记录若不显眼，顶着判据 8 的名义引出去比不写还坏
        lines += [meta["warning"], ""]
    if not seconds:
        # 空跑不是 0s、更不是达标：本模块对空输入给"无从谈起"出口，而不是让 min() 抛错
        lines += ["## 本次运行", "", "- **无观测值**：本轮没有可计时的题，不产出分位读数"
                  "——**不要读成 0s，也不要读成达标**。", ""]
        rows.append(f"| {meta.get('started', '（未记录）')} | {meta['n']} | 无观测 | 无观测 | 无观测 | 空跑 |")
    else:
        warm = seconds[1:] if len(seconds) > 1 else seconds
        over = sum(1 for value in seconds if value >= 2.0)
        lines += ["## 本次运行", "",
                  "| 口径 | P50 | P95 | 最小 | 最大 |", "|---|---|---|---|---|",
                  f"| 全部 {len(seconds)} 条 | {_percentile(seconds, 50):.3f}s | "
                  f"{_percentile(seconds, 95):.3f}s | {min(seconds):.3f}s | {max(seconds):.3f}s |",
                  f"| 剔除首条预热后 {len(warm)} 条 | {_percentile(warm, 50):.3f}s | "
                  f"{_percentile(warm, 95):.3f}s | {min(warm):.3f}s | {max(warm):.3f}s |", "",
                  f"- 首条（含预热）：{seconds[0]:.3f}s",
                  f"- 超过 2s 的条数：{over}", ""]
        rows.append(f"| {meta.get('started', '（未记录）')} | {len(seconds)} | "
                    f"{_percentile(seconds, 50):.3f}s | {_percentile(seconds, 95):.3f}s | {over} "
                    f"| {note} |")
    # 不达标次数只数**全量**运行：判据 8 的门槛是 100 题的读数，把 --limit 的小样本
    # 混进分子会让这个计数既不是全量口径、也不是小样本口径（行本身仍逐条保留）
    full = max((_latency_n(row) for row in rows), default=0)
    hits = sum(1 for row in rows if _latency_n(row) == full and _latency_p95(row) >= 2.0)
    lines += ["## 历史运行记录", "",
              f"- 累计 **{len(rows)}** 次运行（含部分运行）；其中**全量 {full} 题**的运行里"
              f"**不达标（检索段 P95 ≥ 2s）{hits} 次**——部分运行不计入达标数，"
              "但**每一行都原样留在下表**。", "",
              LATENCY_HISTORY_START, "", LATENCY_HISTORY_HEADER, "|---|---|---|---|---|---|",
              *rows, "", LATENCY_HISTORY_END, ""]
    return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    """最近秩分位（与 eval_qa._percentile 同口径；本模块自持一份以免循环导入）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]
