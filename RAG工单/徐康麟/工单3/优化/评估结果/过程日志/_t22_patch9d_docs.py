# -*- coding: utf-8 -*-
"""t22 补丁 9d（文档更正）：登记 EN6 的**采样两态**（收口后新测出的波动，必须如实纠正）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

触发：t22 收口后复跑英文用例，发现 EN6（注册地址）**不是稳定降级**：
    * `_t22_after7.py` 第一次：EN6 = 0.0208（中文表行，`generation.language_gate path=keep_chinese`）→ 降级；
    * `_t22_after7.py` 第二次：EN6 = **0.5588**（英文句 `The registered address … is 湖北省武汉市…A3栋8层.`）→ 达标；
    * `_t22_probe_en1.py EN6 3` 单题连跑：0.0208 / 0.5588 / 0.5588（同一次进程内两态并存）→ 确认是**采样波动**，不是阈值问题。
    两态都满足「可答 + 带引用 + `language=en` + 引用可回溯」；三态用例把 EN6 登记在 `DEGRADED_ALLOWED`，
    达标态与降级态都不判失败（② 只在 ratio<0.5 时要求已登记，见测试文件第 89~96 行）。

因此必须把 §25.5 的「4 条降级」从**单轮数字**纠正为**跨轮两态**，否则文档会与实测冲突。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FACTS = Path(r"E:\gao6gongdan\工单3\部署\配置\环境事实.md")
ACCEPT = Path(r"E:\gao6gongdan\工单3\设计\验收标准.md")

EDITS_FACTS = [
    (
        "§25.5 EN6 行",
        "| EN6 注册地址（兴图） | field | 0.021 | 0.0208 | `path=keep_chinese` | ② 降级（已登记） |",
        "| EN6 注册地址（兴图） | field | 0.021 | **0.0208 / 0.5588（随采样两态）** | `path=keep_chinese`（中文表行）/ 主路径英文句 | ② 降级 或 ① 达标（两态均已登记） |",
    ),
    (
        "§25.5 表后补一条两态说明",
        "  **没有被伪装成达标**；测试文件用三态断言区分（`ENGLISH_BODY_REQUIRED` / `DEGRADED_ALLOWED` / 失败态）。",
        "  **没有被伪装成达标**；测试文件用三态断言区分（`ENGLISH_BODY_REQUIRED` / `DEGRADED_ALLOWED` / 失败态）。\n"
        "* **EN6 是唯一随采样两态的用例**（复跑实测）：0.0208 中文表行（`path=keep_chinese`）与 0.5588 英文句\n"
        "  （`The registered address of Wuhan Xingtu Xinke Electronics Co., Ltd. is 湖北省武汉市…A3栋8层.`）在同一进程内并存（单题连跑 0.0208 / 0.5588 / 0.5588）。\n"
        "  两态都满足「可答 + 带引用 + `language == en` + 引用可回溯」，三态用例把它登记在 `DEGRADED_ALLOWED`：**达标态按①、降级态按②，两态都不允许失败态**。\n"
        "  → 故上表的「3 达标 / 4 降级」是**单轮**数字；跨轮总账为：**EN1 / EN2 / EN7 稳定达标，EN3 / EN4 / EN5 稳定降级，EN6 两态各出现过**。",
    ),
    (
        "§25.7 EN6 归因（去掉「只能靠改写」的说法，改为两态）",
        "   * EN6 另有一重原因：注册地址取值是长中文串（`english_ratio` 0.371 < 0.5），即使逐字核验通过也达不到阈值 → 该题**只能靠 LLM 改写**（继续否决）。",
        "   * EN6 另有一重原因：注册地址取值是长中文串（框架句 `english_ratio` 0.371 < 0.5），确定性框架即使逐字核验通过也达不到阈值；\n"
        "     但该题**存在两态**：LLM 侧的英文框架句能把地址逐字保留着写成英文正文（实测 0.5588 → 达标），与中文表行态（0.0208 → 降级）**随采样交替出现**。\n"
        "     因此 EN6 的正确表述是「**降级或达标，两态均已登记**」，而不是「必然降级」；两态都不新增事实（地址逐字来自证据）。",
    ),
]

EDITS_ACCEPT = [
    (
        "附录 A 降级清单（EN6 两态）",
        "EN3 `0.4156`、EN4 `0.3300`、EN5 `0.0000`、EN6 `0.0208` → **4 条降级（已登记）**",
        "EN3 `0.4156`、EN4 `0.3300`、EN5 `0.0000` 稳定降级；EN6 `0.0208`（中文表行）与 `0.5588`（英文句）**两态随采样** → **登记为「降级或达标，两态均不判失败」**",
    ),
    (
        "附录 A 保证范围行（补 EN6 达标态但不作保证）",
        "EN1 `0.6829`、EN2 `0.7733`、EN7 `0.6071` → **3 条达标**",
        "EN1 `0.6829`、EN2 `0.7733`、EN7 `0.6071` → **3 条稳定达标**（EN6 的达标态 `0.5588` 亦满足同一判据，但**不作保证**）",
    ),
]


def _apply(path: Path, edits: list[tuple[str, str, str]]) -> None:
    """对单个文件按「锚点 → 替换」逐条改写（锚点必须唯一命中，异常即抛）。"""
    body = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for label, old, new in edits:
        anchor = old.replace("\r\n", "\n")
        if new.replace("\r\n", "\n") in body and body.count(anchor) == 0:
            print(f"  [已应用·幂等跳过] {path.name} {label}")
            continue
        count = body.count(anchor)
        if count != 1:
            raise SystemExit(f"{path.name} {label}: 锚点命中 {count} 次（期望 1），已中止，未写盘")
        body = body.replace(anchor, new.replace("\r\n", "\n"))
        print(f"  [已写入] {path.name} {label}")
    path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))


def main() -> int:
    """入口：先改环境事实 §25.5/§25.7，再改设计附录 A。"""
    _apply(FACTS, EDITS_FACTS)
    _apply(ACCEPT, EDITS_ACCEPT)
    for path in (FACTS, ACCEPT):
        data = path.read_bytes()
        print(f"  {path.name}: {len(data)} B, crcrlf={data.count(bytes([13, 13, 10]))}, "
              f"sha256={hashlib.sha256(data).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
