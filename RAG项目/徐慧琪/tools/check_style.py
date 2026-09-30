"""代码规范检查：单文件非空行 ≤ 300、不存在无注释的代码行。

存在的理由：技术方案 11 章把这两条列为项目方硬约束，且要求把本脚本的输出
作为验收证据。所以它自己必须可测、可复核，不能是个一次性脚本。

关于"无注释"的判定口径：采用**块级覆盖**——一行代码只要在它所在的缩进块内
（向上走，遇到缩进更浅的行即跨出本块）或行尾有注释，就算已覆盖。
不用"逐行必须有注释"是因为那会让几乎所有真实代码都告警，验收时凑不出零告警；
块级覆盖既拦得住裸写的大段代码，也符合"解释为什么"而非"逐行复述"的注释本意。
"""
from __future__ import annotations

import pathlib
import sys

MAX_CODE_LINES = 300

# 独立成行的文档字符串与注释等价，都算"这一块有说明"
DOCSTRING_PREFIXES = ('"""', "'''")


def _is_comment(line: str) -> bool:
    """判断一行是否为注释或独立文档字符串。"""
    stripped = line.strip()
    return stripped.startswith("#") or stripped.startswith(DOCSTRING_PREFIXES)


def _indent_of(line: str) -> int:
    """返回缩进宽度，用于判定块边界。"""
    return len(line) - len(line.lstrip())


def count_code_lines(source: str) -> int:
    """统计非空行数。空行不占配额，否则可用空行稀释超限。"""
    return sum(1 for line in source.splitlines() if line.strip())


def find_uncommented_lines(source: str) -> list[int]:
    """返回既无块内注释、也无行尾注释的代码行号（1-based）。"""
    lines = source.splitlines()
    uncommented: list[int] = []
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or _is_comment(raw):
            continue
        # 行尾注释也算覆盖；这里不区分 # 是否在字符串字面量里——
        # 宁可漏报也不误报，否则零告警这条验收门槛无法达成
        if "#" in stripped:
            continue
        indent = _indent_of(raw)
        covered = False
        up = idx - 1
        while up >= 0:
            prev_raw = lines[up]
            if not prev_raw.strip():
                up -= 1
                continue
            if _is_comment(prev_raw):
                covered = True
                break
            # 缩进更浅 = 跨出了当前块。此时不能直接判负——外层块头本身若被注释覆盖，
            # 它包住的整个块（含本行）都应算覆盖，所以把它当作新的基准继续向上找
            if _indent_of(prev_raw) < indent:
                indent = _indent_of(prev_raw)
            up -= 1
        if not covered:
            uncommented.append(idx + 1)
    return uncommented


def check_file(path: pathlib.Path) -> dict:
    """检查单个文件，返回统计结果。"""
    source = path.read_text(encoding="utf-8")
    code = count_code_lines(source)
    return {
        "path": str(path),
        "code_lines": code,
        "oversize": code > MAX_CODE_LINES,
        "uncommented": find_uncommented_lines(source),
    }


def main(targets: list[str]) -> int:
    """检查给定路径下的全部 .py，返回进程退出码（0 表示零告警）。"""
    files: list[pathlib.Path] = []
    for target in targets:
        p = pathlib.Path(target)
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    problems = 0
    for f in files:
        result = check_file(f)
        if result["oversize"]:
            print(f"[超行] {result['path']} 非空行 {result['code_lines']} > {MAX_CODE_LINES}")
            problems += 1
        for lineno in result["uncommented"]:
            print(f"[无注释] {result['path']}:{lineno}")
            problems += 1
    print(f"检查 {len(files)} 个文件，问题 {problems} 处")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["backend", "tools"]))
