"""检查精简版 RAG 的必要运行环境。

在链路中的位置：
    独立的运维脚本，不属于运行时链路。在启动服务/构建知识库之前运行，
    用来确认"依赖装齐了没有、Ollama 和模型准备好没有"。

用法：
    python check_env.py

输出一张 Markdown 表格，逐项列出检查结果，最后给出结论。
退出码：0 = 核心依赖齐全可启动；1 = 仍缺少必要依赖。
    —— 用退出码表达结论，是为了让 CI 或启动脚本能据此自动决定要不要继续。

为什么需要这个脚本：
    本项目依赖面较宽（Python 包 + Ollama 命令行 + 两个模型 + 可选的 MinerU），
    缺任何一项的报错都发生在运行时的深处、且信息晦涩。
    提前在这里一次性查清，能把"启动后报奇怪的错"变成"启动前看到缺什么"。
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass
class Check:
    """单项检查结果。

    字段：
        name:   检查项名称（显示在表格第一列）
        ok:     是否通过
        detail: 详情（版本号、缺失说明、错误原因）
    """

    name: str
    ok: bool
    detail: str


def package_check(module: str, package: str) -> Check:
    """检查一个 Python 包是否已安装，并尽量取出版本号。

    参数：
        module: import 时用的模块名（如 "multipart"）
        package: pip 里的包名（如 "python-multipart"）
    返回：
        Check 结果。

    为什么要分别传模块名和包名：
        两者经常不一致 —— python-multipart 的模块名是 multipart，
        pymupdf 的模块名是 fitz。只传一个名字必然有一边查不对。

    用 find_spec 而不是 try import：
        find_spec 只查"能不能找到"，不真正执行模块代码 ——
        更快，也不会因为某个包 import 时有副作用（如连数据库）而产生干扰。

    取不到版本号但仍算通过（PackageNotFoundError 分支）：
        模块能 import 说明依赖是好的，版本元数据缺失（如开发模式安装）
        不该被判定为失败。这种情况详情写"已安装"。
    """
    if importlib.util.find_spec(module) is None:
        return Check(package, False, "未安装")
    try:
        return Check(package, True, version(package))
    except PackageNotFoundError:
        return Check(package, True, "已安装")


def ollama_check() -> Check:
    """检查 Ollama 是否可用，以及两个必需的模型是否已拉取。

    返回：
        Check 结果。

    三项检查依次进行：
        1. ollama 命令能执行吗（FileNotFoundError = 没装）
        2. 退出码正常吗（非 0 说明服务没起或命令出错）
        3. `ollama list` 的输出里有没有 bge-m3 和 qwen2:7b

    为什么用 subprocess 调命令行而不是调 HTTP 接口：
        "命令不存在"和"服务没起来"是两种不同的故障、修复方法也不同。
        从命令行能区分它们；直接请求 HTTP 端口则两者都表现为连接失败。

    只取错误信息的第一行（splitlines()[0]）：
        Ollama 的报错常是多行，全显示会把表格撑坏。

    大小写归一（result.stdout.lower()）后做包含判断：
        模型名的显示大小写不保证稳定，统一转小写再比更可靠。
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return Check("Ollama", False, "未找到 ollama 命令")
    except Exception as exc:
        return Check("Ollama", False, f"{type(exc).__name__}: {exc}")
    if result.returncode:
        return Check("Ollama", False, (result.stderr or result.stdout or "命令失败").strip().splitlines()[0])

    models = result.stdout.lower()
    required = ["bge-m3", "qwen2:7b"]
    missing = [model for model in required if model not in models]
    return Check("Ollama", not missing, "已安装 bge-m3 / qwen2:7b" if not missing else f"缺少：{', '.join(missing)}")


def checks() -> list[Check]:
    """汇总所有检查项。

    返回：
        Check 列表。

    检查项的取舍：
        Python 版本永远算通过（能跑这个脚本就说明 Python 可用），
        列出它只是为了在报告里显示版本，便于排查版本相关的问题。
        MinerU 标记为可选 —— 它是复杂版面 PDF 的增强解析器，
        缺失时回退 pypdf，不影响核心链路，所以不算"必要依赖"。
    """
    return [
        Check("Python", True, sys.version.splitlines()[0]),
        package_check("fastapi", "fastapi"),
        package_check("uvicorn", "uvicorn"),
        package_check("pymilvus", "pymilvus"),
        package_check("pypdf", "pypdf"),
        package_check("jieba", "jieba"),
        package_check("requests", "requests"),
        package_check("multipart", "python-multipart"),
        Check("MinerU", importlib.util.find_spec("mineru") is not None, "可选；未安装时使用 pypdf"),
        ollama_check(),
    ]


def print_report(items: list[Check]) -> None:
    """把检查结果打成 Markdown 表格。

    参数：
        items: Check 列表

    输出 Markdown 而不是纯文本：
        方便直接粘进文档或 issue 里汇报环境问题，不用再手工排版。
    """
    print("| 检查项 | 状态 | 详情 |")
    print("|---|---|---|")
    for item in items:
        print(f"| {item.name} | {'可用' if item.ok else '缺失'} | {item.detail} |")


def main() -> int:
    """执行检查并给出结论。

    返回：
        0 = 核心依赖齐全；1 = 缺少必要依赖。

    计算结论时把 MinerU 排除在外（name != "MinerU"）：
        它是可选增强项，缺失不该让整个环境检查判定为失败 ——
        否则每次没装 MinerU 都返回 1，CI 就会永远红着，这个退出码也就失去了意义。
    """
    items = checks()
    print_report(items)
    required_failed = [item for item in items if not item.ok and item.name != "MinerU"]
    print("\n核心链路：" + ("可启动" if not required_failed else "仍缺少必要依赖"))
    print("说明：需要先启动 Ollama，并准备 bge-m3 和 qwen2:7b。")
    return 0 if not required_failed else 1


if __name__ == "__main__":
    # 保留命令行入口，exit code 供启动脚本或 CI 判断环境是否就绪
    raise SystemExit(main())
