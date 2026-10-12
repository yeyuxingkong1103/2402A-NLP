# -*- coding: utf-8 -*-
"""从任意工作目录启动本工单脚本，自动将工作目录切到脚本所在目录。
例：python run.py --check；python run.py 研发/app.py --test
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = ("设计", "研发", "测试", "优化", "部署")


def check_project():
    errors = []
    for phase in PHASES:
        if not (ROOT / phase).is_dir():
            errors.append(f"缺少目录：{phase}")
    sources = [ROOT / "run.py"] + [source for phase in PHASES for source in (ROOT / phase).rglob("*.py")]
    for source in sources:
        if "__pycache__" in source.parts:
            continue
        try:
            compile(source.read_text(encoding="utf-8-sig"), str(source), "exec")
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{source.relative_to(ROOT)}：{exc}")
    for filename in ("config.py", "requirements.txt"):
        if not (ROOT / "研发" / filename).is_file():
            errors.append(f"缺少研发/{filename}")
    for error in errors:
        print(f"[失败] {error}", file=sys.stderr)
    if not errors:
        print(f"[通过] {ROOT.name}：五阶段目录及 {len(sources)} 个 Python 文件语法正常")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description="工单脚本启动与目录/语法检查")
    parser.add_argument("--check", action="store_true", help="检查目录与语法，不加载模型、不调用接口")
    parser.add_argument("--list", action="store_true", help="列出脚本路径")
    parser.add_argument("script", nargs="?", help="项目内的脚本相对路径，例如 研发/app.py")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="传给目标脚本的参数")
    args = parser.parse_args()
    if args.check:
        return check_project()
    if args.list or args.script is None:
        for phase in PHASES:
            for source in sorted((ROOT / phase).rglob("*.py")):
                if "__pycache__" not in source.parts:
                    print(source.relative_to(ROOT).as_posix())
        return 0
    source = (ROOT / args.script).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file() or source.suffix != ".py":
        parser.error("脚本必须是本工单目录内存在的 .py 文件")
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    return subprocess.call([sys.executable, "-B", str(source), *forwarded], cwd=str(source.parent))


if __name__ == "__main__":
    raise SystemExit(main())
