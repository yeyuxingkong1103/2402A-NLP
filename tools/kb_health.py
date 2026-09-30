"""知识库健康检查：文件 -> 分块 -> 向量入库 是否一一对上。

用途：入库后快速确认「每个启用角色都有知识块」，并在缺文件 / 缺块时给出明确提示。

用法：
    D:\\an\\envs\\rags_\\python.exe tools/kb_health.py
    D:\\an\\envs\\rags_\\python.exe tools/kb_health.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from role_rag.config import get_config  # noqa: E402
from role_rag.ingest.pipeline import get_ingest_pipeline  # noqa: E402
from role_rag.logging_conf import setup_logging  # noqa: E402
from role_rag.roles import RoleRegistry  # noqa: E402

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库健康检查")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出原始数据")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    config = get_config()

    pipeline = get_ingest_pipeline(config)
    report = pipeline.report()

    files: dict[str, int] = report.get("files", {})
    chunks: dict[str, int] = {
        str(item.get("scope", "")): int(item.get("count", 0))
        for item in report.get("chunks_by_scope", [])
    }

    registry = RoleRegistry.from_config(config)
    enabled = {role.id for role in registry.all() if role.enabled}
    enabled.add("shared")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=" * 66)
    print("知识库健康检查")
    print("=" * 66)
    print(f"知识库根目录 : {report.get('kb_root')}")
    print(f"知识库版本   : kb_version={report.get('kb_version')}")
    print(f"包含作用域   : {', '.join(report.get('scopes', []))}")
    print("-" * 66)
    print(f"{'作用域':<22}{'源文件':>8}{'知识块':>8}   状态")
    print("-" * 66)

    problems: list[str] = []
    for scope in report.get("scopes", []):
        file_count = int(files.get(scope, 0))
        chunk_count = int(chunks.get(scope, 0))
        if chunk_count > 0:
            status = OK
        elif file_count > 0 and scope in enabled:
            status = FAIL + " 有源文件但未入库"
            problems.append(f"{scope}: 有 {file_count} 个源文件但知识块为 0，请执行入库")
        elif scope in enabled:
            status = SKIP + " 未提供源文件"
        else:
            status = SKIP + " 未启用"
        print(f"{scope:<22}{file_count:>8}{chunk_count:>8}   {status}")

    extra = sorted(set(chunks) - set(report.get("scopes", [])))
    if extra:
        print("-" * 66)
        for scope in extra:
            if int(chunks.get(scope, 0)) > 0:
                print(f"{scope:<22}{files.get(scope, 0):>8}{chunks.get(scope, 0):>8}   {WARN} 集合中存在但非当前作用域")

    total_chunks = sum(int(v) for v in chunks.values())
    total_files = sum(int(v) for v in files.values())

    print("-" * 66)
    print(f"合计：源文件 {total_files} 个，知识块 {total_chunks} 个")
    print("=" * 66)

    if problems:
        print(f"{FAIL} 发现 {len(problems)} 个问题：")
        for item in problems:
            print(f"       - {item}")
        return 1

    if total_chunks == 0:
        print(f"{WARN} 知识库为空，请先执行：python tools/ingest_cli.py --all --recreate")
        return 1

    print(f"{OK} 知识库状态正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
