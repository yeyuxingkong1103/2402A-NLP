"""知识库入库命令行。

示例：
    python tools/ingest_cli.py --all --recreate        # 重建并入库全部启用角色 + 公共库
    python tools/ingest_cli.py --roles scientist       # 只入库某个角色
    python tools/ingest_cli.py --file data/kb/lawyer/01_contract_basics.md --role lawyer
    python tools/ingest_cli.py --report                # 只查看知识库现状
    python tools/ingest_cli.py --delete-scope lawyer   # 删除某角色的知识块
    python tools/ingest_cli.py --reset                 # 重建 Milvus 集合（清空全部）
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
from role_rag.logging_conf import get_logger, setup_logging  # noqa: E402

logger = get_logger("tools.ingest_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role RAG_try 知识库入库")
    parser.add_argument("--all", action="store_true", help="入库全部启用角色 + 公共库")
    parser.add_argument("--roles", type=str, default="", help="逗号分隔的角色 id（如 scientist,lawyer）")
    parser.add_argument("--file", type=str, default="", help="指定单个文件入库")
    parser.add_argument("--role", type=str, default="", help="配合 --file 使用的角色 id")
    parser.add_argument("--recreate", action="store_true", help="先删除该作用域下的旧块再入库")
    parser.add_argument("--report", action="store_true", help="只输出知识库现状")
    parser.add_argument("--delete-scope", type=str, default="", help="删除指定作用域的全部知识块")
    parser.add_argument("--reset", action="store_true", help="重建 Milvus 集合（清空知识块与长期记忆）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = get_config()
    pipeline = get_ingest_pipeline(config)

    if args.report:
        print(json.dumps(pipeline.report(), ensure_ascii=False, indent=2))
        return 0

    if args.reset:
        pipeline.reset()
        print("已重建 Milvus 集合。")
        return 0

    if args.delete_scope:
        removed = pipeline.delete_scope(args.delete_scope)
        print(f"已删除作用域 {args.delete_scope} 的知识块 {removed} 条。")
        return 0

    if args.file:
        if not args.role:
            print("使用 --file 时必须同时指定 --role", file=sys.stderr)
            return 2
        path = Path(args.file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        result = pipeline.ingest_files([path], role=args.role, scope=args.role)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    roles = [item.strip() for item in args.roles.split(",") if item.strip()] or None
    if not args.all and not roles:
        print("请指定 --all 或 --roles，或使用 --report / --file / --reset", file=sys.stderr)
        return 2

    def progress(stage: str, info: dict) -> None:
        print(f"  · {stage}: {info}")

    summary = pipeline.ingest_all(roles=roles, recreate=args.recreate, progress=progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
