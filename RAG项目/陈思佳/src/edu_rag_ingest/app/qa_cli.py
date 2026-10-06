from __future__ import annotations

"""RAG 命令行问答入口，用于不启动前端时直接调试问答流程。"""

import argparse
import json
import sys

from ..config.config import load_config
from ..generation.rag_service import TeacherRAGService


def main() -> None:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="九年级语文教师 RAG 问答")
    parser.add_argument("question", help="教师问题")
    parser.add_argument("--config", default="configs/crawler.yaml", help="配置文件路径")
    parser.add_argument("--top-k", type=int, default=None, help="检索数量")
    args = parser.parse_args()

    config = load_config(args.config)
    service = TeacherRAGService(config)
    result = service.ask(args.question, top_k=args.top_k)
    print(result.answer)
    print("\n--- 引用来源 ---")
    print(json.dumps(result.citations, ensure_ascii=False, indent=2))


def _force_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
