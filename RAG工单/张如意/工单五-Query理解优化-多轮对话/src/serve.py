# -*- coding: utf-8 -*-
"""
工单05 FastAPI 服务启动脚本（多轮对话 HTTP 接口）
工单编号：人工智能NLP-RAG-Query理解优化任务

直接复用 rag_core.api 内置的服务（未修改共享代码），其中与本工单相关的接口：
    POST /api/chat       多轮对话：按 session_id 维护上下文，自动做指代消解
    POST /api/feedback   用户反馈：点赞/点踩写回自适应重排器
    POST /api/ask        单轮问答（调试用，返回检索 trace）
    GET  /api/health     健康检查（高可用/容器编排）
    GET  /api/metrics    运行指标（请求数、P95 延迟、LLM 用量）
    GET  /               内置 Web 问答界面

启动：
    python serve.py                        # 0.0.0.0:8000
    python serve.py --port 8080 --reload
    python serve.py --build                # 先建索引再启动

自测（另开终端）：
    curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
         -d "{\\"session_id\\":\\"demo\\",\\"question\\":\\"这个公司的法定代表人是谁？\\"}"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wo05_common import COLLECTION, index_ready          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 API 服务（rag_core.api）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="代码热重载（开发用）")
    ap.add_argument("--build", action="store_true", help="启动前先执行建索引")
    args = ap.parse_args()

    if args.build:
        from build_index import build
        build()
    elif not index_ready(COLLECTION):
        print("[警告] 未检测到索引，接口会返回空检索结果。")
        print("       建议先运行：python build_index.py（或加 --build 参数）")

    try:
        import uvicorn
    except ImportError:
        print("[错误] 未安装 uvicorn，请执行：pip install 'uvicorn[standard]' fastapi")
        return 1

    print("=" * 78)
    print("工单05 多轮对话服务（rag_core.api + FastAPI）")
    print("=" * 78)
    print(f"  · 服务地址：http://127.0.0.1:{args.port}/")
    print(f"  · 多轮对话：POST /api/chat      （字段：session_id / question）")
    print(f"  · 用户反馈：POST /api/feedback  （字段：session_id / question / chunk_id / helpful）")
    print(f"  · 健康检查：GET  /api/health    运行指标：GET /api/metrics")
    print(f"  · 接口文档：http://127.0.0.1:{args.port}/docs")
    print("-" * 78)
    print("五轮脚本自测顺序（session_id 保持不变即可保持上下文）：")
    from rag_core import config
    for i, q in enumerate(config.MULTI_TURN_SCRIPT, 1):
        print(f"    {i}. {q}")
    print("-" * 78)

    uvicorn.run("rag_core.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
