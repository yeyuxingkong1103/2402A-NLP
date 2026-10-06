# -*- coding: utf-8 -*-
"""
工单06 Web 服务启动脚本（混合检索配置在线生效）
工单编号：人工智能NLP-RAG-混合检索任务

功能：
  1. 启动前确保索引就绪（向量库集合 prospectus + BM25 倒排索引）；
  2. 启动 rag_core.api 内置 Web 界面（多轮对话 + 点赞/点踩反馈）；
  3. Web 端可在每次提问时切换检索策略与重排器：
       POST /api/ask
       {"question": "...", "strategy": "hybrid|vector|fulltext",
        "reranker": "none|tfidf|llm|adaptive|cascade",
        "top_k": 5, "return_trace": true}

运行：
    python 工单06-混合检索/src/serve.py                 # 默认 0.0.0.0:8000
    python 工单06-混合检索/src/serve.py --port 8080
    python 工单06-混合检索/src/serve.py --rebuild       # 先重建索引再启动
浏览器访问 http://localhost:8000/

说明：Web 服务固定使用默认嵌入模型 bge-large-zh-v1.5（集合 prospectus，
      与 rag_core.api / pipeline 的默认约定一致）；m3e-base 用于
      vector_search.py 的离线模型对比。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="工单06 混合检索问答服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--rebuild", action="store_true", help="启动前重建索引")
    parser.add_argument("--no-prebuild", action="store_true",
                        help="跳过索引检查（已确认索引就绪时加快启动）")
    args = parser.parse_args()

    from build_index import DEFAULT_MODEL, build_all, ensure_index

    if args.rebuild:
        print("[启动] 重建向量索引 + BM25 索引…")
        build_all(rebuild=True, verbose=True)
    elif not args.no_prebuild:
        ensure_index([DEFAULT_MODEL], verbose=True)

    from rag_core.embed import set_model
    set_model(DEFAULT_MODEL)          # 保证查询向量与库内向量同一模型

    print("=" * 72)
    print("工单06 招股说明书混合检索问答服务")
    print(f"界面地址 : http://localhost:{args.port}/")
    print(f"接口文档 : http://localhost:{args.port}/docs")
    print("策略切换 : POST /api/ask 传 strategy（vector/fulltext/hybrid）"
          "与 reranker（none/tfidf/llm/adaptive/cascade）")
    print("=" * 72)

    import uvicorn
    uvicorn.run("rag_core.api:app", host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
