# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
统一入口（CLI）：

    python main.py ingest          # 1. PDF 解析
    python main.py build           # 2. LightRAG 知识图谱构建（增量演示）
    python main.py stats           # 3. 图谱统计
    python main.py export-graph    # 4. 知识图谱可视化导出
    python main.py compare         # 5. RAG vs LightRAG 检索对比 + RAGAS 评估
"""

import sys

import pdf_ingest
import lightrag_build
import rag_baseline
import compare_eval
import export_graph


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "ingest":
        pdf_ingest.ingest_all()
    elif cmd == "build":
        lightrag_build.main()
    elif cmd == "stats":
        sys.argv = ["lightrag_build.py", "--stats"]
        lightrag_build.main()
    elif cmd == "export-graph":
        export_graph.main()
    elif cmd == "compare":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        compare_eval.asyncio.run(compare_eval.run_compare(limit=limit))
    else:
        print(f"未知命令：{cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
