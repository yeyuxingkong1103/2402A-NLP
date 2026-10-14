from __future__ import annotations

import os
from pathlib import Path

from corpus import load_zip_corpus
from graph_rag import DATA_DIR, save_index


DEFAULT_ZIP = Path(r"C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition.zip")


def main() -> None:
    zip_path = Path(os.getenv("CCF_ZIP_PATH", str(DEFAULT_ZIP)))
    print(f"读取金融年报：{zip_path}")
    chunks = load_zip_corpus(zip_path)
    print(f"生成 {len(chunks)} 个文本块，正在构建知识图谱和TF-IDF索引...")
    rag = save_index(chunks, DATA_DIR)
    print(f"完成：{len(rag.graph['nodes'])} 个节点，{len(rag.graph['edges'])} 条关系。")


if __name__ == "__main__":
    main()

