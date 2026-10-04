# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
主程序：构建金融知识图谱（Graph RAG），基于 eval_question.md 检索回答，
并输出解析出的知识图谱结构。
用法：
  python app.py --graph     # 输出知识图谱结构
  python app.py --qa        # 运行 eval_question 问题集
"""
import sys
import glob
import json
import os

import config
from graph_builder import KnowledgeGraph
from graph_rag import GraphRAG
from llm import LLM


def load_documents():
    docs = []
    if os.path.isdir(config.CCF_TXT_DIR):
        for f in sorted(glob.glob(os.path.join(config.CCF_TXT_DIR, "*.txt"))):
            with open(f, encoding="utf-8", errors="ignore") as fp:
                docs.append(fp.read())
    elif os.path.isdir(config.CCF_PDF_DIR):
        import pymupdf
        for f in sorted(glob.glob(os.path.join(config.CCF_PDF_DIR, "*.pdf"))):
            d = pymupdf.open(f)
            docs.append("\n".join(p.get_text() for p in d))
            d.close()
    return docs


def load_questions():
    qs = []
    with open(os.path.join(os.path.dirname(__file__), "eval_question.md"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("|") and "Question" not in line and "---" not in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= 2:
                    qs.append(cells[1])
    return qs


def main():
    docs = load_documents()
    print(f"[GraphRAG] 加载文档 {len(docs)} 篇，开始构建知识图谱...")
    kg = KnowledgeGraph()
    kg.build_from_documents(docs)
    print(f"[GraphRAG] {kg.summary()}")

    if "--graph" in sys.argv:
        print("\n========== 知识图谱结构 ==========")
        g = kg.to_dict()
        for n in g["nodes"][:40]:
            print(f"  节点[{n['type']}] {n['id']}")
        for e in g["edges"][:40]:
            print(f"  关系: {e['source']} --[{e['relation']}]--> {e['target']} (权重{e['weight']})")
        with open("knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)
        print("知识图谱已保存到 knowledge_graph.json")
        return

    rag = GraphRAG(kg, docs)
    llm = LLM()
    print("\n========== Graph RAG 问答 ==========")
    for q in load_questions():
        res = rag.answer(q, llm)
        print(f"\nQ: {q}")
        print(f"A: {res['answer'][:180]}")
        print(f"子图实体({len(res['subgraph_nodes'])}): {', '.join(res['subgraph_nodes'][:8])}")


if __name__ == "__main__":
    main()
