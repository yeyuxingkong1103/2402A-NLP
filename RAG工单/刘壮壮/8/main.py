# -*- coding: utf-8 -*-
"""
main.py — 基于 Graph RAG 实现金融问答
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

使用 ccf_competition 金融研报构建知识图谱，并基于图谱检索返回答案所在文本块：
  - 实体/关系抽取（LLM / 规则）
  - networkx 知识图谱构建
  - 图谱检索（实体链接 + 邻域扩展）
  - 知识图谱结构输出与可视化

运行：
  python main.py                  # 构建图谱 + 检索问答 + 输出图谱结构
  python main.py --query "xxx"    # 单问题
  python main.py --use-llm        # 用 LLM 抽取实体关系（需配置 API）
  python main.py --render         # 渲染图谱图片
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import Config, split_text, make_llm, answer  # noqa: E402
from graph_rag import GraphRAG  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
TXT_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "txt")


def load_chunks(txt_dir: str, limit_per_file: int = 200) -> list:
    """加载年报并分块（每份文档限制块数以控制图谱规模）。"""
    files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read()
        name = os.path.basename(f)
        parts = split_text(text, 400, 60)[:limit_per_file]
        chunks.extend([f"[{name}] {p}" for p in parts])
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--use-llm", action="store_true", help="用 LLM 抽取实体关系")
    parser.add_argument("--render", action="store_true", help="渲染图谱图片")
    args = parser.parse_args()

    config = Config()
    llm = make_llm(config)

    print(f"[加载] {TXT_DIR}")
    chunks = load_chunks(TXT_DIR, limit_per_file=200)
    print(f"  共 {len(chunks)} 块")

    print("[构建知识图谱]")
    gr = GraphRAG(llm if args.use_llm else None)
    gr.build(chunks, use_llm=args.use_llm)
    stats = gr.stats()
    print(f"  节点 {stats['nodes']}，边 {stats['edges']}，三元组 {stats['triples']}")

    if args.render:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.png")
        try:
            gr.render(out, query=args.query)
            print(f"图谱已渲染：{out}")
        except Exception as e:
            print(f"渲染失败（需安装 matplotlib）：{e}")

    if args.query:
        hits = gr.retrieve(args.query, top_k=5)
        print(f"\n问题：{args.query}")
        print(f"命中实体：{gr.query_entities(args.query)}")
        print(f"回答：{answer(args.query, hits, llm)}")
        print("\n检索到的文本块：")
        for h in hits[:3]:
            print(f"  ({h['score']:.1f}) {h['text'][:90]}")
        return

    print("\n" + "=" * 70)
    for q in QUESTIONS:
        hits = gr.retrieve(q["question"], top_k=5)
        ans = answer(q["question"], hits, llm)
        ents = gr.query_entities(q["question"])
        print(f"\n【问题】{q['question']}")
        print(f"  [命中实体] {ents}")
        print(f"  [答案] {ans[:100]}")
        print(f"  [文本块] {hits[0]['text'][:70] if hits else '无'}")
    print("\n" + "=" * 70)

    # 输出知识图谱结构（实体/关系示例）
    print("\n知识图谱结构（示例三元组，前 15 条）：")
    for h, r, t in gr.triples[:15]:
        print(f"  {h} --[{r}]--> {t}")


if __name__ == "__main__":
    main()
