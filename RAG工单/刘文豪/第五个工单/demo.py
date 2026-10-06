# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-Query 理解优化任务
演示：工单规定的多轮对话（指代消解、省略补全、主体切换、图像问题）
运行：python demo.py
"""
from pathlib import Path

from rag_chat import ChatSession, load_collections

TURNS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",                # 指代：他 → 兴图新科
    "这个公司的法定代表人是谁？",                                # 省略：这个公司 → 兴图新科
    "那武汉力源信息技术股份有限公司呢？",                        # 主体切换 + 省略（问法定代表人）
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",  # 图像问题
]


def main():
    session = ChatSession(load_collections())
    lines = ["# 工单05 演示结果：多轮对话与 Query 理解", "",
             "每轮先做查询改写（指代消解/省略补全/主体切换），再做多源检索与生成", ""]
    for i, q in enumerate(TURNS, 1):
        r = session.ask(q)
        print(f"—— 第{i}轮 ——")
        print(f"原始问题：{q}")
        print(f"改写查询：{r['rewritten']}")
        print(f"回答：{r['answer'][:160]}\n")
        lines += [f"## 第{i}轮", f"**原始问题**：{q}", f"**改写后查询**：{r['rewritten']}", "",
                  f"**回答**：{r['answer']}", ""]
    Path("多轮对话演示.md").write_text("\n".join(lines), encoding="utf-8")
    print("已保存 -> 多轮对话演示.md")


if __name__ == "__main__":
    main()
