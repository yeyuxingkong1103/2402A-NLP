# -*- coding: utf-8 -*-
"""
工单05：Query理解优化 —— 多轮对话
工单编号：人工智能NLP-RAG-Query理解优化任务
功能：在RAG问答基础上支持多轮对话。
  每轮先用 LLM 结合对话历史做 Query 改写（指代消解"他/这个公司"、
  省略补全"那XX公司呢"），再用改写后的完整问题走检索问答流水线。
  知识库：双文档库 zgs_all_v1（工单03）+ 图像语义库 zgs2_img_v1（工单04）。
运行：python run_chat.py          （交互模式）
      python run_chat.py --demo   （复现工单验收对话并生成记录）
"""
import sys
import os
import json
import time

# 当前脚本目录，用于输出对话记录文件
HERE = os.path.dirname(os.path.abspath(__file__))
# 公共模块目录加入 sys.path，跨工单目录复用 00-公共模块
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

from vector_store import VectorStore  # 自研 numpy 向量库
from rag_engine import QA_PROMPT  # 统一问答提示词模板（工单01定义）
from ollama_client import client  # Ollama 本地模型客户端
from query_optimize import multi_retrieve, keyword_rerank  # 多查询检索与关键词重排（工单02/03）

TOP_K = 5

# Query 改写 Prompt：多轮指代消解 + 省略补全（含few-shot示例）
REWRITE_PROMPT = """你是多轮对话的查询改写器。请结合对话历史，把用户最新问题改写成一个
独立、完整、无指代的检索查询。

要求：
1. 消解指代："他/该公司/这个公司" → 替换为具体公司名；
2. 补全省略："那XX公司呢？" → 继承上一轮问题的意图，替换主体为公司XX，
   例如上一轮问"这个公司的法定代表人是谁？"，新问题"那武汉力源信息技术股份有限公司呢？"
   应改写为"武汉力源信息技术股份有限公司的法定代表人是谁？"；
3. 保留新问题中的新信息（新公司名、新指标），只继承历史问题的意图框架；
4. 只输出改写后的查询语句本身，不要解释；
5. 若最新问题本身已完整，则原样输出。

【示例】
对话历史：
Q1: 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
最新问题：他参与的哪个工程荣获了国家科技进步一等奖？
改写输出：武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？

对话历史：
Q1: 这个公司的法定代表人是谁？
最新问题：那武汉力源信息技术股份有限公司呢？
改写输出：武汉力源信息技术股份有限公司的法定代表人是谁？

【对话历史】
{history}

【最新问题】
{question}

【改写后的查询】"""


class MultiTurnRAG:
    """多轮对话 RAG"""

    def __init__(self):
        self.stores = []  # 多库列表：文本库 + 图像语义库都参与检索
        # 依次加载两个知识库索引，存在的才加入检索范围
        for name in ("zgs_all_v1", "zgs2_img_v1"):
            s = VectorStore.load(name)
            if s is not None:
                self.stores.append(s)
        self.history = []  # [{"q": 问, "a": 答}, ...]

    def _rewrite(self, question):
        """结合历史改写查询（首轮不改写）"""
        # 首轮没有历史，不存在指代/省略，直接原样返回
        if not self.history:
            return question
        # 拼接最近3轮历史（历史太长会干扰改写且浪费token），回答截断150字
        hist = "\n".join(f"Q{i+1}: {h['q']}\nA{i+1}: {h['a'][:150]}"
                         for i, h in enumerate(self.history[-3:]))
        try:
            # temperature=0.0 保证改写确定性；num_predict=120 改写只需一句话
            rewritten = client.generate(
                REWRITE_PROMPT.format(history=hist, question=question),
                temperature=0.0, num_predict=120)
            # 去掉首尾空白和模型可能加的引号/书名号包裹；改写为空则回退原问题
            return rewritten.strip().strip('"「」') or question
        except Exception:
            return question  # 改写失败时兜底用原问题，不阻塞问答

    def ask(self, question):
        """一轮问答：改写 → 多库检索 → 生成"""
        # 先做 Query 改写，得到无指代、完整的检索查询
        rewritten = self._rewrite(question)

        hits = []  # 汇总所有库的检索命中
        for s in self.stores:
            # 每个库走 multi_retrieve（查询扩展多路召回），fetch_k=12 先多召回再重排
            hits += multi_retrieve(s, [rewritten], top_k=8, fetch_k=12)
        # 关键词重排 + 按分数降序取前8，控制上下文长度
        hits = keyword_rerank(rewritten, hits)
        hits = sorted(hits, key=lambda h: -h["score"])[:8]

        # 拼装上下文：每片段带序号、来源库与页码
        context = "\n\n".join(
            f"[片段{i+1} | {h['source']} 第{h['page']}页]\n{h['text']}"
            for i, h in enumerate(hits))
        # 用改写后的完整问题生成回答：低温度保证事实稳定，600 token 足够
        answer = client.chat(
            [{"role": "user", "content": QA_PROMPT.format(context=context, question=rewritten)}],
            temperature=0.1, num_predict=600)
        # 把原始问题与答案写入历史，供下一轮改写参考（注意存原始问而非改写后）
        self.history.append({"q": question, "a": answer})
        # 返回改写结果、答案和前3个来源（改写结果单独返回用于记录展示）
        return {"rewritten": rewritten, "answer": answer,
                "sources": [{"page": h["page"], "source": h["source"],
                             "score": round(h["score"], 3)} for h in hits[:3]]}


# 复现工单验收的5轮演示对话：覆盖指代消解（他/这个公司）与省略补全（那XX公司呢）
DEMO_DIALOG = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]


def run_demo():
    engine = MultiTurnRAG()
    # Markdown 报告头部：标题、时间、说明
    lines = ["# 工单05：多轮对话检索记录",
             "",
             f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "",
             "说明：每轮先做指代消解/省略补全的Query改写（改写结果附在回答后），再检索双文档+图像语义库。",
             ""]
    # 逐轮跑演示对话，enumerate 从1开始编号
    for i, q in enumerate(DEMO_DIALOG, 1):
        print(f"\nQ{i}: {q}")
        r = engine.ask(q)
        print(f"A{i}: {r['answer'][:300]}")
        print(f"   [改写为] {r['rewritten']}")  # 打印改写结果便于核对改写质量
        # 逐轮追加到报告：问题、答案、来源、改写后的Query
        lines += [f"## Q{i}：{q}", "", f"**A{i}：** {r['answer']}", "",
                  f"来源：" + " | ".join(f"{s['source']}第{s['page']}页" for s in r["sources"]),
                  f"Query改写：{r['rewritten']}", "", "---", ""]
    # 写出对话记录 Markdown
    with open(os.path.join(HERE, "对话记录-工单05.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n已生成: 对话记录-工单05.md")


def run_interactive():
    engine = MultiTurnRAG()
    print("多轮对话模式（输入 q 退出）")
    while True:
        q = input("\n你 > ").strip()
        if q.lower() in ("q", "quit", "exit"):  # 退出指令
            break
        if not q:  # 空输入跳过，不进流水线
            continue
        r = engine.ask(q)
        print(f"\n{r['answer']}")
        # 展示检索来源，便于人工核对答案依据
        print(f"【来源】" + " | ".join(f"{s['source']}第{s['page']}页" for s in r["sources"]))


if __name__ == "__main__":
    # 带 --demo 参数复现验收对话并生成记录，否则进入交互模式
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
