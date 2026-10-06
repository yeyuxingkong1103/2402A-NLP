# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-Query 理解优化任务
多轮对话 RAG 核心：
  1) Query 理解/改写：根据对话历史做指代消解（他/这个公司→公司名）、省略补全（"那X呢？"补全句型）、
     主体切换识别 —— 由 qwen2.5:7b 完成，输出独立完整的检索查询
  2) 多源检索：兴图文本 + 力源文本/表格 + 力源图像 三个集合（复用工单03/04索引，见 index_paths.txt）
  3) 带历史生成：答案可指代前文，同时自含关键事实
LLM 与嵌入均通过本机 Ollama 官方 SDK（固定端点 127.0.0.1:11434）
"""
import re

import chromadb
import ollama

GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
TOP_K = 6
MIN_SCORE = 0.30
DOC_ALIAS = {"zhaogu1": "招股说明书1（兴图新科）", "zhaogu2": "招股说明书2（力源信息）"}

COMPANY_NAMES = ("武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司")


def clean_query(question: str) -> str:
    """查询清洗（长度阈值策略，工单02/05 实测结论）：
    - 内容丰富的长查询（去公司名后仍>=15字）：去掉公司全称/固定前缀，让内容块胜出
      （如"来自军用领域的收入…"、"荣获国家科技进步一等奖的工程…"）；
    - 短的主体型查询（如"法定代表人是谁"）：公司名是对基本情况/概览块的强判别特征
      （带名 0.74+ vs 无名 0.6x），保留原名。"""
    q = re.sub(r"根据?.{0,20}?招股(说明书|意向书)[，,]?", "", question)
    for name in COMPANY_NAMES:
        q = q.replace(name, "")
    q = q.replace("报告期内，", "").strip("，, 。")
    return q if len(q) >= 15 else question

REWRITE_PROMPT = """你是对话查询改写器。根据对话历史，把用户的最新问题改写成一个独立、完整、无指代的检索查询：
1. 指代消解："他/它/这个公司/该公司" → 具体公司名称；
2. 省略补全："那X呢？" → 按上一问的句型补全为关于X的完整问题；
3. 主体切换：明确新的主语公司；
4. 只输出改写后的问题本身，不要解释。

【对话历史】
{history}

【最新问题】{question}

【改写后的查询】"""

ANSWER_PROMPT = """你是证券文档问答助手。规则：
1. 仅根据【检索上下文】回答【问题】，不得编造；
2. 只要上下文中出现了与问题对应的内容，就必须直接给出答案，禁止以"不够全面"为由拒答；
3. 上下文确实完全没有答案时，才回答"根据文档内容未找到"；
4. 标注【图像内容】的片段来自图表页 OCR 与图结构分析，是结构/数值问题的权威依据；
5. 统计"销售处"个数时，仅统计名称以"销售处"结尾的部门框；
6. 问题包含多个小问时逐一回答；回答末尾标注出处（招股说明书X 第X页/表/图）。

【检索上下文】
{context}

【问题】{question}

【回答】"""


def load_collections():
    """返回 ({'zhaogu1':col, 'zhaogu2':col}, img_col_or_None)，按 index_paths.txt 配置"""
    by_name, img_col = {}, None
    for line in open("index_paths.txt", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        path, coll = line.split("|")
        col = chromadb.PersistentClient(path=path).get_collection(coll)
        if "img" in coll:
            img_col = col
        else:
            by_name[coll.split("_")[0]] = col
    return by_name, img_col


def embed_query(text: str):
    return ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]


def llm(prompt: str) -> str:
    resp = ollama.generate(model=GEN_MODEL, prompt=prompt, stream=False,
                           options={"temperature": 0.1, "seed": 42})
    return resp["response"].strip()


def retrieve(cols, query: str, top_k: int = TOP_K):
    """按集合配额检索：每个文本集合各取前2（两家公司的事实都进上下文且总量聚焦，
    7B 模型在 >6 块混合上下文下注意力明显劣化），图像集合取前2。"""
    vec = embed_query(clean_query(query))
    text_hits, img_hits = [], []
    for i, col in enumerate(cols):
        res = col.query(query_embeddings=[vec], n_results=2)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = round(1 - dist, 4)
            if meta.get("kind") == "image":
                if score >= 0.25:
                    img_hits.append({"text": doc, "doc": meta["doc"], "page": meta["page"],
                                     "kind": "image", "score": score})
            elif score >= MIN_SCORE:
                text_hits.append({"text": doc, "doc": meta["doc"], "page": meta["page"],
                                  "kind": meta.get("kind", "text"), "score": score})
    text_hits.sort(key=lambda h: -h["score"])
    img_hits.sort(key=lambda h: -h["score"])
    return text_hits[:top_k - 2] + img_hits[:2]


def format_history(history):
    return "\n".join(f"问：{q}\n答：{a[:120]}" for q, a in history) or "（无，这是第一问）"


class ChatSession:
    """多轮对话会话：维护历史 + 每轮查询改写"""

    def __init__(self, cols):
        self.cols = cols
        self.history = []  # [(question, answer)]

    def ask(self, question: str):
        rewritten = question if not self.history else llm(
            REWRITE_PROMPT.format(history=format_history(self.history), question=question))
        hits = retrieve(self.cols, rewritten)
        ctx = "\n\n".join(
            f"[{DOC_ALIAS[h['doc']]} 第{h['page']}页{' 图表' if h['kind'] == 'image' else ' 表格' if h['kind'] == 'table' else ''}] {h['text']}"
            for h in hits)
        ans = llm(ANSWER_PROMPT.format(context=ctx, question=rewritten))
        self.history.append((question, ans))
        return {"rewritten": rewritten, "answer": ans, "hits": hits}
