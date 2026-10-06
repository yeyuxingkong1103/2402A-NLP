# -*- coding: utf-8 -*-
"""
RAG 问答引擎
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：端到端问答流水线 —— Query向量化 → 向量检索Top-K → 组装Prompt → LLM生成。
     同时提供"纯LLM直接回答"接口，用于工单01要求的 RAG vs 纯LLM 对比。
"""
import time  # 计时：分别统计检索耗时与生成耗时

from config import DEFAULT_TOP_K      # 默认召回条数 5
from ollama_client import client      # Ollama 单例：做 LLM 生成
from vector_store import VectorStore  # 向量库：加载索引与检索

# 问答 Prompt 模板：要求仅基于检索上下文作答，注明引用页码
# 注意 {context}/{question} 是 str.format 占位符，改这段文本时勿引入其他花括号
QA_PROMPT = """你是金融文档问答助手。请仅根据下面的参考上下文回答用户问题。
要求：
1. 答案必须来自参考上下文，不要编造；上下文中没有的信息请回答"根据文档内容未找到相关信息"。
2. 回答简洁准确，可直接引用数字与专有名词。
3. 回答末尾标注信息来源页码，格式如：（来源：第X页）

【参考上下文】
{context}

【用户问题】
{question}

【回答】"""


class RAGEngine:
    """RAG 问答引擎"""

    def __init__(self, index_name):
        # 启动即从磁盘加载已构建好的向量索引（千级分块，加载耗时可控）
        self.store = VectorStore.load(index_name)
        if self.store is None:
            # 索引缺失时给出明确指引，避免用户面对空检索结果一头雾水
            raise RuntimeError(f"索引 {index_name} 不存在，请先运行 build_index.py 构建索引")

    def ask(self, question, top_k=DEFAULT_TOP_K, with_context=False):
        """RAG 问答
        返回: {"answer", "sources", "time_cost", "retrieved"(可选)}
        """
        t0 = time.time()  # 总计时起点
        # 第一步：检索——内部会把 question 向量化后做余弦相似度 Top-K
        hits = self.store.search(question, top_k=top_k)
        t_retrieval = time.time() - t0  # 单独记录检索耗时（性能对比指标）

        # 第二步：把命中的片段拼成带"编号+页码+相似度"标头的上下文，
        # 标头信息帮助 LLM 在回答末尾正确标注来源页码
        context = "\n\n".join(
            f"[片段{i+1} | 第{h['page']}页 | 相似度{h['score']:.3f}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        prompt = QA_PROMPT.format(context=context, question=question)  # 填充模板占位符

        t1 = time.time()  # 生成计时起点
        answer = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度：事实型问答要稳定、不发散
            num_predict=600,  # 限制最大生成 token，控制响应时长
        )
        t_gen = time.time() - t1  # 生成耗时（主要瓶颈在本地 LLM 推理）

        # 组装结构化结果：sources 供前端/报告展示命中依据
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {"page": h["page"], "source": h["source"], "score": round(h["score"], 4)}
                for h in hits
            ],
            "time_cost": round(time.time() - t0, 2),  # 端到端总耗时
            "retrieval_time": round(t_retrieval, 2),
            "generation_time": round(t_gen, 2),
        }
        if with_context:
            # 调试/评测时需要查看原始命中片段才开启，正常问答不返回（省体积）
            result["retrieved"] = hits
        return result

    # ── 纯 LLM 直接回答（无检索，用于对比） ─────────────────
    def pure_llm_ask(self, question):
        """不经过检索，直接用 LLM 回答（对比基准）"""
        # 不喂任何上下文，测出"模型自身知识"回答招股书问题的能力下限
        prompt = (
            f"请回答以下金融文档相关问题，简洁准确：\n\n【问题】{question}\n\n【回答】"
        )
        t0 = time.time()
        # num_predict=400 比带检索时更短：基线回答无需引用页码
        answer = client.chat([{"role": "user", "content": prompt}], temperature=0.1, num_predict=400)
        return {"question": question, "answer": answer, "time_cost": round(time.time() - t0, 2)}
