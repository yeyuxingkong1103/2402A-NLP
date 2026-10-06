# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
Query 理解模块：多轮对话中的查询改写（指代消解 + 省略补全）。
把带代词/省略的当前问句，结合历史对话改写成"可独立检索的完整问句"，
解决"他""这个公司""那XX呢"等指代导致的检索失败。
"""
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手，服务于招股说明书多轮问答。
请结合【历史对话】，把用户【当前问题】改写成一句"不依赖上下文即可独立检索"的完整问题。

改写规则：
1. 消解代词：把"他/它/该公司/这个公司"等替换为历史中对应的具体公司名。
2. 补全省略：若当前问题是"那XX公司呢？""XX公司呢？"，补全为历史中最近一次问的完整意图（如"XX公司的法定代表人是谁？"）。
3. 保持原意，不要回答问题，不要添加历史中没有的信息。
4. 只输出改写后的问题，不要任何解释。"""


class QueryRewriter:
    """多轮查询改写器"""

    def __init__(self, max_history_turns=4):
        self.max_history = max_history_turns

    def rewrite(self, question, history):
        """
        question: 当前用户问题
        history: [{"role":"user"/"assistant","content":...}, ...]
        返回改写后的独立问句；首轮或改写失败时返回原问题
        """
        if not history:
            return question
        hist_text = "\n".join(
            f"{'用户' if h['role']=='user' else '助手'}：{h['content']}"
            for h in history[-self.max_history * 2:]
        )
        prompt = f"""【历史对话】
{hist_text}

【当前问题】
{question}

请输出改写后的独立问题："""
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            out = (resp.choices[0].message.content or "").strip().split("\n")[0].strip()
            return out or question
        except Exception as e:
            print(f"[Query改写] 失败，使用原问题: {e}")
            return question


if __name__ == "__main__":
    rw = QueryRewriter()
    hist = [{"role": "user", "content": "武汉兴图新科电子股份有限公司来自军用领域的收入是多少？"},
            {"role": "assistant", "content": "报告期内……"}]
    print(rw.rewrite("他参与的哪个工程荣获了国家科技进步一等奖？", hist))
