# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
答案生成模块。
"""
import os


class LLM:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def generate(self, prompt):
        if self.api_key:
            return self._call_api(prompt)
        return self._retrieval_only(prompt)

    def _call_api(self, prompt):
        import urllib.request
        import json
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]

    def _retrieval_only(self, prompt):
        marker = "参考上下文：\n"
        if marker in prompt:
            ctx = prompt.split(marker, 1)[1]
            for tail in ("\n\n请根据", "\n请根据"):
                if tail in ctx:
                    ctx = ctx.split(tail, 1)[0]
            return ctx.strip()
        return prompt


def build_rag_prompt(question, contexts):
    ctx = "\n\n".join(f"[{i + 1}]\n{c}" for i, (c, _) in enumerate(contexts))
    return (f"你是金融招股说明书问答助手。请依据参考上下文回答，答案准确、简洁。\n"
            f"参考上下文：\n{ctx}\n\n用户问题：{question}\n请给出答案：")
