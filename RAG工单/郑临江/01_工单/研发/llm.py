# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
答案生成模块：封装 LLM 调用。
- 配置了 OPENAI_API_KEY / DASHSCOPE_API_KEY 时走真实大模型；
- 未配置时使用“检索式”降级：直接返回检索到的相关文本作为答案，
  保证离线环境可运行、可演示。
"""
import os


class LLM:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        if self.api_key:
            return self._call_api(prompt)
        return self._retrieval_only(prompt)

    def _call_api(self, prompt: str) -> str:
        import urllib.request
        import json
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _retrieval_only(self, prompt: str) -> str:
        # prompt 中已包含“参考上下文”，此处直接抽取上下文作为答案（检索式回答）
        marker = "参考上下文：\n"
        if marker in prompt:
            ctx = prompt.split(marker, 1)[1]
            # 去掉后续的指令部分
            for tail in ("\n\n请根据", "\n请根据"):
                if tail in ctx:
                    ctx = ctx.split(tail, 1)[0]
            return ctx.strip()
        return prompt


def build_rag_prompt(question: str, contexts) -> str:
    """根据问题与召回的上下文构建 prompt。"""
    ctx = "\n".join(f"[{i + 1}] {c}" for i, (c, _) in enumerate(contexts))
    return (
        f"你是金融招股说明书问答助手。请仅依据下方参考上下文回答用户问题，"
        f"答案要准确、简洁。\n"
        f"参考上下文：\n{ctx}\n\n"
        f"用户问题：{question}\n请给出答案："
    )
