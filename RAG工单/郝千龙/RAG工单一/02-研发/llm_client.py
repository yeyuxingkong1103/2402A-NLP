# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【LLM生成组件 · llm_client.py】OpenAI 兼容客户端（同步+异步），支持 RAG/基线双模式与中英双语
# 编写日期：2026-09-28   修订日期：2026-10-04
from typing import List
import openai

import config


class LLMClient:
    """OpenAI 兼容 LLM 客户端：默认 DeepSeek，可换 vLLM/Qwen/OpenAI"""

    _sync = None
    _async = None

    def __init__(self):
        if LLMClient._sync is None:
            LLMClient._sync = openai.OpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY,
                timeout=config.LLM_TIMEOUT,
            )
        self.client = LLMClient._sync

    @classmethod
    def async_client(cls):
        """惰性创建异步客户端（供 FastAPI async 路由使用，避免阻塞事件循环）"""
        if cls._async is None:
            cls._async = openai.AsyncOpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY,
                timeout=config.LLM_TIMEOUT,
            )
        return cls._async

    def _create_kwargs(self) -> dict:
        """公共生成参数"""
        return {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }

    # ---------- 同步 ----------
    def chat(self, prompt: str) -> str:
        """同步 chat completion"""
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **self._create_kwargs(),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {e}")
            return f"[LLM 调用失败] {e}"

    def answer_with_context(self, question: str, contexts: List[str],
                            lang: str = "zh") -> str:
        """带检索上下文的 RAG 生成"""
        ctx = "\n\n".join(f"[片段{i+1}] 第{c_t[1]}页\n{c_t[0]}"
                          for i, c_t in enumerate(contexts))
        prompt = config.get_rag_prompt(lang).format(context=ctx, question=question)
        return self.chat(prompt)

    def answer_baseline(self, question: str, lang: str = "zh") -> str:
        """无检索的纯 LLM 基线回答（对比用）"""
        prompt = config.get_baseline_prompt(lang).format(question=question)
        return self.chat(prompt)

    # ---------- 异步 ----------
    async def achat(self, prompt: str) -> str:
        """异步 chat completion，高并发下不阻塞事件循环"""
        try:
            resp = await self.async_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **self._create_kwargs(),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM 异步调用失败: {e}")
            return f"[LLM 调用失败] {e}"

    async def aanswer_with_context(self, question: str, contexts: List[tuple],
                                   lang: str = "zh") -> str:
        """异步 RAG 生成；contexts: [(text, page), ...]"""
        ctx = "\n\n".join(f"[片段{i+1}] 第{p}页\n{t}"
                          for i, (t, p) in enumerate(contexts))
        prompt = config.get_rag_prompt(lang).format(context=ctx, question=question)
        return await self.achat(prompt)

    async def aanswer_baseline(self, question: str, lang: str = "zh") -> str:
        """异步基线回答"""
        prompt = config.get_baseline_prompt(lang).format(question=question)
        return await self.achat(prompt)


if __name__ == "__main__":
    cli = LLMClient()
    q = "武汉兴图新科电子股份有限公司法定代表人是谁？"
    print("基线:", cli.answer_baseline(q))
    print("RAG :", cli.answer_with_context(q, [("法定代表人为程家明", 22)]))

# ====================================================================
# 技术备注：
# 1. RAG：Prompt 限定仅依据检索片段回答并要求标注页码，缓解幻觉、支持溯源。
# 2. 异步客户端配合 FastAPI 的 async/await，使单机并发吞吐显著提升。
# 3. Transformer：LLM 主干为多层 Transformer 解码器，自回归生成答案。
# 4. Fine-tuning：可对 LLM 做 LoRA 领域适配，默认免微调即可满足验收。
# ====================================================================
