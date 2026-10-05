import json
import re

from openai import AsyncOpenAI

from .config import Settings
from .schemas import QueryPlan
from .vector_store import SearchHit


class DeepSeekClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.deepseek_api_key or "missing", base_url=settings.deepseek_base_url)

    async def plan(self, question: str) -> QueryPlan:
        if not self.settings.deepseek_api_key:
            return QueryPlan(search_queries=[question], entities=[question])
        response = await self.client.chat.completions.create(model=self.settings.deepseek_model, temperature=0, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content": "你是招股说明书检索规划器。只输出 JSON，字段为 language, intent, entities, years, sub_questions, search_queries, needs_clarification, clarification_question。不要回答问题。"},
            {"role": "user", "content": question},
        ])
        try:
            return QueryPlan.model_validate(json.loads(response.choices[0].message.content or "{}"))
        except Exception:
            return QueryPlan(search_queries=[question], entities=[question])

    async def answer(self, question: str, hits: list[SearchHit], language: str) -> tuple[str, str | None]:
        if not hits:
            return "招股说明书中未找到足够依据回答该问题。", "没有检索到相关证据"
        context = "\n\n".join(f"[证据{i}] PDF第{hit.page_number}页｜{hit.section_title}\n{hit.content}" for i, hit in enumerate(hits, 1))
        context = context[: self.settings.max_context_chars]
        system = "只依据给定证据回答。不得补充证据之外的事实。回答简洁，数字、单位、年份必须原样保留；在句末使用[证据N]引用。证据不足时明确说证据不足。"
        if language == "en":
            system += " Answer in English, but preserve Chinese source quotations when useful."
        if not self.settings.deepseek_api_key:
            return f"根据招股说明书检索结果：{hits[0].content} [证据1]", None
        response = await self.client.chat.completions.create(model=self.settings.deepseek_model, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": f"问题：{question}\n\n证据：\n{context}"}])
        answer = response.choices[0].message.content or ""
        cited = re.findall(r"\[证据(\d+)\]", answer)
        if not cited:
            return answer, "回答未包含有效证据引用"
        return answer, None
