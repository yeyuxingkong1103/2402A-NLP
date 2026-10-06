# app/core/chain_service.py
"""LangChain 集成层：把框架的抽象落到本项目。

框架选型说明：候选里 Dify / RagFlow 是需要独立部署的平台，会取代整个应用；
LlamaIndex 偏重建索引、LightRAG 偏图检索，都与本项目已有的自研链路重叠。
**LangChain** 最合适——它已是项目依赖，且六大功能能逐项对应到现有实现：

    Models    ChatOpenAI 包装 DeepSeek（本节）
    Prompts   ChatPromptTemplate + MessagesPlaceholder（本节）
    Chains    LCEL 组装「提示词 → 模型 → 取正文」（本节）
    Memory    未套用框架接口——短期记忆在 memory_service.py，本层只接收拼好的 messages
    Indexes   保留自研检索（见下方说明）
    Agents    保留自研路由（见下方说明）

**为什么 Indexes 与 Agents 不套用框架**：本项目的检索是「多查询 × 多路召回
（稠密 + 稀疏 + 法条元数据）→ 加权 RRF → 交叉编码器精排」，LangChain 的
EnsembleRetriever 只支持多检索器等权融合，表达不了元数据路与权重；
Agents 的 tool-calling 在本项目里就是一次确定性的规则路由（识别到法条号
才走元数据路），套上 Agent 只会引入不确定性。这两块的取舍写在这里，
避免后来者以为漏掉了。
"""
from typing import Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

logger = logging.getLogger(__name__)

from app.config import settings


# ============================================================
# Models：把 DeepSeek 包装成 LangChain 的聊天模型
# ============================================================
class DeepSeekChatModel(BaseChatModel):
    """DeepSeek 的 LangChain 适配。

    没有直接用 langchain_openai.ChatOpenAI，原因是 deepseek-flash 是**推理模型**：
    它会先产出 reasoning_content，max_tokens 给小了正文会是空字符串，
    而 finish_reason 可能是 stop 而不是 length。这种补重试逻辑属于本项目特有的
    处理，写在自定义模型里比打补丁到通用类上更清楚。
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=settings.LLM_API_KEY,
                      base_url=settings.LLM_BASE_URL,
                      timeout=self.timeout)

    @staticmethod
    def _to_openai(messages: Sequence[BaseMessage]) -> List[dict]:
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        out = []
        for m in messages:
            role = role_map.get(m.type)
            if role:
                out.append({"role": role, "content": m.content})
        return out

    def _generate(self, messages: Sequence[BaseMessage], stop=None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs) -> ChatResult:
        payload = self._to_openai(messages)
        client = self._client()

        content = ""
        for attempt, budget in enumerate((self.max_tokens,
                                          self.max_tokens * 2, 8192)):
            resp = client.chat.completions.create(
                model=self.model, messages=payload,
                temperature=self.temperature, max_tokens=budget)
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if content:
                break
            if attempt < 2:
                logger.warning("LangChain 链路：正文为空(finish_reason=%s)，"
                               "以 max_tokens=%s 重试", choice.finish_reason, budget)

        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=content))])

    def _stream(self, messages: Sequence[BaseMessage], stop=None,
                run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        client = self._client()
        stream = client.chat.completions.create(
            model=self.model, messages=self._to_openai(messages),
            temperature=self.temperature, max_tokens=self.max_tokens, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def get_chat_model(temperature: Optional[float] = None) -> DeepSeekChatModel:
    return DeepSeekChatModel(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )


# ============================================================
# Prompts + Chains：LCEL 组装生成链路
# ============================================================
def build_prompt_template() -> ChatPromptTemplate:
    """与 app/core/prompt.py 同构，但用 LangChain 的模板表达。

    好处是变量缺失会在组装阶段就报错，而不是拼出一个残缺的字符串发给模型。
    """
    return ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}"),
    ])


def build_answer_chain(temperature: Optional[float] = None):
    """LCEL 链：提示词 → 模型 → 取正文。

    同步与流式共用同一条链——`.invoke()` 取完整回答，`.stream()` 逐段产出。

        chain = build_answer_chain()
        chain.invoke({"system_prompt": ..., "history": [...], "question": ...})
        for piece in chain.stream({...}): ...
    """
    return build_prompt_template() | get_chat_model(temperature) | StrOutputParser()
