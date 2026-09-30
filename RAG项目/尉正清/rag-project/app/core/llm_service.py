# app/core/llm_service.py
"""大模型调用封装（DeepSeek）。

注意：deepseek-flash 是**推理模型**，会先产出 reasoning_content 再产出正文。
若 max_tokens 给小了，token 会被推理过程耗尽，正文返回空字符串。

正文为空有两种成因，重试逻辑对**两者都生效**（不看 finish_reason）：
  - finish_reason=length：推理吃光了 token（放大预算可解）
  - finish_reason=stop：模型只产出了推理就停止（敏感话题上实测出现过）
只认 length 会漏掉后一种，因此这里只要正文为空就重试。
"""
import threading
from typing import List, Optional

from openai import OpenAI

from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LLMService:
    _inst: Optional["LLMService"] = None
    _lock = threading.Lock()

    def __init__(self):
        if not settings.LLM_API_KEY:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，请检查 .env 或环境变量")
        self.client = OpenAI(api_key=settings.LLM_API_KEY,
                             base_url=settings.LLM_BASE_URL,
                             timeout=settings.LLM_TIMEOUT)
        self.model = settings.LLM_MODEL
        logger.info("LLM 就绪: %s @ %s", self.model, settings.LLM_BASE_URL)

    @classmethod
    def instance(cls) -> "LLMService":
        if cls._inst is None:
            with cls._lock:
                if cls._inst is None:
                    cls._inst = cls()
        return cls._inst

    # ---------------- 基础调用 ----------------
    def chat(self, messages: List[dict],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             retry_on_empty: bool = True) -> str:
        """同步对话，返回正文（已剥离推理内容）。"""
        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        def _call(mt: int):
            return self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=mt)

        try:
            resp = _call(max_tokens)
        except Exception as e:
            logger.error("调用大模型失败: %s", e)
            raise

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()

        # 正文为空有两种成因：推理吃光 token（finish_reason=length），
        # 或模型只产出推理就停止（finish_reason=stop）。两者都要重试。
        if not content and retry_on_empty:
            for bigger in (max_tokens * 2, 8192):
                if bigger <= max_tokens:
                    continue
                logger.warning("正文为空(finish_reason=%s)，以 max_tokens=%s 重试",
                               choice.finish_reason, bigger)
                resp = _call(bigger)
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                if content:
                    break

        if not content:
            logger.warning("大模型返回空正文, finish_reason=%s", choice.finish_reason)
        return content

    # ---------------- 专用小任务 ----------------
    def summarize(self, text: str, max_chars: int = 300) -> str:
        """把一段对话压缩成记忆摘要。"""
        prompt = (
            "请把下面的对话压缩成一段不超过 %d 字的记忆摘要，"
            "保留用户的关键信息、诉求、结论与情绪倾向，去掉寒暄。\n"
            "直接输出摘要正文，不要加任何前缀。\n\n%s" % (max_chars, text))
        try:
            out = self.chat([{"role": "user", "content": prompt}],
                            temperature=0.2, max_tokens=2048)
            return out.strip()
        except Exception as e:                              # pragma: no cover
            logger.warning("生成记忆摘要失败: %s", e)
            return text[:max_chars]

    def rewrite_query(self, question: str, history_lines: List[str]) -> str:
        """多轮追问改写：把「那他呢」这类指代补全成独立可检索的问题。"""
        if not history_lines:
            return question
        history = "\n".join(history_lines[-6:])
        prompt = (
            "下面是一段对话历史和一个追问。请把追问改写成**不依赖上下文**、"
            "可以独立检索的完整问题，保留原意，补全其中的指代与省略。\n"
            "只输出改写后的问题本身，不要解释，不要加引号。\n"
            "如果追问本身已经完整，原样输出。\n\n"
            "【对话历史】\n%s\n\n【追问】\n%s" % (history, question))
        try:
            out = self.chat([{"role": "user", "content": prompt}],
                            temperature=0.1, max_tokens=1024)
            out = out.strip().strip('"“”')
            return out or question
        except Exception as e:                              # pragma: no cover
            logger.warning("query 改写失败，使用原问题: %s", e)
            return question


def get_llm() -> LLMService:
    return LLMService.instance()
