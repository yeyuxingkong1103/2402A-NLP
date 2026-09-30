# -*- coding: utf-8 -*-
"""大模型客户端：Ollama 的 OpenAI 兼容接口。"""
import re
from collections.abc import Iterator

from openai import OpenAI

from ..core import config
from ..core.logging import get_logger

log = get_logger("llm")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key=config.OLLAMA_API_KEY,
            timeout=180,
        )
    return _client


def chat(messages: list[dict], temperature: float | None = None,
         max_tokens: int | None = None) -> str:
    """非流式生成。"""
    resp = get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_stream(messages: list[dict], temperature: float | None = None,
                max_tokens: int | None = None) -> Iterator[str]:
    """流式生成，逐段 yield 文本增量。"""
    stream = get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------- 查询改写
# 改写要花一次完整的 LLM 调用（实测约 2.4s）。多数问句本身已完整，
# 改写纯属浪费，所以先用启发式判断「这句话脱离上下文看不看得懂」。
_PRONOUN_RE = re.compile(
    r"(他|她|它|这个|那个|这些|那些|上述|刚才|前面|上面|此法|该法|这种方法)"
)
_CONJ_STARTS = ("那", "还有", "所以", "但是", "不过", "然后", "接着", "继续", "再多说")


def needs_rewrite(question: str) -> bool:
    """判断问句是否依赖上下文。返回 False 可省掉一次 LLM 调用。"""
    q = question.strip()
    if len(q) <= 12:                      # 过短，多半是省略句
        return True
    if _PRONOUN_RE.search(q):             # 含代词
        return True
    if q.startswith(_CONJ_STARTS):        # 承接上文的连词开头
        return True
    return False


REWRITE_PROMPT = """你是检索查询改写助手。根据对话历史，把用户最新的一句话改写成**独立、完整、适合检索**的查询。

要求：
- 消解代词指代（他/它/这个/那个 → 具体名词）
- 补全省略的主语和语境
- 只输出改写后的查询本身，不要任何解释、标点外的修饰或引号
- 如果最新一句本身已经完整，原样输出

对话历史：
{history}

用户最新一句：{question}

改写后："""


def rewrite_query(question: str, memory: list[dict],
                  max_tokens: int = 80) -> str:
    """结合对话历史改写查询，用于指代消解。失败时原样返回。"""
    if not memory or not needs_rewrite(question):
        return question

    history_lines = []
    for m in memory[-4:]:
        role = "用户" if m.get("role") == "user" else "助手"
        content = (m.get("content") or "").strip()[:200]
        if content:
            history_lines.append(f"{role}：{content}")
    if not history_lines:
        return question

    try:
        out = chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(
                history="\n".join(history_lines), question=question)}],
            temperature=0.0, max_tokens=max_tokens,
        )
        out = out.strip().strip('"').strip("'").splitlines()[0].strip()
        # 改写结果异常（过长/为空）时放弃
        if not out or len(out) > len(question) * 4 + 40:
            return question
        return out
    except Exception as e:
        log.warning("查询改写失败，使用原查询: %s", str(e)[:100])
        return question


def health() -> dict:
    """探活。

    OLLAMA_BASE_URL 来自环境变量，先校验 scheme 再发起请求——
    允许 http/https，拒绝 file:// 等，避免探活端点被用来访问任意 URL。
    """
    base = config.OLLAMA_BASE_URL.rstrip("/")
    if not re.match(r"^https?://", base):
        return {"ok": False, "error": f"OLLAMA_BASE_URL 仅支持 http(s)，当前: {base[:60]}"}
    try:
        import httpx
        r = httpx.get(f"{base}/models", timeout=5, follow_redirects=False)
        models = [m["id"] for m in r.json().get("data", [])]
        return {"ok": config.LLM_MODEL in models, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}
