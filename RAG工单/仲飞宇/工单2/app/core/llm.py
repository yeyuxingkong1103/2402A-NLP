"""LLM 抽象层：OpenAI 兼容客户端，默认指向本地 Ollama。

通过 LLM_PROVIDER 切换：
- ollama        默认，走 Ollama 的 OpenAI 兼容端点 /v1
- openai_compat 任意 OpenAI 兼容服务商（硅基流动 / DeepSeek / 豆包 / 千问）
- dummy         离线冒烟测试用，回显用户问题，不发起网络请求
"""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from .config import Settings
from .errors import EmptyAnswerError
from .logging_config import get_logger

log = get_logger("llm")

# OpenAI 消息格式：list[{"role": ..., "content": ...}]
Messages = list[dict]


def _warn_if_truncated(finish_reason: str | None) -> None:
    """回答被长度上限截断时留痕。

    只警告、不抛异常：被截断的回答多数仍有价值，转成 502 会比现状更糟。
    这个字段此前全项目从没读过，截断完全静默。

    **别高估这条告警的覆盖面**：实测（qwen3:8b, n_ctx=16384）prompt 超窗时，
    Ollama 是靠丢弃最老轮次消化掉的，`finish_reason` 仍然是 `stop`——那种失败这
    里抓不到，得靠 templates.build_messages 那条「系统消息吃满预算」的告警。
    真正落到 `length` 的是生成阶段挪不动上下文的情况（见 pipeline.EmptyAnswerError
    的 docstring：content 为空、推理段却有一两百字），与本应用未设 max_tokens 的
    调用方式叠加后并不常见。
    """
    if finish_reason == "length":
        log.warning(
            "回答被长度截断（finish_reason=length）：生成阶段没拿到足够空间，"
            "回答不完整，且会被写入短期记忆影响下一轮"
        )


# 推理模型把思维链放在哪个字段：各家不一，而 openai SDK 的 extra 字段不一定挂成
# 属性（pydantic v2 要看 extra 策略），所以下面按名字逐个试，再兜底翻 model_extra。
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thinking")


def extract_reasoning(message) -> str:
    """从 chat completion 的 message 里取推理段原文，取不到返回空串。

    ⚠️ 别拿它当「有没有思考」的判据：普通模型当然没有这个字段，返回空串是正常的。
    """
    for name in _REASONING_FIELDS:
        v = getattr(message, name, None)
        if isinstance(v, str) and v.strip():
            return v
    extra = getattr(message, "model_extra", None) or {}
    for name in _REASONING_FIELDS:
        v = extra.get(name)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _first_choice(resp):
    """取第一个 choice；一个都没有时按「模型没产出内容」处理。

    `resp.choices[0]` 直接下标会在上游返回 HTTP 200 + `choices: []` 时抛 IndexError
    （中转、内容过滤、上游抖动都会给出这种形态）。那不是"依赖掉线"，而是"模型没答"，
    接口层对这两种情况的语义完全不同：前者 503、后者 502 + 明确提示。
    流式路径遇到同样的块是直接跳过的，这里跟着对齐，别让同一个上游在两条路径上
    得到两种错误分类。
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise EmptyAnswerError("上游返回空 choices（模型未产出任何内容）")
    return choices[0]


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: OpenAI | None = None
        if settings.llm_provider != "dummy":
            # 显式给超时与重试，别吃 SDK 默认值（整请求 600s、连接错误与 5xx 自动重试 2 次）：
            # Ollama 挂住不返回时，每个 /chat 会占住一个线程池 worker 很久，够多的这种请求
            # 会把线程池占满——随后 /chat、/knowledge/upload 乃至 /health（ping 也走网络）
            # 一起排队。重试对"慢"没有帮助，只会把占用时间翻倍，所以 retries=0。
            self._client = OpenAI(
                base_url=settings.llm_effective_base_url,
                api_key=settings.llm_api_key or "ollama",
                timeout=settings.llm_timeout,
                max_retries=0,
            )

    def ping(self) -> bool:
        """探测 LLM 服务是否可达。"""
        # dummy 恒 True：没有外部依赖可探。它只表示「不需要 LLM 也能跑」，不代表模型可用
        # ——/health 上看到 ok 不等于换了 provider 也能出真答案。
        if self.settings.llm_provider == "dummy":
            return True
        try:
            self._client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 不可达（%s）: %s", self.settings.llm_effective_base_url, exc)
            return False

    def chat(self, messages: Messages) -> str:
        """非流式对话，返回完整文本。

        内容为空时返回空串、**不在这里抛错**：空答案的语义统一由 pipeline 判（postprocess
        清洗完仍为空才抛 EmptyAnswerError，接口层译成 502）。若这里自己抛，流式那条路径
        就没有对应物，同一个「模型没答」会在两条路径上得到两种错误分类。
        """
        if self.settings.llm_provider == "dummy":
            return self._dummy_reply(messages)
        # 不设 max_tokens：留给服务端默认值（Ollama 下即无限），被截断的情况由
        # _warn_if_truncated 留痕。temperature 等参数在 chat / chat_with_reasoning / stream
        # 三处各写了一份，要改就得三处一起改，别只动这一处。
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            stream=False,
        )
        choice = _first_choice(resp)
        _warn_if_truncated(choice.finish_reason)
        return choice.message.content or ""

    def chat_with_reasoning(self, messages: Messages) -> tuple[str, str]:
        """同 chat()，但额外返回推理段原文：(content, reasoning)。

        chat() 把推理段丢掉是**对的**——问答链路只要正文，思维链混进去反而污染答案。
        但评测/审计需要它：判分模型给出「faithfulness=0」时，只看分数分不清是它误判
        还是被测模型真的在编，必须看它给的理由。实测 qwen3:8b 当 judge 时 content 是
        干净的 '1'/'0'，理由全在推理段里。

        不支持推理段的服务商（绝大多数 OpenAI 兼容端点）返回空串，不报错。
        """
        # 这里和 chat() 各写了一遍 create，没抽公共函数：返回值形状不同（这里要留着 message
        # 对象才取得到推理段），抽出来还得再包一层。代价是参数要手工保持一致。
        if self.settings.llm_provider == "dummy":
            return self._dummy_reply(messages), ""
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            stream=False,
        )
        choice = _first_choice(resp)
        _warn_if_truncated(choice.finish_reason)
        return choice.message.content or "", extract_reasoning(choice.message)

    def stream(self, messages: Messages) -> Iterator[str]:
        """流式对话，逐段产出文本增量。

        只管转达增量，不做重试、也不在整条流为空时补错：空产出交给下游的
        EmptyAnswerError（与 chat() 同一套语义），保证「模型没答」在两条路径上表现一致。
        """
        # dummy 下把整段回复一次性吐出：SSE 管道、前端分块渲染都还能走通，
        # 只是没有逐字效果，离线联调够用。
        if self.settings.llm_provider == "dummy":
            yield self._dummy_reply(messages)
            return
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            stream=True,
        )
        finish_reason: str | None = None
        for chunk in resp:
            if not chunk.choices:
                # 中转/内容过滤会发这种块（HTTP 200 但 choices 为空）：跳过即可，
                # 全流都不带内容时下游会走 EmptyAnswerError，与非流式口径一致。
                continue
            # finish_reason 只在最后一个有 choices 的块里给；取到就留着，供收尾告警
            finish_reason = chunk.choices[0].finish_reason or finish_reason
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        _warn_if_truncated(finish_reason)

    @staticmethod
    def _dummy_reply(messages: Messages) -> str:
        # 只回显最后一条 user 消息，不回显 system：system 里塞着人设和 TOP_K 条检索资料，
        # 整段回显会把前端聊天区刷满资料，反而看不出「提问有没有被正确拼进去」。
        # 想看完整拼装结果请查日志或用 tests 里的 build_messages 用例。
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return (
            f"[离线 dummy 回复] 收到你的问题：{last_user}\n"
            "当前 LLM 未配置（LLM_PROVIDER=dummy），请启动 Ollama 或配置在线 API 后重试。"
        )
