"""生成编排。**本包唯一对外的生成入口。**

    await generate(question, passages, mode=...) -> GenerationResult   # 一次性（CLI）
    start(question, passages, mode=...) -> Generation                  # 流式（服务端）

两者走**同一个** `Generation` 实现：`generate` 只是把流耗光再返回结果。这样
"CLI 验证的链路"与"服务端跑的链路"不可能是两条 —— 而 CLI 存在的全部意义就是
验证服务端跑的那条。

---

## 流式与引用校验的先后

`docs/05` §3.1.7 要求做引用白名单校验，而校验需要**完整正文**。流式输出没法等 ——
所以两者是分阶段的：

    流式阶段：模型吐什么就发什么（**未校验**）
    收尾阶段：校验 → `done` 帧的 `answer_text` 是**已校验**的版本

前端 `frontend/js/transcript.js` 的 `renderDone` 会用终帧的 `answer_text` 整体
替换流式渲染的内容 —— S7 的规格里写明"渐进渲染只是预览，最终显示的必须与服务端
逐字一致"。**引用校验正是这条设计要挡的差异**，它不是一个需要消除的副作用。

代价：若模型引用了不存在的编号，用户会在终帧到达时看到那几个标记消失。这是
正确的行为，不是故障。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

from . import DEFAULT_MODE, MODES, PromptError
from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, call_model, stream_model
from .prompt import render
from .result import GenerationResult, build_sources, confidence_of, parse_answer
from .store import save_result

logger = logging.getLogger(__name__)

__all__ = ["generate", "start", "Generation"]


class Generation:
    """一次生成的进行时。**流式路径的唯一入口。**

    用法（服务端）：

        gen = start(question, passages, mode=mode, config=config)
        async for delta in gen.deltas():
            yield sse_event(EV_TOKEN, {"text": delta})
        result = gen.result          # 迭代结束后必定可用

    `deltas()` 正常结束后 `result` 一定非 None。抛 `LLMError` 时 `result` 保持
    None —— 那表示"没生成出可采信的东西"，调用方走拒答。
    """

    def __init__(
        self,
        question: str,
        passages: list,
        *,
        mode: str = DEFAULT_MODE,
        answer_id: str | None = None,
        config=None,
        save: bool = True,
    ) -> None:
        if mode not in MODES:
            raise PromptError("未知的回答模式：%r（可选：%s）" % (mode, " / ".join(MODES)))

        self.question = question
        self.passages = passages
        self.mode = mode
        self.answer_id = answer_id or str(uuid.uuid4())
        self.save = save
        self.result: GenerationResult | None = None

        # `render` 在片段为空时抛 PromptError —— 空上下文 MUST NOT 走到模型
        # （docs/05 §4.3）。放在构造函数里而不是 `deltas()` 里，是为了让调用方
        # 在**发出任何 SSE 帧之前**就知道参数不对。
        self.prompt_text = render(question, passages, mode=mode)

        from backend.api.config import AppConfig  # 延迟导入：避免 config→generate 的环

        cfg = config or AppConfig()
        self._api_key = cfg.agicto_api_key
        self._base_url = cfg.llm_base_url or DEFAULT_BASE_URL
        self._model = cfg.llm_model or DEFAULT_MODEL
        self._timeout = cfg.request_timeout_s or 10.0

    async def deltas(self) -> AsyncIterator[str]:
        """逐段产出**模型原始**增量，并在结束时把校验后的结果放进 `self.result`。

        ⚠️ 产出的是原始文本，**引用标记尚未校验**。原因见模块文档字符串。
        """

        parts: list[str] = []
        async for delta in stream_model(
            self.prompt_text,
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
        ):
            parts.append(delta)
            yield delta

        self._finish("".join(parts))

    def _finish(self, raw: str) -> None:
        answer_text, used_ids, failure = parse_answer(raw, self.passages)

        self.result = GenerationResult(
            answer_text=answer_text,
            used_citation_ids=used_ids,
            is_citation_failure=failure,
            raw_model_output=raw,
            answer_id=self.answer_id,
            question=self.question,
            mode=self.mode,
            sources=build_sources(self.passages, used_ids),
            confidence=confidence_of(self.passages),
            model=self._model,
        )

        if failure:
            logger.warning(
                "generation_citation_failure answer_id=%s reason=%s used=%d",
                self.answer_id,
                "资料不足哨兵" if not used_ids else "无有效引用",
                len(used_ids),
            )

        if self.save:
            save_result(self.result)


async def generate(
    question: str,
    passages: list,
    *,
    mode: str = DEFAULT_MODE,
    answer_id: str | None = None,
    config=None,
    save: bool = True,
) -> GenerationResult:
    """一次性生成（CLI 与脚本用）。失败时抛 `LLMError`。"""

    generation = Generation(
        question, passages, mode=mode, answer_id=answer_id, config=config, save=save
    )
    async for _ in generation.deltas():
        pass

    assert generation.result is not None  # deltas() 正常返回则必然已填
    return generation.result


def start(
    question: str,
    passages: list,
    *,
    mode: str = DEFAULT_MODE,
    answer_id: str | None = None,
    config=None,
    save: bool = True,
) -> Generation:
    """创建一次流式生成。**服务端用这个。**"""

    return Generation(
        question, passages, mode=mode, answer_id=answer_id, config=config, save=save
    )


def run(coro):
    """在无事件循环的上下文里跑协程（CLI 用）。**服务端 MUST NOT 调用。**"""

    return asyncio.run(coro)
