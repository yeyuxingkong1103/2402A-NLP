# -*- coding: utf-8 -*-
"""
大模型客户端封装

包含两个相互独立的客户端：
    1. ChatLLM       —— 文本生成（deepseek-v4-flash），用于生成问答答案
    2. VisionLLM     —— 视觉理解（通义千问 VL），用于 V2 解析 PDF 图表

两者均走 OpenAI 兼容接口，因此共用同一套调用逻辑。

设计要点（对应 ADR-009）：
    视觉模型是本项目唯一存在外部不确定性的组件，因此抽象为可插拔 Provider：
    更换视觉模型只需修改 .env 中的 VISION_BASE_URL / VISION_API_KEY / VISION_MODEL，
    **无需改动任何业务代码**。
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """大模型调用异常"""


# ===========================================================================
# 文本生成
# ===========================================================================

class ChatLLM:
    """文本生成客户端（deepseek-v4-flash）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.timeout = timeout or settings.llm_timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """懒加载 OpenAI 兼容客户端"""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise LLMError(
                "未配置 DEEPSEEK_API_KEY，请在 .env 中填写后重试"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("需要 openai 库，请先安装：pip install openai") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=2,
        )
        return self._client

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式对话补全：产出结构化事件。

        事件形态（dict）：
            {"kind": "reasoning", "chars": 1284}   —— 推理进度（只报字数，内容不外泄）
            {"kind": "content",   "text": "…"}     —— 正式答复的增量文字

        与 chat() 的关系：
            chat() 保持原样（一次性返回），评测脚本、压测、非流式接口都还在用它；
            本方法是**新增的并行的流式实现**，两者共用同一套鉴权、模型与参数配置。

        两个关键处理与 chat() 保持一致：
            1. 推理模型的 reasoning_content 属于模型内部过程，按 ADR-028 不对外暴露 ——
               这里只累计它的长度用于日志，不往下传，前端只会看到正式答复。
            2. 整段结束后若一个字正文都没产出（推理吃光 token 预算），
               必须显式抛 LLMError，不能把"空答案"当成功返回。
        """
        client = self._get_client()
        started = time.time()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
                stream=True,
            )
        except Exception as exc:
            logger.error("大模型流式调用失败：%s", exc)
            raise LLMError(f"大模型调用失败：{exc}") from exc

        pieces: List[str] = []
        finish_reason = ""
        reasoning_chars = 0
        reported_chars = 0
        reported_at = time.time()
        try:
            for event in stream:
                choices = getattr(event, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = str(choice.finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    reasoning_chars += len(reasoning_piece)
                    # 推理阶段实时上报「思考进度」（只报字数，不报内容）：
                    # 推理模型的思考过程可能持续十几秒，期间正文是空的，
                    # 若什么都不报，用户会以为卡死了。这里做节流，每累计 50 字
                    # 或距上次上报满 0.4 秒才发一条，避免事件过密。
                    now = time.time()
                    if (reasoning_chars - reported_chars >= 50) or (now - reported_at >= 0.4):
                        reported_chars = reasoning_chars
                        reported_at = now
                        yield {"kind": "reasoning", "chars": reasoning_chars}
                piece = getattr(delta, "content", None)
                if piece:
                    pieces.append(piece)
                    yield {"kind": "content", "text": piece}
        except Exception as exc:
            logger.error("大模型流式调用中断：%s", exc)
            raise LLMError(f"大模型流式调用中断：{exc}") from exc

        answer = "".join(pieces).strip()
        logger.info(
            "大模型流式生成完成 | 模型=%s | 耗时=%.2fs | 正文=%d字 | 推理=%d字 | finish_reason=%s",
            self.model, time.time() - started, len(answer), reasoning_chars, finish_reason or "?",
        )
        if not answer:
            raise LLMError(
                "模型输出被长度限制截断，未能产出正式答复"
                "（推理过程占满了 token 预算，请提高 LLM_MAX_TOKENS）"
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发起一次对话补全，返回回答正文。

        异常统一包装为 LLMError，由上层决定降级策略。
        """
        client = self._get_client()
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
                stream=False,
            )
        except Exception as exc:
            logger.error("大模型调用失败：%s", exc)
            raise LLMError(f"大模型调用失败：{exc}") from exc

        elapsed = time.time() - started
        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", "") or ""
        content = (message.content or "").strip()
        reasoning = (getattr(message, "reasoning_content", "") or "").strip()

        usage = getattr(response, "usage", None)
        logger.info(
            "大模型生成完成 | 模型=%s | 耗时=%.2fs | 输入token=%s | 输出token=%s"
            " | finish_reason=%s",
            self.model, elapsed,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            finish_reason or "?",
        )

        # ★ 空响应必须显式报错，不得静默返回空字符串 ★
        # deepseek-v4-flash 是推理模型：先输出 reasoning_content 再输出 content。
        # 当推理过程吃光 max_tokens 预算时，finish_reason 为 "length" 且 content 为空。
        # 旧实现直接返回空串，用户会拿到一个「没有任何内容」的答案且系统毫无异常提示
        # （实测已产生 2 条空答案记录，并导致一次评测样本被 RAGAS 判为 nan）。
        if not content:
            # 推理内容只用于服务端排查，绝不返回给调用方（见 ADR-028）
            if reasoning:
                logger.warning(
                    "模型未产出正式答复，仅有推理内容 | finish_reason=%s | "
                    "推理内容前 800 字：%s",
                    finish_reason or "?", reasoning[:800],
                )
            else:
                logger.warning(
                    "模型返回空内容 | finish_reason=%s | 无 reasoning_content",
                    finish_reason or "?",
                )
            if finish_reason == "length":
                raise LLMError(
                    "模型输出被长度限制截断，未能产出正式答复"
                    "（推理过程占满了 token 预算，请提高 LLM_MAX_TOKENS）"
                )
            raise LLMError("模型返回了空内容")

        return content


# ===========================================================================
# 视觉理解（V2 使用）
# ===========================================================================

class VisionLLM:
    """
    视觉理解客户端（通义千问 VL，可插拔）

    V1 不调用本类；V2 用它解析 PDF 中的图表、流程图。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.vision_api_key
        self.base_url = base_url or settings.vision_base_url
        self.model = model or settings.vision_model
        self.timeout = timeout or settings.vision_timeout
        self._client: Any = None

    @property
    def available(self) -> bool:
        """是否已配置可用的视觉模型"""
        return bool(self.api_key) and settings.vision_enabled

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise LLMError("未配置 VISION_API_KEY，无法调用视觉模型")

        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=1,
        )
        return self._client

    @staticmethod
    def _to_data_url(image_path: Path) -> str:
        """把本地图片编码为 data URL（部分兼容接口不接受本地路径）"""
        suffix = image_path.suffix.lower().lstrip(".") or "png"
        if suffix == "jpg":
            suffix = "jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/{suffix};base64,{encoded}"

    def describe_image(
        self,
        image_path: Path,
        *,
        caption: str = "",
        context: str = "",
        prompt: Optional[str] = None,
    ) -> str:
        """
        解析单张图片，返回中文语义描述。

        参数：
            image_path : 图片文件路径
            caption    : 原图注（作为提示，提升描述准确度）
            context    : 所在章节上下文
            prompt     : 自定义提示词，默认使用通用图表解析提示词

        用途（V2）：把图表、流程图的语义转成文本，补充进知识库 chunk，
        使「按图提问」能够被检索到。
        """
        if not image_path.exists():
            raise LLMError(f"图片不存在：{image_path}")

        default_prompt = (
            "你是技术标准文档的图表解析助手。请用中文详细描述这张图片的内容，"
            "重点关注：图表的类型（流程图/结构图/表格/示意图）、"
            "图中出现的所有文字标签与数值、各元素之间的逻辑关系与流向。"
            "描述要完整到：读者不看原图，仅凭你的描述就能理解这张图表达了什么。"
            "直接输出描述内容，不要添加「这张图展示了」之类的开场白。"
        )
        hints: List[str] = []
        if caption:
            hints.append(f"原图注：{caption}")
        if context:
            hints.append(f"所在章节：{context}")
        text_prompt = "\n".join([prompt or default_prompt] + hints)

        client = self._get_client()
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": self._to_data_url(image_path)}},
                        {"type": "text", "text": text_prompt},
                    ],
                }],
                max_tokens=1024,
            )
        except Exception as exc:
            logger.error("视觉模型调用失败：%s | 图片=%s", exc, image_path.name)
            raise LLMError(f"视觉模型调用失败：{exc}") from exc

        description = (response.choices[0].message.content or "").strip()
        logger.info("视觉解析完成 | 图片=%s | 耗时=%.1fs | 描述长度=%d",
                    image_path.name, time.time() - started, len(description))
        return description


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_chat_llm: Optional[ChatLLM] = None
_vision_llm: Optional[VisionLLM] = None


def get_chat_llm() -> ChatLLM:
    """获取文本生成客户端单例"""
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatLLM()
    return _chat_llm


def get_vision_llm() -> VisionLLM:
    """获取视觉理解客户端单例"""
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = VisionLLM()
    return _vision_llm
