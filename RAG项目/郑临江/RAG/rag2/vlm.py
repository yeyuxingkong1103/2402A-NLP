# -*- coding: utf-8 -*-
"""多模态视觉语言模型（VLM）图片理解模块（默认接本地 Ollama 视觉模型）。

与 ``ocr.py`` 的区别：OCR 只把图片里的**文字**抽出来；本模块让视觉模型真正
「看懂」图片——描述内容、看图问答，适用于照片、示意图、图表等非纯文字图片。

实现要点：

1. **OpenAI 兼容多模态消息格式**：把图片编码成 ``data:image/...;base64,...``，
   以 ``{"type": "image_url", "image_url": {"url": <data url>}}`` 放进 user 消息的
   ``content`` 列表，随文本一起发给模型（Ollama 的 /v1 兼容端点支持该格式）；
2. **复用 LLMClient**：底层走 ``rag2.llm_client.LLMClient``（默认连 Ollama），
   仅把 ``reasoning_effort`` 置 None（视觉模型不是思考链模型，避免未知参数报错）；
3. **懒加载**：首次真正调用时才创建 OpenAI 客户端，import 本模块零开销。

运行环境：

    - 依赖本地 Ollama 已启动，且已 ``ollama pull qwen2.5vl:7b``（或其它视觉模型）；
    - 模型名在 config.yaml 的 ``vlm.model`` 里配置。

用法示例：

    from rag2.vlm import describe_image, answer_image, ImageDescriber

    print(describe_image("病害图片.jpg"))
    print(answer_image("病害图片.jpg", "这张图是什么病？如何防治？"))

    desc = ImageDescriber.from_config(load_config().vlm)   # 未启用时返回 None
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterator, Sequence

from .logging_config import get_logger

logger = get_logger("vlm")

DEFAULT_PROMPT = (
    "请详细描述这张图片的内容：主体、场景、关键细节，以及它可能传达的信息或类别。"
    "用简体中文分点说明。"
)

# 常见图片扩展名 → MIME（mimetypes 在 Windows 上有时识别不准，这里显式兜底）
_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".ico": "image/x-icon",
}


def _guess_mime(image: str | Path) -> str:
    suffix = Path(image).suffix.lower()
    return _IMAGE_MIME.get(suffix) or mimetypes.guess_type(str(image))[0] or "image/jpeg"


def encode_image_data_url(image: str | Path, mime: str | None = None) -> str:
    """把本地图片编码为 ``data:<mime>;base64,<...>``（多模态消息用）。"""
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    mime = mime or _guess_mime(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def resolve_image_data_url(image: str | Path, mime: str | None = None) -> str:
    """统一返回 data URL：已是 data URL 直接返回，否则按本地文件编码。"""
    s = str(image)
    if s.startswith("data:"):
        return s
    return encode_image_data_url(image, mime)


def image_content_part(image: str | Path, mime: str | None = None) -> dict[str, Any]:
    """生成 OpenAI 兼容的图片 content 片段（直接拼进 messages）。"""
    return {"type": "image_url", "image_url": {"url": resolve_image_data_url(image, mime)}}


class ImageDescriber:
    """视觉语言模型客户端（描述图片 / 看图问答）。"""

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 300.0,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.prompt = prompt or DEFAULT_PROMPT
        self._llm: Any = None

    @classmethod
    def from_config(cls, vlm_config: Any) -> "ImageDescriber | None":
        """从 ``config.vlm``（VLMConfig）构建；未启用或为 None 时返回 None。"""
        if vlm_config is None or not getattr(vlm_config, "enabled", False):
            return None
        return cls(
            model=vlm_config.model,
            base_url=vlm_config.base_url,
            api_key=vlm_config.api_key,
            temperature=vlm_config.temperature,
            max_tokens=vlm_config.max_tokens,
            prompt=getattr(vlm_config, "prompt", None),
        )

    def _client(self) -> Any:
        if self._llm is None:
            from .llm_client import LLMClient

            self._llm = LLMClient(
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                reasoning_effort=None, timeout=self.timeout,
            )
            logger.info("初始化 VLM：%s @ %s", self.model, self.base_url)
        return self._llm

    # ------------------------------------------------------------ 底层多模态调用
    def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        """非流式多模态生成（messages 的 content 可以是文本或「文本 + 图片」列表）。"""
        return self._client().chat(messages, **kwargs)

    def stream(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """流式多模态生成，逐段产出文本增量。"""
        return self._client().stream(messages, **kwargs)

    # ------------------------------------------------------------ 高层封装
    def _ask(self, image: str | Path, prompt: str) -> str:
        content = [{"type": "text", "text": prompt}, image_content_part(image)]
        return self.chat([{"role": "user", "content": content}]).strip()

    def describe(self, image: str | Path, prompt: str | None = None) -> str:
        """生成图片内容描述。"""
        return self._ask(image, prompt or self.prompt)

    def answer(self, image: str | Path, question: str) -> str:
        """看图问答：围绕图片内容回答一个问题。"""
        q = (question or "").strip() or "请描述这张图片。"
        return self._ask(image, q)

    def describe_batch(
        self, images: Sequence[str | Path], prompt: str | None = None
    ) -> list[str]:
        """批量描述多张图片（按顺序返回描述文本列表）。"""
        return [self.describe(img, prompt) for img in images]


# ---------------------------------------------------------------------- 便捷函数
def describe_image(image: str | Path, prompt: str | None = None, **kwargs: Any) -> str:
    """便捷函数：描述一张图片（等价 ``ImageDescriber(**kw).describe(...)``）。"""
    return ImageDescriber(**kwargs).describe(image, prompt)


def answer_image(image: str | Path, question: str, **kwargs: Any) -> str:
    """便捷函数：看图问答。"""
    return ImageDescriber(**kwargs).answer(image, question)


__all__ = [
    "ImageDescriber",
    "describe_image",
    "answer_image",
    "encode_image_data_url",
    "resolve_image_data_url",
    "image_content_part",
    "DEFAULT_PROMPT",
]
