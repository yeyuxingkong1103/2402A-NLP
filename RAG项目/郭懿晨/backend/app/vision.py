from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib import error, request
import base64
import json

import pypdfium2 as pdfium

from backend.app.mineru import MinerUBlock


@dataclass(frozen=True)
class VisionResult:
    """VLM 识别结果。"""

    source_span: str
    markdown: str


class OpenAIVisionClient:
    """OpenAI 兼容视觉接口。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """用视觉模型把图片转成 Markdown。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个文档视觉解析助手，只输出 Markdown。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(image_path)}"
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        chat_url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        chat_request = request.Request(chat_url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(chat_request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"VLM 调用失败: {exc}") from exc

        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content).strip()

    def _encode_image(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")


@dataclass(frozen=True)
class VisionBlockSelection:
    """需要送入视觉模型的块。"""

    block: MinerUBlock
    image_path: Path


VISION_LABELS = {"table", "figure", "image", "formula", "caption"}


def select_blocks_for_vision(blocks: list[MinerUBlock]) -> list[VisionBlockSelection]:
    """挑选需要走 VLM 的非文本块。"""
    selections: list[VisionBlockSelection] = []
    for block in blocks:
        if _should_use_vision(block):
            selections.append(VisionBlockSelection(block=block, image_path=Path(f"page-{block.page}.png")))
    return selections


def _should_use_vision(block: MinerUBlock) -> bool:
    label = block.label.lower()
    return label in VISION_LABELS or label not in {"text", "正文", "paragraph"}


def merge_vision_blocks(blocks: list[MinerUBlock], selections: list[VisionBlockSelection], results: list[VisionResult]) -> list[MinerUBlock]:
    """把 VLM Markdown 合并回原始块。"""
    result_by_span = {item.source_span: item.markdown for item in results}
    merged: list[MinerUBlock] = []
    for block in blocks:
        markdown = result_by_span.get(block.source_span)
        if markdown:
            merged.append(
                block.model_copy(
                    update={
                        "text": markdown,
                        "vision_text": markdown,
                        "vision_applied": True,
                        "vision_model": "openai-compatible-vlm",
                        "raw_text": block.text,
                    }
                )
            )
        else:
            merged.append(block)
    return merged


def render_pdf_pages(pdf_path: Path, pages: set[int], output_dir: Path) -> dict[int, Path]:
    """把指定页渲染成图片。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    rendered: dict[int, Path] = {}
    for page_number in sorted(pages):
        page = document.get_page(page_number - 1)
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        image_path = output_dir / f"page-{page_number}.png"
        pil_image.save(image_path)
        rendered[page_number] = image_path
    return rendered
