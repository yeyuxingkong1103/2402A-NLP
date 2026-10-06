from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Protocol

from backend.app.mineru import MinerUBlock
from backend.app.vision import VisionResult, OpenAIVisionClient, render_pdf_pages


TEXTUAL_LABELS = {
    "text",
    "title",
    "paragraph",
    "para",
    "caption",
    "header",
    "footer",
    "reference",
    "note",
}

VISION_LABELS = {"table", "figure", "image", "formula", "scan", "scanned", "ocr_error"}


class VisionClient(Protocol):
    def describe_image(self, image_path: Path, prompt: str) -> str:
        """把图片转成 Markdown。"""


@dataclass(slots=True)
class VisionBlockSelection:
    """需要交给 VLM 处理的块。"""

    block: MinerUBlock
    image_path: Path


def select_blocks_for_vision(blocks: list[MinerUBlock]) -> list[VisionBlockSelection]:
    """挑出非纯文本块供视觉模型处理。"""
    selections: list[VisionBlockSelection] = []
    for block in blocks:
        label = str(block.label or "").strip().lower()
        if label in TEXTUAL_LABELS:
            continue
        if label in VISION_LABELS or label:
            selections.append(
                VisionBlockSelection(
                    block=block,
                    image_path=Path(f"page-{block.page}.png"),
                )
            )
    return selections


def merge_vision_blocks(
    blocks: list[MinerUBlock],
    selections: list[VisionBlockSelection],
    results: list[VisionResult],
) -> list[MinerUBlock]:
    """把 VLM 结果回填到原始块中。"""
    result_map = {result.source_span: result.markdown for result in results}
    selected_spans = {selection.block.source_span for selection in selections}
    merged: list[MinerUBlock] = []

    for block in blocks:
        markdown = result_map.get(block.source_span)
        if block.source_span in selected_spans and markdown:
            merged.append(
                block.model_copy(
                    update={
                        "raw_text": block.text,
                        "text": markdown,
                        "vision_text": markdown,
                        "vision_model": "openai-compatible-vlm",
                        "vision_applied": True,
                    }
                )
            )
        else:
            merged.append(block)

    return merged


class MinerUVLMPostProcessor:
    """MinerU 后处理 VLM 增强器。"""

    def __init__(self, client: VisionClient, prompt: str | None = None) -> None:
        self.client = client
        self.prompt = prompt or "请把这张文档图片转换成 Markdown，保留表格结构和必要说明。"

    def process(self, pdf_path: Path, blocks: list[MinerUBlock]) -> list[MinerUBlock]:
        selections = select_blocks_for_vision(blocks)
        if not selections:
            return blocks

        pages = {item.block.page for item in selections}
        with tempfile.TemporaryDirectory(prefix="vlm-") as temp_dir:
            rendered_pages = render_pdf_pages(pdf_path, pages, Path(temp_dir))
            results: list[VisionResult] = []
            for selection in selections:
                image_path = rendered_pages.get(selection.block.page)
                if image_path is None:
                    continue
                markdown = self.client.describe_image(image_path, self.prompt)
                if not markdown.strip():
                    continue
                results.append(
                    VisionResult(
                        source_span=selection.block.source_span,
                        markdown=markdown.strip(),
                    )
                )
        return merge_vision_blocks(blocks, selections, results)
