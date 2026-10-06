from pathlib import Path

from backend.app.config import AppSettings
from backend.app.mineru import MinerUBlock
from backend.app.mineru_vlm import (
    MinerUVLMPostProcessor,
    VisionBlockSelection,
    VisionClient,
    merge_vision_blocks,
    select_blocks_for_vision,
)
from backend.app.vision import VisionResult


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls = []

    def describe_image(self, image_path: Path, prompt: str) -> str:
        self.calls.append({"image_path": image_path, "prompt": prompt})
        return "| 列1 | 列2 |\n| --- | --- |\n| A | B |"


def test_select_blocks_for_vision_only_picks_non_text_blocks():
    blocks = [
        MinerUBlock(page=1, label="text", text="正文", source_span="page=1:block=1"),
        MinerUBlock(page=1, label="table", text="表格占位", source_span="page=1:block=2"),
        MinerUBlock(page=2, label="figure", text="图像占位", source_span="page=2:block=1"),
    ]

    selection = select_blocks_for_vision(blocks)

    assert [item.block.label for item in selection] == ["table", "figure"]


def test_merge_vision_blocks_replaces_text_with_markdown():
    blocks = [
        MinerUBlock(page=1, label="text", text="正文", source_span="page=1:block=1"),
        MinerUBlock(page=1, label="table", text="表格占位", source_span="page=1:block=2"),
    ]
    selection = [VisionBlockSelection(block=blocks[1], image_path=Path("page-1.png"))]
    results = [
        VisionResult(
            source_span="page=1:block=2",
            markdown="| 列1 | 列2 |\n| --- | --- |\n| A | B |",
        )
    ]

    merged = merge_vision_blocks(blocks, selection, results)

    assert merged[0].text == "正文"
    assert merged[1].text == "| 列1 | 列2 |\n| --- | --- |\n| A | B |"
    assert merged[1].label == "table"
    assert merged[1].vision_applied is True


def test_vision_result_keeps_source_span():
    result = VisionResult(source_span="page=1:block=2", markdown="说明")

    assert result.source_span == "page=1:block=2"
    assert result.markdown == "说明"


def test_app_settings_exposes_vlm_configuration_fields():
    settings = AppSettings()

    assert hasattr(settings, "vlm_enabled")
    assert hasattr(settings, "vlm_base_url")
    assert hasattr(settings, "vlm_api_key")
    assert hasattr(settings, "vlm_model")
    assert hasattr(settings, "vlm_timeout")
