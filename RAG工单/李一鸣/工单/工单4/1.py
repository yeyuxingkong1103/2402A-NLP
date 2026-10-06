"""Work order 04: add image/figure extraction and multimodal retrieval hooks."""

from __future__ import annotations

import base64
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, make_chunks


@dataclass
class FigureAsset:
    source: str
    page: int
    image_path: str
    caption: str = ""
    ocr_text: str = ""


class VisionModel(Protocol):
    def describe(self, image_bytes: bytes, prompt: str) -> str: ...


def extract_images(pdf_path: str, output_dir: str = "figures") -> list[FigureAsset]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF or replace extract_images with your PDF renderer") from exc
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    assets = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            for image_index, image in enumerate(page.get_images(full=True)):
                payload = document.extract_image(image[0])
                suffix = payload.get("ext", "png")
                image_path = target / f"page-{page_number:04d}-{image_index:02d}.{suffix}"
                image_path.write_bytes(payload["image"])
                assets.append(FigureAsset(pdf_path, page_number, str(image_path)))
    return assets


def enrich_figures(assets: list[FigureAsset], vision: VisionModel | None = None) -> list[FigureAsset]:
    if vision is None:
        return assets
    for asset in assets:
        asset.ocr_text = vision.describe(Path(asset.image_path).read_bytes(), "Extract all text and explain the figure.")
        asset.caption = vision.describe(Path(asset.image_path).read_bytes(), "Write a concise searchable caption.")
    return assets


def multimodal_documents(assets: list[FigureAsset]) -> list[dict[str, Any]]:
    return [
        {
            "source": asset.source,
            "page": asset.page,
            "text": f"Figure caption: {asset.caption}\nOCR: {asset.ocr_text}",
            "metadata": {"content_type": "figure", "image_path": asset.image_path},
        }
        for asset in assets
        if asset.caption or asset.ocr_text
    ]


def image_payload(path: str) -> str:
    """Encode an image for an OpenAI-compatible vision request."""
    data = Path(path).read_bytes()
    digest = hashlib.md5(data).hexdigest()  # useful for cache keys
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}#cache={digest}"
