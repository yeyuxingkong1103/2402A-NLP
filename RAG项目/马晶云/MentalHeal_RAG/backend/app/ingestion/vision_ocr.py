from pathlib import Path
from typing import Any

import fitz
import numpy as np

from .schemas import ExtractedPage
from .text_processing import normalize_text


class VisionOcrPdfParser:
    name = "paddleocr-vl"

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu

    def parse(self, path: Path) -> dict[int, str]:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR-VL is required. Install paddleocr and compatible paddlepaddle packages."
            ) from error

        device = "gpu" if self.use_gpu else "cpu"
        pipeline = PaddleOCRVL(device=device)
        results: dict[int, str] = {}
        with fitz.open(path) as document:
            for index, page in enumerate(document, start=1):
                image = self._page_image(page)
                output = pipeline.predict(image)
                text = self._result_text(output)
                if text:
                    results[index] = text
        return results

    def apply(self, pages: list[ExtractedPage], vision_by_page: dict[int, str]) -> None:
        for page in pages:
            page.vision_ocr_text = normalize_text(vision_by_page.get(page.page_number, ""))

    @staticmethod
    def _page_image(page: fitz.Page) -> Any:
        matrix = fitz.Matrix(2.0, 2.0)
        image = page.get_pixmap(matrix=matrix, alpha=False).pil_image()
        return np.asarray(image)

    @classmethod
    def _result_text(cls, result: Any) -> str:
        values: list[str] = []
        cls._collect_text(result, values)
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(value)
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
        return "\n".join(unique)

    @classmethod
    def _collect_text(cls, value: Any, output: list[str]) -> None:
        if isinstance(value, str) and value.strip():
            output.append(value)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"text", "rec_text", "rec_texts", "content", "markdown"}:
                    cls._collect_text(item, output)
                elif isinstance(item, (dict, list, tuple)):
                    cls._collect_text(item, output)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                cls._collect_text(item, output)
            return
        payload = getattr(value, "json", None)
        if payload is not None:
            cls._collect_text(payload() if callable(payload) else payload, output)
