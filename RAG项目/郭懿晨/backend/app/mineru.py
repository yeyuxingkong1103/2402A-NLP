import os
import re
import sys
import tempfile
from pathlib import Path

from loguru import logger

current_dir = str(Path(__file__).resolve().parent)
removed_current_dir = False
if current_dir in sys.path:
    sys.path.remove(current_dir)
    removed_current_dir = True

from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
from mineru.data.data_reader_writer import FileBasedDataWriter
from pydantic import BaseModel

if removed_current_dir:
    sys.path.insert(0, current_dir)

try:
    import safetensors
    import transformers.modeling_utils as transformers_modeling_utils
except ImportError:
    safetensors = None
else:
    _original_safe_open = safetensors.safe_open

    def _wrap_safe_open(*args, **kwargs):
        file_handle = _original_safe_open(*args, **kwargs)
        original_metadata = file_handle.metadata

        class _WrappedFileHandle:
            def __init__(self, inner):
                self._inner = inner

            def metadata(self):
                metadata = original_metadata()
                if not metadata:
                    return {"format": "pt"}
                return metadata

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._inner.__exit__(exc_type, exc, tb)

        return _WrappedFileHandle(file_handle)

    safetensors.safe_open = _wrap_safe_open
    transformers_modeling_utils.safe_open = _wrap_safe_open


class MinerUBlock(BaseModel):
    """MinerU 解析出的结构化块。"""

    page: int
    label: str
    text: str
    source_span: str
    raw_text: str | None = None
    vision_text: str | None = None
    vision_model: str | None = None
    vision_applied: bool = False


class MinerUParser:
    """MinerU 解析适配器。"""

    def __init__(
        self,
        backend: str = "pipeline",
        model_source: str = "local",
        formula_enable: bool = False,
        table_enable: bool = False,
    ) -> None:
        self.parse_method = "auto" if backend == "pipeline" else backend
        self.model_source = model_source
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    def parse_pdf(self, pdf_path: Path) -> list[MinerUBlock]:
        """解析 PDF 并返回结构化块。"""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        try:
            parsed_blocks = self._parse_with_mineru(pdf_path)
            if parsed_blocks:
                return parsed_blocks
            logger.warning("MinerU returned no blocks for {}, falling back to pypdf.", pdf_path)
        except Exception as exc:
            logger.warning("MinerU parse failed for {}, falling back to pypdf: {}", pdf_path, exc)
        return self._fallback_parse_pdf(pdf_path)

    def _parse_with_mineru(self, pdf_path: Path) -> list[MinerUBlock]:
        """用 MinerU 解析 PDF。"""
        parsed_docs: list[list[MinerUBlock]] = []

        with tempfile.TemporaryDirectory(prefix="mineru-") as temp_dir:
            image_writer = FileBasedDataWriter(str(Path(temp_dir) / "images"))
            os.environ["MINERU_MODEL_SOURCE"] = self.model_source

            def on_doc_ready(doc_index: int, model_list: list[object], middle_json: dict, ocr_enable: bool) -> None:
                parsed_docs.append(self._parse_middle_json(middle_json))

            pdf_bytes = pdf_path.read_bytes()
            doc_analyze_streaming(
                [pdf_bytes],
                [image_writer],
                [None],
                on_doc_ready,
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                client_side_output_generation=False,
            )

        return parsed_docs[0] if parsed_docs else []

    def _fallback_parse_pdf(self, pdf_path: Path) -> list[MinerUBlock]:
        """MinerU 不可用时，退回到 pypdf 文本提取。"""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("缺少 pypdf，无法执行 PDF 文本兜底解析。") from exc

        reader = PdfReader(str(pdf_path))
        blocks: list[MinerUBlock] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = self._normalize_fallback_text(page.extract_text() or "")
            if not text:
                continue
            for block_index, paragraph in enumerate(self._split_fallback_paragraphs(text), start=1):
                blocks.append(
                    MinerUBlock(
                        page=page_number,
                        label="正文",
                        text=paragraph,
                        source_span=f"page={page_number}:fallback={block_index}",
                        raw_text=paragraph,
                    )
                )
        return blocks

    def _normalize_fallback_text(self, text: str) -> str:
        """清理 pypdf 提取文本中的多余空白。"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _split_fallback_paragraphs(self, text: str) -> list[str]:
        """把页面文本拆成可入库的段落。"""
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
        return paragraphs or ([text] if text else [])

    def _parse_middle_json(self, content: dict[str, object]) -> list[MinerUBlock]:
        """从 MinerU middle.json 中提取块。"""
        blocks: list[MinerUBlock] = []
        pdf_info = content.get("pdf_info") if isinstance(content, dict) else None
        if not isinstance(pdf_info, list):
            return blocks

        for page_index, page_info in enumerate(pdf_info):
            if not isinstance(page_info, dict):
                continue
            page_number = int(page_info.get("page_no", page_info.get("page_idx", page_index))) + 1
            page_blocks = self._extract_page_blocks(page_info)
            for block_position, block in enumerate(page_blocks, start=1):
                if not isinstance(block, dict):
                    continue
                text = self._extract_text(block).strip()
                if not text:
                    continue
                raw_index = block.get("index", block_position)
                try:
                    block_index = int(raw_index)
                except (TypeError, ValueError):
                    block_index = block_position
                label = str(block.get("type") or block.get("sub_type") or block.get("category") or "text")
                blocks.append(
                    MinerUBlock(
                        page=page_number,
                        label=label,
                        text=text,
                        source_span=f"page={page_number}:block={block_index}",
                    )
                )
        return blocks

    def _extract_page_blocks(self, page_info: dict[str, object]) -> list[object]:
        """提取单页中的块列表。"""
        for key in ("preproc_blocks", "segmented_content_blocks", "blocks", "para_blocks"):
            value = page_info.get(key)
            if isinstance(value, list):
                return value
        return []

    def _extract_text(self, node: object) -> str:
        """递归提取块文本。"""
        parts: list[str] = []
        self._collect_text(node, parts)
        return " ".join(part for part in parts if part)

    def _collect_text(self, node: object, parts: list[str]) -> None:
        """收集节点中的文本片段。"""
        if node is None:
            return
        if isinstance(node, str):
            stripped = node.strip()
            if stripped:
                parts.append(stripped)
            return

        if isinstance(node, list):
            for item in node:
                self._collect_text(item, parts)
            return

        if isinstance(node, dict):
            for key in ("content", "text", "code_body", "markdown"):
                value = node.get(key)
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        parts.append(stripped)
            for key in ("lines", "spans", "blocks", "items", "list_items"):
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        self._collect_text(item, parts)
