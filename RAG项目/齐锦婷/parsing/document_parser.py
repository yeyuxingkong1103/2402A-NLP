import base64
import json
from pathlib import Path
from typing import Any

import fitz
import httpx

from backend.app.core.config import get_settings


settings = get_settings()


class DocumentParser:
    """负责把原始 PDF 解析成统一的列表结构。"""

    async def parse(self, file_path: Path, document_id: int) -> list[dict[str, Any]]:
        # 优先调用远程 MinerU，因为 MinerU 更擅长保留 PDF 的版面结构。
        if settings.mineru_api_url and not settings.mineru_api_url.startswith("请在这里"):
            try:
                # 调用远程 MinerU，并接收统一格式的解析块。
                mineru_blocks = await self._parse_with_mineru(file_path, document_id)
                # 远程服务返回空列表时不能当作解析成功，必须进入本地降级或报错。
                if mineru_blocks:
                    return mineru_blocks
                # 把空结果转换成异常，进入下面的本地降级逻辑。
                raise RuntimeError("远程 MinerU 返回了空解析结果")
            except Exception:
                # 如果远程 MinerU 失败，并且允许降级，就继续使用本地基础解析。
                if not settings.mineru_fallback_to_local:
                    raise
        # 本地降级解析只抽取文本，保证没有 MinerU 时项目仍然能跑通。
        local_blocks = self._parse_with_pymupdf(file_path)
        # 如果本地 PDF 有文字，就直接使用文字解析结果。
        if local_blocks:
            return local_blocks
        # 扫描型 PDF 没有文字层时，尝试把页面图片交给 Qwen-VL 识别。
        qwen_blocks = await self._parse_scanned_pdf_with_qwen(file_path)
        # 如果 Qwen-VL 也没有返回内容，就抛出明确异常，不允许空文档入库。
        if qwen_blocks:
            return qwen_blocks
        # 返回空结果前直接报错，让任务状态显示为 failed。
        raise RuntimeError("MinerU、本地文字解析和 Qwen-VL 都没有解析出有效内容")

    async def _parse_scanned_pdf_with_qwen(self, file_path: Path) -> list[dict[str, Any]]:
        # 创建图片解析器，用于调用 qwen-vl-max。
        image_parser = ImageTableParser()
        # 准备保存每一页的视觉解析结果。
        blocks: list[dict[str, Any]] = []
        # 打开扫描型 PDF。
        with fitz.open(file_path) as pdf:
            # 逐页处理，避免一次请求传入过大的 PDF。
            for page_index, page in enumerate(pdf):
                # 使用两倍缩放渲染页面，提高小字体识别效果。
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                # 把页面渲染成 PNG 二进制数据。
                image_bytes = pixmap.tobytes("png")
                # 转成 data URL，DashScope 可以直接读取这种图片地址。
                image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
                try:
                    # 调用 Qwen-VL 识别当前页面文字、表格和图片内容。
                    text = await image_parser.describe(image_data_url)
                except Exception:
                    # 单页识别失败时跳过这一页，继续处理剩余页面。
                    continue
                # 只有识别出有效文字才保存为一个文本块。
                if text and text.strip():
                    blocks.append({"type": "text", "text": text, "page": page_index + 1, "title": ""})
        # 返回扫描 PDF 的视觉解析结果。
        return blocks

    async def _parse_with_mineru(self, file_path: Path, document_id: int) -> list[dict[str, Any]]:
        # 读取文件字节，作为 multipart 表单上传给远程 MinerU 服务。
        files = {"file": (file_path.name, file_path.read_bytes(), "application/pdf")}
        # 传递文档编号，方便远程服务在日志中定位是哪份文档。
        data = {"document_id": str(document_id)}
        # 使用 Bearer Token 方式放置 MinerU 密钥。
        headers = {"Authorization": f"Bearer {settings.mineru_api_key}"}
        # 设置超时时间，避免大文档解析时请求过早断开。
        timeout = httpx.Timeout(settings.mineru_timeout_seconds)
        # 创建异步 HTTP 客户端，避免阻塞 FastAPI 事件循环。
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(settings.mineru_api_url, files=files, data=data, headers=headers)
        # 如果远程服务返回错误状态码，这里会抛出异常进入降级逻辑。
        response.raise_for_status()
        # 约定远程 MinerU 返回 JSON；不同服务格式可只改这里的适配代码。
        payload = response.json()
        # 如果返回值中有 blocks 字段，就直接使用 blocks。
        blocks = payload.get("blocks", payload if isinstance(payload, list) else [])
        # 统一字段，保证后面的清洗和分块不关心 MinerU 原始格式。
        return [self._normalize_block(block) for block in blocks]

    def _parse_with_pymupdf(self, file_path: Path) -> list[dict[str, Any]]:
        # 准备一个列表，用来存放每页解析出来的文本块。
        blocks: list[dict[str, Any]] = []
        # 打开 PDF 文件，PyMuPDF 会逐页读取内容。
        with fitz.open(file_path) as pdf:
            # page_index 从 0 开始，所以展示页码时要加 1。
            for page_index, page in enumerate(pdf):
                # 提取当前页的纯文本。
                text = page.get_text("text")
                # 如果这一页没有文字，就跳过，避免生成空块。
                if not text.strip():
                    continue
                # 把当前页文本放入统一结构。
                blocks.append({"type": "text", "text": text, "page": page_index + 1, "title": ""})
        # 返回所有页面的文本块。
        return blocks

    def _normalize_block(self, block: dict[str, Any]) -> dict[str, Any]:
        # MinerU 的不同版本字段可能不同，这里统一内容类型。
        block_type = block.get("type") or block.get("content_type") or "text"
        # 统一文本内容字段，方便后续处理。
        text = block.get("text") or block.get("content") or block.get("html") or ""
        # 统一页码字段，没有页码就记为 0。
        page = int(block.get("page") or block.get("page_number") or 0)
        # 统一标题路径字段，用于后续结构化分块。
        title = block.get("title") or block.get("title_path") or ""
        # 返回项目内部统一格式。
        return {"type": block_type, "text": text, "page": page, "title": title}


class ImageTableParser:
    """负责使用 qwen-vl-max 解析图片或复杂表格。"""

    async def describe(self, image_url_or_path: str) -> str:
        # 如果没有配置 DashScope 密钥，直接返回空字符串，由上层记录跳过。
        if not settings.dashscope_api_key or settings.dashscope_api_key.startswith("请在这里"):
            return ""
        # DashScope 兼容 OpenAI 风格的多模态接口，这里用 HTTP 直接调用。
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        # 构造请求头，把 API Key 放在 Authorization 中。
        headers = {"Authorization": f"Bearer {settings.dashscope_api_key}", "Content-Type": "application/json"}
        # 构造多模态消息，让模型把图片或表格转成可检索文本。
        payload = {
            "model": settings.qwen_vl_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请把这张法律文档图片或表格转换成准确、简洁、可检索的中文文本。"},
                        {"type": "image_url", "image_url": {"url": image_url_or_path}},
                    ],
                }
            ],
        }
        # 设置超时，避免图片解析长时间卡住。
        timeout = httpx.Timeout(120)
        # 发送请求给 qwen-vl-max。
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        # 如果失败，上层会捕获并记录跳过。
        response.raise_for_status()
        # 解析模型返回的文字。
        data = response.json()
        # 取第一条候选答案。
        return data["choices"][0]["message"]["content"]


def save_json(path: Path, data: Any) -> None:
    # 确保目标目录存在，避免写文件时报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 ensure_ascii=False 保留中文，方便你直接打开文件查看。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
