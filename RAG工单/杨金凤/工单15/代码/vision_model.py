"""
视觉问答旁路：检测到图片类问题时，直接抽 PDF 页面图像调用 GLM-4V。
"""
import os
import re
import base64
import logging
from typing import Optional, Tuple, List

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz

from rag.utils.storage_factory import STORAGE_IMPL

# 智谱 API
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_VISION_MODEL = "glm-4v-flash"  # 免费版；可改 "glm-4v" 或 "glm-4v-plus"


def is_visual_question(question: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    识别视觉问题，返回 (是否视觉, 页码, 图号)
    例：'在文件中第11页图3中，...' -> (True, 11, 3)
    """
    page_match = re.search(r'第\s*(\d+)\s*页', question)
    fig_match = re.search(r'图\s*(\d+)', question)
    if not page_match and not fig_match:
        return False, None, None
    page = int(page_match.group(1)) if page_match else None
    figure = int(fig_match.group(1)) if fig_match else None
    return True, page, figure


def extract_page_image(bucket: str, filename: str, page_num: int, dpi: int = 200) -> Optional[bytes]:
    """从 MinIO 取 PDF，抽出指定页为 PNG 字节"""
    try:
        binary = STORAGE_IMPL.get(bucket, filename)
        if not binary:
            logging.warning(f"[vision] empty binary: {bucket}/{filename}")
            return None
        doc = fitz.open(stream=binary, filetype="pdf")
        if page_num < 1 or page_num > len(doc):
            logging.warning(f"[vision] page {page_num} out of [1,{len(doc)}]")
            return None
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception as e:
        logging.exception(f"[vision] extract_page_image failed: {e}")
        return None


def glm4v_chat(question: str, image_bytes: bytes,
               api_key: str, model: str = DEFAULT_VISION_MODEL,
               timeout: int = 60) -> str:
    """调用智谱 GLM-4V 回答问题"""
    if not api_key:
        return "[视觉模型未配置 API Key]"
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(ZHIPU_API_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.exception(f"[vision] glm-4v failed: {e}")
        return f"[视觉模型调用失败: {e}]"


def _get_zhipu_key(tenant_id: str) -> Optional[str]:
    """从数据库中读取租户已配置的智谱 API Key"""
    try:
        from api.db.services.llm_service import TenantLLMService
        rows = TenantLLMService.query(tenant_id=tenant_id, llm_factory="ZHIPU-AI")
        for r in rows:
            if r.api_key:
                return r.api_key
    except Exception as e:
        logging.exception(f"[vision] read key failed: {e}")
    return None


def _list_pdf_docs(kb_id: str) -> List[str]:
    """用底层 ORM 查询知识库里的 PDF 文件名"""
    try:
        from api.db.db_models import Document
        q = (Document
             .select(Document.name)
             .where((Document.kb_id == kb_id) & (Document.status == "1")))
        return [d.name for d in q if d.name and d.name.lower().endswith(".pdf")]
    except Exception as e:
        logging.exception(f"[vision] list pdf failed: {e}")
        return []


def try_vision_answer(question: str, tenant_id: str, kb_ids: List[str]) -> Optional[str]:
    """
    尝试用 GLM-4V 回答视觉问题。
    - 非视觉问题 -> 返回 None（走正常 RAG 流程）
    - 视觉问题   -> 返回答案字符串
    """
    is_visual, page, figure = is_visual_question(question)
    if not is_visual:
        return None
    logging.info(f"[vision] detected visual question, page={page}, figure={figure}")
    if page is None:
        page = 1

    api_key = _get_zhipu_key(tenant_id)
    if not api_key:
        logging.warning("[vision] no ZHIPU-AI key")
        return None

    for kb_id in kb_ids:
        names = _list_pdf_docs(kb_id)
        logging.info(f"[vision] kb {kb_id} has {len(names)} pdf: {names[:5]}")
        for name in names:
            img = extract_page_image(kb_id, name, page)
            if img:
                logging.info(f"[vision] got page {page} of {name} ({len(img)}B), calling glm-4v")
                return glm4v_chat(question, img, api_key=api_key)
    logging.warning("[vision] no PDF page found")
    return None