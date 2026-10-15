# -*- coding: utf-8 -*-
# config.py 工单15 跨模态检索全局配置
import re

# ========= RAGFlow API配置 =========
RAGFLOW_BASE_URL = "http://127.0.0.1:8000/api/v1"
RAGFLOW_API_KEY = "填入你的RAGFlow api‑key"
KB_ID = "填入你的知识库ID"

# ========= 检索参数配置 =========
TOP_K = 8
RRF_ALPHA_QUERY = 60          # 原始query支路RRF权重
RRF_ALPHA_AUGMENT = 120       # 图纸增强查询支路RRF权重（图文场景调高）
ENABLE_MULTIMODAL_RERANK = True

# ========= 图纸引用正则，识别：图3、第11页图、第XX页图示 =========
FIGURE_REF_PATTERN = re.compile(
    r"第(\d+)页.*?(图(\d+)|图示)|图(\d+)",
    re.IGNORECASE
)
