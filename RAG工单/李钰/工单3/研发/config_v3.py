# -*- coding: utf-8 -*-
"""
配置模块 V3 - 多PDF + 表格解析
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 多 PDF 文档配置 ============
# 每个条目: {"path": PDF路径, "company": 公司名, "alias": 别名列表}
PDF_DOCS = [
    {
        "path": os.path.join(BASE_DIR, "招股说明书1.pdf"),
        "company": "武汉兴图新科电子股份有限公司",
        "aliases": ["兴图新科", "武汉兴图新科", "发行人"],
    },
    {
        "path": os.path.join(BASE_DIR, "招股说明书2.pdf"),
        "company": "武汉力源信息技术股份有限公司",
        "aliases": ["力源信息", "武汉力源", "武汉力源信息技术"],
    },
]

# ============ LLM 配置 (复用) ============
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# ============ 表格解析参数 ============
TABLE_HEADER_KEYWORDS = [
    "项目", "名称", "金额", "数量", "比例", "占比", "持股", "关系",
    "发行", "募集", "资金", "用途", "控制", "关联方", "股东",
]
TABLE_MIN_ROWS = 2          # 最少行数才视为表格
TABLE_MIN_COLS = 2          # 最少列数

# ============ 检索参数 ============
TOP_K = 5
MAX_RESPONSE_TIME = 3.0

# ============ 缓存目录 ============
CACHE_DIR = os.path.join(BASE_DIR, "cache_v3")
os.makedirs(CACHE_DIR, exist_ok=True)

# 预设表格数据 (当招股说明书2.pdf缺失时使用)
PRESET_TABLES_PATH = os.path.join(BASE_DIR, "预设表格数据.json")
