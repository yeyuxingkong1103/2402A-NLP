# -*- coding: utf-8 -*-
"""
配置模块 V4 - 多模态图像解析
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 多模态配置 ============
# CLIP 模型 (可选, 用于图像-文本跨模态检索)
# 首次使用会自动下载 (~300MB)
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "clip-ViT-B-32")
CLIP_AVAILABLE = False  # 运行时检测

# 多模态 LLM (可选, GPT-4V / Claude Vision 等)
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-vision-preview")

# 通用 LLM (复用)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# ============ 图片提取参数 ============
# 是否截取页面作为图像 (PDF 无单独图片时的降级)
RENDER_PDF_TO_IMAGE = True

# ============ PDF 配置 (复用 V3) ============
PDF_DOCS = [
    {
        "path": os.path.join(BASE_DIR, "招股说明书1.pdf"),
        "company": "武汉兴图新科电子股份有限公司",
        "aliases": ["兴图新科", "武汉兴图新科"],
    },
    {
        "path": os.path.join(BASE_DIR, "招股说明书2.pdf"),
        "company": "武汉力源信息技术股份有限公司",
        "aliases": ["力源信息", "武汉力源"],
    },
]

# ============ 缓存目录 ============
CACHE_DIR = os.path.join(BASE_DIR, "cache_v4")
os.makedirs(CACHE_DIR, exist_ok=True)

# 预设图像描述 (招股说明书2.pdf 缺失时使用)
PRESET_IMAGES_PATH = os.path.join(BASE_DIR, "预设图像描述.json")

# 图像查询关键词 (命中则优先走图像检索)
IMAGE_QUERY_KEYWORDS = [
    "图", "图表", "图像", "结构", "增长", "饼图", "柱状图", "折线图",
    "分布", "比例图", "架构图", "示意图", "组织架构", "流程图",
    "曲线图", "趋势图", "构成图", "应用结构", "组成", "构成",
    "销售部", "部门构成", "销售处",
]
