# -*- coding: utf-8 -*-
# 【全局配置 · config.py】集中管理工单四图像解析与图文检索系统的可调参数
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""全局配置：PDF解析、图像抽取、OCR、多模态降级、检索阈值全部集中配置。"""
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """系统配置数据类（纯CPU离线环境可直接运行）。"""

    # ---------- 语料文档 ----------
    # 键=文档标签，值=PDF文件名；招股书1=兴图新科，招股书2=力源信息
    pdf_docs: Dict[str, str] = field(default_factory=lambda: {
        "招股说明书1": "招股说明书1.pdf",
        "招股说明书2": "招股说明书2.pdf",
    })
    # 各文档主体公司全称（用于给检索块补主语上下文，避免跨公司串答）
    doc_companies: Dict[str, str] = field(default_factory=lambda: {
        "招股说明书1": "武汉兴图新科电子股份有限公司",
        "招股说明书2": "武汉力源信息技术股份有限公司",
    })

    # ---------- PDF 文本/表格解析 ----------
    header_footer_patterns: List[str] = field(default_factory=lambda: [
        r"招股意向书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股意向书\s*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{1,2}-\d{1,2}-\d{1,3}\s*$",
    ])
    table_fallback_chars: int = 60   # 单页文本少于该字数时回退pdfplumber表格
    noise_words: tuple = ("北京八维信息集团", "八维教育", "rengongzhinengliumin",
                          "人工智能刘敏")
    heading_pattern: str = (r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、"
                            r"|（[一二三四五六七八九十]+）|\(+[一二三四五六七八九十]+\)+"
                            r"|\d+[\.、])")

    # ---------- 分块 ----------
    chunk_size: int = 450
    chunk_overlap: int = 80

    # ---------- 图像抽取 ----------
    img_max_reuse_pages: int = 3      # xref复用页数超过该值视为页眉水印并丢弃
    img_min_width: int = 90           # 有效图片最小像素宽
    img_min_height: int = 90          # 有效图片最小像素高
    img_min_area: int = 12000         # 有效图片最小像素面积
    img_min_ratio: float = 0.4        # 长宽比下限（过滤细长扫描条）
    img_max_ratio: float = 4.5        # 长宽比上限
    thumb_max_side: int = 360         # 缩略图长边像素
    figure_min_drawings: int = 10     # 矢量图页最少绘图指令数
    figure_title_pattern: str = (r"组织结构图|组织架构图|股权结构图|关系图|增长图|"
                                 r"架构图|流程图|分布图|示意图|产业链图|构成图")
    figure_ref_pattern: str = r"如下图|如下图示|下图所示|见下图|图示如下"
    caption_band_pt: float = 75.0     # 图题与图形bbox的最大纵向间距
    nearby_band_pt: float = 130.0     # 邻近正文纵向关联带宽
    vector_render_zoom: float = 2.5   # 矢量图区域渲染缩放倍数
    image_store_dirname: str = "image_store"
    ocr_cache_name: str = "ocr_cache.json"

    # ---------- OCR / 多模态 ----------
    enable_ocr: bool = True           # 建库期是否对图像做离线OCR
    ocr_engine: str = "paddleocr"     # 本机已缓存PP-OCRv6模型；不可用时自动降级
    # 本机~/.cache/huggingface无CLIP缓存且禁止联网，图像编码默认关闭（伪多模态降级）
    enable_image_encoder: bool = False
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"

    # ---------- 检索 ----------
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 3
    # bge-reranker-large在纯CPU上对30候选推理约8~10秒，超出3秒SLA；
    # 默认用等价信号的离线精排（RRF+IDF覆盖度+长短语+类型加权），毫秒级
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-large"
    rerank_truncate: int = 160
    image_query_pattern: str = (r"图|图表|结构图|柱状图|饼图|流程图|示意图|增长率|"
                                r"组织|部门|销售处|构成")
    image_chunk_boost: float = 0.35   # 图像类问题对图像证据块的精排加分
    response_timeout_s: float = 3.0
    # bge=优先本地BGE（缺失时自动降级TF-IDF）；tfidf=强制TF-IDF（优化基线实验用）
    fallback_embedding: str = "bge"

    # ---------- 路径 ----------
    base_dir: str = ""
    root_dir: str = ""
    index_dir: str = ""
    image_dir: str = ""

    def __post_init__(self) -> None:
        """补齐研发目录、工单根目录与索引/图像库路径。"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = os.path.dirname(self.base_dir)
        self.index_dir = os.path.join(self.base_dir, "index_store")
        self.image_dir = os.path.join(self.base_dir, self.image_store_dirname)


# 全局单例配置
CONFIG = Config()
