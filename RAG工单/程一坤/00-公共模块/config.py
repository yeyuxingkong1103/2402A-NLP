# -*- coding: utf-8 -*-
"""
公共配置模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：工单01-13 共用的路径、模型、Ollama 服务配置
"""
import os  # 用于读取环境变量与拼接文件路径（跨平台用 os.path.join 而非手写分隔符）

# ── 目录配置 ────────────────────────────────────────────────
# 项目根目录：__file__ 是本文件所在路径，向上跳两级即得到"程一坤"目录
# （abspath 先转绝对路径，避免相对路径下 dirname 结果不稳定）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 程一坤 目录
# 附件目录（工单原始 PDF 位置，按实际机器路径可覆盖）
# 支持用环境变量 RAG_ATTACH_DIR 覆盖默认路径，方便换机器部署时不用改代码
ATTACH_DIR = os.environ.get(
    "RAG_ATTACH_DIR",
    r"C:\Users\92842\Desktop\RAG 工单\附件",  # r 前缀原始字符串，防止路径中的反斜杠被当转义符
)
# 各工单使用的 PDF 附件路径：用 os.path.join 拼接保证 Windows/Linux 兼容
PDF_ZGS1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")   # 工单01/02/05/06/12 使用
PDF_ZGS2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")   # 工单03/04/12 使用
CCF_PDF_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "pdf")   # 工单07/08 使用
CCF_TXT_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "txt")   # 工单07 可直接用txt

# 向量索引持久化目录（各工单共用，避免重复向量化）
# 放在 BASE_DIR 下，JSON 索引文件集中管理，二次运行可直接 load 复用
INDEX_DIR = os.path.join(BASE_DIR, "index_store")

# ── Ollama 服务配置 ─────────────────────────────────────────
# Ollama 本地服务地址：同样支持环境变量覆盖；注意必须用 127.0.0.1 而非 localhost
# （localhost 在部分环境会解析成 IPv6 ::1 导致 Ollama 连不上）
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")     # 生成模型
EMBED_MODEL = os.environ.get("EMBED_MODEL", "dengcao/bge-m3:567m") # 向量模型(1024维)
VL_MODEL = os.environ.get("VL_MODEL", "qwen2.5vl:7b")              # 多模态模型(工单04)

# ── 分块 / 检索参数 ─────────────────────────────────────────
CHUNK_SIZE = 500      # 分块目标长度（字符）：过大超出检索粒度，过小丢失上下文语义
CHUNK_OVERLAP = 80    # 相邻分块重叠长度：防止关键句被切在块边界导致检索漏召
DEFAULT_TOP_K = 5     # 默认召回条数：喂给 LLM 的上下文片段数量，过多会稀释注意力
EMBED_BATCH = 32      # 向量化批大小：每处理 32 条打印一次进度，兼顾速度与反馈
