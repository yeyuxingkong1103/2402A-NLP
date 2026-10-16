# -*- coding: utf-8 -*-
"""核心业务包：人工智能NLP-RAG 项目-金融问答系统部署（RAG 内核由 工单7 只读复用）。

模块清单（离线解析链路 T2/T4）：
    config        配置与语料发现（含工单4 的图像块接入契约常量）
    logging_conf  自实现结构化 JSON 日志（禁 loguru）
    errors        统一异常体系
    text_utils    文本/数字/分词/命中判定工具
    table_parser  表格归一化与跨页续表合并（合并单元格 None / 续表检测 / 表头继承）
    pdf_parser    PDF 逐页解析与产物落盘
    chunker       分块、关键词与元数据（工单4 新增 image 块接入与 id 口径）
    vector_store  向量索引（FAISS 主用 + numpy 兜底）
    bm25_index    自实现 BM25
    embedder      嵌入后端（本地 Ollama bge-m3:latest，1024 维）
    milvus_store  Milvus 适配器接口（可选后端；FAISS 为主用）

在线问答链路（T5，本包新增模块；契约见 设计/接口设计.md §4.2）：
    query_understanding  问句理解（语言/意图/答案形态/主体路由）
    retriever            混合检索（FAISS + BM25 → RRF → 轻量重排）＋ 证据页补全 ＋ 词表外实体判定
    reranker             八维确定性重排（词法/字段/答案形态/模态/主体/极性/数值邻近/身份簇）
    answer_composer      **证据化答案合成器**（逐项完整枚举的确定性骨架；禁止编造）
    generator            可插拔后端（ollama / openai 兼容 / extractive）＋ 原子保全护栏
    llm_client           Ollama / OpenAI 兼容流式客户端（探测 ≤0.5 s、不重试）＋ 预热
    citation             引用构造（1-based 物理页码 + 模态标注 + 内部引用串）
    answerability        可答性闸门与拒答（证据不足统一回「不清楚」）
    language             中英文双路径（语言判定 + 英文问句→中文检索式）
    conversation         多轮会话与指代消解
    prompts_loader       提示词文件加载（研发/app/prompts/*.txt）
    qa_engine            端到端问答编排（缓存 / 计时 / 引用 / 结构化日志）
"""
