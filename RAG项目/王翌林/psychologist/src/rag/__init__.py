"""RAG 包：解析、分块、向量化、检索、重排、提示词。

本包是"心理医生 RAG 陪伴系统"的引擎层，全部由「纯函数 + 无状态模块」组成，
即：同样的输入永远得到同样的输出，不持有会话状态，也不依赖 Web 框架。
各子模块职责一览：
    parser.py      —— 文档解析（PDF/TXT/MD/DOCX → 纯文本），四级降级链
    chunker.py     —— 文本分块（6 种策略：固定/句子/段落/标题/语义/父子块）
    embedder.py    —— BGE-M3 向量化（双检锁单例懒加载）
    retriever.py   —— 检索漏斗（召回 → 重排 → 阈值过滤）
    reranker.py    —— 交叉编码器精排（sigmoid 归一化）
    prompt.py      —— 提示词模板（三层记忆注入 + 三角色 system prompt）
    ocr_loader.py  —— MinerU / PaddleOCR 外部进程调用（WSL venv）
    _paddle_runner.py —— PaddleOCR 独立运行脚本（流式增量写盘）
    pipeline.py    —— 组合根（只做编排转发，不含业务逻辑）

分层约定：rag 层只依赖 db / core，不依赖 services / api，避免循环依赖。
"""