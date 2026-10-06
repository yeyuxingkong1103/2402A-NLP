# Query 理解优化（多轮对话）—— 工单五

> 工单编号：人工智能NLP-RAG-Query 理解优化任务

本项目基于工单四（图文表融合 RAG）复制而来，在 `/home/dabaie/code/工单/工单五` 目录下进行多轮对话能力开发。

## 项目概述

- **基础版本**：工单四已实现文本+表格+图像三路融合 RAG，单轮问答准确率 100%
- **新增能力**：多轮对话（会话管理 + 指代消解 + 问句改写），支持"他/这个公司"代词消解、"那X呢"实体切换与问法复用
- **优化目标**：问答准确率 ≥ 90%、指代消解正确率 ≥ 90%、响应时间 ≤ 3 秒、支持中英文问答、用户反馈
- **优化技术**：基于规则的确定性指代消解（不引入额外 LLM 调用，保障低延迟）+ RAGEngineV4 复用

## 目录结构

```
工单四/
├── app/                    # Streamlit 前端
├── src/                    # 核心源码
│   ├── optimization/       # 工单二优化模块（混合检索/重排/查询改写等）
│   ├── table_parser/       # 工单三表格解析模块（pdfplumber/camelot/结构化）
│   └── image_parser/       # 工单四图像解析模块（PyMuPDF提取/CLIP嵌入/图像描述/OCR）
├── scripts/                # 部署与初始化脚本
├── tests/                  # 测试用例
├── data/
│   ├── parsed/             # PDF 解析 JSON（工单二产出）
│   ├── parsed_v3/          # 工单三表格解析产出（结构化表格 JSON）
│   ├── tables/             # 抽取的表格 CSV/JSON
│   ├── chunks/             # 分块结果
│   ├── optimized/          # 评估基线/优化后指标
│   ├── images/             # 工单四提取的 PDF 图像
│   └── image_descriptions/ # 工单四图像语义描述/结构化信息 JSON
├── docs/
│   ├── 00_工单四任务说明.md   # 工单四任务说明（16 个测试问题、验收标准）
│   └── screenshots/        # 测试截图（工单四）
└── 附件/                   # 源文档 PDF（招股说明书1、2）
```

## 快速开始

```bash
# 启动（端口 8005 API / 8505 UI）
bash scripts/start_v5.sh

# 停止
bash scripts/stop_v5.sh

# 运行工单五测试
python -m pytest tests/test_conversation_engine.py tests/test_api_v5.py -v

# 5 轮多轮对话评估
python scripts/evaluate_v5.py
```

## 文档索引

| 文档 | 说明 |
|------|------|
| docs/00_工单五任务说明.md | 工单五目标、5 轮验收对话、验收标准 |
| docs/05_测试报告.md | 工单五测试报告（评估结果 + 消解策略验证） |
| docs/06_部署文档.md | 部署说明（API/UI 端点、启动脚本） |
