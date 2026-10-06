# PDF文档的图像内容解析及检索优化（工单四）

> 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

本项目基于工单三（表格解析及检索优化版 RAG 问答系统）复制而来，在 `/home/dabaie/code/工单/工单四` 目录下进行图像内容解析与检索优化开发。

## 项目概述

- **基础版本**：工单三已实现混合检索（BM25+向量）、重排序（bge-reranker）、查询改写、多文档知识库、PDF 表格结构化解析、表格感知检索、图文（文本+表格）RRF 融合等能力
- **新增能力**：PDF 图像提取（组织结构图/图表/流程图等）、多模态图像语义解析（Chinese-CLIP 图像嵌入 + BLIP/Qwen-VL 图像描述 + OCR 辅助）、图像向量化入库（Milvus collection `rag_images`）、图像感知检索、文本+表格+图像三路融合检索
- **优化目标**：检索准确率 ≥ 90%、图像类问题准确率 ≥ 90%、响应时间 ≤ 3 秒、支持中英文问答、系统稳定容错、支持高并发
- **优化技术**：PyMuPDF 图像提取 + Chinese-CLIP 图像嵌入 + BLIP/Qwen-VL 图像描述 + PaddleOCR 辅助、混合检索（BM25+向量）、重排序（bge-reranker）、查询改写、图像感知检索

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
# 启动（端口 8003 API / 8503 UI）
bash scripts/start_v3.sh

# 停止
bash scripts/stop_v3.sh

# 初始化（MySQL + Milvus + PDF 入库，含表格/图像解析）
bash scripts/install_v3.sh

# 运行测试
bash scripts/run_tests.sh
```

## 文档索引

| 文档 | 说明 |
|------|------|
| docs/00_工单四任务说明.md | 工单四目标、16 个测试问题、验收标准 |
| docs/00_工单三任务说明.md | 工单三目标、14 个测试问题、验收标准（基础参考） |
| docs/00_优化任务说明.md | 工单二目标、问题列表、验收标准（基础参考） |
| docs/08_部署文档.md | 部署说明 |
