# PDF文档的表格解析及检索优化（工单三）

> 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本项目基于工单二（已优化 RAG 问答系统）复制而来，在 `/home/dabaie/code/工单/工单三` 目录下进行表格解析与检索优化开发。

## 项目概述

- **基础版本**：工单二已实现混合检索（BM25+向量）、重排序（bge-reranker）、查询改写、语义分块、多语言支持等优化能力
- **新增能力**：多文档知识库（招股说明书1 + 招股说明书2）、PDF 表格结构化解析、表格感知检索
- **优化目标**：检索准确率 ≥ 90%、表格类问题准确率 ≥ 90%、响应时间 ≤ 3 秒、支持中英文问答、系统稳定容错、支持高并发
- **优化技术**：pdfplumber + camelot + PaddleOCR（兜底）+ 自研表格结构化、混合检索（BM25+向量）、重排序（bge-reranker）、查询改写、表格感知检索

## 目录结构

```
工单三/
├── app/                    # Streamlit 前端
├── src/                    # 核心源码
│   ├── optimization/       # 工单二优化模块（混合检索/重排/查询改写等）
│   └── table_parser/       # 工单三表格解析模块（pdfplumber/camelot/结构化）
├── scripts/                # 部署与初始化脚本
├── tests/                  # 测试用例
├── data/
│   ├── parsed/             # PDF 解析 JSON（工单二产出）
│   ├── parsed_v3/          # 工单三表格解析产出（结构化表格 JSON）
│   ├── tables/             # 抽取的表格 CSV/JSON
│   ├── chunks/             # 分块结果
│   └── optimized/          # 评估基线/优化后指标
├── docs/
│   ├── 00_工单三任务说明.md   # 工单三任务说明
│   └── screenshots/        # 测试截图（工单三）
└── 附件/                   # 源文档 PDF（招股说明书1、2）
```

## 快速开始

```bash
# 启动（端口 8001 API / 8501 UI）
bash scripts/start.sh

# 停止
bash scripts/stop.sh

# 初始化（MySQL + Milvus + PDF 入库，含表格解析）
bash scripts/init_all.sh

# 运行测试
bash scripts/run_tests.sh
```

## 文档索引

| 文档 | 说明 |
|------|------|
| docs/00_工单三任务说明.md | 工单三目标、14 个测试问题、验收标准 |
| docs/00_优化任务说明.md | 工单二目标、问题列表、验收标准（基础参考） |
| docs/07_优化报告.md | 工单一优化报告（基础参考） |
| docs/08_部署文档.md | 部署说明 |
