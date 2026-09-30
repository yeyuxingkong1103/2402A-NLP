# 基于 PDF 文档的问答系统

> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 项目简介

基于《招股说明书1.pdf》构建的 RAG 问答系统。

- PDF 解析：548 页 + 477 表格
- 向量检索：bge-base-zh-v1.5（768 维）+ FAISS
- Web 界面：Gradio 6.x

## 快速开始

pip install -r requirements.txt

export HF_ENDPOINT=https://hf-mirror.com

python3 app.py

访问 http://127.0.0.1:7860

## 性能指标

| 指标 | 结果 | 要求 |
|------|------|------|
| 响应时间 | 0.01~0.02s | ≤3s |
| PDF 解析 | 548 页 / 477 表格 | 支持 |
| 向量索引 | 2465 块 / 768 维 | - |

## 版本

v1.0 — 2025 年 2 月
