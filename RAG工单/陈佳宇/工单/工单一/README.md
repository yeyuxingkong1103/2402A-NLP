# 招股说明书RAG智能问答系统
基于RAG检索增强生成技术，实现PDF招股书文档问答。输入自然语言问题，系统自动检索PDF文档相关内容，结合大模型生成答案，不编造文档外信息。

## 项目架构
PDF文档解析 → 文本分块 → Embedding向量编码 → Milvus向量库存储 → 用户提问向量检索 → 上下文拼接 → DeepSeek大模型生成回答 → Web前端展示

## 环境依赖
Python >=3.10
```bash
pip install pymupdf sentence-transformers pymilvus flask requests