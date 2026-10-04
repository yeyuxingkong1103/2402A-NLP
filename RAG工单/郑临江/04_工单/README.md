# 工单 04：PDF 文档的图像内容解析及检索优化

**工单编号**：人工智能NLP-RAG-图像内容解析及检索优化

## 一、项目简介

在工单 01~03 基础上，增加 PDF 图像内容的语义解析：提取 PDF 内嵌图片，
使用 **多模态模型（CLIP 或多模态大模型）** 理解图像语义（组织结构图、
市场增长图等），支持图像类问题（id 5、6）的检索回复。

## 二、技术方案

1. PyMuPDF 提取页面内嵌图片并保存；
2. 图像语义解析（三选一，自动降级）：
   - 多模态大模型（OpenAI 兼容 vision API）生成图片描述；
   - CLIP 图文匹配，识别图片语义类型；
   - 上下文降级：用图片所在页文字描述；
3. 图像描述作为结构化检索块，与文本、表格统一建立 BM25 索引。

## 三、目录结构

```
04_工单/
├── app.py           # 主程序
├── config.py        # 配置（含图像类问题、CLIP 标签）
├── pdf_parser.py    # 文本/表格/图像提取
├── image_parser.py  # 图像语义解析（CLIP / 多模态LLM）
├── retriever.py     # BM25 检索
├── llm.py           # LLM 封装
├── images/          # 提取出的图片（运行时生成）
├── requirements.txt
└── README.md
```

## 四、运行

```bash
pip install -r requirements.txt

# 使用多模态大模型需配置：
#   set OPENAI_API_KEY=xxx  （或 DASHSCOPE_API_KEY）
# 使用 CLIP 需额外安装 transformers + torch

python app.py
```

## 五、验收对照

- 图像语义解析使用多模态模型（CLIP / 多模态大模型）实现；
- 图像类问题（id 5、6）能返回图中信息；
- 问答准确率 ≥ 90%，响应时间 ≤ 3 秒；
- 代码注释含工单编号：人工智能NLP-RAG-图像内容解析及检索优化。
