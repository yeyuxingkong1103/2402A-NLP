# 基于 PDF 文档的 RAG 问答系统

这是一个精简版 RAG 项目，用 Streamlit 提供页面，使用 MinerU 解析 PDF，使用 LangChain 编排文本切分、FAISS 向量检索、通义 DashScope Embedding 和 LLM。

## 功能

- 支持输入本地 PDF 路径或上传 PDF
- 使用 MinerU 将 PDF 解析为 Markdown
- 使用 LangChain 切分文本并构建 FAISS 向量索引
- 输入单个问题并生成答案
- 上传 `questions.json` 批量生成答案
- 支持下载 `answers.json` 和 MinerU Markdown

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置 API Key

本项目默认读取 `.streamlit/secrets.toml`：

```toml
DASHSCOPE_API_KEY = "你的通义API Key"
MINERU_API_KEY = "你的MinerU API Key"
```

也可以使用环境变量配置同名 Key。

## 启动

```bash
streamlit run app.py
```

## 使用方式

1. 打开页面后确认 PDF 路径，或上传 `1.pdf`
2. 在页码范围中输入本次解析范围，例如 `1-20`、`75-94` 或 `1-333`。超过 20 页会自动分批解析并合并知识库
3. 点击“构建 / 重建知识库”
4. 在“单问题问答”里输入问题，或在“批量问题”里上传 `questions.json`
5. 查看答案和检索片段，批量模式可下载 `answers.json`

## 问题文件格式

```json
[
  {"id": 260, "question": "问题内容"},
  {"id": 95, "question": "问题内容"}
]
```

## 说明

代码集中在 `app.py`，避免过度拆分。后续如需更换解析器、OpenAI、Ollama 或其他模型，只需要替换 `parse_pdf_with_mineru`、`make_embeddings` 和 `make_llm`。
