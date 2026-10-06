# 基于 PDF 文档的 RAG 问答系统优化版

本项目是工单 02 的优化实现，在基础 RAG 系统上增强解析、切分、检索和批量评测能力。

## 优化点

- 使用 MinerU 解析 PDF，避免中文乱码。
- 支持 `1-333` 这种长页码范围，自动按 20 页分批解析。
- 增加 Markdown 缓存，相同 PDF 和页码范围不用重复调用 MinerU。
- 使用 LangChain 的 Markdown 标题切分和递归切分，保留章节结构。
- 使用 FAISS 的 MMR 检索，降低重复片段，提高召回多样性。
- 批量输出包含答案和检索依据，方便人工评估准确率。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动

```bash
D:/AN3/python -m streamlit run app.py
```

## 使用流程

1. 确认 PDF 路径，默认是 `C:\Users\bin\Desktop\作业\招股说明书1-无水印.pdf`。
2. 输入页码范围，例如 `1-333`。
3. 点击“构建 / 重建优化知识库”。
4. 单问题测试，或上传 `questions.json` 进行批量评测。
5. 下载 `optimized_answers.json` 作为优化版答案文件。

## 文件说明

- `app.py`：优化版主程序。
- `questions.json`：批量问题文件。
- `requirements.txt`：依赖文件。
- `优化说明报告.md`：工单 02 优化说明。
- `cache/`：MinerU Markdown 缓存，自动生成，不需要提交。
- `.streamlit/secrets.toml`：本地 API Key，不要提交。
