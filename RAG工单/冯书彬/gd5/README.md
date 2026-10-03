# Query 理解优化 RAG 系统

本项目对应工单 05，在文本、表格、图像多索引 RAG 基础上，增加 Query 理解优化能力。

## 核心优化

- 对用户问题进行 Query 分类：`text`、`table`、`image`、`mixed`。
- 使用 Qwen 对问题进行改写，生成更适合检索的 Query。
- 生成多个子查询，提高召回覆盖率。
- 根据 Query 类型路由到文本、表格、图像索引。
- 多路检索结果去重融合。
- 页面展示 Query 理解结果，批量输出保留 `query_analysis`。

## 启动

```bash
cd C:\Users\bin\Desktop\作业\gd5
D:/AN3/python -m streamlit run app.py
```

## 使用流程

1. 确认 PDF 路径，默认是 `C:\Users\bin\Desktop\作业\招股说明书2.pdf`。
2. 输入页码范围，例如 `1-333`。
3. 点击“构建 / 重建 Query 优化知识库”。
4. 在“单问题问答”中输入问题，查看答案和 Query 理解结果。
5. 上传 `questions.json`，在“批量评测”中导出 `query_optimized_answers.json`。
6. 在“Query理解”页签单独测试问题分类和改写。

## 文件说明

- `app.py`：Query 理解优化版主程序。
- `questions.json`：批量问题文件。
- `requirements.txt`：依赖列表。
- `Query理解优化说明报告.md`：项目说明报告。
- `cache/`：MinerU Markdown 缓存，不需要提交。
- `.streamlit/secrets.toml`：本地 API Key，不要提交。
