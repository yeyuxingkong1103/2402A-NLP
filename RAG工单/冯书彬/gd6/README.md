# 混合检索 RAG 系统

本项目对应工单 06，在 Query 理解优化基础上，增加 FAISS 向量检索与 TF-IDF 关键词检索的混合召回和加权融合。

## 核心优化

- 保留 Query 分类、改写和子查询扩展。
- 保留文本、表格、图像多索引向量检索。
- 新增 TF-IDF 关键词检索。
- 使用向量分数和 TF-IDF 分数加权融合。
- 结果保留 `vector`、`tfidf` 来源和混合分数。
- 页面支持调整向量检索权重和 TF-IDF 权重。

## 启动

```bash
cd C:\Users\bin\Desktop\作业\gd6
D:/AN3/python -m streamlit run app.py
```

## 使用流程

1. 确认 PDF 路径，默认是 `C:\Users\bin\Desktop\作业\招股说明书2.pdf`。
2. 输入页码范围，例如 `1-333`。
3. 点击“构建 / 重建混合检索知识库”。
4. 在“单问题问答”中测试问题。
5. 上传 `questions.json`，在“批量评测”中导出 `hybrid_answers.json`。
6. 在“混合检索调试”页签查看召回来源和分数。

## 文件说明

- `app.py`：混合检索主程序。
- `questions.json`：批量问题文件。
- `requirements.txt`：依赖列表。
- `混合检索说明报告.md`：项目说明报告。
- `cache/`：MinerU Markdown 缓存，不需要提交。
- `.streamlit/secrets.toml`：本地 API Key，不要提交。
