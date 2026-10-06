# PDF 文档图像内容解析及检索优化

本项目对应工单 04，在文本检索和表格检索基础上，增加 PDF 图像内容解析和图像类问题检索优化。

## 核心优化

- 使用 MinerU 解析 PDF，保留 Markdown 中的图片引用。
- 自动抽取 `![](...)` 图片引用及其相邻上下文。
- 建立文本、表格、图像三个 FAISS 索引。
- 对图、图片、示意图、流程、结构、IC、芯片等问题优先检索图像索引。
- 批量输出答案时保留依据类型，方便核查图像来源。

## 启动

```bash
cd C:\Users\bin\Desktop\作业\gd4
D:/AN3/python -m streamlit run app.py
```

## 使用流程

1. 确认 PDF 路径，默认是 `C:\Users\bin\Desktop\作业\招股说明书2.pdf`。
2. 输入页码范围，例如 `1-333`。
3. 点击“构建 / 重建图像优化知识库”。
4. 在“单问题问答”中测试图像类问题。
5. 上传 `questions.json`，在“批量评测”中导出 `image_optimized_answers.json`。
6. 在“图像片段”页签查看抽取到的图片引用和上下文。

## 文件说明

- `app.py`：图像优化版主程序。
- `questions.json`：16 条问题，包含新增图像类问题 `id: 5, 6`。
- `requirements.txt`：依赖列表。
- `图像优化说明报告.md`：项目说明报告。
- `cache/`：MinerU Markdown 缓存，不需要提交。
- `.streamlit/secrets.toml`：本地 API Key，不要提交。
