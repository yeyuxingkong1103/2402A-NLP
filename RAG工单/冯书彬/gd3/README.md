# PDF 文档表格解析及检索优化

本项目对应工单 03，在工单 02 的 RAG 优化基础上，进一步增强 PDF 表格解析和表格类问题检索能力。

## 核心优化

- 使用 MinerU 开启表格识别，将 PDF 解析为 Markdown。
- 自动抽取 Markdown 表格，生成独立表格片段。
- 建立文本索引和表格索引两个 FAISS 向量库。
- 对金额、年份、收入、利润、比例等问题优先检索表格索引。
- 使用 MMR 检索，提升召回多样性。
- 批量输出答案时保留依据类型和内容，方便核查表格来源。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动

```bash
cd C:\Users\bin\Desktop\作业\gd3
D:/AN3/python -m streamlit run app.py
```

## 使用方式

1. 确认 PDF 路径，默认是 `C:\Users\bin\Desktop\作业\招股说明书2.pdf`。
2. 输入页码范围，例如 `1-333`。
3. 点击“构建 / 重建表格优化知识库”。
4. 在“单问题问答”中测试表格问题。
5. 上传 `questions.json` 后，在“批量评测”中生成 `table_optimized_answers.json`。
6. 在“表格片段”页签查看 MinerU 解析出的表格内容。

## 文件说明

- `app.py`：表格优化版主程序。
- `questions.json`：14 条问题，包含 4 条表格类问题。
- `requirements.txt`：依赖列表。
- `表格优化说明报告.md`：项目说明报告。
- `cache/`：MinerU Markdown 缓存，不需要提交。
- `.streamlit/secrets.toml`：本地 API Key，不要提交。
