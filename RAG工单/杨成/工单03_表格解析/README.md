# 工单 03：PDF 表格解析及检索优化

## 内容
- `main.py`：Markdown、TSV、空格分隔文本表格及 PDF 表格解析。
- `common/`：本工单独立运行所需的共享基础模块。
- `tests/`：本工单测试。
- `人工智能NLP-RAG项目-03-*.pdf`：原始工单。

## 安装
```bash
pip install -e .
pip install -e ".[table]"
```

## 运行
```bash
python main.py --help
python main.py --input table.md --source report.pdf --page 8
```

## 测试
```bash
python -m pytest -q
```
