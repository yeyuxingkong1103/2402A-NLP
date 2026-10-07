# 工单 01：基于 PDF 文档的问答系统

## 内容
- `main.py`：PDF/TXT 建索引与问答 CLI。
- `common/`：本工单独立运行所需的解析、索引、embedding、LLM 和数据模型。
- `tests/`：本工单测试。
- `人工智能NLP-RAG项目-01-*.pdf`：原始工单。

## 安装
```bash
pip install -e .
```

配置 API（可选）：复制 `.env.example` 中的变量到当前环境。未配置 API 时使用本地检索式回答。

## 运行
```bash
python main.py build --input <文档.pdf或文档.txt> --index data/index.json
python main.py ask --index data/index.json --question "营业收入是多少？"
python main.py --help
```

## 测试
```bash
python -m pytest -q
```
