# 工单 02：PDF 问答系统优化

## 内容
- `main.py`：混合检索、关键词覆盖率、向量评分、去重和重排。
- `common/`：本工单独立运行所需的共享基础模块。
- `tests/`：本工单测试。
- `人工智能NLP-RAG项目-02-*.pdf`：原始工单。

## 安装与测试
```bash
pip install -e .
python -m pytest -q
python main.py --help
```

`optimize_results(query, chunks, top_k=5)` 可在 Python 中直接调用。未配置 API 时仍可使用本地 embedding 和检索。
