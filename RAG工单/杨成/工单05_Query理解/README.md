# 工单 05：Query 理解优化

## 内容
- `main.py`：Query 分类、实体抽取、改写、扩展、多轮上下文和优化检索委托。
- `common/`：本工单独立运行所需的共享基础模块。
- `tests/`：本工单测试。
- `人工智能NLP-RAG项目-05-*.pdf`：原始工单。

## 安装与运行
```bash
pip install -e .
python main.py --help
python main.py --query "2024年平安银行营业收入是多少" --index data/index.json
```

## 测试
```bash
python -m pytest -q
```
