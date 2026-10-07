# 工单 04：PDF 图像内容解析及检索优化

## 内容
- `main.py`：PDF 图片提取、图像校验、OCR、视觉描述和图像索引。
- `common/`：本工单独立运行所需的共享基础模块。
- `tests/`：本工单测试。
- `人工智能NLP-RAG项目-04-*.pdf`：原始工单。

## 安装
```bash
pip install -e .
pip install -e ".[image]"
```

## 运行
```bash
python main.py --help
python -m 工单04_图像解析 --help
```

未配置 `OPENAI_API_KEY` 时不会请求远程视觉接口；OCR/视觉依赖缺失时不会伪造识别结果。

## 测试
```bash
python -m pytest -q
```
