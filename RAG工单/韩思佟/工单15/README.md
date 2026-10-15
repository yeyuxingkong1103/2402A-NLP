# 工单15 · 优化技术图纸与文本的跨模态检索流程

工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程

## 一、工单目标

剖析 RAGFlow 检索架构，优化查询理解（视觉引用识别）+ 三路召回融合 + 跨模态重排 + Prompt，6 个图文混合问题准确率 100%。

## 二、怎么跑

```bash
python 工单15/code/test_cross_modal.py --tag before
python 工单15/code/test_cross_modal.py --tag after
```

## 三、产出物
- `工单15/技术分析文档_跨模态检索优化.md`
- `工单15/code/cross_modal_retrieval.py`
- `工单15/产出物/`

## 五、需要截图的地方

见同目录 `截图清单.md`。