# 工单9 · Graph RAG 优化

工单编号：人工智能NLP-RAG-Graph RAG 优化

## 一、工单目标

从 Prompt(实体关系)/检索/生成三层优化，RAGAS context precision ≥0.8、context recall ≥0.9。

## 二、怎么跑

```bash
python scripts/w09_graphrag_optimize.py
```

## 三、产出物
- `outputs/w09/工单09_优化报告.md`
- `outputs/w09/优化前_评估.json`
- `outputs/w09/优化后_评估.json`

## 四、实测结果（本机跑出）
- `优化前_评估.json`：答案准确率 **0%**、上下文精度 0.10、上下文召回 0.10
- `优化后_评估.json`：答案准确率 **30%**、上下文精度 0.62、上下文召回 0.62

## 五、需要截图的地方

见同目录 `截图清单.md`。