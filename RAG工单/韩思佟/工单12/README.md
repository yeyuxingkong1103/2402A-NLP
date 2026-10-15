# 工单12 · LightRAG 优化

工单编号：人工智能NLP-RAG-LightRAG 优化

## 一、工单目标

用 LightRAG 实现招股书1/2 的知识图谱与双层检索，比对 RAG / LightRAG 的检索结果与 RAGAS 指标。

## 二、怎么跑

```bash
python scripts/w12_lightrag_compare.py --max-chunks 400
```

## 三、产出物
- `outputs/w12/工单12_LightRAG对比报告.md`
- `outputs/w12/招股书知识图谱.html`
- `outputs/w12/RAG_评估.json`
- `outputs/w12/LightRAG_评估.json`
- `outputs/w12/检索结果对比.json`

## 四、实测结果（本机跑出）
- `LightRAG_评估.json`：答案准确率 **20%**、上下文精度 0.54、上下文召回 0.54
- `RAG_评估.json`：答案准确率 **40%**、上下文精度 0.85、上下文召回 0.85

## 五、需要截图的地方

见同目录 `截图清单.md`。