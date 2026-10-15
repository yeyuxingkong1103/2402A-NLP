# 工单8 · 基于Graph RAG实现金融问答

工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答

## 一、工单目标

基于年报构建金融知识图谱，问题 → 实体链接 → 子图 → 原文块 → 生成；并导出知识图谱可视化。

## 二、怎么跑

```bash
python scripts/w08_graphrag.py --max-chunks 200
```

## 三、产出物
- `outputs/w08/工单08_GraphRAG报告.md`
- `outputs/w08/金融知识图谱.html`
- `outputs/w08/知识图谱统计.json`
- `outputs/w08/GraphRAG问答结果.json`

## 四、实测结果（本机跑出）
- `评估汇总.json`：答案准确率 **10%**、上下文精度 0.71、上下文召回 0.71

## 五、需要截图的地方

见同目录 `截图清单.md`。