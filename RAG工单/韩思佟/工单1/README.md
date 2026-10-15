# 工单1 · 基于PDF文档的问答系统

工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 一、工单目标

基于《招股说明书1.pdf》搭建问答系统：Query理解 → 检索 → 生成 → 交互界面；10 个测试题给出 RAG 答案并与纯 LLM 答案对比，用 RAGAS 体系评估。

## 二、怎么跑

```bash
python -m src.build_kb --name share1_plain --pdf data/招股说明书1.pdf --strategy plain
python scripts/w01_pipeline_demo.py
python -m src.cli --kb share1_plain --show-context
```

## 三、产出物
- `outputs/w01/工单01_结果报告.md`
- `outputs/w01/问答结果_含纯LLM对比.json`
- `outputs/w01/评估汇总.json`
- `outputs/w01/RAG_vs_纯LLM_对比.md`
- `outputs/w01/知识库统计.json`

## 四、实测结果（本机跑出）
- `评估汇总.json`：答案准确率 **30%**、上下文精度 0.18、上下文召回 0.38

## 五、需要截图的地方

见同目录 `截图清单.md`。