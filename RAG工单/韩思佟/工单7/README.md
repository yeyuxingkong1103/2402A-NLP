# 工单7 · 功能测试及评估

工单编号：人工智能NLP-RAG-功能测试及评估

## 一、工单目标

用 ccf_competition 的 9 份年报 + sample_questions 模板整理 10 个测试问题，输出检索结果、RAGAS 评估与问题分析。

## 二、怎么跑

```bash
python scripts/build_all.py ccf
python scripts/w07_eval_ccf.py
```

## 三、产出物
- `outputs/w07/工单07_测试与评估报告.md`
- `outputs/w07/10题检索结果.json`
- `outputs/w07/评估汇总.json`
- `outputs/w07/问题分析.json`

## 四、实测结果（本机跑出）
- `评估汇总.json`：答案准确率 **40%**、上下文精度 0.68、上下文召回 0.68

## 五、需要截图的地方

见同目录 `截图清单.md`。