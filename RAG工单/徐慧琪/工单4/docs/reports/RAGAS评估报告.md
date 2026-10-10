# RAGAS 评估报告

- 运行模式：`full_04`

## 执行状态：已执行

## 一、指标结果

| 指标 | 数值 | 含义 |
| --- | --- | --- |
| faithfulness | 0.4702 | 答案是否忠于检索上下文（抗幻觉） |
| answer_relevancy | 0.8874 | 答案与问题的相关度 |
| context_precision | 0.7381 | 检索上下文的精确度 |
| context_recall | 0.7598 | 检索上下文的召回率 |

> ⚠️ 以下指标有样本未成功打分（`raise_exceptions=False` 下记 NaN，均值已跳过这些样本）：faithfulness 缺 3/16 条。

## 二、说明

- 评估语料：16 道工单题目
- 样本数：16
- ground truth：`questions.py` 中的人工校核标准答案要点
- 评判模型：`deepseek-v4-flash`（`temperature=0`，但仍存在 LLM 打分波动）
