# 工单 09：Graph RAG 优化

**工单编号**：人工智能NLP-RAG-Graph RAG优化任务

## 一、项目简介

基于工单 08 搭建的 GraphRAG，从 **prompt 构建实体/关系、检索、生成** 三个
层面进行优化，使用 RAGAS 的上下文精度（context precision）与上下文召回
（context recall）指标对比优化前后效果。目标：精度 ≥ 0.8，召回 ≥ 0.9。

## 二、优化点

| 层面 | 基线 | 优化 |
|------|------|------|
| 实体/关系抽取 | 规则 + 粗粒度共现 | prompt 引导 LLM 精确抽取（限定实体/关系类型） |
| 检索 | 单一路径 | 图谱邻居扩展 + 文本检索联合召回 |
| 生成 | 普通拼接 | 结合图谱结构与文档片段，对齐参考答案 |

## 三、目录结构

```
09_工单/
├── app.py           # 优化前后对比
├── graph_builder.py # 基线 + 优化图谱构建（prompt 抽取）
├── graph_rag.py     # 图谱检索
├── evaluation.py    # RAGAS 上下文精度/召回
├── llm.py / config.py
├── requirements.txt
└── README.md
```

## 四、运行

```bash
pip install -r requirements.txt
python app.py
```

## 五、验收对照

- 问题分析 + 优化技术方案；
- RAGAS 指标：context precision ≥ 0.8、context recall ≥ 0.9；
- 输出优化前后指标变化；
- 代码注释含工单编号：人工智能NLP-RAG-Graph RAG优化任务。
