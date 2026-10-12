# 工单 09 — Graph RAG 优化

> 工单编号：人工智能NLP-RAG-Graph RAG 优化任务

基于工单 08 的 GraphRAG，从「实体关系抽取 / 检索 / 生成」三个层面优化，并用 RAGAS 风格指标（上下文精度、上下文召回）对比优化前后。目标：精度 ≥ 0.8，召回 ≥ 0.9。

## 优化方案

| 层面 | 优化前（工单 08） | 优化后 |
| --- | --- | --- |
| 抽取 | 通用规则抽取，噪声三元组多 | 限定实体类型与关系字典，抑制噪声 |
| 检索 | 1-hop 邻域扩展 | 2-hop 扩展 + 实体度数加权（直接命中实体加权 1.5，高度数实体降权） |
| 生成 | 直接拼接上下文 | 保留出处、按关系组织上下文 |

## 运行

```bash
pip install -r requirements.txt
cd 9
python main.py                 # 对比优化前后 context precision / recall
python main.py --query "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？"
```

## 文件

```
9/
├── main.py        # 优化对比 + 评估
├── graph_rag.py   # 图谱构建/检索（含 OptimizedGraphRAG）
├── questions.py   # 金融问答问题集
├── rag_core.py
└── README.md
```

## 验收要点

- 问题分析及优化技术方案
- 比对 Graph RAG 检索生成结果与 sample_questions.pdf 参考答案
- context precision ≥ 0.8，context recall ≥ 0.9
- 响应时间 ≤ 3 秒
