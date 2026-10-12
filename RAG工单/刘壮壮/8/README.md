# 工单 08 — 基于 Graph RAG 实现金融问答

> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

使用 ccf_competition 金融研报构建知识图谱，并基于 Graph RAG 检索返回答案所在文本块，支持知识图谱可视化展示。

## 功能

1. **知识图谱构建**：实体/关系抽取（LLM 结构化抽取，规则兜底）+ networkx 有向图
2. **图谱检索**：Query 实体链接 → 邻域扩展（hop 可调）→ 召回关联文本块
3. **图谱可视化**：导出 GraphML / matplotlib 渲染 PNG
4. **问答**：图谱召回的文本块 + LLM 生成答案

## 文件

```
8/
├── main.py       # 主程序：构建图谱 + 检索问答 + 输出图谱结构
├── graph_rag.py  # 图谱构建 / 检索 / 可视化
├── questions.py  # 金融问答问题集
├── rag_core.py
└── README.md
```

## 运行

```bash
pip install -r requirements.txt
cd 8
python main.py                          # 构建图谱 + 10 问题检索 + 输出图谱结构
python main.py --query "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？"
python main.py --use-llm                # 用 LLM 抽取实体关系（需配置 .env）
python main.py --render                 # 渲染知识图谱图片 graph.png
```

## 说明

- 默认用规则抽取（公司/人物实体 + 同句共现建边），离线可运行。
- 配置 LLM 接口后，`--use-llm` 用大模型做更准确的三元组抽取。
- `graph_rag.py` 提供 `export_graphml()` 导出图谱，可用 Gephi 等工具可视化。
- 工单提及的 `eval_question.md` 未随附件提供，已用 sample_questions.pdf 金融问题替代，可在 `questions.py` 中替换。

## 验收要点

- 知识图谱可视化展示实体、关系
- 输出 eval_question 问题的检索结果
- 交互界面友好，支持中英文
