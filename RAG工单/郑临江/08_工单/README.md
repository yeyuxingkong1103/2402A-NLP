# 工单 08：基于 Graph RAG 实现金融问答

**工单编号**：人工智能NLP-RAG-基于Graph RAG实现金融问答

## 一、项目简介

使用 `ccf_competition.zip` 中的金融研报构建 **知识图谱（Graph RAG）**，
根据用户问题检索知识库并返回答案所在文本块，支持问答与知识图谱可视化。

## 二、技术方案

1. **实体抽取**：公司（9 家）、人物（董事长/行长等）、金融指标；
2. **关系抽取**：共现关系 + 规则关系（高管、财务指标）；
3. **图谱存储**：节点 + 带权关系；
4. **图谱检索**：从问题抽取实体 → 邻居扩展（1~2 跳）→ 关联文本块 → 生成答案；
5. **可视化**：图谱结构输出为 JSON（可扩展 networkx/matplotlib 渲染）。

## 三、目录结构

```
08_工单/
├── app.py           # 主程序（--graph 输出图谱 / --qa 问答）
├── graph_builder.py # 实体/关系抽取 + 知识图谱
├── graph_rag.py     # 图谱检索与问答
├── llm.py           # LLM 封装
├── eval_question.md # 评估问题集
├── config.py
├── requirements.txt
└── README.md
```

## 四、运行

```bash
pip install -r requirements.txt
python app.py --graph   # 输出知识图谱结构
python app.py --qa      # 运行 eval_question 问答
```

## 五、验收对照

- 使用 Graph RAG 技术实现；
- 知识图谱可视化展示实体、关系（`--graph` 输出 + knowledge_graph.json）；
- 基于 eval_question.md 检索，输出结果；
- 代码注释含工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答。
