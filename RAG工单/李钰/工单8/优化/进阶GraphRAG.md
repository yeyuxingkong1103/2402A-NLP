# 进阶 Graph RAG 方案

> 工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答

## 一、V8 当前方案

| 能力 | 实现 | 依赖 |
|------|------|------|
| 知识图谱 | NetworkX / 内置 SimpleGraph | networkx(可选) |
| 实体抽取 | 规则词典 + 正则 | 零依赖 |
| 关系抽取 | 规则模式匹配 | 零依赖 |
| 实体链接 | jieba 分词 + 节点搜索 | jieba |
| 子图检索 | BFS k-hop 遍历 | 零依赖 |
| 图谱融合 | 0.5×Graph + 0.3×Text | V6 |
| 可视化 | HTML Canvas / D3.js | 零依赖 |
| 预设图谱 | financial_kg.json | 零依赖 |

## 二、进阶实体/关系抽取

### 2.1 LLM Zero-Shot 抽取
```python
prompt = f"""从以下文本中抽取实体和关系。

实体类型: Company(公司), Person(人物), Product(产品), Standard(标准), Financial(财务数据), Industry(行业)
关系类型: 法定代表人/持股比例/控股股东/参与制定/主营/注册资本/下属/属于

文本: {chunk}

JSON 格式输出: {{"entities": [...], "relations": [...]}}"""

entities = call_llm(prompt)
```

### 2.2 spaCy + 自定义 NER
```python
import spacy
nlp = spacy.load("zh_core_web_sm")
# 训练自定义实体
ner = nlp.get_pipe("ner")
ner.add_label("FINANCIAL_DATA")
```

### 2.3 RE (关系抽取) 模型
```python
# REBEL / T5-based RE
from transformers import pipeline
re_pipeline = pipeline("text2text-generation", model="Babelscape/rebel-large")
```

## 三、进阶图谱存储

### 3.1 Neo4j
```bash
pip install neo4j
# 启动 Neo4j: docker run -p 7474:7474 -p 7687:7687 neo4j
```
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
def query_graph(tx, cypher):
    return list(tx.run(cypher))
```

### 3.2 NebulaGraph (开源分布式)
### 3.3 本地 JSON + NetworkX (当前方案, 已够用)

## 四、进阶 Graph RAG 检索

### 4.1 Graph RAG 2.0 (Microsoft)
```python
# 社区发现 + 摘要 + 图谱驱动检索
from graphrag.query.global_search import global_search
answer = global_search(query, config)
```

### 4.2 LightRAG
```python
# 双索引: 图索引 + 向量索引
from lightrag import LightRAG
rag = LightRAG()
rag.insert(pdf_text)
answer = rag.query("武汉力源的控股股东")
```

### 4.3 Graph + Vector Hybrid
```
1. 实体链接 → 图谱节点
2. 子图遍历 → 关联文本块集合
3. 文本块向量化 → 向量检索重排
4. 融合得分 = α×Graph + β×Vector + γ×LLMRerank
```

## 五、进阶可视化

### 5.1 ECharts Graph
```javascript
echarts.init(dom).setOption({series: [{type: 'graph', layout: 'force', ...}]})
```

### 5.2 Cytoscape.js
```javascript
cytoscape({elements: data.elements})
```

### 5.3 Gephi (离线分析)

## 六、V8 已足够达标

预设知识图谱 + 规则抽取 + BFS 子图检索 + LLM 融合,
已覆盖工单验收的全部技术要求:
- 知识图谱可视化 ✅
- 图谱检索 + 文本检索融合 ✅
- 问答 + 图谱结构展示 ✅
- 多语言 (中文/英文) ✅
