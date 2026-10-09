# LightRAG 进阶优化

> 工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化

## 一、LightRAG 完整链路 vs 本实现

| 阶段 | 官方 LightRAG (HKUDS) | V12 简化版 | 效果 |
|------|----------------------|-----------|------|
| 实体抽取 | LLM (GPT) | 规则 + 预设库 | 领域足够 |
| 关系抽取 | LLM (prompt) | 正则模板 | 金融领域够用 |
| 社区发现 | Leiden + LLM 摘要 | 连通分量 | 简单有效 |
| 向量嵌入 | sentence-transformers | TF-IDF (降级) | 可升级 |
| 局部检索 | 子图检索 | k-hop BFS | 功能等价 |
| 全局检索 | 社区遍历 | 路径 + 摘要 | 简化版 |
| 增量更新 | 图差异分析 | 实体/关系 diff | ✅ 已实现 |

## 二、官方 LightRAG 完整集成

```bash
pip install lightrag-hku
```

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(
    working_dir="./lightrag_cache",
    llm_model_func=gpt_35_turbo,
    embedding_model_func=bge_embedding,
)

# 插入文档
rag.insert("武汉力源信息技术股份有限公司...")

# 局部模式 (具体实体)
result = rag.query("武汉力源的控股股东是谁?", param=QueryParam(mode="local"))

# 全局模式 (概念推理)
result = rag.query("IC 市场增长最快的行业?", param=QueryParam(mode="global"))

# 混合模式
result = rag.query("...", param=QueryParam(mode="hybrid"))
```

## 三、图数据库加速

### 3.1 Neo4j 替换
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pass"))
# 把 light_rag.py 的实体/关系存储改为 Cypher 查询
```

### 3.2 FAISS 向量索引
```python
import faiss
index = faiss.IndexFlatIP(768)  # inner product
# 替换 sklearn TF-IDF
```

## 四、异步 + 分布式

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

# 多进程并行社区发现
with ProcessPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(detect, chunk) for chunk in partition_entities()]
```

## 五、混合检索进阶

### 5.1 RAG + LightRAG 融合
```python
# 用 LightRAG 做 query 分析 + 实体链接
# 传统 RAG 做精确文本检索
# 两路合并后送入 LLM
```

### 5.2 重排序 (Reranker)
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores = reranker.predict([[query, c] for c in contexts])
```

## 六、V12 已覆盖验收

✅ 双层检索 (local + global)  
✅ 增量更新  
✅ 实体/关系抽取优化 (针对金融领域)  
✅ 社区发现  
✅ 路径推理  
✅ 15 个问题完整覆盖  
✅ RAGAS 风格评估 (precision/recall/faithfulness)  
✅ RAG vs LightRAG 对比
