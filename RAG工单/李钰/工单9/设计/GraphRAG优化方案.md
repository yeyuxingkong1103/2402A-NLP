# Graph RAG V8→V9 优化方案

> 工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
> 基线: V8 Graph RAG (预设图谱 + 规则抽取 + BFS 检索)

## 一、V8 问题诊断

| 层面 | V8 实现 | 问题 | 影响 |
|------|---------|------|------|
| **实体抽取** | 规则词典+正则 | 覆盖不全, 未知实体无法识别 | context recall 低 |
| **关系抽取** | 5 种固定模式 | 关系类型有限, 多跳关系缺失 | context precision 低 |
| **子图检索** | BFS k-hop | 无路径质量评估, 子图噪音大 | precision ↓ |
| **实体链接** | jieba 分词匹配 | 歧义实体无消歧, 链接不准 | precision ↓ |
| **生成** | 单 Prompt 拼接 | 图谱+文本上下文无优先级 | 答案质量不稳定 |
| **评估** | 无 RAGAS | 缺少 context precision/recall 量化 | 无法验收 |

## 二、V9 三层面优化

### 2.1 Prompt / 实体关系层面
```
V8: 规则词典 (30+ 实体模式)
V9: 
  1. 扩展词典 (50+ 模式 + 金融术语同义词)
  2. LLM Zero-Shot 抽取 (可选, 提升覆盖)
  3. 实体消歧 (同名实体 → 上下文区分)
  4. 关系推断 (实体对自动推断关系类型)
```

### 2.2 检索层面
```
V8: BFS k-hop 子图遍历
V9:
  1. 路径质量评估 (关系权重 + 路径长度 + 实体重要性)
  2. 实体消歧链接 (上下文感知)
  3. 子图剪枝 (去除弱关联节点, 提升 precision)
  4. 图谱+文本重排 (图谱匹配优先)
  5. 融合策略: α×GraphPath + β×Text + γ×LLMRerank
```

### 2.3 生成层面
```
V8: 简单拼接图谱+文本 → LLM
V9:
  1. 上下文打包 (图谱结构化 → 文本块去重 → 优先级排序)
  2. 分层 Prompt (先图谱推理 → 后文本验证)
  3. Faithfulness 约束 (答案必须包含图谱证据)
  4. 引用图谱结构 (输出中引用实体关系路径)
```

## 三、评估指标 (RAGAS 风格)

### 3.1 Context Precision / Recall
```
Context Precision = |relevant_context ∩ retrieved_context| / |retrieved_context|
Context Recall    = |relevant_context ∩ retrieved_context| / |relevant_context|
```

### 3.2 其他 RAGAS 指标
| 指标 | 含义 | 目标 |
|------|------|------|
| **Context Precision** | 检索到的上下文中有多少是相关的 | ≥ 0.8 |
| **Context Recall** | 所有相关上下文中有多少被检索到 | ≥ 0.9 |
| **Faithfulness** | 答案中有多少可从上下文推断 | ≥ 0.9 |
| **Answer Relevance** | 答案与问题的相关程度 | ≥ 0.8 |

## 四、代码组织

```
工单9/
├── 设计/    (本文件)
├── 研发/
│   ├── config_v9.py              # 优化配置
│   ├── entity_extractor_v2.py    # 实体+关系抽取 V2
│   ├── knowledge_graph_v2.py     # 增强图谱 (路径+权重+消歧)
│   ├── graph_retriever_v2.py     # 路径检索+剪枝+重排
│   ├── ragas_evaluator.py        # RAGAS 风格评估 (context P/R)
│   ├── qa_engine_v9.py           # V9 整合
│   ├── app_v9.py                 # Flask 入口 (端口 5008)
│   └── templates_v9/index_v9.html # 聊天+对比界面
├── 部署/    (启动脚本)
├── 优化/    (Neo4j / 更多 LLM 抽取)
└── 测试/    (V8 vs V9 对比)
```

## 五、预期提升

| 指标 | V8 | V9 目标 |
|------|----|---------|
| Context Precision | ~0.65 | **≥ 0.80** |
| Context Recall | ~0.75 | **≥ 0.90** |
| Faithfulness | ~0.80 | **≥ 0.90** |
| 响应时间 | < 1s | ≤ 3s |
