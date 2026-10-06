# 需求规格说明书 · 基于 RAG 的角色扮演系统

## 1. 项目背景

构建一个基于 RAG（检索增强生成）的多角色聊天机器人系统，以提示词模板 + 多源知识库支撑「特定角色」的对话能力，覆盖客服、社交 NPC、医疗、心理、法律、证券投资、理财、科学普及、英语学习等场景。参考产品：character.ai。

## 2. 角色定义

| code | 名称 | 领域 | 主要数据来源 |
| --- | --- | --- | --- |
| teacher | 英语教师 | 英语学习 | 中英翻译句对 TSV |
| npc_friend | 虚拟朋友 | 社交 NPC | 中英例句库 |
| customer_service | 学习客服 | 客服 | 系统说明、FAQ |
| doctor | 医生 | 医疗 | 国家卫健委、高血压治疗指南 |
| psychologist | 心理医生 | 心理咨询 | 心理评估量表、干预方法 |
| lawyer | 律师 | 法律 | 法律条文（311 件）、司法解释、裁判文书网判例 |
| stock_advisor | 证券投资顾问 | 证券投资 | 东方财富网等公开资料 |
| financial_planner | 金融理财师 | 个人理财 | 理财、税务、保险资料 |
| scientist | 科学家 | 科学普及 | 公开科学资料 |

## 3. 功能需求

### 3.1 用户模块
- FU-01 用户注册（用户名 + 密码，PBKDF2-SHA256 哈希）
- FU-02 用户登录，签发 HMAC-SHA256 token
- FU-03 多用户隔离，每个用户独立会话与短/长期记忆

### 3.2 角色模块
- FR-01 多角色提示词模板（见上表）
- FR-02 角色选择 / 切换
- FR-03 角色信息持久化（MySQL/SQLite `roles` 表）

### 3.3 知识库与检索
- FK-01 主数据集加载（TSV 句对）
- FK-02 BM25 检索（中文 jieba + 英文分词）
- FK-03 向量召回（BGE-m3 + Milvus）
- FK-04 混合检索：BM25 ∪ 向量，分数融合
- FK-05 多路召回：Milvus / MySQL / Redis / Neo4J / MongoDB / 互联网 / ClickHouse（部分预留）
- FK-06 重排序：BGE-rerank
- FK-07 余弦相似度 / BM25 得分阈值过滤
- FK-08 Query 改写（大模型扩写）
- FK-09 知识库动态更新：上传 txt / pdf / tsv 入库
- FK-10 PDF 处理：PyMuPDF 去水印 + PDFPlumber 表格
- FK-11 分块：固定长度 + 段落 + 句子 + 语义（预留标题/父子块）
- FK-12 知识库优化：去重、删除低质量文档（KnowledgeDoc 表）

### 3.4 对话生成
- FC-01 多轮对话（大模型无记忆，需 Redis 短期记忆 + Milvus 长期记忆）
- FC-02 流式输出
- FC-03 LangChain 编排（Prompt / Model / Retriever / Memory / Chain / Agent 工具描述）
- FC-04 兜底：未配置 LLM_API_KEY 时走「检索讲解模式」

### 3.5 接口与界面
- FI-01 HTTP JSON API（FastAPI）
- FI-02 Web 界面（Streamlit）
- FI-03 CLI（main.py）

### 3.6 评测与测试
- FE-01 RAGAS 评测：faithfulness / answer_relevancy / context_precision / context_recall
- FE-02 单元测试 pyunit / pytest
- FE-03 集成测试
- FE-04 接口测试：Postman / APIpost
- FE-05 压力测试：JMeter（QPS、负载均衡）

## 4. 业务流程

### 4.1 注册登录流程
```
用户 → 输入用户名/密码 → 后端校验唯一性 → PBKDF2 哈希存库 → 返回 token
登录 → 校验密码 → 签发 HMAC token → 后续请求带 Authorization: Bearer <token>
```

### 4.2 对话主流程
```
用户提问 → Query 改写 → 混合检索（BM25 + Milvus） → 多路召回 → 重排 →
拼接提示词（system + 检索资料 + 短期记忆 + 用户问题） → 大模型生成 →
写入短/长期记忆 + 持久化消息 → 返回答案（可选流式）
```

### 4.3 知识库动态更新
```
上传文件 → 按类型解析（TSV/文本/PDF） → 分块 → 入 BM25 → 入 Milvus → 记录 KnowledgeDoc
```

## 5. 业务规则

- BR-01 大模型生成必须优先引用检索资料，资料缺失时显式声明，禁止编造数据/法条/剂量。
- BR-02 各角色系统提示词约束其只扮演对应身份，不跨界（如医生不提供法律建议）。
- BR-03 短期记忆保留最近 `SHORT_MEMORY_TURNS` 轮，超过自动 LTrim。
- BR-04 长期记忆与知识库分离：长期记忆存储用户历史，知识库存放领域资料。
- BR-05 关键领域（医疗/法律/投资）必须附带「不替代专业服务」提示。
- BR-06 多用户会话隔离：`chat:{user_id}:{role_code}:{session_id}` 作为 Redis Key。

## 6. 非功能需求

- NF-01 性能：单机 QPS 通过 JMeter 压测；负载均衡（http-proxy / nginx）支持横向扩展。
- NF-02 可用性：Redis / Milvus / BGE 未启用时自动回退到 SQLite + 进程内字典 + BM25 单路。
- NF-03 可维护性：Python logging 滚动日志；开发 / 测试 / 生产三套环境（APP_ENV）。
- NF-04 安全：密码 PBKDF2-SHA256、token HMAC-SHA256、CORS、文件上传白名单。
- NF-05 可部署：install.sh / run.sh / shutdown.sh 一键化。

## 7. 数据集

| 类别 | 来源 | 用途 |
| --- | --- | --- |
| 中英翻译句对 | TSV（本地） | 英语教师主数据 |
| 高血压治疗指南 | 国家卫健委 | 医生角色 |
| 法律条文 / 司法解释 / 判例 | GitHub / HuggingFace / ModelScope / 中国裁判文书网 | 律师角色 |
| 股票行情 / 财报 | 东方财富网 | 证券投资顾问 |
| 公开科学资料 | 各期刊/科普站 | 科学家 |

## 8. 思维导图（文本版）

```
RAG 角色扮演系统
├── 用户
│   ├── 注册/登录/多用户隔离
│   └── 短期记忆 Redis / 长期记忆 Milvus
├── 角色（提示词模板）
│   ├── 客服 / NPC / 教师 / 医生 / 心理 / 律师 / 投资 / 理财 / 科学家
│   └── 角色信息存 MySQL
├── 知识库
│   ├── 主数据 TSV / 动态上传 txt|pdf|tsv
│   ├── 分块 + 去水印 + PDF 表格
│   └── 向量化 BGE-m3 / 重排 BGE-rerank
├── 检索
│   ├── BM25（jieba + 英文分词）
│   ├── Milvus 向量召回
│   ├── 多路召回 / 重排 / 得分过滤 / Query 改写
│   └── 知识库去重 / 删除低质量文档
├── 生成
│   ├── 大模型（本地 vLLM / 在线 DeepSeek/豆包/Claude/...）
│   ├── LangChain 编排 / 流式输出 / 兜底讲解模式
└── 接口与测试
    ├── Web Streamlit / HTTP FastAPI / CLI
    ├── RAGAS 评测 / Postman / JMeter
    └── 日志 / 三环境 / 部署脚本
```
