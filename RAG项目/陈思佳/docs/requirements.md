# 基于 RAG 的教师教学辅助系统需求规格说明书

## 1. 项目定位

本项目是一个面向教师的教育领域 RAG 系统，用于帮助教师基于教材、教案、课程标准、题库、教学资料等知识库内容，完成智能问答、备课辅助、教案生成、试题生成和教学资源检索。

系统核心不是普通聊天机器人，而是一个 **基于学校或教师个人教学资料库的智能教学助手**。

## 2. 项目目标

系统需要实现以下目标：

- 支持教师上传教育资料并构建知识库。
- 支持 PDF、Word、TXT、Markdown、图片类文档解析。
- 支持 OCR、表格解析、版面解析等复杂文档处理。
- 支持基于知识库的教学问答。
- 支持教案、练习题、教学活动等内容生成。
- 回答必须尽量基于知识库，并展示引用来源。
- 支持多轮对话，保留最近上下文。
- 支持 RAG 优化，包括混合检索、多路召回、重排序、Query 改写等。
- 支持 RAG 评测，使用 RAGAS 评估系统效果。
- 支持本地部署或云服务器部署。

## 3. 目标用户

### 3.1 主要用户

- 中小学教师
- 教研组教师
- 学科负责人
- 教务人员

### 3.2 第一版建议用户范围

MVP 第一版建议只做：

- 教师端
- 单个教师或少量教师使用
- 以教学资料问答和备课辅助为核心

暂不做：

- 学生端
- 家长端
- 医生、律师、金融等其他角色
- 社交聊天类角色扮演
- 股票、医疗、法律等垂直领域知识库

## 4. 核心业务场景

### 4.1 教学资料问答

教师上传教材、课标、教案、题库后，可以直接提问：

- “五年级语文《落花生》的教学重点是什么？”
- “根据课程标准，三年级英语口语教学有哪些要求？”
- “帮我总结一下七年级数学一元一次方程的知识点。”
- “这份教案有哪些可以优化的地方？”
- “根据这份资料，设计 5 道课堂练习题。”

### 4.2 备课辅助

教师输入年级、学科、章节后，系统生成：

- 教学目标
- 教学重点
- 教学难点
- 教学过程
- 课堂活动
- 课堂练习
- 作业设计
- 板书设计
- 教学反思建议

### 4.3 试题生成

教师选择学科、年级、章节、知识点、题型和难度，系统生成：

- 单选题
- 多选题
- 判断题
- 填空题
- 简答题
- 应用题
- 阅读理解题
- 答案
- 解析
- 对应知识点

### 4.4 教学资源检索

教师可以根据关键词、章节、知识点检索已有资料：

- 教材片段
- 课程标准内容
- 教案内容
- 题库内容
- 教研资料
- 教学案例

## 5. 系统整体架构

系统分为两个主要部分：

```text
离线知识库构建部分
+
在线 RAG 问答部分
```

### 5.1 离线部分

离线部分负责把教师上传的资料处理成可检索的知识库。

流程如下：

```text
文档上传
→ 文档解析
→ 文本清洗
→ 文档分块
→ 元数据提取
→ 向量化
→ 存入向量数据库
→ 建立检索索引
```

### 5.2 在线部分

在线部分负责接收教师问题并生成答案。

流程如下：

```text
教师提问
→ 问题预处理
→ 问题向量化
→ 混合检索
→ 多路召回
→ 重排序
→ 构造提示词
→ 调用大模型
→ 后处理
→ 返回答案和引用来源
```

## 6. 功能需求

### 6.1 用户管理

#### 功能说明

系统需要支持教师账号登录和基础用户管理。

#### 功能点

- 用户注册 / 创建
- 用户登录
- 用户退出
- 用户信息管理
- 用户角色管理
- 密码加密存储
- 登录状态保持

#### 用户角色

第一版建议保留两个角色：

| 角色 | 权限 |
|---|---|
| 管理员 | 管理用户、管理所有资料、查看系统状态 |
| 教师 | 上传资料、管理自己的知识库、进行问答、生成教案和试题 |

后续可扩展：

| 角色 | 权限 |
|---|---|
| 教研组长 | 管理本学科或本年级共享资料 |

### 6.2 文档上传与管理

#### 功能说明

教师可以上传教育资料，系统自动解析并入库。

#### 支持文档类型

第一版建议支持：

- PDF
- Word：`.docx`
- TXT
- Markdown：`.md`
- 图片：`.jpg`、`.png`

后续扩展支持：

- Excel
- PPT
- 扫描版 PDF
- 含复杂表格 PDF
- 含图片、公式、图文混排资料

#### 资料类型

上传时需要选择资料类型：

- 教材
- 课程标准
- 教案
- 题库
- 课件
- 教研资料
- 教学案例
- 学生常见问题
- 作业资料
- 校本课程资料

#### 元数据字段

每个文档需要保存：

- 文档 ID
- 文件名
- 文件类型
- 学科
- 年级
- 教材版本
- 册次
- 章节
- 知识点
- 资料类型
- 上传人
- 上传时间
- 文件状态
- 权限范围
- 原始文件路径

#### 文档状态

文档处理状态包括：

- 待解析
- 解析中
- 解析成功
- 解析失败
- 已入库
- 已删除

### 6.3 文档解析

#### 功能说明

系统需要对不同类型教育资料进行文本提取。

#### PDF 解析方案

根据文档复杂程度选择不同工具：

| 场景 | 推荐方案 |
|---|---|
| 普通文本 PDF | PyMuPDF / fitz |
| 表格较多 PDF | pdfplumber |
| 扫描版 PDF | PaddleOCR |
| 复杂版面 PDF | MinerU |
| 图文混排 PDF | MinerU / Paddle-OCR-VL |
| 图片型资料 | PaddleOCR / Paddle-OCR-VL |
| 特殊复杂资料 | 多模态大模型辅助解析 |

#### Word 解析方案

- 使用 `python-docx` 提取正文。
- 保留标题层级。
- 尽量保留表格内容。
- 提取段落、标题、列表信息。

#### 表格解析

对于教材表格、课程标准表格、题库表格，需要尽量保留：

- 表头
- 行列关系
- 单元格内容
- 表格上下文说明

可选方案：

- pdfplumber
- MinerU
- Paddle-OCR-VL

#### OCR 要求

对于扫描版 PDF 或图片资料，需要支持 OCR。

推荐工具：

- PaddleOCR
- Paddle-OCR-VL
- 多模态大模型

OCR 结果需要进行：

- 错别字清洗
- 多余换行处理
- 页眉页脚过滤
- 水印内容过滤
- 重复文本去除

### 6.4 文档清洗

#### 功能说明

解析出的文本需要进行清洗，减少噪声，提高检索效果。

#### 清洗规则

- 去除页眉页脚
- 去除页码
- 去除水印
- 去除重复空格
- 去除无意义换行
- 合并断裂句子
- 过滤乱码
- 删除低质量文本片段
- 保留章节标题
- 保留知识点结构
- 保留题目、答案、解析关系

#### PDF 去水印

如果 PDF 中存在水印，需要尽量处理：

- 文本水印可通过规则过滤。
- 图片水印可作为后续优化项。
- 不建议第一版强依赖复杂去水印能力。

### 6.5 文档分块

#### 功能说明

文档需要切分成适合向量检索的小片段。

#### 支持分块方式

- 固定长度分块
- 按句子分块
- 按段落分块
- 按标题分块
- 按章节分块
- 语义分块

#### 教育场景推荐策略

不同资料采用不同分块方式：

| 资料类型 | 推荐分块方式 |
|---|---|
| 教材 | 按章节 + 段落分块 |
| 课程标准 | 按标题层级分块 |
| 教案 | 按教学环节分块 |
| 题库 | 按题目为单位分块 |
| 表格 | 表格整体保存，必要时转 Markdown |
| 教学案例 | 按段落或语义分块 |

#### 分块字段

每个 chunk 需要保存：

- chunk_id
- document_id
- chunk_index
- 原文内容
- 摘要
- 学科
- 年级
- 章节
- 知识点
- 页码
- 来源文件
- 创建时间
- 更新时间

#### 分块优化

后续可加入：

- 父子块
- 摘要块
- 章节块
- 语义块
- Query 改写块
- 数据增强块

### 6.6 向量化

#### 功能说明

系统需要将文档 chunk 转换为向量，用于语义检索。

#### 推荐向量化模型

第一版推荐：

- BGE-M3

原因：

- 支持中文效果较好
- 支持多语言
- 适合通用 RAG 场景
- 可用于密集向量检索

后续可选：

- bge-large-zh
- Qwen Embedding
- text2vec
- Jina Embeddings

#### 向量化内容

建议向量化内容包含：

```text
标题 + 学科 + 年级 + 章节 + 知识点 + 正文内容
```

这样可以提升教育场景下的召回准确率。

### 6.7 向量数据库

#### 功能说明

系统需要使用向量数据库存储文档向量和对应文本。

#### 推荐方案

第一版推荐：

- Milvus

如果原型阶段想降低部署复杂度，也可以临时使用：

- Chroma
- FAISS
- pgvector

但根据当前需求，正式方案建议采用：

- Milvus + MySQL + Redis

#### Milvus Collection 设计

Milvus 中每条数据建议包含：

| 字段 | 说明 |
|---|---|
| id | 主键，自增 ID |
| vector | 文本向量 |
| content | 向量对应原文 |
| summary | 文本摘要 |
| document_id | 文档 ID |
| document_name | 文档名称 |
| subject | 学科 |
| grade | 年级 |
| chapter | 章节 |
| knowledge_point | 知识点 |
| source | 文档来源 |
| page_number | 页码 |
| created_at | 创建时间 |
| updated_at | 修改时间 |

#### 混合检索支持

Milvus 需要支持：

- 向量检索
- BM25 关键词检索
- 元数据过滤
- 相似度得分过滤

### 6.8 在线问答

#### 功能说明

教师输入自然语言问题，系统从知识库检索相关内容并调用大模型生成回答。

#### 问答流程

```text
教师输入问题
→ 判断问题所属学科、年级、章节
→ Query 清洗
→ Query 改写 / 扩写
→ 问题向量化
→ 混合检索
→ 多路召回
→ 重排序
→ 相似度过滤
→ 拼接上下文
→ 套用提示词模板
→ 调用大模型
→ 后处理
→ 返回答案、引用来源
```

#### 支持问题类型

- 知识点解释
- 教材内容查询
- 课程标准查询
- 教案优化
- 教学活动设计
- 课堂练习生成
- 试题生成
- 作业设计
- 教学反思建议
- 学生常见问题答复

### 6.9 检索模块

#### 功能说明

检索模块负责从知识库中找到与教师问题最相关的资料。

#### 检索方式

系统需要支持：

- 向量检索
- BM25 关键词检索
- 混合检索
- 多路召回
- 元数据过滤
- 相似度阈值过滤

#### 多路召回来源

第一版建议来源：

- Milvus 向量库
- MySQL 文档元数据
- Redis 最近对话上下文

后续可扩展：

- MongoDB
- Neo4j
- ClickHouse
- 互联网搜索
- 学校已有资源库

#### 检索过滤条件

教师可以按以下条件筛选：

- 学科
- 年级
- 教材版本
- 册次
- 章节
- 知识点
- 资料类型
- 上传人
- 权限范围

### 6.10 重排序

#### 功能说明

初步召回的内容需要通过重排序模型进行精排，提高最终上下文质量。

#### 推荐模型

- BGE-rerank

#### 重排序流程

```text
初步召回 Top 30
→ BGE-rerank 精排
→ 选取 Top 5 到 Top 10
→ 输入大模型上下文
```

#### 排序依据

- 与问题的相关性
- 文档权威性
- 学科匹配度
- 年级匹配度
- 章节匹配度
- 内容完整性

### 6.11 提示词模板

#### 功能说明

系统需要根据不同教学任务使用不同提示词模板。

#### 通用问答提示词模板

```text
你是一个面向教师的教学助手。
请严格根据给定的知识库内容回答问题。
如果知识库内容不足，请明确说明“当前资料不足，无法确定”。
不要编造教材内容、课程标准、考试要求或政策内容。

【教师问题】
{question}

【知识库内容】
{context}

【回答要求】
1. 回答要准确、清晰、适合教师使用。
2. 优先引用知识库内容。
3. 如果涉及教学建议，可以在依据知识库的基础上适度扩展。
4. 输出结构化内容。
5. 最后列出引用来源。

请生成回答：
```

#### 教案生成提示词模板

```text
你是一个有经验的教师备课助手。
请根据教师输入和知识库资料生成一份教学设计。

【教学信息】
学科：{subject}
年级：{grade}
章节：{chapter}
课时：{lesson_hours}

【教师补充要求】
{requirements}

【知识库资料】
{context}

【输出格式】
1. 教学目标
2. 教学重点
3. 教学难点
4. 教学准备
5. 教学过程
6. 课堂活动
7. 巩固练习
8. 课堂小结
9. 作业设计
10. 板书设计
11. 教学反思建议

要求：
- 内容适合一线教师直接参考。
- 不要脱离知识库资料随意编造。
- 如果资料不足，请标明哪些部分需要教师补充。
```

#### 试题生成提示词模板

```text
你是一个教师出题助手。
请根据知识库资料生成符合要求的练习题。

【出题要求】
学科：{subject}
年级：{grade}
章节：{chapter}
知识点：{knowledge_point}
题型：{question_type}
题目数量：{question_count}
难度：{difficulty}
是否需要答案：{need_answer}
是否需要解析：{need_explanation}

【知识库资料】
{context}

【输出要求】
每道题包含：
1. 题目
2. 选项：如适用
3. 答案
4. 解析
5. 对应知识点
6. 难度等级

要求：
- 题目必须符合指定年级和知识点。
- 不要生成超纲内容。
- 题目表达清楚，无歧义。
```

### 6.12 大模型调用

#### 支持方式

系统需要支持两种大模型接入方式。

#### 在线 API

可选：

- DeepSeek
- 通义千问
- 豆包
- Claude
- ChatGPT
- Gemini
- 硅基流动

#### 本地部署

可选：

- vLLM
- SGLang
- Xinference

可部署模型：

- Qwen 系列
- DeepSeek 系列
- 其他中文能力较好的开源模型

#### 第一版建议

如果希望快速完成项目，建议第一版优先使用：

- 在线 API：DeepSeek / 通义千问 / Claude
- Embedding：BGE-M3
- Rerank：BGE-rerank
- 向量库：Milvus

本地大模型部署可以放到第二阶段。

### 6.13 多轮对话与记忆

#### 功能说明

大模型本身没有真正长期记忆，系统需要通过数据库和缓存管理上下文。

#### 短期记忆

使用 Redis 保存最近聊天记录。

建议保存：

- 最近 N 轮用户问题
- 最近 N 轮助手回答
- 当前会话 ID
- 当前学科、年级、章节上下文

Redis 数据结构可用：

- String
- List
- Hash
- Set
- ZSet

教育问答中推荐使用：

- List 保存最近对话
- Hash 保存会话元信息
- String 保存临时状态

#### 长期记忆

使用 MySQL 保存：

- 会话记录
- 消息记录
- 用户收藏
- 教师生成的教案
- 教师生成的试题

使用 Milvus 保存：

- 教学资料知识库
- 对话摘要向量：可选
- 教师个人常用资料向量：可选

### 6.14 后处理

#### 功能说明

大模型生成结果返回给用户前，需要进行后处理。

#### 后处理内容

- 正则表达式替换
- 去除无效标记
- 格式规范化
- 引用来源格式整理
- 敏感内容过滤
- 空回答检测
- 幻觉风险提示
- Markdown 格式修复
- 教案和试题结构校验

#### 示例

如果模型输出中出现无效占位符：

```text
[来源未知]
```

可以替换为：

```text
未检索到明确来源
```

### 6.15 RAG 优化

#### 优化方向

系统后续需要支持多种 RAG 优化策略。

#### 检索优化

- 混合检索
- 多路召回
- 重排序
- Query 改写
- Query 扩写
- 相似度得分过滤
- 元数据过滤
- 章节过滤
- 学科过滤
- 年级过滤

#### 分块优化

- 固定长度分块
- 标题分块
- 语义分块
- 父子块
- 摘要块
- 按题目分块
- 按教学环节分块

#### 数据优化

- 数据清洗
- 去重
- 删除低质量文档
- 摘要增强
- 知识点标注
- 文档结构化
- 表格转 Markdown
- OCR 纠错

#### 模型优化

- Embedding 模型优化
- BGE-M3 微调：后续可选
- Rerank 模型优化
- 大模型 Prompt 优化
- 大模型参数优化

#### 生成结果优化

- 后处理
- 引用校验
- 答案结构化
- 事实一致性检查
- 超纲内容检查
- 教学适用性检查

#### 性能优化

- Redis 缓存
- 流式输出
- 异步文档解析
- 批量向量化
- 向量索引优化
- 代码逻辑优化
- 硬件优化
- 负载均衡

### 6.16 RAG 评测

#### 功能说明

系统需要使用 RAGAS 对 RAG 效果进行评估。

#### 评测工具

- RAGAS

#### 评测指标

建议评估：

- Faithfulness：答案忠实度
- Answer Relevancy：答案相关性
- Context Precision：上下文精确率
- Context Recall：上下文召回率
- Context Relevancy：上下文相关性

#### 教育场景额外指标

建议增加人工评估维度：

- 教学适用性
- 是否符合年级水平
- 是否符合学科表达
- 是否超纲
- 是否可直接用于备课
- 是否引用来源明确

#### 评测数据集

第一版可以人工构造教育测试集：

| 问题 | 标准答案 | 参考文档 | 学科 | 年级 | 章节 |
|---|---|---|---|---|---|

示例：

```text
问题：五年级语文《落花生》的教学重点是什么？
标准答案：……
参考文档：五年级语文教材 / 教案
学科：语文
年级：五年级
章节：落花生
```

## 7. 数据库设计

### 7.1 MySQL

MySQL 用于保存结构化业务数据。

#### 用户表 `users`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 用户 ID |
| username | varchar | 用户名 |
| password_hash | varchar | 密码哈希 |
| real_name | varchar | 教师姓名 |
| role | varchar | 用户角色 |
| subject | varchar | 学科 |
| grade | varchar | 年级 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

#### 文档表 `documents`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 文档 ID |
| title | varchar | 文档标题 |
| file_name | varchar | 原始文件名 |
| file_path | varchar | 文件路径 |
| file_type | varchar | 文件类型 |
| subject | varchar | 学科 |
| grade | varchar | 年级 |
| textbook_version | varchar | 教材版本 |
| volume | varchar | 册次 |
| chapter | varchar | 章节 |
| knowledge_point | varchar | 知识点 |
| document_type | varchar | 资料类型 |
| status | varchar | 解析状态 |
| uploaded_by | bigint | 上传人 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

#### 文档分块表 `document_chunks`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | chunk ID |
| document_id | bigint | 文档 ID |
| chunk_index | int | 分块序号 |
| content | text | 原文内容 |
| summary | text | 摘要 |
| page_number | int | 页码 |
| vector_id | varchar | Milvus 向量 ID |
| metadata | json | 元数据 |
| created_at | datetime | 创建时间 |

#### 会话表 `conversations`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 会话 ID |
| user_id | bigint | 用户 ID |
| title | varchar | 会话标题 |
| subject | varchar | 学科 |
| grade | varchar | 年级 |
| chapter | varchar | 章节 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

#### 消息表 `messages`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 消息 ID |
| conversation_id | bigint | 会话 ID |
| role | varchar | user / assistant |
| content | text | 消息内容 |
| citations | json | 引用来源 |
| created_at | datetime | 创建时间 |

#### 教案表 `lesson_plans`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 教案 ID |
| user_id | bigint | 用户 ID |
| subject | varchar | 学科 |
| grade | varchar | 年级 |
| chapter | varchar | 章节 |
| title | varchar | 教案标题 |
| content | longtext | 教案内容 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

#### 试题表 `questions`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | bigint | 题目 ID |
| user_id | bigint | 用户 ID |
| subject | varchar | 学科 |
| grade | varchar | 年级 |
| chapter | varchar | 章节 |
| knowledge_point | varchar | 知识点 |
| question_type | varchar | 题型 |
| difficulty | varchar | 难度 |
| question_content | text | 题目内容 |
| answer | text | 答案 |
| explanation | text | 解析 |
| created_at | datetime | 创建时间 |

## 8. 接口需求

### 8.1 用户接口

```text
POST /api/auth/login
POST /api/auth/logout
GET  /api/users/me
```

### 8.2 文档接口

```text
POST   /api/documents/upload
GET    /api/documents
GET    /api/documents/{id}
DELETE /api/documents/{id}
POST   /api/documents/{id}/parse
GET    /api/documents/{id}/chunks
```

### 8.3 问答接口

```text
POST   /api/chat
GET    /api/conversations
GET    /api/conversations/{id}
DELETE /api/conversations/{id}
```

### 8.4 教案接口

```text
POST   /api/lesson-plans/generate
GET    /api/lesson-plans
GET    /api/lesson-plans/{id}
DELETE /api/lesson-plans/{id}
```

### 8.5 试题接口

```text
POST   /api/questions/generate
GET    /api/questions
GET    /api/questions/{id}
DELETE /api/questions/{id}
```

### 8.6 RAG 评测接口

```text
POST /api/evaluation/ragas
GET  /api/evaluation/reports
GET  /api/evaluation/reports/{id}
```

## 9. 技术选型

### 9.1 后端

推荐：

- Python
- FastAPI
- LangChain / LlamaIndex
- Python logging
- Pytest / unittest

### 9.2 前端

推荐：

- React + Vite
- Ant Design
- Markdown 渲染组件
- 流式输出展示组件

### 9.3 数据库与中间件

推荐：

- MySQL：业务数据
- Redis：短期记忆、缓存、最近对话
- Milvus：向量数据库
- MongoDB：可选，用于非结构化记录
- Neo4j：可选，后续做知识图谱时使用

### 9.4 RAG 组件

推荐：

- BGE-M3：向量化模型
- BGE-rerank：重排序模型
- Milvus：向量数据库
- BM25：关键词检索
- RAGAS：RAG 评测
- LangChain / LlamaIndex：RAG 编排

### 9.5 文档解析组件

推荐：

- PyMuPDF / fitz
- pdfplumber
- PaddleOCR
- Paddle-OCR-VL
- MinerU
- python-docx

### 9.6 大模型

支持两类：

在线 API：

- DeepSeek
- 通义千问
- 豆包
- Claude
- ChatGPT
- Gemini
- 硅基流动

本地部署：

- vLLM
- SGLang
- Xinference
- Qwen 系列模型
- DeepSeek 系列模型

## 10. 页面设计

### 10.1 登录页

功能：

- 用户名登录
- 密码登录
- 登录失败提示
- 记住登录状态

### 10.2 工作台

展示：

- 最近问答
- 最近上传资料
- 知识库文档数量
- 今日问答次数
- 常用功能入口
- 系统状态

### 10.3 知识库管理页

功能：

- 上传文档
- 文档列表
- 文档搜索
- 文档筛选
- 查看解析状态
- 查看 chunk
- 删除文档
- 重新解析
- 编辑元数据

### 10.4 智能问答页

功能：

- 对话输入框
- 流式回答
- 引用来源展示
- 多轮对话
- 学科筛选
- 年级筛选
- 章节筛选
- 历史会话
- 复制回答

### 10.5 教案生成页

功能：

- 输入学科、年级、章节
- 输入课时数量
- 输入教学要求
- 生成教案
- 编辑教案
- 保存教案
- 导出 Markdown / Word

### 10.6 试题生成页

功能：

- 选择学科、年级、章节
- 选择知识点
- 选择题型
- 选择难度
- 选择题目数量
- 生成题目
- 显示答案和解析
- 保存题目

### 10.7 RAG 评测页

功能：

- 上传测试集
- 执行 RAGAS 评测
- 查看评测指标
- 查看问题级评测结果
- 导出评测报告

### 10.8 系统设置页

功能：

- 大模型配置
- Embedding 模型配置
- Rerank 模型配置
- Milvus 配置
- MySQL 配置
- Redis 配置
- 文档解析配置

## 11. 测试需求

### 11.1 单元测试

使用：

- `unittest`
- `pytest`

测试内容：

- 文档解析
- 文本清洗
- 分块逻辑
- 向量化调用
- 检索逻辑
- 重排序逻辑
- Prompt 构造
- 后处理函数

### 11.2 接口测试

使用：

- Postman
- Apipost
- Python requests

测试内容：

- 登录接口
- 上传接口
- 问答接口
- 教案生成接口
- 试题生成接口
- RAG 评测接口

### 11.3 集成测试

测试完整流程：

```text
上传教育资料
→ 文档解析
→ 分块
→ 向量化
→ 入库
→ 提问
→ 检索
→ 重排序
→ 生成回答
→ 返回引用
```

### 11.4 压力测试

使用：

- JMeter

测试指标：

- QPS
- 平均响应时间
- 最大响应时间
- 并发用户数
- 错误率
- CPU 使用率
- 内存使用率

## 12. 部署需求

### 12.1 部署环境

支持：

- Ubuntu
- WSL + Ubuntu
- 腾讯云
- 阿里云
- 算力云

开发环境：

```text
Windows 11 + WSL + Ubuntu
```

生产环境建议：

```text
Ubuntu Server
```

### 12.2 部署组件

需要部署：

- Python 环境
- FastAPI 后端
- React 前端
- MySQL
- Redis
- Milvus
- Embedding 模型
- Rerank 模型
- 大模型 API 或本地大模型服务

### 12.3 部署脚本

需要提供：

```text
install.sh
run.sh
shutdown.sh
```

### 12.4 部署文档

需要提供：

```text
README.md
```

内容包括：

- 项目介绍
- 环境要求
- 安装步骤
- 配置说明
- 启动方式
- 停止方式
- 接口测试方式
- 常见问题

## 13. MVP 第一版范围

为了保证项目能尽快落地，第一版建议只做核心链路。

### 13.1 第一版必做

- 教师登录
- 文档上传
- PDF / Word / TXT 解析
- OCR 基础支持
- 文本清洗
- 文档分块
- BGE-M3 向量化
- Milvus 入库
- 教师问答
- 混合检索
- BGE-rerank 重排序
- 提示词模板
- 大模型调用
- 引用来源展示
- Redis 保存最近对话
- MySQL 保存用户、文档、会话
- Python logging 日志
- Postman / Apipost 接口测试
- README 部署说明

### 13.2 第一版可选

- 教案生成
- 试题生成
- RAGAS 评测
- 流式输出
- MinerU 复杂 PDF 解析
- Paddle-OCR-VL
- JMeter 压力测试

### 13.3 第一版暂不做

- 学生端
- 家长端
- 医疗、法律、金融等角色
- Character.ai 类角色扮演
- 股票数据
- 判例知识库
- 医疗指南知识库
- 多角色聊天机器人
- Neo4j 知识图谱
- 复杂微调
- LlamaFactory 微调
- 大规模本地模型训练

## 14. 推荐开发阶段

### 第一阶段：RAG 主流程

目标：跑通教师知识库问答。

内容：

- 后端框架搭建
- MySQL 表设计
- Milvus 接入
- Redis 接入
- 文档上传
- 文档解析
- 文档分块
- 向量化
- 检索
- 重排序
- 大模型生成
- 返回引用来源

### 第二阶段：教师功能增强

目标：让系统更适合教学场景。

内容：

- 教案生成
- 试题生成
- 问答历史
- 文档分类管理
- 学科、年级、章节过滤
- 流式输出

### 第三阶段：RAG 优化

目标：提高回答质量。

内容：

- Query 改写
- 多路召回
- 父子块
- 摘要增强
- 文档去重
- 低质量文本过滤
- Prompt 优化
- 后处理增强

### 第四阶段：评测与部署

目标：完成可交付项目。

内容：

- RAGAS 评测
- 单元测试
- 接口测试
- 集成测试
- JMeter 压力测试
- Ubuntu 部署
- 部署脚本
- README 文档

## 15. 最终交付物

项目最终应包含：

- 需求规格说明书
- 技术架构设计文档
- 功能设计文档
- 数据库设计文档
- 接口文档
- RAG 流程设计文档
- 前端代码
- 后端代码
- 文档解析模块
- 向量入库模块
- 检索与重排序模块
- Prompt 模板
- RAGAS 评测代码
- 单元测试代码
- 接口测试结果
- 压力测试结果
- 部署脚本
- README 部署文档

## 16. 推荐最终技术方案

如果确认按这个方向做，建议最终技术栈定为：

```text
前端：React + Vite + Ant Design
后端：FastAPI
数据库：MySQL
缓存：Redis
向量库：Milvus
Embedding：BGE-M3
Rerank：BGE-rerank
文档解析：PyMuPDF + pdfplumber + PaddleOCR + MinerU
RAG 编排：LangChain 或 LlamaIndex
RAG 评测：RAGAS
接口测试：Postman / Apipost
压力测试：JMeter
部署环境：Ubuntu / WSL Ubuntu
```

## 17. 待确认事项

在正式开始开发前，需要确认以下内容：

1. 第一版是否按 `React + FastAPI + MySQL + Redis + Milvus` 实现。
2. 大模型第一版使用在线 API，还是本地部署模型。
3. 是否第一版就必须支持 OCR。
4. 是否第一版必须实现教案生成和试题生成。
5. 是否第一版必须实现 RAGAS 评测页面。
6. 是否优先实现后端接口，再实现前端页面。
7. 是否部署目标为 `Windows 11 + WSL + Ubuntu`。
