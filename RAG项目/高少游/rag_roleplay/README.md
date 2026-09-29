# RAG 角色扮演系统

基于检索增强生成（RAG）的多角色法律/心理/社交问答系统，支持知识库检索、角色扮演、流式对话。

## 项目架构

```
rag_roleplay/
├── main.py                 # FastAPI 主入口，路由分发
├── config.py               # 全局配置（端口、模型、集合路由）
├── db_milvus.py            # Milvus 向量数据库（入库、检索、长期记忆）
├── db_mysql.py             # MySQL 业务层（角色、会话记录）
├── db_redis.py             # Redis 短期记忆（对话历史滑动窗口）
├── retriever.py            # 混合检索（向量 + BM25 + CrossEncoder 重排）
├── llm_client.py           # DeepSeek API 调用封装
├── prompt_templates.py     # 3 个角色的提示词模板 + 建议问题
├── doc_parser.py           # 文档解析（PDF/DOCX/WPS/TXT/OCR）
├── text_splitter.py        # 文本切分（5 种策略）
├── post_processor.py       # 检索结果后处理（去重、压缩）
├── kb_update.py            # 知识库增删改
├── batch_ingest_pdfs.py    # 批量入库脚本
├── logger.py               # 日志配置
├── requirements.txt        # Python 依赖
├── deploy/                 # Linux 部署脚本（install.sh + start.sh）
├── 前端版/                  # DeepSeek 风格前端
│   ├── index.html
│   ├── style.css
│   └── app.js
├── rag_roleplay_test.jmx   # JMeter 压力测试文件
└── README.md               # 本文件
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 向量数据库 | Milvus 2.x | 多集合隔离，按角色路由 |
| 关系数据库 | MySQL 8.0 | 角色、会话记录 |
| 缓存 | Redis 7.x | 对话历史滑动窗口（20条，2小时过期） |
| Embedding | Ollama + bge-m3 | 1024 维向量，本地推理 |
| 重排序 | bge-reranker-v2-m3 | CrossEncoder 二次排序 |
| LLM | DeepSeek API | 流式输出 |
| 后端 | FastAPI + Uvicorn | 异步，支持流式响应 |
| 前端 | 原生 HTML/CSS/JS | DeepSeek 风格，无框架依赖 |

## 多集合架构

```
Milvus:
├── rag_legal              # 法律顾问（民法典/刑法/治安管理处罚法等）
├── rag_psychology          # 心理专家（心理学知识）
├── rag_companion           # 虚拟朋友（聊天素材）
└── rag_long_term_memory    # 长期记忆（所有角色共用）
```

不同角色检索不同集合，互不干扰：

| 角色 | role_key | 检索集合 | 温度 |
|------|----------|----------|------|
| 法律顾问 | lawyer | rag_legal | 0.3 |
| 心理专家 | psychologist | rag_psychology | 0.7 |
| 虚拟朋友 | virtual_friend | rag_companion | 0.8 |

## 数据流

```
用户提问
  │
  ├─ 1. Redis 取对话历史（最近 20 条）
  ├─ 2. Milvus 向量检索（top_k * 3 候选）
  ├─ 3. BM25 关键词检索（top_k * 3 候选）
  ├─ 4. RRF 融合两路结果
  ├─ 5. CrossEncoder 重排序（取 top_k）
  ├─ 6. 后处理（去重、上下文压缩）
  ├─ 7. 拼接 Prompt（角色设定 + 检索结果 + 历史 + 问题）
  ├─ 8. DeepSeek API 流式生成
  ├─ 9. Redis 保存对话历史
  └─ 10. MySQL 保存会话记录
```

## 快速开始

### 1. 环境准备

```powershell
# Python 3.10+ 环境
pip install -r requirements.txt
```

### 2. 服务依赖

| 服务 | 端口 | 说明 |
|------|------|------|
| MySQL | 3306 | 存角色和会话数据 |
| Redis | 6379 | 存对话历史 |
| Milvus | 19530 | 向量数据库 |
| Ollama | 11434 | 本地 Embedding 推理 |

### 3. 配置

编辑 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key
MYSQL_PASSWORD=your_password
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. 拉取 Embedding 模型

```powershell
ollama pull bge-m3
```

### 5. 入库

```powershell
# 法律文档入库到 rag_legal 集合
python batch_ingest_pdfs.py --role lawyer

# 心理学文档入库到 rag_psychology 集合
python batch_ingest_pdfs.py --role psychologist --dir /path/to/psychology/pdfs

# 聊天素材入库到 rag_companion 集合
python batch_ingest_pdfs.py --role virtual_friend --dir /path/to/companion/pdfs
```

### 6. 启动

```powershell
python main.py
```

### 7. 访问

| 地址 | 说明 |
|------|------|
| http://localhost:8000 | 前端界面（自动跳转） |
| http://localhost:8000/app/index.html | 前端界面（直连） |
| http://localhost:8000/docs | Swagger API 文档 |
| http://localhost:8000/health | 健康检查 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| GET | /roles | 列出所有角色（含 role_key 和建议问题） |
| POST | /chat | 对话（一次性返回） |
| POST | /chat/stream | 对话（流式返回） |
| POST | /ingest/text | 文本入库 |
| POST | /ingest/files | 文件上传入库 |
| POST | /cases | 判例检索 |
| POST | /memory/save | 保存长期记忆 |
| POST | /memory/clear | 清空短期记忆 |
| GET | /kb/sources | 列出知识库来源 |
| DELETE | /kb/sources/{source} | 删除指定来源 |
| PUT | /kb/update | 更新知识库内容 |
| POST | /kb/rebuild | 重建集合 |

## 文档解析支持

| 格式 | 解析方式 | 依赖 |
|------|----------|------|
| .txt | 直接读取 | 无 |
| .md | 直接读取（按标题切分） | 无 |
| .docx | python-docx 逐段落提取 | python-docx |
| .pdf | PyMuPDF 逐页提取 | PyMuPDF |
| .pdf（扫描件） | PaddleOCR 识别 | paddleocr |
| .wps / .doc | WPS/Word COM 另存为 txt | win32com |
| .xlsx / .xls | openpyxl 逐 Sheet 转 Markdown | openpyxl（注释中） |
| .pptx | python-pptx 逐页提取 | python-pptx（注释中） |
| 图片 | VLM 视觉模型描述 | openai（注释中） |

## 文本切分策略

| 策略 | 适用场景 |
|------|----------|
| 递归切分（默认） | 通用文本，按 ["\n\n", "\n", "。", "！", "？", "；"] 递归 |
| Markdown 标题切分 | .md 文件，按 #/##/### 分块 |
| 固定长度切分 | 纯文本无结构 |
| 语义切分 | 长文档，按语义边界分块 |
| 滑动窗口切分 | 避免句子被截断，重叠 50 字 |

## 前端功能

- DeepSeek 风格界面，居中窄栏布局
- 角色切换时建议问题动态变化
- 流式输出，AI 回答逐字显示
- 引用资料展示
- 清空对话记忆
- 响应式适配手机

## JMeter 测试

```powershell
# 用 JMeter 打开测试文件
jmeter -t rag_roleplay_test.jmx
```

测试文件包含 10 个接口的冒烟测试（1 线程 1 次循环）和 3 接口的负载测试（10 线程 10 次循环，默认禁用）。

### 冒烟测试结果（1 线程 1 次循环，异常率 0%）

| 接口 | 样本数 | 平均值 | 中位数 | 90%线 | 95%线 | 99%线 | 最小值 | 最大值 | 吞吐量 | 接收KB/s |
|------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| 01-健康检查 | 1 | 91ms | 91ms | 91ms | 91ms | 91ms | 91ms | 91ms | 629/sec | 0.09 |
| 02-角色列表 | 1 | 9ms | 9ms | 9ms | 9ms | 9ms | 9ms | 9ms | 629/sec | 0.01 |
| 03-知识库来源 | 1 | 31ms | 31ms | 31ms | 31ms | 31ms | 31ms | 31ms | 625/sec | 0.01 |
| 04-文本入库 | 1 | 8,891ms | 8,891ms | 8,891ms | 8,891ms | 8,891ms | 8,891ms | 8,891ms | 6.8/min | 0.01 |
| 05-判例检索 | 1 | 12,884ms | 12,884ms | 12,884ms | 12,884ms | 12,884ms | 12,884ms | 12,884ms | 4.7/min | 0.39 |
| 06-对话(非流式) | 1 | 22,552ms | 22,552ms | 22,552ms | 22,552ms | 22,552ms | 22,552ms | 22,552ms | 2.7/min | 0.02 |
| 07-对话(流式) | 1 | 19,739ms | 19,739ms | 19,739ms | 19,739ms | 19,739ms | 19,739ms | 19,739ms | 3.0/min | 0.01 |
| 08-保存长期记忆 | 1 | 733ms | 733ms | 733ms | 733ms | 733ms | 733ms | 733ms | 82/sec | 0.01 |
| 09-清空短期记忆 | 1 | 8ms | 8ms | 8ms | 8ms | 8ms | 8ms | 8ms | 667/sec | 0.01 |
| 10-删除测试数据 | 1 | 1,660ms | 1,660ms | 1,660ms | 1,660ms | 1,660ms | 1,660ms | 1,660ms | 36/sec | 0.01 |
| **总计** | **10** | **6,659ms** | **733ms** | **19,739ms** | **19,739ms** | **19,739ms** | **8ms** | **22,552ms** | **8.5/min** | **0.05** |

### 性能分析

| 类别 | 接口 | 平均响应时间 | 说明 |
|------|------|-------------|------|
| 极快 (<100ms) | 健康检查、角色列表、清空短期记忆 | 8-91ms | 纯内存/简单查询，性能优秀 |
| 正常 (100ms-2s) | 知识库来源、保存长期记忆、删除数据 | 31-1660ms | 数据库读写，可接受 |
| 慢 (2s-10s) | 文本入库 | 8.9s | Ollama embedding + Milvus 写入 |
| 很慢 (10s-20s) | 判例检索 | 12.9s | 向量检索 + BM25 + CrossEncoder 重排 |
| 极慢 (>20s) | 对话(非流式/流式) | 19.7-22.6s | 检索全链路 + DeepSeek API 调用 |

### 瓶颈分析

1. **对话接口最慢（19-23 秒）**：完整链路 = 向量检索 + BM25 + RRF融合 + CrossEncoder重排 + DeepSeek API 生成
2. **判例检索慢（12.9 秒）**：不调 LLM 但仍需向量检索 + BM25 + 重排，Ollama 在 CPU 上跑 embedding 是主要瓶颈
3. **文本入库慢（8.9 秒）**：单条文本走 embedding + Milvus 写入，批量入库时更明显
4. **简单接口极快（<100ms）**：健康检查、角色列表、清空记忆，不涉及 AI 组件

### 优化方向

| 优先级 | 优化项 | 预期效果 |
|--------|--------|----------|
| P0 | Ollama 换 GPU 推理 | embedding 从秒级降到毫秒级 |
| P0 | DeepSeek API 加流式 + 超时控制 | 对话响应时间降 30-50% |
| P1 | BM25 索引持久化到磁盘 | 启动时间缩短，不影响请求时延 |
| P1 | 检索结果缓存（Redis） | 相同 query 秒级返回 |
| P2 | CrossEncoder 延迟加载 | 首次检索快 10-30 秒 |
| P2 | MySQL 连接池 | 高并发下稳定 |

### 注意事项

- 以上结果为单线程单次循环，样本量 n=1，统计置信度有限
- 对话接口会调 DeepSeek API 产生费用
- 负载测试组（10 线程 10 次循环）默认禁用，需要时手动启用
- 建议用 100+ 次循环 + 10-50 并发线程重新测试以获得更准确数据

## 开发环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 |
| Python | 3.10 |
| 虚拟环境 | H:\an\envs\langchain2 |
| 解释器路径 | H:\an\envs\langchain2\python.exe |
| 项目路径 | C:\Users\30274\Desktop\1\rag_roleplay |

## 已知问题

1. **PaddleOCR 未安装**：扫描件 PDF 无法入库（劳动合同法）
2. **长期记忆集合名已修复**：原硬编码 `long_term_memory` 已改为读配置
3. **BM25 索引每次启动重建**：数据量大时启动慢，后续可持久化
4. **Redis 对话历史无 token 截断**：20 条消息可能超 LLM 上下文窗口

## 后续优化方向

- [ ] 安装 PaddleOCR，处理扫描件 PDF
- [ ] Redis 对话历史加 token 计数截断
- [ ] BM25 索引持久化到磁盘
- [ ] 检索结果缓存（Redis）
- [ ] MySQL 连接池（SQLAlchemy pool）
- [ ] LLM 调用加超时和重试
- [ ] 对话超窗口时自动摘要存入长期记忆
- [ ] GPU 加速（PyTorch GPU + vLLM）
