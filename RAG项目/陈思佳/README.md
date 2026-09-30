# 教育 RAG 教师助手

面向九年级语文教师的知识库问答系统：从课程标准、教学案例和试题资料中检索内容，结合通义千问生成带引用的回答。

## 一、项目结构

```text
Education-RAG/
├── README.md                    # 项目说明和操作入口
├── requirements.txt             # Python 依赖
├── start_education_rag.bat      # Windows 一键启动
├── configs/                     # 应用配置
│   └── crawler.yaml
├── data/                        # 知识库原始资料、解析结果和 chunk
│   ├── raw/                     # 原始 PDF、Markdown 等资料
│   ├── mineru/                  # MinerU 原始 ZIP、JSON、图片等完整产物
│   ├── parsed/                  # MinerU 解析后的 Markdown
│   ├── chunks/                  # 向量化前的 JSONL chunk
│   └── source_manifest.json     # 资料分类和来源信息
├── deploy/                      # Docker 服务配置
│   ├── docker-compose.redis.yml
│   └── docker-compose.milvus.yml
├── docs/                        # 项目需求文档
├── frontend/                    # React + Vite 前端
├── scripts/                    # 辅助脚本
├── src/                         # Python 后端和 RAG 核心代码
└── tests/                       # Python 测试
```

`volumes/` 是 Docker 的本地持久化数据目录，不要随意删除。`frontend/node_modules/` 和 `frontend/dist/` 是前端依赖、构建产物，不属于源代码。

## 二、第一次使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. 配置密钥

推荐在项目根目录创建 `.env`，不要把密钥直接写入代码：

```env
DASHSCOPE_API_KEY=你的通义千问APIKey
MINERU_API_KEY=你的MinerUAPIKey
```

本地 BGE-M3 模型路径和 Milvus 配置位于 `configs/crawler.yaml`。

### 3. 启动 Docker 服务

确保 Docker Desktop 已启动，然后在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.redis.yml up -d
docker compose -f deploy/docker-compose.milvus.yml up -d
```

服务地址：

```text
Redis:  localhost:6379
Milvus: localhost:19530
Attu:   http://localhost:3000
```

### 4. 启动后端

```bash
uvicorn src.edu_rag_ingest.app.rag_api_backend:app --host 0.0.0.0 --port 8000
```

健康检查：

```text
http://localhost:8000/health
```

### 5. 启动前端

另开一个终端：

```bash
cd frontend
npm run dev
```

浏览器打开：

```text
http://localhost:5173
```

Windows 用户也可以直接双击：

```text
start_education_rag.bat
```

## 三、已有知识库的使用

当前项目已经准备好资料和向量数据：

```text
data/chunks/document_chunks.jsonl
Milvus collection: edu_rag_teacher_chunks
```

启动 Redis、Milvus、后端和前端后即可直接问答。

## 四、重新处理资料

### 本地资料解析和分块

把资料放入 `data/raw/`，按目录分类，例如：

```text
data/raw/
├── teaching_cases/   # 教学案例、教学设计
├── exams/            # 期中、期末、单元试题
└── entrance_exam/    # 中考真题、模拟题
```

执行：

```bash
python -m src.edu_rag_ingest.ingestion.pipeline \
  --config configs/crawler.yaml \
  --input-dir data/raw
```

输出到：

```text
data/mineru/                    # MinerU 完整产物：ZIP、JSON、图片、表格等
 data/parsed/                    # 兼容 RAG 流程的 Markdown
 data/chunks/document_chunks.jsonl
```

每次处理 PDF 或 Word 时，程序会：

1. 将 MinerU 返回的原始 ZIP 保存到 `data/mineru/<文件名>.zip`；
2. 将 ZIP 完整解压到 `data/mineru/<文件名>/`；
3. 保留其中的 Markdown、JSON、图片、表格和其他资源；
4. 将主要 Markdown 复制到 `data/parsed/<文件名>.md`，继续供清洗、分块和向量化使用。

### 向量化并写入 Milvus

确认 Milvus 已启动，并确认 `configs/crawler.yaml` 中的 BGE-M3 路径正确：

```bash
python -m src.edu_rag_ingest.retrieval.milvus_ingest \
  --config configs/crawler.yaml \
  --chunks data/chunks/document_chunks.jsonl
```

入库程序会先按 `chunk_id` 查询 collection 中已存在的记录，只对新增 chunk 进行向量化和写入；重复执行不会重复入库。配置中的 `drop_existing: false` 表示增量入库时保留已有 collection。只有明确需要完全重建 collection 时，才将其改为 `true`。

### 只重新构建 chunk

如果只是修改清洗或分块逻辑，不需要重新解析文件：

```bash
python -m src.edu_rag_ingest.ingestion.rebuild_chunks \
  --config configs/crawler.yaml \
  --parsed-dir data/parsed
```

## 五、命令行问答

```bash
python -m src.edu_rag_ingest.app.qa_cli \
  "第四学段（7～9年级）的阅读教学要求是什么？" \
  --config configs/crawler.yaml \
  --top-k 5
```

## 六、常用接口

```text
GET    /health
GET    /api/sessions
POST   /api/sessions
GET    /api/sessions/{session_id}/messages
DELETE /api/sessions/{session_id}
DELETE /api/sessions/{session_id}/messages
POST   /api/chat
```

示例：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"九年级语文第四学段的阅读教学要求是什么？","top_k":5}'
```

## 七、测试和构建

```bash
python -m pytest -q
python -m compileall -q src
```

前端构建：

```bash
cd frontend
npm run build
```

## 八、注意事项

- 只处理公开可访问且允许合理使用的教育资料。
- 不要把 `.env`、API Key 或个人资料提交到代码仓库。
- 不要删除 `data/` 和 `volumes/`，除非确认不再需要当前知识库和 Docker 数据。
- 详细需求说明见 `docs/requirements.md`。
