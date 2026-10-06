# RAG 角色扮演系统

基于 RAG（检索增强生成）的多角色、多用户、多轮对话聊天机器人。
支持医生、律师、客服、NPC、金融顾问、心理咨询师等角色，从知识库（Milvus）检索相关文档，
交给大模型以该角色人设生成回答并标注来源。

## 特性

- **混合检索**：向量召回（BGE-m3）+ BM25 关键词召回，RRF 多路融合
- **重排序**：BGE-rerank 精排（无模型时退回本地关键词精排）
- **Query 改写**：结合对话历史做共指消解 / 多查询扩写
- **多轮对话**：Redis 短期记忆 + Milvus 长期记忆（用户画像/对话摘要）
- **流式输出**：SSE 流式返回
- **知识库动态更新**：增量添加 / 按来源删除 / 批量重建
- **多角色**：预设模板 + 自定义提示词创建，JSON 文件持久化
- **大模型**：在线 API（DeepSeek/Qwen/豆包/OpenAI）+ 本地 vLLM

## 项目结构

```
rag_roleplay/
├── config/config.yaml          # 配置文件
├── docs/
│   ├── requirements.md         # 需求规格说明书
│   ├── technical.md            # 技术设计文档
│   └── api.md                  # 接口文档
├── data/
│   ├── raw/                    # 原始文档（PDF/txt/md/json）
│   └── roles/                  # 角色 JSON 持久化
├── logs/                       # 日志
├── models/bge-m3/              # 本地向量化模型
├── src/
│   ├── api/routes.py           # FastAPI 路由
│   ├── rag/                    # 文档解析/分块/向量化/向量库/检索/重排序/生成
│   │   ├── bm25.py             # BM25 稀疏索引
│   │   └── query_rewriter.py   # Query 改写/扩写
│   ├── memory/                 # 短期(Redis) / 长期(Milvus)记忆
│   ├── llm/                    # 大模型客户端
│   ├── role/                   # 角色管理 + 提示词模板
│   ├── config/                 # 配置加载
│   └── utils/                  # 日志/工具
├── static/                     # 前端页面
├── scripts/
│   ├── build_index.py          # 构建知识库索引
│   ├── deploy.sh               # 部署脚本
│   └── start.sh                # 启动脚本
├── tests/                      # 单元/接口测试
├── main.py                     # 主入口
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖（建议用 conda 环境，如 E:\anaconda3\envs\myenv）
pip install -r requirements.txt

# 2. 配置 config/config.yaml（大模型 API key、Milvus/Redis 地址等）

# 3. 启动外部服务（Milvus: 19530, Redis: 6379）

# 4. 构建知识库索引
python scripts/build_index.py            # 重建：python scripts/build_index.py --drop

# 5. 启动服务
python main.py
# 浏览器打开 http://localhost:8000
```

> 本机向量化模型指向 `models/bge-m3`（本地路径，避免 HuggingFace 拉取），
> 已转换 `model.safetensors`；详见 `scripts/convert_to_safetensors.py`。

## 测试

```bash
pytest -q          # 运行全部单元/接口测试
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 聊天（多轮） |
| POST | `/api/chat/stream` | 流式聊天（SSE） |
| POST | `/api/role/create` | 创建角色 |
| GET  | `/api/role/list` | 角色列表 |
| GET  | `/api/role/{role_id}` | 角色详情 |
| POST | `/api/knowledge/add` | 添加文档 |
| POST | `/api/knowledge/build` | 批量构建知识库 |
| GET  | `/api/knowledge/stats` | 知识库统计 |
| GET  | `/api/knowledge/list` | 知识库来源列表 |
| DELETE | `/api/knowledge/{source}` | 删除某来源文档 |
| DELETE | `/api/session/{session_id}` | 清除会话 |
| GET  | `/api/health` | 健康检查 |

完整请求/响应示例见 `docs/api.md`，启动后也可访问 `/docs` 查看 Swagger。

## 文档

- 需求规格说明书：`docs/requirements.md`
- 技术设计文档：`docs/technical.md`
- 接口文档：`docs/api.md`
