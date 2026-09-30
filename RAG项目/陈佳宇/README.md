# 基于RAG的角色扮演系统

## 项目简介
基于检索增强生成（RAG）技术的角色扮演对话系统，支持多用户、多角色，具备混合检索、重排序、多轮记忆、知识库动态更新、流式输出、前端页面等能力。

## 技术架构
用户 → FastAPI接口 → Query改写 → 混合检索（稠密向量+BM25）→ BGE-Rerank重排序 → 提示词模板 → DeepSeek大模型 → 后处理 → 返回

## 技术组件
| 组件 | 选型 | 用途 |
|------|------|------|
| 大模型 | DeepSeek API | 对话生成 |
| 向量数据库 | Milvus | 稠密向量存储与检索 |
| 嵌入模型 | bge-base-zh-v1.5（本地） | 文本向量化 |
| 重排序模型 | bge-reranker-base（本地） | 检索结果精排 |
| 缓存/记忆 | Redis | 短期对话记忆 |
| 关系数据库 | MySQL | 用户、角色信息 |
| 文档解析 | PyMuPDF + pdfplumber | PDF文本/表格提取 |
| Web框架 | FastAPI | HTTP接口 |
| 鉴权 | JWT | API接口鉴权 |
| 前端 | 原生HTML+JS | 网页聊天界面 |

## 功能特性
- 多用户登录/注册（MySQL存储，密码加密）
- 多角色切换（Alice/医生/律师，可扩展，角色CRUD）
- 混合检索（稠密向量 + BM25关键词）
- BGE-Rerank重排序
- Redis短期对话记忆（最近5轮）
- 知识库动态更新（添加/导入/列表/删除）
- PDF/TXT文档导入，多种分块策略
- Query改写（提升召回率）
- 回答后处理（正则清理）
- 流式输出（SSE打字机效果）
- JWT API鉴权
- 前端HTML聊天页面
- Python logging日志
- FastAPI HTTP接口 + Swagger文档
- RAG评测（5项指标）

## 快速部署
bash install.sh    # 一键安装
bash run.sh        # 启动服务
bash shutdown.sh   # 停止服务

## 前端页面
启动后浏览器访问：http://127.0.0.1:8000
- 登录 → 选择角色 → 开始对话
- 支持流式打字机效果

## 接口文档
启动后访问：http://127.0.0.1:8000/docs

### 主要接口
| 接口 | 方法 | 鉴权 | 说明 |
|------|------|------|------|
| /api/health | GET | 否 | 健康检查 |
| /api/register | POST | 否 | 用户注册 |
| /api/login | POST | 否 | 用户登录，返回Token |
| /api/characters | GET | 否 | 角色列表 |
| /api/characters/{id} | GET | 是 | 角色详情 |
| /api/characters | POST | 是 | 新增角色 |
| /api/characters/{id} | PUT | 是 | 更新角色 |
| /api/characters/{id} | DELETE | 是 | 删除角色 |
| /api/chat | POST | 是 | 对话（RAG检索+生成） |
| /api/chat/stream | POST | 是 | 流式对话（SSE） |
| /api/knowledge/add | POST | 是 | 添加知识库文本 |
| /api/knowledge/list | GET | 是 | 知识库文档列表 |
| /api/knowledge/delete | POST | 是 | 删除知识库文档 |
| /api/knowledge/import | POST | 是 | 导入PDF/TXT文件 |
| /api/history/clear | POST | 是 | 清空对话历史 |

### 对话接口示例
请求：
POST /api/chat
Header: Authorization: Bearer {token}
{
"character_name": "Alice",
"query": "Alice 今年多大？"
}
响应：
{
"code": 0,
"msg": "success",
"data": {
"answer": "20 岁。",
"character": "Alice",
"retrieved_docs": ["..."],
"elapsed": 2.85
}
}


## 项目结构
rag_character/
├── main.py                  # 命令行交互版
├── app.py                   # FastAPI 接口版
├── rag_core.py              # RAG 核心逻辑（共用）
├── vector_store_milvus.py   # Milvus 向量库（混合检索 + 重排）
├── db.py                    # MySQL 数据库操作
├── auth.py                  # JWT 鉴权
├── config.py                # 统一配置读取
├── pdf_parser.py            # PDF 解析 + 分块
├── logger.py                # 日志模块
├── ragas_eval.py            # RAG 评测脚本
├── static/
│   └── index.html           # 前端页面
├── tests/                   # 测试文件
├── scripts/                 # 临时 / 初始化脚本
├── logs/                    # 日志目录
├── install.sh               # 环境安装脚本
├── run.sh                   # 启动脚本
├── shutdown.sh              # 停止脚本
├── requirements.txt         # Python 依赖
├── .env                     # 环境变量配置
├── README.md                # 项目说明
├── DESIGN.md                # 技术设计文档
└── ragas_eval_result.json   # 评测结果


## 评测结果
| 指标 | 得分 |
|------|------|
| 上下文精确率 | 1.0000 |
| 回答准确率 | 1.0000 |
| 回答相关性 | 1.0000 |
| 忠实度 | 1.0000 |
| 上下文召回率 | 0.7333 |

## 后续优化方向
- 分块优化（语义分块、父子块）提升召回率
- 接入BGE-M3实现原生稀疏向量混合检索
- 知识库长期记忆（对话内容自动沉淀到Milvus）
- 压力测试（Jmeter）
- 负载均衡（Nginx）
- Docker部署
- 单元测试（pytest）

