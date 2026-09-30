# MentalHeal RAG

基于 RAG 的心理健康网页版 AI 问答机器人。

当前项目目标：先从 0 到 1 做出一个可以演示、可以迭代的心理健康陪伴助手。系统只使用心理健康相关资料作为知识库，不再扩展律师、金融、NPC、股票、医学诊断等其他领域。系统会读取本地 PDF，通过 OCR、文档解析、分块、向量化、Milvus 检索、BGE-rerank 重排序、DeepSeek 大模型生成回答，并支持 Redis 短期记忆、MySQL 结构化数据和 RAGAS 评测。

## 一期核心功能

- 心理健康陪伴助手聊天。
- 本地 PDF 知识库导入。
- MinerU Cloud API 文档解析和 MinerU 内部 OCR。
- BGE-m3 向量化。
- Milvus 向量数据库存储和检索。
- BM25 + 向量混合检索。
- BGE-rerank 精排。
- DeepSeek API 生成回答。
- Redis 保存最近聊天记录。
- MySQL 保存用户、角色、文档、会话等数据。
- RAGAS 做 RAG 效果评测。

## 项目目录

```text
backend/                  后端代码
  app/
    api/                  接口路由
    core/                 配置、日志、安全
    db/                   MySQL、Redis、Milvus 连接
    models/               数据库模型
    schemas/              请求和响应结构
    services/             业务服务
    rag/                  RAG 主流程
    ingestion/            PDF 解析、OCR、分块、向量化
    memory/               短期记忆、长期记忆
    security/             内容安全和心理危机检测
    main.py               FastAPI 入口
  tests/                  测试代码
frontend/                 前端代码
docs/                     项目文档
scripts/                  安装、启动、运维脚本
data/                     本地数据目录
configs/                  配置文件模板
logs/                     日志
notebooks/                实验 notebook
```

## 重要文档

- 需求规格说明书：`docs/requirements/requirements_spec.md`

## Milvus 管理地址

项目固定使用唯一的 Attu 管理页面：

```text
http://localhost:8001
```

Attu 容器通过 Docker 网络连接 Milvus：

```text
milvus-standalone:19530
```

不要使用 `http://localhost:3000`，该重复 Attu 实例已停用。查看当前容器状态：

```bash
docker ps --filter name=milvus-attu-local
```

## 密钥配置

不要把真实密钥写进代码或提交到仓库。

请复制 `.env.example` 为 `.env`，然后在 `.env` 里填写自己的真实配置：

```bash
cp .env.example .env
```

需要配置的关键项：

- `DEEPSEEK_API_KEY`
- `MINERU_API_KEY`
- `MYSQL_*`
- `REDIS_*`
- `MILVUS_*`

## 本地虚拟环境

项目已创建 `.venv` 虚拟环境，使用 Python 3.10.11 arm64，适配 macOS Apple Silicon 的 PaddleOCR 生态。

进入环境：

```bash
source .venv/bin/activate
```

检查核心依赖：

```bash
python -m pip check
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
python -c "from paddleocr import PaddleOCR, PaddleOCRVL; print('OCR OK')"
```

运行 PDF 解析：

```bash
PYTHONPATH=backend python -m app.ingestion.cli
```

只解析单个 PDF：

```bash
PYTHONPATH=backend python -m app.ingestion.cli --file "data/raw/你的文件.pdf"
```

## 一期推荐开发顺序

1. 启动 FastAPI 后端健康检查。
2. 连接 MySQL、Redis、Milvus。
3. 封装 DeepSeek 调用。
4. 封装 PDF 解析和 OCR。
5. 实现文档分块和 BGE-m3 向量化。
6. 写入 Milvus。
7. 实现 `/api/v1/chat` RAG 问答接口。
8. 做 Vue 聊天页面。
9. 增加 RAGAS 评测。
10. 写部署脚本和测试报告。

## 心理健康安全边界

本系统是心理健康陪伴和科普工具，不是医生、心理咨询师或急救服务。系统不能做正式诊断，不能替代专业医疗建议。遇到自伤、自杀、伤害他人等高风险表达时，应优先给出安全提醒，并建议用户联系当地紧急服务、专业机构或可信任的人。
