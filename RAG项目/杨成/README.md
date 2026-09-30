# 基层高血压 RAG 专家系统

面向基层医疗场景的中文高血压知识库问答原型。系统使用 MinerU 解析本地 PDF，使用 BGE-m3 向量化，使用 Milvus 执行 Dense + BM25 混合检索，经 RRF 和 BGE-reranker-v2-m3 重排后调用 DeepSeek 生成带来源提示的回答。

> 这是课程/实验性质的医疗 RAG 原型，不能替代医生诊断、处方或治疗决策。

## 快速了解

```text
PDF → MinerU → 父子分块 → 表格行拆分 → BGE-m3
   → Milvus Dense + BM25 → RRF → BGE Reranker
   → DeepSeek → 引用检查 / 低相关性兜底 → API / 前端
```

当前知识库：

- 原始 PDF：6 份。
- Milvus collection：`hypertension_rag`。
- Dense 向量维度：1024。
- 已验收实体：约 691 条。
- 评测集：25 条。
- 单元测试：73 个通过。

完整架构、逐文件职责、数据流、接口、评测和讲解提纲见 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)。

## 目录

```text
configs/rag.yaml                # 模型、Milvus、检索和文档身份配置
data/raw/                       # 原始 PDF
data/parsed/                   # MinerU 解析输出
data/processed/chunks/          # 父子分块结果
data/processed/tables/          # 表格行拆分结果
data/processed/vectors/         # 向量和 prepared JSONL
data/processed/eval_dataset.json # 25 条评测问题
data/processed/eval_results.json # 批量评测结果
data/processed/ragas_result.json # RAGAS 结果
data/runs/                      # 检索、重排、问答实验输出
data/runtime/                   # SQLite 会话数据
scripts/                        # 解析、入库、检索、API 和评测脚本
tests/                          # unittest 测试
web/                            # 前端和设计预览
```

`yl/` 是本地 Python/Conda 环境，不是业务源码。由于脚本之间存在直接导入关系，当前没有贸然移动 `scripts/` 下的文件。

## 安装

```bash
python -m pip install -r requirements.txt
```

准备 `.env`：

```text
MINERU_TOKEN=你的 MinerU token
DEEPSEEK_API_KEY=你的 DeepSeek API key
```

`.env` 仅保存在本机，不要提交或打印真实密钥。默认本地模型路径和 Milvus 参数见 [configs/rag.yaml](configs/rag.yaml)。

## 启动 API 与前端

先确认 Milvus Standalone 正在 `127.0.0.1:19530` 监听，然后执行：

```bash
python -m uvicorn api_server:app --app-dir scripts --host 127.0.0.1 --port 8000
```

访问：

- 前端：<http://127.0.0.1:8000/>
- 设计预览：<http://127.0.0.1:8000/design-preview>
- 健康检查：<http://127.0.0.1:8000/health>
- API 文档：<http://127.0.0.1:8000/docs>

当前正式前端为方案 A“医疗专业版”；设计预览页另外展示方案 B“现代 AI 对话版”和方案 C“深色专家工作台”。

## 主要命令

初始化或检查 collection：

```bash
python scripts/init_milvus.py --host localhost --port 19530 --collection hypertension_rag
```

增量导入 prepared records：

```bash
python scripts/insert_to_milvus.py \
  --uri http://127.0.0.1:19530 \
  --collection hypertension_rag \
  --upsert \
  --inputs data/processed/vectors/prepared_*.jsonl
```

> `--upsert` 会覆盖相同稳定 ID。新增文档应先完成解析、分块、向量化和 `prepare_records.py` 校验。

全量重建 collection（破坏性操作）：

```bash
python scripts/rebuild_milvus.py \
  --uri http://127.0.0.1:19530 \
  --collection hypertension_rag \
  --drop-existing \
  --inputs data/processed/vectors/prepared_guideline_2025.jsonl
```

只有明确需要删除并重建 collection 时才使用 `--drop-existing`。

批量评测与 RAGAS：

```bash
python scripts/batch_eval.py
python scripts/run_ragas.py
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## API 概览

- `GET /`：聊天前端。
- `GET /design-preview`：三套视觉方案预览。
- `GET /health`：健康检查。
- `POST /chat`：问答，返回答案、来源、延迟和 `answer_mode`。
- `GET /sessions?user_id=...`：会话列表。
- `GET /sessions/{session_id}/messages?user_id=...`：会话消息。
- `POST /memories`、`GET /memories`、`DELETE /memories/{memory_id}`：长期记忆管理。

`answer_mode=rag` 表示回答使用了本地知识库；`answer_mode=llm_fallback` 表示低相关性或无命中时使用通用大模型知识，回答中会明确说明不是基于数据库指南。

## 检索参数

- Dense Top-20。
- BM25 Top-20。
- RRF `k=60`。
- 融合后 Top-10 进入重排。
- BGE-reranker-v2-m3 最终 Top-5。
- 最高 rerank 分数低于 `0.3` 时切换到 LLM fallback。

主键由以下规则生成：

```text
sha256(document_id + ":" + child_id)[:32]
```

这样可以跨路径稳定增量导入，并避免不同文档的子块 ID 冲突。

## 数据与清理边界

必须保留：

- `data/raw/` 原始 PDF。
- 当前 Milvus 对应的 prepared/vector 文件。
- `data/runtime/memory.sqlite3` 中的会话数据（其中可能包含隐私）。

暂不应直接删除：

- `yl/`，当前可能是正在使用的本地 Python 环境。
- `data/runs/`，其中保存历史实验结果。
- `data/parsed/`、`chunks/`、`tables/`，这些虽然可以重建，但对复现和讲解有价值。

可以清理的缓存：

- `__pycache__/`。
- `*.pyc`。
- 已确认无用途的 `bash.exe.stackdump`。

## 安全边界

当前 API 没有认证，CORS 配置适合本地演示而非公开部署。真实环境至少需要身份认证、访问控制、HTTPS、敏感数据加密、审计、限流和人工医学审核。长期记忆和大模型 fallback 都不是医学证据；药物、诊断和治疗结论应由专业人员依据最新权威指南核对。
