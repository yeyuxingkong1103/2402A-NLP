# RAG 知识问答系统

本项目是本地 RAG PDF 问答系统 MVP。用户上传 PDF 后，后端执行清洗、分块、向量化、入库，再通过问答接口返回带页码引用的答案。

## 本地运行

1. 复制 `.env.example` 为 `.env`，并确认本地模型路径正确。
2. 安装依赖：`pip install -r requirements.txt`
3. 启动 Ollama，并确认模型 `deepseek-r1:7b` 可用。
4. 确认 MinerU 可通过 `D:\an\envs\mineru\python.exe -m mineru` 调用。
5. 启动 API：`python backend/app/main.py`
6. 打开接口文档：`http://127.0.0.1:8000/docs`
7. 打开前端页面：`http://127.0.0.1:8000/`

### MinerU 依赖与兼容补丁

- MinerU 3.4.x 的 PP-DocLayoutV2 模型依赖 **transformers 5.x** 的 `hgnet_v2`
  backbone（4.x 不提供，会导致 layout 检测全空并静默回退 pypdf）。
- 若在 `pip install mineru` / 升级 mineru 后重新初始化环境，请运行一次：
  `python scripts/patch_mineru_tf5.py`
  它会为 `mineru/model/layout/pp_doclayoutv2.py` 应用 3 处与 transformers 5.x
  兼容的幂等补丁（config 校验顺序、decoder head 位置、加载后 buffer 重置）。
- 当前已验证组合：`transformers==5.7.0` + `FlagEmbedding==1.4.2` +
  `sentence-transformers==5.7.0` + `mineru==3.4.5`（见 `requirements.txt`）。

## 主要接口

- `POST /api/files/upload`
- `POST /api/tasks/{task_id}/start`
- `GET /api/tasks/{task_id}`
- `GET /api/documents`
- `GET /api/vector-store`
- `GET /api/categories`
- `POST /api/chat`
- [接口文档](./docs/接口文档.md)

## v2 检索重排

v2 在 RRF 召回后新增两阶段重排：

```text
question → RRF 召回(candidate_k=30, 带 dense)
  → 粗排（BGE-M3 dense 余弦，coarse_top_k=10）
  → 精排（bge-reranker-v2-m3，final_top_k=6）
  → min_retrieval_score 过滤 → 引用 → Ollama 回答
```

- 粗排打分方式可配置：`dense_cosine` / `rrf_score` / `hybrid`（`COARSE_SCORER`）
- `RERANK_ENABLED=false` 时退化为 v1 行为
- 相关配置见 `.env.example`（`RETRIEVAL_CANDIDATE_K`、`COARSE_TOP_K`、`FINAL_TOP_K`、`RERANKER_MODEL_PATH` 等）
- 评测：`python scripts/run_sf6_eval.py --variant both`（对比 recall/MRR/延迟，产物落 `eval/results/`）。测评直接对当前真实检索库打分，**需先上传并构建对应文档**，不再自动建库。

## 目录说明

- `backend/`：FastAPI 后端、检索、问答、构建流水线
- `frontend/`：前端单页界面
- `docs/`：需求、计划、任务、架构和版本记录
- `data/`：本地 PDF 和 Qdrant 数据
- `eval/`：评测集与基线结果
