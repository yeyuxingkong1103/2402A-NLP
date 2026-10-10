# 部署 · 04 常见问题 FAQ 与容错

> 工单编号：人工智能NLP-RAG-PDF 文档的表格解析及检索优化

---

## 一、常见问题 FAQ

### Q1：启动界面报 `ModuleNotFoundError: No module named 'src'`

工程以 `src` 为包名，需把工程根目录加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH="C:\Users\30274\Desktop\工单代码\工单三"
streamlit run app.py
```

### Q2：报 `No module named 'faiss'` / `langchain_ollama`

依赖未装齐，执行 `pip install -r requirements.txt`（确认已 `conda activate langchain2`）。

### Q3：报 `ConnectionError` / 连接 `localhost:11434` 失败

Ollama 未启动。另开终端执行 `ollama serve`，并验证：

```powershell
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```

### Q4：首次提问特别慢（10 s+）

首次会加载向量库 + embedding 模型。评估脚本默认**预热**；
界面用 `st.cache_resource` 缓存引擎，**第二次提问即恢复毫秒级**。

### Q5：出现 `FAISS AVX2` 警告

使用 `faiss-cpu` 的正常提示，**不影响功能**，可忽略。

### Q6：向量库保存/加载失败（中文路径）

Windows 版 faiss 的 C++ 层无法打开含中文的路径。`knowledge_base.save_local/load_local`
已做兼容：非 ASCII 路径时先写临时 ASCII 目录再拷贝回。若仍失败，可把工程放到纯英文路径。

### Q7：问"力源信息"的问题却答成"兴图新科"

跨文档串味。确认提问中**带公司名**（或简称"力源/兴图"），系统据此路由文档；
并确认 `.env` 中 `DOC_FILTER_ENABLE=1`。

### Q8：答案里出现"列4/列6"等占位表头

个别复杂合并表头未完全还原。可在 `src/config.py` 调整
`TABLE_HEADER_ROWS`（多级表头行数）后 `python scripts\build_kb.py --force` 重建。

### Q9：`matplotlib` 报缺少字体 / 中文乱码

`make_figures.py` 已指定 `Microsoft YaHei`。Windows 自带该字体；
若在精简系统运行，安装中文字体或改用 `SimHei`。

### Q10：`make_video.py` 报 `No module named 'cv2'`

视频合成需 `opencv-python` + `Pillow`：

```powershell
pip install opencv-python pillow
```

### Q11：如何切换"抽取式"与"大模型生成"？

改 `.env`：`ANSWER_MODE=extractive`（默认，快、稳、零幻觉）或 `llm`（本地生成）。

### Q12：如何只重建某一个向量库？

```powershell
python scripts\build_kb.py            # 只重建优化后
python scripts\build_kb.py --baseline # 只重建优化前
```

---

## 二、容错与降级设计

系统针对"高可用、容错"要求，在每层都设了兜底：

| # | 异常场景 | 处理策略 | 位置 |
|---|---|---|---|
| 1 | 用户空输入 | 返回空分析，不抛异常 | `query_understanding.analyze` |
| 2 | 向量检索失败 | 记 warning，返回空列表，**仅用 BM25** | `retriever._vector_retrieve` |
| 3 | BM25 查询为空（无有效词） | 返回空列表，**仅用向量** | `retriever.BM25Index.search` |
| 4 | 文档路由过滤过度 | 过滤后候选 < 5 → **退回全量候选**，防召回塌陷 | `retriever.retrieve` |
| 5 | 表格块解析为空 | **退回正文句子级抽取** | `answer_builder.build_answer` |
| 6 | 字段型过滤无命中 | **退回最佳表整行** | `answer_builder._table_answer` |
| 7 | 无任何候选 | 返回"根据招股说明书内容无法回答该问题。" | `answer_builder.build_answer` |
| 8 | PDF 文件不存在 | 抛 `FileNotFoundError`（明确错误，不静默） | `pdf_parser` / `table_parser` |
| 9 | LLM 调用失败 | 捕获异常，返回失败提示，**不中断服务** | `qa_engine._invoke` |
| 10 | 中文路径 FAISS 保存失败 | 临时 ASCII 目录桥接 | `knowledge_base.save_local` |
| 11 | 大库一次性嵌入超时 | 按 50 条分批请求 Ollama | `knowledge_base._BatchedEmbeddings` |

---

## 三、多级降级链路

```
优化链路（表格结构化 + 表格感知重排 + 抽取式）
   │  任一环节异常
   ▼
降级① 表格专用路径 → 句子级抽取（表格块为空时）
   │
   ▼
降级② 文档路由过滤失效 → 全量候选（候选过少时）
   │
   ▼
降级③ 向量失败 → 仅 BM25（或反之）
   │
   ▼
兜底 无候选 → "根据招股说明书内容无法回答该问题。"
```

> 设计原则：**任何单点异常都不导致整体不可用**，且绝不编造答案。

---

## 四、性能与稳定性说明

| 项 | 说明 |
|---|---|
| 平均响应 | 1.475 s（优化后，含冷启动的 4 道题） |
| 稳态响应 | 0.1~0.2 s（纯检索 + 打分，无模型推理） |
| 最大响应 | 2.279 s ≤ 3 s ✅ |
| 并发 | Streamlit 单进程演示级；生产可横向扩展为 API 服务 |
| 数据安全 | 全本地，无外发请求 |