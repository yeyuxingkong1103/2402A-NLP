# 医疗RAG角色扮演系统 · 医小助

> 对标 Character.ai，基于 RAG 架构的医疗智能助手：多角色切换（西医/中医/心理医生）、
> BM25+向量混合检索、BGE-reranker 精排、Redis 短期记忆 + Milvus 长期记忆、急症安全拦截。

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 向量库 | Milvus 2.4 | 知识库 + 长期记忆 |
| 向量化 | BGE-m3（本地部署，1024 维） | 文本→向量 |
| 重排 | bge-reranker-v2-m3（本地部署） | 检索结果精排 |
| 框架 | LangChain | RAG 链编排 / DeepSeek 对接 |
| 短期记忆 | Redis 7（List + TTL） | 多轮对话历史 |
| 长期记忆 | Milvus user_memory | 用户摘要召回 |
| 关键词检索 | BM25（rank_bm25 + jieba） | 混合检索关键词路径 |
| 大模型 | DeepSeek deepseek-chat（流式） | 答案生成 |
| 界面 | Streamlit（WSL Ubuntu / Python 3.10） | Web 演示 |
| 评测 | RAGAS 0.2 + pytest | 质量评测 / 单元测试 |
| PDF | PyMuPDF（页眉页脚/水印/表格）；MinerU 扫描件解析（可选） | 文档解析 |
| 图片识别 | PaddleOCR（可选，未装自动降级） | 化验单/药盒/说明书识图问答 |

## 功能清单

- 🏥 **三角色切换**：西医医生 / 中医 / 心理医生，人设、禁用术语、检索偏向各自独立
- 💬 **多轮问诊**：Redis 短期记忆 + LLM 省略句改写，按角色隔离的多会话
- 🔎 **三路增强检索**：BM25 + Milvus 向量 → RRF 融合（向量×2 权重）→ 角色偏向/多路召回 → reranker 精排
- 🚨 **急症拦截**：胸痛等关键词直接提示拨打 120，不进 RAG
- ❓ **不完整先追问**：断句/缺宾语给可点选项，不硬答；无关问题走生活方式兜底
- 💊 **安全用药**：只讲指南内药物、不越权处方，药名与知识库白名单比对
- 📚 **引用溯源**：每条回答带来源文件、页码、相关度，命中句高亮
- 🧠 **长期记忆**：会话摘要入 Milvus，后续对话自动召回
- 📝 **知识库动态更新**：追加 PDF/TXT 增量入库；MinerU 支持无文本层扫描版 PDF
- 📷 **图片识图问答**：上传化验单/药盒照片，OCR 文本拼入检索走 RAG（可选，自动降级）
- 👍👎 **反馈收集**：差评落 data/feedback.jsonl；日志按天滚动落 logs/app.log

## 快速开始（WSL2 Ubuntu）

### 首次部署（一条命令完成全部初始化）

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY；确认 EMBEDDING_MODEL 本地路径
bash scripts/deploy.sh        # 起容器→装依赖→解析PDF→全量入库
```

### 日常启动

```bash
bash scripts/start.sh         # 起基础设施容器 + Streamlit
# 浏览器打开 http://localhost:8501
```

### 也可手动分步执行

```bash
docker compose up -d                              # Milvus/Redis
python3 scripts/02_parse_guideline.py             # PDF → 清洗文本
python3 scripts/03_init_kb.py                     # 切块向量化入库
streamlit run src/main.py                         # 启动界面
```

### 测试与评测

```bash
python3 -m pytest tests -q                        # 17 个离线单元测试
python3 scripts/05_ragas_eval.py                  # RAGAS 在线评测（需 Milvus+API Key）
```

## 模型本地化（重要）

两个大模型均为**本地路径加载**，不要写 HF 仓库名（否则每次启动联网下载 2.3G）：

| .env 配置项 | 当前值 |
|-------------|--------|
| EMBEDDING_MODEL | `/mnt/d/models（文本嵌入模型）/bge_m3` |
| RERANKER_MODEL | `models/bge-reranker-v2-m3`（项目内） |
| EMBEDDING_DEVICE | `cuda`（无 GPU 改 `cpu`） |

> Streamlit 运行在 WSL，路径必须用 `/mnt/d/...` 格式；Windows 下测试用 `D:/...`。

## 项目结构

```
.
├── README.md                    # 本文件
├── requirements.txt             # 依赖
├── docker-compose.yml           # Milvus(etcd/minio)/Redis 一键启动
├── .env / .env.example          # 配置（模型路径、连接、阈值、API Key）
├── .streamlit/config.toml       # 主题 + WSL 文件监听关闭
├── docs/                        # 项目文档（开发流程对齐）
│   ├── 01_可行性研究.md
│   ├── 02_需求规格说明书.md
│   ├── 03_概要设计.md
│   ├── 04_测试报告.md
│   └── 05_接口说明.md            # 全部模块函数入参/出参
├── data/
│   ├── guidelines/              # 两份卫健委指南 PDF + 解析文本
│   ├── roles.json               # 3 个角色人设
│   └── sample_images/           # 化验单/药盒实测图片
├── models/bge-reranker-v2-m3/   # 本地重排模型
├── scripts/
│   ├── deploy.sh / start.sh     # 部署 / 启动 shell 脚本
│   ├── 02_parse_guideline.py ~ 04_update_kb.py   # 数据管线
│   ├── 05_ragas_eval.py         # RAGAS 评测
│   └── 06_mineru_kb.py          # MinerU 扫描版 PDF 入库（可选）
├── src/                         # 22 个模块，每个 ≤150 行（pytest 自动守护）
│   ├── main.py                  # Streamlit 入口/装配/预热
│   ├── ui_*.py                  # 侧栏/对话区/消息/样式
│   ├── rag_chain.py             # 急症→改写→检索→重排→4 模式
│   ├── retrieval/reranker/bm25_index/citation.py
│   ├── ingestion/pdf_parser.py  # 入库管线 / PDF 解析
│   ├── image_ocr.py             # PaddleOCR 图片→文本（可选）
│   ├── memory.py                # Redis 短期 + Milvus 长期记忆
│   └── role/intent/query_tools/prompts/safety/chat_session/config.py
└── tests/
    ├── test_core.py             # 9 个：约束/BM25/入库/Prompt/角色
    └── test_retrieval_safety.py # 8 个：意图/安全/引用
```

## 开发流程（对齐课程要求）

```
可行性研究 → docs/01   需求规格 → docs/02   概要设计 → docs/03
详细设计   → 各模块 docstring + docs/05 接口说明
开发调试   → src/（logging 入口/出口日志 + try/except 兜底）
单元测试   → pytest 17 用例（含 150 行/24 文件守护）
集成/评测  → 手工验收清单 docs/04 + RAGAS 脚本
部署维护   → deploy.sh/start.sh + docker-compose + logs/app.log
```

## 常见问题

**Q: 启动卡在 embedder？**
A: 检查 .env 的 EMBEDDING_MODEL 是本地路径（WSL 用 /mnt/d/...），联网仓库名会触发 2.3G 下载。

**Q: Milvus/Redis 连不上？**
A: `docker compose up -d` 后等约 30 秒；`docker ps` 确认容器在跑。

**Q: reranker 一直"未启用"？**
A: 冷启动约 27 秒，后台预热中；看侧栏状态或 logs/app.log。失败会自动回退向量分（阈值 0.45）。

**Q: 改了 .env 不生效？**
A: 配置在进程启动时加载，重启 streamlit 进程即可。
