# 部署就绪度核查与缺口登记（批次 40 前置）

- 核查时间：2026-09-23 11:00~11:20
- 核查方式：静态核对「脚本 + 配置 + 数据包 + 文档」四类交付物与代码实际读取点，逐条给证据
  （文件:行号 或 命令实测输出）
- **边界**：本次**未改动任何代码、数据、文档**。本文件只是登记；所有修复建议待裁决。
  （冻结令生效中：改动业务代码会让 `eval_20260922_212626_predeploy_baseline` 失效）

## 一、结论

脚本骨架与文档基本齐备，但存在 **4 个阻塞项**——其中 2 个会让"装完即不可用"，
2 个会让"装完不是评测过的那套数据"。另 5 个次要缺口。

| 级别 | 项 | 一句话 |
|---|---|---|
| 🔴 阻塞 | 1 | `data/labor_law_processed/` 是批次 37 的旧数据包（1627 块），5 篇含页脚样板 |
| 🔴 阻塞 | 2 | `install.sh` 的「导入→索引」顺序在全新库上必然产出 **0 向量** |
| 🔴 阻塞 | 3 | 全链路没有任何一步安装 Node.js / `npm install`，前端起不来 |
| 🔴 阻塞 | 4 | `install.sh` 不建 `chunk_summaries` 表也不生成摘要（486 条缺失） |
| ⚠️ 次要 | 5 | `.env.production` 18 处占位符待填真实值（需你提供） |
| ⚠️ 次要 | 6 | 法规效力状态/公布施行日期未初始化 → 时效过滤失效 |
| ⚠️ 次要 | 7 | `docs/部署文档.md` 与实际交付物多处不一致（文档你改，我只登记） |
| ⚠️ 次要 | 8 | 压测从未执行（方案就绪、零数据） |
| ⚠️ 次要 | 9 | 无 Dockerfile / docker-compose；`rag-backend:latest` 与前端 EXE 在本机找不到 |

## 二、已就绪（清单）

| 交付物 | 状态 |
|---|---|
| `scripts/deploy/install.sh` | ✅ 20.6KB，七步幂等、门槛检查（CPU≥2/内存≥8G 硬门/磁盘≥40G）、`--dry-run`、失败即停 + 修复提示、口令走环境变量不打印 |
| `scripts/deploy/run.sh` | ✅ 依赖服务三项探活 → pid 判重 → 后端 uvicorn → 前端 next → 双健康检查（60s/30s）→ 打印地址日志 |
| `scripts/deploy/shutdown.sh` / `nginx.conf` | ✅ nginx 反代 `/`→前端、`/api/`→后端，SSE 关缓冲 |
| `backend/requirements.txt` | ✅ 43 行全部 `==` 锁版本，口径=本地跑全量测试的环境（批次 22 裁决） |
| `backend/.env.example` | ✅ 8KB，键清单权威来源 |
| `.env.production` | ✅ 模板齐全（含 production 9 个必填键的校验说明），待填 18 处 |
| `data/` | ✅ 原始 HTML 11 篇 840K + 处理后包 2.5M + PDF 样本 960K + 同义词表 + 100 题评测集 |
| `scripts/migrations/` | ✅ 11 个脚本 + README 写明"重建库执行顺序"（含批次 38/39 的状态回填） |
| `scripts/e2e/` | ✅ auth / review / long_term_memory / create_admin + `run_integration_tests.py` |
| `scripts/loadtest/` | ✅ JMeter `.jmx`（10/50/100 三档）+ `first_token_latency.py` + 成本估算 |
| 评测 | ✅ 部署前基线已归档 + 口径说明 + 分块/消融实验 |
| 前端 | ✅ 本机 `npm run build` 通过（Next.js 15.5.7，10/10 页） |

## 三、阻塞项详述

### 🔴 1. 数据包是旧版：`data/labor_law_processed/` 仍是批次 37 状态

**证据（实测）**

- 包内文件 mtime **全部为 2026-09-20 10:55**，无一处晚于批次 38/39（09-23 09:12~09:51 重导）。
- 逐包统计 `document_chunks.jsonl`，合计 **1627** 块（= 批次 37 的规模；当前库是 1557）。
- 5 个包的 chunk 正文里命中特征词 `版权所有 / ICP备 / 责任编辑 / 总机 / 所在位置`：

| 包目录 | 文档 | 旧包块数 | 含样板块 |
|---|---|---|---|
| `472681-doc-695bae71…` | doc1《最高法典型案例》 | 69 | 6（含 `所在位置：…来源：…字号：`） |
| `472691-doc-de9bc3a0…` | doc2《解释（二）和典型案例》 | 81 | 5（含 `责任编辑：刘帆`、`总机：67550114`） |
| `282121-doc-5ed23721…` | doc7《司法解释（一）》 | 183 | 5（含 `责任编辑：韩绪光`、`总机：67550114`） |
| `997e66171cf55d21…` | doc8《调解仲裁法》 | 167 | 2（`中华人民共和国最高人民法院 版权所有`） |
| `content_5711284-…` | doc11《工资支付暂行规定》 | 77 | 5（含 `【打印】/【我要纠错】/链接：/京ICP备`） |
| 其余 6 篇 | — | 1040 | 0 |

**影响**：`install.sh` 第 f 步用 `--packages-root data` 直接导入这批包 →
部署出来的知识库退回到批次 37：页脚样板混入正文、doc1 的整页抓取垃圾 parent 回归。
批次 38/39 的清洗成果**在部署路径上丢失**（库内是对的，交付物是旧的）。

**根因**：批次 38/39 重导时重建包写在了 `%TEMP%/b38_out`、`%TEMP%/b39_out`，
**没有回写 `data/labor_law_processed/`**（当时 `rebuild_pkgs.py` 以 TEMP 为输出目录）。

**修复路径（已验证可精确复现基线）**：把 5 个重建包替换回 `data/labor_law_processed/`：

| 来源 | 文档 | 块数 |
|---|---|---|
| `%TEMP%/b38_out/472681-doc-695bae71eb6908bd532d576c` | doc1 | 52 |
| `%TEMP%/b39_out/282121-doc-5ed237217d8a942f9d8e816e` | doc7 | 170 |
| `%TEMP%/b39_out/472691-doc-de9bc3a0496114ef111e8110` | doc2 | 65 |
| `%TEMP%/b39_out/997e66171cf55d219c613ec18dc370-…` | doc8 | 166 |
| `%TEMP%/b39_out/content_5711284-doc-f85f272456449d…` | doc11 | 54 |

算术校验：`1627 − (69+183+81+167+77) + (52+170+65+166+54) = 1557` —— **与库内、与评测基线完全一致**。
包结构已核对：新旧包文件集一致（各 6 个文件：`documents.jsonl` / `document_versions.jsonl` /
`document_chunks.jsonl` / `crawl_records.jsonl` / `manifest.json` / `raw/<hash>.html`），可直接覆盖。

> 注：`%TEMP%/b38_out` 那个包确认为**批次 38 的最终产物**（52 块，与报告一致）；
> 不要把 `%TEMP%/rebuild_check/`（70 块）或 `%TEMP%/clean_bak/*`（69 块）当新包。

### 🔴 2. 全新库上「导入→索引」必然产出 0 向量

**证据（代码级）**

1. `backend/app/db/import_service.py:116-119`：新导入版本 `version_status` 一律 `pending_review`，
   `processing_status='awaiting_embedding'`（注释「阶段6：新版本一律待审核」）。
2. `backend/app/db/vector_index_service.py:52-57`：索引的查询条件是
   `processing_status == 'awaiting_embedding'` **AND** `version_status == APPROVED`。
3. `backend/app/cli/index_legal_documents.py:77-88`：`--recreate-collection` 只把
   `processing_status` 重置为 `awaiting_embedding`（第 85 行），**不碰** `version_status`。
4. 检索侧同样只读已发布：`retrieval/vector_search.py:239`、`retrieval/keyword_search.py:129`
   都是 `DocumentVersion.version_status == APPROVED`。

**影响**：全新库全是 `pending_review` → 索引选中 0 行 → `status="empty"`，
但 CLI 返回 0（**表面成功**）。装完系统能起、`/health/ready` 可能也过，
但**任何问题都检索不到**，表现为全部拒答。

**缺失的那一步**：把 11 篇置为 `approved`。当前唯一置 approved 的入口是
`app/review/review_service.py:154`（审核发布接口，需管理员逐篇操作）；
存量库当年是用 `scripts/migrations/migrate_review_status.py` 批量置的——
而 **`install.sh` 没有调用它**（`grep -n migrate scripts/deploy/install.sh` 无命中）。

**修复建议（三选一，待裁决）**
- (a) `install.sh` 第 f 步之后插入 `migrate_review_status.py`（+ 保留逐篇审核的产品语义，另加 `--auto-approve` 开关）；
- (b) 新增一个 CLI `app/cli/publish_all.py`，把「批量发布」变成受控命令；
- (c) 在 `import_mysql` 增加 `--initial-approve`（仅限首次建库）。

无论选哪个，**顺序必须是：导入 → 发布 → 建摘要 → 索引**（当前脚本是「导入 → 索引」，顺序也错）。

### 🔴 3. 没有任何一步安装 Node.js / `npm install`

**证据**

- `grep -n -i "node\|npm" scripts/deploy/install.sh` → **无命中**（全脚本零提及）。
- `scripts/deploy/run.sh:143-145`：`frontend/node_modules` 不存在时直接 `die`
  （提示"先执行 `cd frontend && npm install && npm run build`"）。
- `scripts/deploy/run.sh:146-149`：缺 `.next/BUILD_ID` 时会尝试 `npm run build`。
- `frontend/package.json`：无 `engines` 字段（未声明 Node 版本要求）。

**影响**：干净 Ubuntu 上按 install.sh 装完 → `run.sh` 在启动前端时 `die`，整条启动链中断。
（本机 `.next/` 是 Windows 构建产物，不能直接搬到 Linux；`node_modules` 含平台相关二进制，同样不可搬。）

**修复建议**：install.sh 增加一步「安装 Node.js（NodeSource 20.x LTS）+ `npm ci` + `npm run build`」，
并把 Node 版本写进 `frontend/package.json` 的 `engines` 以固化口径（Next.js 15 要求 ≥18.18）。

### 🔴 4. 不建 `chunk_summaries` 表、不生成摘要

**证据**

- `install.sh` 只调用 `app.cli.import_mysql` 与 `app.cli.index_legal_documents`；
  建表脚本是 `scripts/migrations/migrate_chunk_summaries.py`，生成脚本是
  `app/cli/summarize_chunks.py` —— **两者都不在 install.sh 里**。
- `summarize_chunks.py:80-83` 同样只处理 `version_status == APPROVED` 的 parent。
- 检索侧会联表读摘要挂到 `RetrievedArticle`（批次 37 引入）。

**影响**：新库 `chunk_summaries` 表不存在或为空 → 检索拿不到 parent 摘要，
与**评测基线口径不一致**（基线是 486/486 全带摘要），部署后"前后对比"不成立。

**修复建议**：install.sh 在「发布之后、索引之前/之后」插入建表 + `summarize_chunks.py`；
注意该步会真实调用 LLM（486 条，参考批次 37 的耗时与成本）。

## 四、次要缺口

### ⚠️ 5. `.env.production` 18 处占位符

`install.sh:339-341` 会 `grep -q "<"` 拦截未替换占位符。必须由你提供真实值的键：

| 组 | 键 |
|---|---|
| MySQL | `MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `DATABASE_URL` |
| Redis | `REDIS_URL` |
| Milvus | `MILVUS_HOST` |
| Embedding | `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` |
| Reranker | `RERANKER_API_BASE_URL` / `RERANKER_API_KEY` |
| LLM | `LLM_API_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` |
| SMTP | `SMTP_HOST` / `SMTP_USERNAME` / `SMTP_PASSWORD` / `SMTP_FROM_EMAIL` |

（`backend/.env.example` 是键清单权威来源；production 会强制校验「模型 API + 邮件」9 个必填键，
缺一个就启动失败。填值时注意本机写入 `sk-` 类密钥会被过滤器改写，建议分片拼接后核对落盘。）

### ⚠️ 6. 法规效力状态 / 公布施行日期未初始化

`install.sh` 未调用 `backfill_law_dates.py` / `migrate_law_status_vocabulary.py` /
`backfill_status_6laws.py`。新库 `law_versions.status` 与日期为空 →
时效过滤按"未知→True"放行（`vector_index_service` 口径），`as_of_date` 题（9 条）行为与基线不一致。

### ⚠️ 7. `docs/部署文档.md` 与实际交付物不一致（文档你改，我只登记）

| 文档写的 | 实际 |
|---|---|
| §4 目录结构含根 `docker-compose.yml`、根 `nginx.conf` | **都不存在**（nginx.conf 在 `scripts/deploy/`） |
| §4 含 `scripts/health-check.sh`、`scripts/backup.sh` | **都不存在** |
| §10.1/§10.2 用 `docker compose up -d --build` / `up api worker frontend` | 实际路线是 `install.sh` + conda + `run.sh`；仓库无 compose/Dockerfile |
| §10.3 `./scripts/health-check.sh`、§10.4 `./scripts/shutdown.sh` | 实际是 `scripts/deploy/shutdown.sh`；health-check 脚本不存在 |
| §4 写 `data/` 下有 `documents/`、`parsed/`、`backups/` | 实际是 `labor_law_raw/`、`labor_law_processed/`、`pdf_samples/`、`evaluation/` |
| 全文零处提及 Node.js / npm | 而 run.sh 依赖 npm |
| §23 行「项目不使用 Docker Compose 编排应用」 | 与 §10.1/10.2 自相矛盾 |

### ⚠️ 8. 压测从未执行

`scripts/loadtest/`（`.jmx` + `first_token_latency.py` + README）写得很完整，
但**无任何结果产物**：仓库内无 `.jtl`、无 `report_*/`、`reports/` 无压测报告；
本机也**未安装 JMeter / Java**（`which jmeter java` 均无）。
→ 部署后的"性能前后对比"目前没有基线可对。**更正**：此前我口头说"压测零覆盖"不准确，
准确说法是「**方案就绪、从未执行、零数据**」。

### ⚠️ 9. 镜像与 EXE 产物在本机找不到

- 仓库内**无 Dockerfile、无 docker-compose**（`find` 全仓无命中）。
- 本机 docker 只有 `milvusdb/milvus:v3.0.0`、`redis*`、`zilliqa/attu` 等基础镜像，
  **没有 `rag-backend:latest`**。
- 桌面与下载目录**找不到「财问 Setup 1.0.0.exe」**。

若这两个产物在别的机器/服务器上，请告诉我路径我做登记；若已废弃，则部署物清单里应删掉
（`install.sh` 路线本身不需要后端镜像，它用 conda + uvicorn 直接跑源码）。

## 五、建议的最小修复顺序（待你裁决，我未动手）

1. **补数据包**：5 个重建包覆盖回 `data/labor_law_processed/` → 校验合计 1557、0 样板残留。
2. **补 install.sh 的初始化链**：置 approved → 建 `chunk_summaries` 表 → 生成摘要 → 再索引；
   并把「导入 → 索引」改成「导入 → 发布 → 摘要 → 索引」。
3. **补 Node.js 步骤**：install.sh 装 Node 20 LTS + `npm ci` + `npm run build`。
4. **补法规元数据回填**：`backfill_law_dates.py` / `migrate_law_status_vocabulary.py` / `backfill_status_6laws.py`。
5. **填 `.env.production`**（密钥由你提供，我只做落盘核对）。
6. 干跑验证：`install.sh --dry-run` 全绿 + 在真实 Ubuntu 上跑通一次（含健康检查与一问一答）。
7. 压测在部署机执行（方案已就绪）。
8. `docs/部署文档.md` 对齐（你来改）。

> 说明：第 1 步是**数据产物替换**（不涉代码），第 2~4 步是**部署脚本改动**。
> 按冻结令，第 2~4 步属"改动业务代码/交付脚本"，需你明确放行后我才动手；
> 若你希望保持基线干净，也可把这几步写成**部署期操作手册**（由你或运维按步骤执行）。

## 六、本次核查未做（边界）

- 未改动任何代码 / 脚本 / 配置 / 数据 / 文档
- 未在真实 Ubuntu 上跑 install.sh（本机是 Windows，且 install.sh 硬门槛要求 Ubuntu + ≥8G 内存）
- 第 2 条为**代码路径推断**（file:line 证据充分），未在空库上实测；如需实证，
  可在本机用临时库名跑一次 `import_mysql` + `index_legal_documents` 观察
  `indexed_versions=0`（会新建库表，需你同意后执行）

## 七、09-23 14:00 补充核验（回答"任务都完成了吗"）

### 7.1 交付任务台账核销（TaskList 抽取）

平台任务列表共 211 条：188 completed / **9 in_progress** / **14 pending**。
抽查 6 条长期挂着的未关闭条目，**实际工作均已做完**（属状态陈旧，非工作缺失）：

| 条目 | 核验证据 | 判定 |
|---|---|---|
| #4~#10 文件重构系列 | `backend/app/chat/streaming.py` 在位、`retrieval/assembly.py` 在位、`memory/short_term.py` 在位、`ingest/law_metadata_extractor.py` 已改名 | ✅ 已完成 |
| #124 拆分 run_eval.py | 实测 **217 行**（≤300） | ✅ 已完成 |
| #137 拆分 calibrate_refusal.py | 实测 **127 行**（≤300） | ✅ 已完成 |
| #138 清 `.dev_bak25b/` | `ls -d .dev_bak*` 无输出 | ✅ 已清理 |
| #133 答辩/讲解材料 | 桌面 `法律RAG-考核清单逐项对照表.md`（12.6KB，09-22 16:16） | ✅ 已交付 |
| #84/#158 根目录残留 | 根目录已无临时脚本/备份目录 | ✅ 已处置 |

### 7.2 新增登记项（冻结期只登记不改）

| # | 项 | 证据 | 影响 |
|---|---|---|---|
| 10 | `frontend/.next/` **内嵌本机绝对路径** | `.next/required-server-files.json:102,292,295` 写着 `C:\\Users\\92842\\Desktop\\rag\\frontend`；`.next/standalone/` 同样 | **进一步坐实阻塞项 3**：`.next` 连同 `node_modules` 都必须在 Linux 上重新构建，不可搬 |
| 11 | `backend/app/retrieval/vector_search.py` **306 行** | `wc -l` 实测 | 超"单文件 ≤300 行"约束 6 行；属批次 25 之后的自然增长，**冻结期不修**，登记待部署后处理 |
| 12 | 根目录两个运行日志 | `e2e_backend.log`（38KB）、`e2e_review_backend.log`（38KB），09-23 09:49 批次 39 跑 e2e 时由 `scripts/e2e/*.py` 生成 | 脚本自身会重建（非仓库内容），可清可留；不影响交付 |

### 7.3 已登记限制的复核（避免重复记账）

`scripts/e2e/*.py` 硬编码本机路径（`BACKEND_DIR` / `PY` / `RAG`，如 `e2e_auth_persist.py:32-33`）——
**这不是新缺口**：`scripts/e2e/README.md:108` 已明确写"换机器需改这三处"。
且 e2e 脚本**不是部署必需步骤**（`install.sh` 不调用），故不升级为阻塞项。

### 7.4 当前数据包状态（未变）

复核 `data/labor_law_processed/`：仍是 **1627 块**、目录 mtime `2026-09-20 10:55` → 阻塞项 1 **依然在位，未替换**。
