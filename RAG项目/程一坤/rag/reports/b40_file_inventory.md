# 项目文件用途总览 + 冗余代码审计（批次 40）

- 时间：2026-09-23 14:09~14:50
- 方法：AST 静态解析 247 个 `.py`（提取模块 docstring + 顶层符号）+ 全库名字引用图 +
  配置键读取比对 + 前端引用扫描；候选逐条人工核实（打印上下文/装饰器/全库出现位置）
- **边界**：本文档**只诊断不改**（冻结期），所有"可删"结论均给证据，未动任何代码

---

## 一、规模总览（733 个文件）

| 目录 | 文件数 | 性质 | 行数 |
|---|---|---|---|
| `backend/` | 212 | 后端源码 205 py + 3 测试夹具 html + 4 配置 | 32,951 |
| `reports/` | 315 | **产物**（140 md 报告 / 80 txt 证据 / 75 json 评测结果 / 20 log） | 522,601 |
| `data/` | 89 | 数据（22 原始 HTML / 48 数据包 jsonl / 13 json / 4 PDF / 2 其他） | 18,462 |
| `frontend/` | 37 | 前端源码 27 ts·tsx + 10 配置 | 7,229 |
| `scripts/` | 36 | 25 py + 5 md + 3 sh + 3 其他（jmx/json/conf） | 6,458 |
| `docs/` | 20 | 文档 | 7,260 |
| `evaluation/` | 18 | 评测代码 17 py + README | 2,596 |
| 根目录 | 6 | `.env`×4 + `README.md` + `CLAUDE.md` | 207 |

> 说明：`frontend/node_modules`（340M）与 `.next`（197M）是构建产物，不计入文件清单。

---

## 二、逐文件用途

### 2.1 `backend/app/` —— 应用源码（约 100 个模块）

#### `app/`（根）
| 文件 | 行 | 用途 |
|---|---|---|
| `__init__.py` | 1 | 包标记 |
| `main.py` | 165 | FastAPI 应用入口：路由挂载、请求 ID 中间件、全局异常处理、`/health/live`、`/health/ready` |
| `errors.py` | 113 | 业务异常基类 `BusinessException`、错误码枚举与便捷构造器（bad_request/not_found/…） |

#### `app/api/` —— HTTP 接口层
| 文件 | 行 | 用途 |
|---|---|---|
| `__init__.py` | 1 | 包标记 |
| `chat.py` | 245 | **问答 SSE 主接口** + 服务端分帧工具（`encode_sse`/`split_answer_chunks`） |
| `chat_persistence.py` | 244 | 问答持久化编排：短期记忆获取、一问一答落库、会话摘要触发 |
| `legal_search.py` | 195 | 法律检索 HTTP 接口（契约校验 + 结果序列化） |
| `memory.py` | 173 | 长期记忆接口（列表 / 删除 / 用户级开关） |
| `review.py` | 174 | 管理员审核与发布接口 |
| `session_routes.py` | 262 | 会话管理 REST：创建 / 本人列表 / 历史消息 / 删除 |
| `users_me.py` | 57 | 当前登录用户信息接口 |

#### `app/auth/` —— 认证
| 文件 | 行 | 用途 |
|---|---|---|
| `current_user.py` | 56 | FastAPI 依赖：取令牌并校验会话 |
| `mailer.py` | 129 | 邮件发送协议与两实现：`InMemoryMailer`（开发）/ `SMTPMailer`（生产真发） |
| `router.py` | 165 | 认证 HTTP：发码 / 注册 / 登录 / 改密 / 登出 |
| `schemas.py` | 80 | 认证请求响应模型 + 邮箱白名单校验器 |
| `service.py` | 228 | 认证核心：注册/登录/改密/验证码/令牌签发 + `ENVIRONMENT=test` 固定码旁路 |
| `session_store.py` | 124 | 会话令牌的 Redis 读写（创建/读取/撤销） |
| `sql_store.py` | 116 | 认证用户主体的 MySQL 存储 |

#### `app/chat/` —— 问答链路（本项目核心）
| 文件 | 行 | 用途 |
|---|---|---|
| `bootstrap.py` | 87 | 默认问答服务装配（LLM/检索/记忆实例组装） |
| `chat_history.py` | 175 | 会话与消息读取侧（列表 + 历史消息） |
| `chat_runtime.py` | 57 | 聊天存储的生产装配（MySQL 会话工厂） |
| `chat_store.py` | 280 | 聊天持久化写入侧 + 归属校验 `assert_session_owned` |
| `chat_title.py` | 51 | 会话标题生成规则 |
| `citation_check.py` | 159 | 引用校验**结构性入口**：异常类 + `check_citations`（语言规则转发到下一文件） |
| `citation_language.py` | 206 | 引用校验**语言层**：表述词表、法规名归一化、逐句结论扫描 |
| `guard.py` | 288 | **护栏**：拒答判定、绝对化表述拦截、免责声明、`resolve_citation_outcome` 统一处置 |
| `memory_hooks.py` | 152 | 记忆读取钩子（长期记忆 + 会话前情） |
| `prompt_builder.py` | 135 | 提示词组装（含 SYSTEM_PROMPT） |
| `result.py` | 43 | 问答结果结构与上下文摘录 |
| `service.py` | 205 | 问答主流程（同步 `chat`） |
| `source_assembly.py` | 37 | 法源清单（citation 事件）组装 |
| `streaming.py` | 187 | **流式问答主流程**（`chat_stream`） |

#### `app/cli/` —— 命令行入口
| 文件 | 行 | 用途 |
|---|---|---|
| `create_admin.py` | 93 | 创建/升级管理员 CLI |
| `demo_ask.py` | 123 | 演示入口：问一题并打印检索统计、法源、回答 |
| `import_mysql.py` | 102 | **数据包导入 MySQL CLI**（单包 / 批量） |
| `index_legal_documents.py` | 184 | **向量索引 CLI**：取待嵌入版本 → 嵌入 → 写 Milvus |
| `summarize_chunks.py` | 204 | 条文一句话摘要生成 CLI（写 `chunk_summaries`） |

#### `app/core/` —— 配置
| 文件 | 行 | 用途 |
|---|---|---|
| `config.py` | 242 | 配置门面：`Settings` + `.env` 文件加载 |
| `config_defaults.py` | 194 | 默认值常量表（批次 25-2 拆出） |
| `config_loader.py` | 268 | 配置读取与校验（按 `ENVIRONMENT` 选文件、production 必填校验） |
| `logging.py` | 29 | 结构化 JSON 日志配置 |

#### `app/crawler/` —— 采集（数据侧）
| 文件 | 行 | 用途 |
|---|---|---|
| `collect_labor_law.py` | 210 | 采集首批劳动法官方页并保存原始文件（入口） |
| `crawl_record.py` | 61 | 采集记录模型 + 正文哈希（增量比对依据） |
| `html_text.py` | 94 | HTML 正文提取与清洗（采集侧） |
| `rate_limiter.py` | 57 | 按来源域名的请求限速 |
| `requester.py` | 288 | 官方页面 HTTP 请求器（重试/UA/超时/大小限制） |
| `response_reader.py` | 54 | 响应流读取与体积上限保护 |
| `robots_policy.py` | 40 | robots.txt 访问策略 |
| `runner.py` | 108 | 采集编排：白名单 → 限速 → 请求 → 落盘 |
| `whitelist.py` | 141 | 官方来源白名单与 URL 规则 |

#### `app/db/` —— 数据访问
| 文件 | 行 | 用途 |
|---|---|---|
| `base.py` | 15 | SQLAlchemy `Base` + `utc_now()` |
| `batch_import.py` | 67 | 批量导入：发现包目录并逐个导入 |
| `chat_models.py` | 115 | 聊天会话/消息表模型 |
| `document_models.py` | 189 | 文档域实体：Document / DocumentVersion / DocumentChunk / ChunkSummary / CrawlRecord / ImportRecord |
| `engine.py` | 15 | 数据库引擎与会话工厂创建 |
| `import_cleanup.py` | 80 | `replace_existing` 模式下的旧版本清理 |
| `import_service.py` | 279 | **导入事务主流程**（新版本一律 `pending_review`） |
| `import_version_status.py` | 84 | 增量判定：new / unchanged / updated |
| **`law_status.py`** | 60 | `law_versions.status` 取值词表**唯一定义处** |
| `law_models.py` | 184 | 法规域实体：Law / LawVersion / Article（条-款-项） |
| `legal_metadata_writer.py` | 179 | 三层法规元数据表写入（含"非空才写"防冲掉人工补录） |
| `milvus_store.py` | 187 | Milvus 集合封装（建表/upsert/查询/删除） |
| `redis_client.py` | 40 | Redis 客户端复用（认证 + 短期记忆共用） |
| `sql_models.py` | 41 | **实体统一导入入口**（不定义表、不写逻辑，聚合 re-export） |
| `user_models.py` | 49 | 用户域实体 |
| **`vector_index_service.py`** | 280 | 待嵌入版本 → 向量写入 Milvus + 孤儿向量回收 |
| **`version_status.py`** | 36 | `document_versions.version_status` 取值词表**唯一定义处** |

#### `app/ingest/` —— 离线入库
| 文件 | 行 | 用途 |
|---|---|---|
| `article_number_rules.py` | 185 | 条文路径生成规则 |
| `article_splitter.py` | 61 | 条文边界切分 |
| `chinese_number.py` | 137 | 中文数字 → 阿拉伯数字 |
| `chunk_fingerprint.py` | 46 | 切块指纹（导入幂等依据） |
| `chunk_text.py` | 51 | 切块文本工具（归一化 / 检索正文拼接） |
| `chunker.py` | 182 | **切块主流程**：条（父块）+ 款/项（子块）两级 |
| `cleaner.py` | 219 | 正文清洗：页脚样板 / 导航 / 短行规则（批次 38/39 重点） |
| `law_date_rules.py` | 134 | 法规日期识别规则 |
| `law_metadata_extractor.py` | 207 | 法规级元数据抽取（状态/类型/机关/名称） |
| `offline_ingest.py` | 277 | **离线入库主流程**（原始文件 → 解析 → 切块 → 打包） |
| `paragraph_items.py` | 117 | 款/项识别 |
| `parser.py` | 274 | 多格式解析入口（HTML / 文本 / PDF）+ TRS 正文区提取 |
| `pdf_parser.py` | 90 | PDF 解析编排（MinerU 主 + Qwen-VL 兜底） |
| `run_offline_ingest.py` | 124 | 离线入库 CLI（manifest 驱动） |
| `title_normalizer.py` | 117 | 文档标题归一化（站点后缀剥离 + 黑名单） |

#### `app/memory/` —— 记忆
| 文件 | 行 | 用途 |
|---|---|---|
| `long_term.py` | 233 | Milvus 长期记忆读写（跨会话事实/偏好） |
| `memory_schema.py` | 134 | 长期记忆集合管理 + `MemoryRecord` + `LongTermMemoryError` |
| `memory_settings.py` | 73 | 用户级记忆开关存储 |
| `memory_summarize.py` | 50 | 写入前的事实提炼 |
| `short_term.py` | 184 | Redis 短期记忆：最近消息 + 摘要 + 节流状态 |
| `summary_policy.py` | 30 | 摘要触发阈值与长度上限（单点调参） |
| `summary_service.py` | 206 | 会话摘要写入侧（窗口满后压缩更早轮次） |

#### `app/models/` —— 外部服务适配
| 文件 | 行 | 用途 |
|---|---|---|
| `embedding.py` | 195 | 嵌入客户端（SiliconFlow bge-m3） |
| `http_retry.py` | 257 | **共用** HTTP 请求 + 指数退避重试工具 |
| `llm.py` | 250 | 大模型适配器（OpenAI 兼容，含流式） |
| `mineru.py` | 245 | MinerU 适配器（PDF 结构化解析主通道） |
| `mineru_api.py` | 108 | MinerU API 层（鉴权头 + 重试 + 传输实现） |
| `mineru_flow.py` | 154 | MinerU 上传/轮询/下载编排 |
| `mineru_result.py` | 150 | MinerU 结果解析（zip → Markdown / 页数） |
| `qwen_vl.py` | 300 | Qwen-VL 适配器（PDF 渲染 + OCR，兜底通道） |
| `reranker.py` | 160 | 重排适配器（bge-reranker-large） |

#### `app/pipeline/` —— 数据包
| 文件 | 行 | 用途 |
|---|---|---|
| `hashing.py` | 13 | 数据包文件 SHA256（全项目唯一实现） |
| `package_models.py` | 163 | 数据包记录模型 + 稳定 ID |
| `package_validator.py` | 181 | 数据包完整性校验 |
| `package_writer.py` | 89 | 数据包原子发布 |

#### `app/retrieval/` —— 检索（本项目第二核心）
| 文件 | 行 | 用途 |
|---|---|---|
| `assembly.py` | 233 | 检索链路装配与依赖协议 |
| `context_builder.py` | 207 | 检索结果数据结构 + 注入 LLM 的法源清单 |
| `filters.py` | 107 | 检索前元数据过滤（法域/时间点/文书类型/时效） |
| `fusion.py` | 163 | RRF 融合与去重 |
| `item_convert.py` | 110 | RankedItem 与召回结果转换纯函数 |
| `keyword_search.py` | 301 | 关键词检索（jieba + BM25 + 法条号精确加分） |
| `parent_collapse.py` | 251 | 父块解析与归并（子块命中升级为父块视角） |
| `query_rewrite.py` | 260 | 基于短期上下文的查询改写 |
| `service.py` | 267 | **检索服务总入口**：串联完整链路 |
| `synonym_expansion.py` | 255 | 法律术语同义扩写 |
| `vector_search.py` | 307 | **向量召回主体** → 取正文 → 重排 |

#### `app/review/` —— 审核发布
| 文件 | 行 | 用途 |
|---|---|---|
| `review_detail.py` | 115 | 审核详情查询（版本定位 + 元数据 + 分块预览） |
| `review_service.py` | 221 | 审核核心：列表 / 通过 / 驳回 + 索引触发 + 向量回收 |

### 2.2 `backend/tests/` —— 测试（80 个文件）

| 文件 | 行 | 覆盖对象 |
|---|---|---|
| `conftest.py` | 242 | 共享夹具（建会话、造数据包、FakeRedis、批量置 approved） |
| `demo_retrieval_pipeline.py` | 136 | ⚠️ 验收演示脚本（**非测试**，pytest 不收集） |
| `test_auth_api.py` | 271 | 认证接口契约 |
| `test_auth_sql_store.py` | 182 | 认证落库（重启后仍可登录） |
| `test_auth_test_bypass.py` | 125 | test 环境固定码旁路 + 三环境守卫 |
| `test_batch_import.py` | 61 | 批量导入发现与幂等 |
| `test_chat_api.py` | 355 | 问答 SSE 接口契约（事件序列） |
| `test_chat_endpoint_shared.py` | 55 | 两个客户端共用 endpoint 的锁定 |
| `test_chat_memory_injection.py` | 232 | 长期记忆注入问答 |
| `test_chat_persistence.py` | 146 | 落库 model 字段取当轮配置 |
| `test_chat_service.py` | 138 | 问答主流程 |
| `test_chat_sessions_api.py` | 429 | 会话管理接口契约 |
| `test_chat_stream_service.py` | 170 | 流式问答（全替身） |
| `test_check_services.py` | 170 | check_services 的自检（假绿灯回归） |
| `test_chunk_summary.py` | 263 | chunk_summaries 表 / 摘要入索引 / sources 组装 |
| `test_chunker_paragraphs.py` | 210 | 条-款-项切块语义 |
| `test_citation_check.py` | 248 | 引用校验 |
| `test_citation_guard_rules.py` | 113 | "无引用结论"规则细分 |
| `test_citation_outcome_shared.py` | 110 | 统一处置阶梯 |
| `test_citation_position.py` | 60 | 引用位置（句号前后）归一化 |
| `test_cleaner_boilerplate.py` | 106 | 页脚样板过滤 + TRS 正文区（批次 38） |
| `test_cleaner_headings.py` | 83 | 短行规则不误删标题（批次 22） |
| `test_cleaner_site_footer.py` | 186 | 站点级页脚覆盖 + 真实 HTML fixture（批次 39） |
| `test_cli_contract.py` | 74 | CLI 用到 ChatResult 字段的契约 |
| `test_content_hash_stability.py` | 246 | 正文哈希忽略动态时间戳 |
| `test_context_builder_metadata.py` | 159 | 上下文组装与引用字段补全 |
| `test_create_admin.py` | 75 | create_admin CLI |
| `test_document_title_extraction.py` | 54 | 标题站点后缀剥离 |
| `test_email_validator.py` | 98 | 邮箱白名单三份合一后的行为锁定 |
| `test_embedding_retry.py` | 164 | 嵌入重试 |
| `test_env_file_selection.py` | 181 | 按 ENVIRONMENT 选 .env 的加载逻辑 |
| `test_guard.py` | 107 | 拒答与表述拦截 |
| `test_guard_zero_citation_tier.py` | 79 | 零引用分级三场景 |
| `test_guardrail_graded.py` | 160 | 护栏分级处置 |
| `test_http_retry.py` | 403 | 共用 HTTP 重试工具 |
| `test_import_chunk_fingerprint.py` | 248 | 导入幂等与切块指纹 |
| `test_import_incremental.py` | 237 | 增量导入三态 |
| `test_import_mysql_cli.py` | 64 | 导入 CLI |
| `test_index_legal_documents_cli.py` | 125 | 索引 CLI |
| `test_keyword_search_filters.py` | 103 | 关键词检索过滤与排名 |
| `test_law_metadata_dates.py` | 123 | 生效/公布日期抽取增强 |
| `test_law_models.py` | 303 | 法规域模型约束与唯一键 |
| `test_law_status.py` | 247 | 效力状态词表单一来源锁定 |
| `test_legal_meta.py` | 162 | 条文路径工具 |
| `test_legal_metadata_writer.py` | 229 | 重导入不冲掉人工补录（批次 26-C5） |
| `test_legal_search_api.py` | 208 | 检索接口契约 |
| `test_llm_stream.py` | 124 | 流式 LLM 客户端 |
| `test_long_term_memory.py` | 339 | 长期记忆存储层 |
| `test_mailer_selection.py` | 183 | SMTP/内存邮件按环境选择 |
| `test_memory_api.py` | 167 | 长期记忆接口 |
| `test_milvus_store.py` | 190 | Milvus 封装（schema / upsert / 存在性） |
| `test_mineru_client.py` | 513 | MinerU 客户端 |
| `test_mysql_import_service.py` | 66 | 导入服务三态与失败回滚 |
| `test_offline_ingest.py` | 116 | 离线入库主流程 |
| `test_offline_ingest_pdf.py` | 299 | 离线入库 PDF 通道 |
| `test_package_module.py` | 298 | 数据包模型/校验/原子发布 |
| `test_parent_collapse_order.py` | 65 | 父块加载顺序 |
| `test_pdf_config.py` | 88 | PDF 相关配置接线 |
| `test_pdf_parser.py` | 143 | PDF 解析分支（主/兜底） |
| `test_production_smtp_validation.py` | 61 | production SMTP 启动期校验 |
| `test_prompt_builder.py` | 157 | 提示词组装（含防注入、免责声明） |
| `test_prompt_no_drafting.py` | 14 | 提示词禁止代写文书 |
| `test_query_rewrite.py` | 292 | 查询改写与检索链路接入 |
| `test_qwen_vl_client.py` | 380 | Qwen-VL 兜底客户端 |
| `test_refusal_threshold.py` | 103 | 拒答阈值 |
| `test_retrieval_factory_smoke.py` | 86 | 真实装配冒烟 |
| `test_retrieval_filters.py` | 117 | 时效过滤 |
| `test_retrieval_fusion.py` | 112 | RRF 融合 |
| `test_retrieval_service.py` | 411 | 检索服务总入口 |
| `test_review_api.py` | 431 | 审核发布接口 |
| `test_review_service.py` | 371 | 审核能力层 |
| `test_review_status.py` | 106 | 审核状态词表单一来源锁定 |
| `test_session_store.py` | 227 | 会话令牌存储 |
| `test_session_summary.py` | 374 | 会话摘要触发/节流/失败保留 |
| `test_short_term_memory.py` | 159 | Redis 短期记忆 |
| `test_sql_models.py` | 177 | 表结构与唯一键 |
| `test_synonym_expansion.py` | 275 | 术语同义扩写 |
| `test_users_me.py` | 126 | 当前用户接口 |
| `test_vector_index_service.py` | 152 | 索引服务 |
| `test_vector_search_contract.py` | 74 | 向量召回参数契约 |
| `fixtures/court_case_page.html` | — | 真实案例页片段（批次 39） |
| `fixtures/court_interpretation_page.html` | — | 真实司法解释页片段 |
| `fixtures/gov_news_page.html` | — | 中国政府网新闻页片段 |

### 2.3 `evaluation/` —— 评测（18 个文件）

| 文件 | 行 | 用途 |
|---|---|---|
| `aggregate.py` | 141 | 指标汇总（四指标 + 分类型 + 最差样本） |
| `answer_judging.py` | 73 | 回答侧判定原语（拒答口径 + 引用越界审计） |
| `calibrate_refusal.py` | 128 | 拒答阈值校准 CLI |
| `compare_synonym_ablation.py` | 198 | 术语扩写消融对比 |
| `compare_window.py` | 75 | 召回窗口前后对比 |
| `diagnose_rerank.py` | 180 | 重排环节诊断 |
| `faithfulness.py` | 229 | Faithfulness 轻量打分器 |
| `item_runner.py` | 140 | 单题执行（检索→问答→判定） |
| `legal_matching.py` | 96 | 法规名/条号匹配原语 |
| `multi_turn_grading.py` | 191 | multi_turn 判定与报告渲染 |
| `multi_turn_runner.py` | 256 | multi_turn 真实多轮执行器 |
| `refusal_report.py` | 118 | 拒答校准报告渲染 |
| `refusal_scoring.py` | 117 | 分数采集与阈值扫描 |
| `render_report.py` | 92 | 评测报告 Markdown 渲染 |
| `run_config.py` | 114 | 当轮运行变量采集（跨轮对比依据） |
| `run_eval.py` | 218 | **评测运行器主入口** |
| `selfcheck_faithfulness.py` | 113 | 打分器自检 |
| `README.md` | — | 评测口径与用法 |

### 2.4 `scripts/` —— 运维与研究脚本（36 个）

| 文件 | 行 | 用途 |
|---|---|---|
| `_env.py` | 87 | 脚本共用 `.env` 加载器与 MySQL 参数（零明文机密） |
| `check_imports.py` | 114 | 全仓 import 完整性校验 |
| `check_services.py` | 434 | 服务健康自检（Docker/Redis/MySQL/Milvus/外部 API） |
| `deploy/install.sh` | — | Ubuntu 一键安装（七步幂等 + 门槛检查 + dry-run） |
| `deploy/run.sh` | — | 启动（探活 → 后端 → 前端 → 双健康检查） |
| `deploy/shutdown.sh` | — | 停止 |
| `deploy/nginx.conf` | — | nginx 反代（SSE 关缓冲） |
| `e2e/cleanup_e2e_probe_accounts.py` | 124 | 清理 e2e 探针账号与从属数据 |
| `e2e/compare_pdf_vs_html.py` | 320 | PDF 通道 vs HTML 通道等价性对照 |
| `e2e/e2e_auth_persist.py` | 237 | 认证落库 e2e |
| `e2e/e2e_create_admin.py` | 109 | create_admin e2e |
| `e2e/e2e_long_term_memory.py` | 241 | 长期记忆全链路 e2e |
| `e2e/e2e_review_publish.py` | 241 | 审核发布 e2e |
| `e2e/fetch_pdf_samples.py` | 192 | 重新下载 PDF 回归样本 |
| `e2e/pdf_parse_main_path.py` | 225 | PDF 主路径验收 |
| `e2e/pdf_parse_qwen_fallback.py` | 212 | PDF 兜底分支验收 |
| `e2e/run_integration_tests.py` | 133 | **集成测试统一入口** |
| `e2e/README.md` | — | e2e 前置与旁路说明 |
| `eval/multi_turn_summary_check.py` | 297 | 多轮链路探针（验摘要/改写真生效） |
| `eval/README.md` | — | 与 `evaluation/` 的分工说明 |
| `experiments/chunking_strategy_compare.py` | 304 | 分块策略三臂对照实验 |
| `loadtest/first_token_latency.py` | 142 | 首字延迟测量 |
| `loadtest/chat_stream_loadtest.jmx` | — | JMeter 压测计划（10/50/100 并发） |
| `loadtest/README.md` | — | 压测说明与成本估算 |
| `migrations/`（10 脚本 + README） | — | 一次性 DB 迁移/回填（含执行顺序说明） |
| `api-collection/`（json + README） | — | Postman/ApiPost 接口集合 |

### 2.5 其他

| 位置 | 内容 |
|---|---|
| `frontend/src/`（25 个 ts/tsx） | Next.js 15 App Router：`app/` 下页面（chat / login / register / password-reset / admin 等）+ 组件与 API 客户端；**无孤儿文件** |
| `frontend/`（10 配置） | `package.json` / `next.config.ts` / `tsconfig.json` / `tailwind.config.js` / `postcss.config.js` / `.env` / `next-env.d.ts` 等 |
| `docs/`（20 md） | 架构文档、接口文档、开发路线图、部署文档、目录与命名约定、CONTEXT 等 + `archive/{plans,specs}` 历史归档 |
| `data/labor_law_raw/`（22 html） | 11 部法规的原始采集页 |
| `data/labor_law_processed/`（11 包 × 6 文件） | 离线入库产物（documents/versions/chunks/crawl_records/manifest/raw） |
| `data/evaluation/` | 100 题评测集 |
| `data/pdf_samples/`（4 pdf） | PDF 通道回归样本 |
| `data/`（同义词表等 13 json） | 术语同义表（带版本号 + 负向规则）等 |
| `reports/`（315） | 产出物：140 md 报告、80 txt 证据、75 json 评测结果、20 log —— 见 `b40_file_inventory` 附注 |
| 根 6 文件 | `.env`（私有覆盖）/`.env.development`/`.env.test`/`.env.production` + `README.md` + `CLAUDE.md` |

---

## 三、冗余代码审计

### 3.1 结论

| 类别 | 数量 | 处理建议 |
|---|---|---|
| 🔴 **确认无用（可删）** | **28 处**（16 处模块级 + 12 处测试内） | 冻结期**只登记**，部署后统一清理 |
| 🟡 看着像多余、实为**有意保留** | 6 类 | 不要动（有 `# noqa` 或注释说明） |
| 🟢 扫描器**必然误报** | 14 个函数 | 框架注册（装饰器），非死代码 |
| 🟢 重复实现 | 29 组同名函数 | 多为测试/迁移脚本局部复用，建议择机合并 |
| ✅ 无冗余 | 配置键 63 个全被读取；前端 25 文件无孤儿；247 py 无孤儿模块 | — |

### 3.2 🔴 确认无用（逐条带证据）

**A. 死常量（定义后全库零引用，含测试）**

| # | 位置 | 内容 | 证据/说明 |
|---|---|---|---|
| 1 | `app/ingest/cleaner.py:41` | `_SOURCE_PREFIX = "来源："` | 全库零引用；**同文件 `_BOILERPLATE_PREFIXES` 第 35 行已含同样字面量** → 纯重复定义 |
| 2 | `app/ingest/law_metadata_extractor.py:50` | `CASE_MATERIAL_KEYWORDS` | 零引用（实际生效的是下一行 `CASE_MATERIAL_TITLE_KEYWORDS`） |
| 3 | `app/models/mineru.py:59` | `MAX_PAGES = 200` | 零引用。⚠️ **不只是多余**：注释写"超出会被服务端拒绝，这里提前拦下"，但代码里没有这道拦截 → **意图存在、实现缺失** |
| 4 | `app/cli/create_admin.py:29` | `STATUS_NOT_REGISTERED` | 零引用；注释称"测试断言 + CLI 输出共用"，但两边都没用它 |
| 5 | `evaluation/multi_turn_grading.py:21` | `SESSION_WINDOW = 20` | 零引用；注释称"与 chat_persistence 的 max_messages=20 逐字一致"，**没有代码绑定** → 两边真改起来不会同步 |
| 6 | `evaluation/answer_judging.py:20` | `REFUSAL_PHRASES` | 零引用，但**注释明说"旧词表仅留作参考/debug"** → 有意保留，可选删 |

**B. 未使用的 import（AST 证实本文件内既无名字引用，也无字符串注解需要）**

| # | 位置 | 未使用项 |
|---|---|---|
| 7 | `app/chat/guard.py:14-18` | `CitationError`（函数体只用 3 个子类做 `isinstance`） |
| 8 | `app/chat/chat_history.py:22` | `SessionAccessDenied`（实际由 `assert_session_owned` 抛出） |
| 9 | `app/db/vector_index_service.py:9` | `Article`（同行其余 4 个名字都在用） |
| 10 | `app/retrieval/assembly.py:113-117` | 函数内 import 的 `DEFAULT_TABLE_PATH`（用的是 `settings.synonym_table_path`） |
| 11 | `app/memory/long_term.py:20-25` | `LongTermMemoryError`（从 memory_schema 导入后未用） |
| 12 | `app/memory/memory_schema.py:19` | `from typing import Any` |
| 13 | `app/models/mineru_api.py:22,24` | `BytesTransport`、`Transport` |
| 14 | `evaluation/compare_window.py:14` | `import os` |
| 15 | `evaluation/faithfulness.py:19` | `import os` |
| 16 | `scripts/e2e/fetch_pdf_samples.py:24` | `import sys` |
| 17 | `scripts/migrations/backfill_law_dates.py:43,44` | `text`（未用）、`Session`（仅出现在 `# type:` 注释里） |

**C. 测试文件内的未使用 import（12 处，噪音级）**
`test_chat_service.py`/`test_cli_contract.py`/`test_guard.py`/`test_guardrail_graded.py`/`test_retrieval_fusion.py`: `pytest`；
`test_chat_stream_service.py`/`test_guard_zero_citation_tier.py`: `REFUSAL_ANSWER`、`CITATION_FALLBACK_NOTICE`；
`test_import_incremental.py`: `Path`、`Session`；`test_import_mysql_cli.py`: `json`；`test_keyword_search_filters.py`: `KeywordSearcher`；
`test_mailer_selection.py`: `field`；`test_mysql_import_service.py`: `SQLAlchemyError`；`test_review_service.py`: `datetime`；
`test_session_store.py`: `SessionUser`；`demo_retrieval_pipeline.py`: `RetrievalResult`、`RetrievalService`

### 3.3 🟡 看着像多余、实为有意保留（**不要删**）

| 位置 | 形态 | 为什么不能删 |
|---|---|---|
| `app/cli/import_mysql.py:6`、`index_legal_documents.py`、`summarize_chunks.py` | `from app.db import sql_models  # noqa: F401` | **副作用 import**：导入模型以注册 `Base.metadata`（注释已说明） |
| `app/db/sql_models.py`（整文件） | 12 个名字"未使用" | **聚合入口**，供其他模块统一 `from app.db.sql_models import X`；文件头注释已声明"不定义表、不写逻辑" |
| `app/chat/citation_check.py:29` | 9 个名字转发 | 代码内注释：`# noqa: F401 —— 公共名转发，保持既有 import 路径`（拆分前的兼容） |
| `evaluation/run_eval.py:53-75` | 6 个名字 | 代码内注释：**"以上名字在本模块是'再导出'（re-export），不要删"**，且有 6 个脚本确实这样导入 |
| `app/auth/service.py:25` | `SqlAuthStore` | 用于**字符串类型注解** `"InMemoryAuthStore | SqlAuthStore | None"`；运行时不求值，属"可移入 `TYPE_CHECKING`"而非死代码 |

### 3.4 🟢 扫描器必然误报：框架注册函数（14 个，非死代码）

My 的零引用扫描会把这些列出来，但它们是**装饰器注册**、函数名不被显式调用：

| 文件 | 函数 | 装饰器 |
|---|---|---|
| `api/legal_search.py:83` | `search_legal_documents` | `@router.post('/search')` |
| `api/memory.py:115,149` | `delete_memory`、`update_memory_settings` | `@router.delete` / `@router.put` |
| `api/review.py:117,145` | `get_document_review_detail`、`review_document_by_id` | `@router.get` / `@router.post` |
| `api/session_routes.py:62,160` | `create_session`、`list_session_messages` | `@router.post('')` / `@router.get` |
| `api/users_me.py:30` | `get_current_user_info` | `@router.get('/api/v1/users/me')` |
| `auth/router.py:53,68` | `send_register_code`、`send_reset_code` | `@router.post` |
| `main.py:83,93,108` | `startup_event`、两个 `exception_handler` | `@app.on_event` / `@app.exception_handler` |
| `main.py:144,156` | `live_health`、`ready_health` | `@app.get('/health/...')` |

### 3.5 🟢 重复实现（跨文件同名，29 组）

**真重复、建议合并**（复制粘贴痕迹明显）：

| 组 | 位置 | 说明 |
|---|---|---|
| HTTP 重试的 6 组同名测试 | `test_embedding_retry.py` ↔ `test_http_retry.py` | `test_retry_after_timeout_succeeds` 等 6 个函数两处各一份（后者 403 行已覆盖通用工具） |
| 状态词表的 3 组同名测试 | `test_law_status.py` ↔ `test_review_status.py` | `test_status_literals_match_database_vocabulary` 等 3 个 |
| MinerU/Qwen-VL 的 2 组 | `test_mineru_client.py` ↔ `test_qwen_vl_client.py` | `test_429_retried_then_succeeds` 等 |
| `render_markdown` | `evaluation/refusal_report.py:18` ↔ `evaluation/render_report.py:27` | 两份渲染实现（拒答报告 / 评测报告），**建议人工确认是否有意分家** |

**可接受的一次性重复**（迁移/验收脚本各自独立更安全）：
`column_exists`×3（迁移脚本）、`connect`/`fetch_distribution`/`print_distribution`×2、
`build_url`×2（两个 restore 脚本）、`line`/`resolve_pdf`×2~3（e2e PDF 脚本）、`run_cli`×3。

### 3.6 ✅ 检查后确认无冗余

| 检查项 | 结果 |
|---|---|
| 配置键是否有定义未用 | `backend/.env.example` 等 5 个文件共 **63 个键，0 个未被代码读取** |
| 前端是否有孤儿文件 | 25 个 ts/tsx，**0 个孤儿**（`app/` 下 page/layout/route 由文件系统路由生效） |
| 是否有孤儿模块（无人 import） | 247 个 py，**0 个** |
| 注释掉的代码块 | 仅 **4 处**（连续 ≥3 行），量级可忽略 |
| TODO/FIXME/临时标记 | 仅 **2 处** |

---

## 四、其他发现（顺带登记）

1. `evaluation/faithfulness.py:128` 默认 `reports/latest_eval.json` **不存在** → 不带 `--eval-json` 跑必失败（上一轮已登记）
2. `docs/目录与命名约定.md` 被 `evaluation/run_config.py` 注释引用 → 说明项目有命名规范文档，本次整理未违背
3. `app/ingest/pdf_parser.py` 的 `.py` 文件里没有"无 docstring"的解析风险；全部 247 个文件 AST 解析**无一语法错误**

## 五、边界

- 本文档只做静态取证，**未改动、未删除任何代码/配置/数据**
- 3.2 的 28 处"可删"项均为**静态零引用证据**；其中 `MINERU_MAX_PAGES`（#3）建议按"补实现"而非"删常量"处理
- 冻结令仍生效：这些改动待部署验收完成后统一处理
