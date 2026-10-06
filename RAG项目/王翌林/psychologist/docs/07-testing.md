# 07 — 测试策略与报告模板

> 测试类型：单元测试（pytest）、接口测试（Swagger/curl/Postman）、集成测试（pytest + TestClient）、RAG 评测（RAGAS 口径）、压力测试（JMeter 5.6.3）。

---

## 1. 运行方式

```bash
cd /home/dabaie/code/psychologist

# 全量（需 MySQL 可用；Redis/Milvus/管理员不可用时对应用例自动 skip）
PYTHONPATH=/home/dabaie/code/psychologist \
  /home/dabaie/code/my_project/.venv/bin/pytest -v

# 仅纯函数单元测试（无外部依赖）
PYTHONPATH=$(pwd) /home/dabaie/code/my_project/.venv/bin/pytest -v tests/test_unit_chunker.py

# 语法自检
/home/dabaie/code/my_project/.venv/bin/python -m compileall src scripts
```

测试基础设施（`tests/conftest.py`）：

- TestClient 直接基于 `src.main:app`，**不进入 lifespan**（避免预热 BGE-M3/Reranker）；
- **严禁调用真实大模型**：session 级 monkeypatch 屏蔽 `llm_service.chat / chat_stream / simple_complete`；
- 外部服务探测 fixture：MySQL/Redis/Milvus 不可用时对应集成用例 `pytest.skip`；
- 每个用例独立注册临时用户（`pytest_` 前缀）并真实登录取 token。

## 2. 当前用例清单（174 项，全部通过 @ 2026-09-16，阶段 3/4 更新）

| 文件 | 用例数 | 覆盖点 |
| :--- | :--- | :--- |
| `tests/test_unit_chunker.py` | 18 | `estimate_tokens`（空/CJK/ASCII/混合/单调性）、五种分块策略、空输入、未知策略回退、fixed 滑窗重叠、句子边界、段落保留、标题切分、semantic、parent_child 链接、`summarize_chunk` |
| `tests/test_unit_crisis.py` | 17 | 危机词检测（自杀/自残/伤人等）、弱风险正则、转介文案、敏感词脱敏、`ensure_crisis_notice` 兜底、`post_process` 清理与"不诊断/不开药"正则 |
| `tests/test_unit_parser.py` | 16 | `clean_text`（水印行/不可见字符/全角归一）、段落去重、重复行移除、txt/md/GBK 编码解析、目录解析、不支持类型与文件不存在异常、字符数统计 |
| `tests/test_unit_prompt.py` | 14 | 三角色语气差异、提示词互异、安全边界存在、知识片段注入、空命中提示、短/长期记忆注入、危机注入开关、占位符模板填充、context/轮数上限、默认 prompt 查找 |
| `tests/test_unit_security.py` | 12 | bcrypt 哈希与校验、加盐、错密码、非法 hash、access/refresh 签发与区分、有效期、过期/换密钥/篡改/畸形 token 拒绝 |
| `tests/test_unit_personas.py` | 8 | 种子含三角色、开场白与提示词互异、各角色字段完整性、prompt 与 prompt 模块一致、KNOWLEDGE_DIRS 覆盖、sys_roles、数据库角色与种子一致且仅三个 |
| `tests/test_integration_auth.py` | 10 | 注册/登录/资料/改密/刷新 token 流程、access 冒充 refresh 返回 401、登出与审计日志、登录日志、越权、禁用用户登录拒绝、重复注册 409、限流偏好接口 |
| `tests/test_unit_config.py` | 8 | .env 关键值（3307/6379/19530/collection/1024 维/模型路径）、CORS 与紧急电话解析、`validate_security` 通过与占位/弱 JWT、缺 ADMIN_PASSWORD 三种拒绝 |
| `tests/test_unit_memory.py` | 5 | 短期记忆 append/get/clear/轮次、MySQL 回填写回缓存、摘要规则兜底、空消息 |
| `tests/test_unit_rag.py` | 12 | embedder/reranker 单例与懒加载契约、retrieve 混合→稠密降级、空候选、改写开关与空串回退、pipeline 三条转发 |
| `tests/test_integration_redis_memory.py` | 4 | 真实 Redis 写读清、TTL 存在、limit 截取保留最近 N 条、清空后 MySQL 回填 |
| `tests/test_integration_milvus.py` | 5 | 真实插入/稠密检索（同向量>0.99）/混合检索（BM25+RRF）/计数/按 doc 删除、长期记忆写入与检索 |
| `tests/test_integration_models.py` | 6 | BGE-M3 1024 维、归一化、批量、语义区分（焦虑~失眠 > 焦虑~股市）、Reranker 相关片段排第一、分数 ∈(0,1) |
| `tests/test_integration_endpoints.py` | 8 | `/personas` 列表/详情/404、`/conversations` 建/列表/过滤/消息/删除 404、`/users/me` 200/401 |
| `tests/test_integration_chat.py` / `test_integration_knowledge.py` | 8 | 非流式问答+落库、SSE 事件流、危机注入、跨用户会话 403、删除会话 404、知识上传→列表→删除、越权上传 403、删除不存在 404 |

参数化用例展开后共 **174 项**（阶段 3 新增 config/memory/rag 单元与 Redis/Milvus/BGE 集成、端点用例；阶段 4 保持全绿）。运行结果记入第 6 节报告模板。

## 3. 接口测试（手工/Postman/Apifox）

1. 启动服务后打开 `http://127.0.0.1:8000/docs`，按模块逐接口执行；
2. 冒烟流程：注册 → 登录 → 角色列表 → 新建会话 → `/chat` → `/chat/stream` → 历史消息 → 删除会话；
3. 边界用例：超长消息（>4000 拒绝）、他人会话（403）、下架角色聊天（429）、错误密码（401）、重复用户名（409）、无 token 访问（401）。

## 4. RAG 评测（RAGAS 口径）

- 数据集：`data/eval/ragas_dataset.jsonl`，每行 `{"persona_code": "...", "question": "...", "ground_truth": "可选"}`；
- 指标：Faithfulness / Answer Relevancy / Context Precision / Context Recall（0~1，越高越好）；
- 引擎：优先 `ragas`，当前环境与 langchain-community 冲突时自动回退 `builtin_llm_judge`（DeepSeek 判分，口径一致），响应中 `engine` 字段标识实际引擎；
- 执行：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/eval/ragas \
  -H "Authorization: Bearer <admin_token>" -H "Content-Type: application/json" \
  -d '{"persona_id": 1, "limit": 10}'
# 报告输出：data/eval/reports/ragas_report_persona{id}_{时间戳}.json
```

- 建议每角色 ≥20 条样本；`context_recall` 需数据集带 `ground_truth`。

## 5. 压力测试（JMeter 5.6.3，Windows）

工具路径：`D:\JMeter\apache-jmeter-5.6.3\apache-jmeter-5.6.3\bin`（Windows 侧，`jmeter.bat`）。

建议方案：

| 场景 | 线程组 | 说明 |
| :--- | :--- | :--- |
| 登录吞吐 | 50 并发 × 100 次 | 观察 bcrypt 与 MySQL 连接池 |
| 聊天非流式 | 50 并发 × 100 次，间隔 2s | 观察 `/chat` P95 延迟与限流 429 |
| SSE 流式 | 20 并发 | 观察首 token 延迟（目标 < 3s） |
| 知识库检索 | 50 并发 | 观察检索延迟（目标 < 500ms）+ 重排延迟（< 800ms） |

注意：压测前确认 `RATE_LIMIT_PER_MINUTE` 与压测量匹配，避免误判；JMeter 需配 HTTP Header Manager（Bearer token）与 JSON 断言（`code=0`）。

## 6. 测试报告模板

```markdown
# 测试报告

- 项目：基于 RAG 的心理医生多角色陪伴系统
- 版本：app_version=__　分支/提交：__
- 测试人：__　日期：__-__-__
- 环境：Ubuntu 22.04 @ WSL2　Python 3.10.12　MySQL 3307 / Redis 6379 / Milvus 19530

## 1. 单元/集成测试

- 命令：`PYTHONPATH=$(pwd) .venv/bin/pytest -v`
- 结果：__ passed / __ failed / __ skipped（耗时 __s）
- 失败用例与原因：__（无则填"无"）

## 2. 接口冒烟

| 用例 | 预期 | 实际 | 结论 |
| :--- | :--- | :--- | :--- |
| /health 三库健康 | 全 true | | |
| 注册→登录→取资料 | code=0 | | |
| 三角色列表 | 3 条 status=1 | | |
| /chat 正常回复 | answer 非空、references 可溯源 | | |
| /chat/stream SSE | meta→delta*→done | | |
| 危机词输入 | crisis_detected=true 且附转介提示 | | |
| 越权访问他人会话 | 403 | | |
| 未登录访问 /chat | 401 | | |

## 3. RAG 评测（每角色）

| persona | samples | faithfulness | answer_relevancy | context_precision | context_recall | engine |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| humanistic_lin | | | | | | |
| cbt_chen | | | | | | |
| mindfulness_zhou | | | | | | |

## 4. 压测

| 场景 | 并发 | 总请求 | 错误率 | P50 | P95 | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| /chat | | | | | | |
| /chat/stream | | | | | | |

## 5. 问题与风险

1. __

## 6. 结论

- [ ] 达到验收标准（需求文档 11.3），可以发布
```
