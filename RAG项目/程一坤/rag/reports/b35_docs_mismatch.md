# 批次 35：docs/接口文档.md 与当前代码不一致清单

核对基准：`backend/app` 路由实际注册（2026-09-22 逐文件核对）。
当前代码实际端点共 **21 个**（认证 6、用户 2、问答流式 1、会话 4、记忆 3、
管理员审核 3、法律检索 1、健康检查 2 中 users/me 与 memory-settings 计入用户）。

## 一、文档有、代码无（集合与 README 已按实际口径处理）

| # | 文档章节 | 文档端点 | 现状 | 建议 |
|---|---|---|---|---|
| 1 | §4.1~4.4 角色接口 | POST/GET/PUT/DELETE `/api/v1/roles...` | 无任何 roles 路由 | 文档标注"二期规划，未实现"，或删除 |
| 2 | §5.1 创建知识库 | POST 知识库 | 未实现 | 同上 |
| 3 | §5.2 上传文档 | 上传接口 | 未实现（数据走离线导入 import_mysql） | 同上 |
| 4 | §5.3/5.4/5.5 文档状态/重建索引/删除 | 知识库文档管理 | 未实现（重建索引只有 CLI） | 同上；5.4 可指向 `python -m app.cli.index_legal_documents` |
| 5 | §6.1/6.2 采集任务触发/查询 | 采集任务 | 未实现（采集为 CLI/脚本触发） | 同上 |
| 6 | §7.3 非流式问答 | POST `/api/v1/chat/completions` | 未实现（文档自己已注明） | 保留"后续扩展"标注即可 |
| 7 | §9 反馈接口 | POST `/api/v1/messages/{message_id}/feedback` | 未实现 | 标注"未实现" |

## 二、文档有、代码也有，但形状有出入

| # | 位置 | 差异 | 建议 |
|---|---|---|---|
| 8 | §3.1/3.4 请求体 | 文档只写 `{"email": ...}`；**代码 CodeRequest 同样只收 email** ✓ 一致；但 tests 里有带 `purpose` 字段的用法（CodeRequest 忽略未知字段） | 无需改；如担心误导读文档的人，可注明"purpose 字段无效" |
| 9 | §3.2/3.5 请求体字段名 | 文档 3.5 重置密码写 `email/password/code` ✓ 与代码 `ResetPasswordRequest(RegisterRequest)` 一致 | 无 |
| 10 | §3.7 注销响应 | 文档 message 是 "success"；代码实际 message="注销成功"（data.status="logged_out" 一致） | 文档示例的 message 改 "注销成功"（或以实际为准核对） |
| 11 | §7.1 创建会话 | 文档写 character_id 固定 `character_001`；**代码默认值是 `legal-assistant`**（ChatStreamRequest 同） | 文档统一改成 `legal-assistant` |
| 12 | §7.1 响应 | 文档示例无 status 字段 ✓；代码 data 含 session_id/character_id/title/created_at ✓；注意返回码 **201** 文档未写明 | 文档补 201 |
| 13 | §7.2 options.document_types | 文档示例值 `["law","judicial_interpretation","case"]` ✓；代码枚举 law/administrative_regulation/judicial_interpretation/case ✓ 一致 | 无 |
| 14 | §7.5 会话列表响应 | 文档 data 里有 `total/page/page_size`；**代码只返回 `{"items": [...]}`，无 total/page/page_size** | 二选一：文档删掉 total 三键，或代码补（改代码需走正常批次流程） |
| 15 | §7.6 消息历史 model 示例值 | 文档写 `"model": "deepseek-v4-flash"`；库中实际落的是配置值（当前 `deepseek-flash`） | 文档示例改为不带版本号的示意值 |
| 16 | §6.4 审核发布 decision 取值 | 文档未写清取值枚举；代码 decision 为自由字符串（1~16 字），approve/reject 语义由服务层处理 | 文档补 decision 取值说明与示例（批次 35 集合按 approve/reject 写） |
| 17 | §5.6 legal/search | 文档约束与代码一致（kb_labor_law_001 / labor_law / 四类 document_types）✓；但文档未写明 **user_id 字段被忽略**（身份只取认证上下文） | 文档补一句"身份只取令牌" |

## 三、批次 35 交付物对该口径的处置

- `scripts/api-collection/legal_rag_collection.json`：只收录代码真实存在的 21 个端点；
  未实现接口写在集合 description 的"文档与代码不一致"段。
- `README.md` 常用命令全部经 `--help` 或真实执行核对（含
  `index_legal_documents --recreate-collection --prune-orphans` 实际存在）。
- `scripts/loadtest/first_token_latency.py` 已在 dev 后端（8126）真实跑通：
  首字 5398ms / 完整 7120ms（1 次，远程 LLM 正常量级）。
