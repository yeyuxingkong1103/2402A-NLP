# 批次 15 验收报告

日期：2026-09-20　分支：refactor/optimize　解释器：C:/Users/92842/anaconda3/python.exe

## 任务 1：清理 e2e 测试数据

- 脚本：项目根 `cleanup_e2e_probe_accounts.py`（未进 backend/）
- 删除条件（写死）：`password_hash` 非 48 字节 urlsafe-base64（16 字节盐 + 32 字节 PBKDF2 摘要）OR `email LIKE 'e2e%@%'`
- 删除顺序：Milvus 记忆实体（`client.delete(filter=user_id==...)`，未 drop 集合）→ chat_messages → chat_sessions → users 行

### 删前清单（8 行，全部命中死条件）

| id | email | 命中原因 | 记忆 | 会话 | 消息 |
|----|-------|---------|-----|------|-----|
| 2 | e2e_auth_persist_a@qq.com | email | 0 | 1 | 2 |
| 3 | e2e_auth_persist_b@qq.com | email | 0 | 0 | 0 |
| 4 | e2e_auth_persist_a_08344844@qq.com | email | 0 | 1 | 2 |
| 5 | e2e_auth_persist_b_08344844@qq.com | email | 0 | 0 | 0 |
| 7 | e2e_auth_persist_a_f42a70b5@qq.com | email | 0 | 1 | 2 |
| 8 | e2e_auth_persist_b_f42a70b5@qq.com | email | 0 | 0 | 0 |
| 20 | e2e-mem-a@test.local | hash无效+email | 14 | 0 | 0 |
| 21 | e2e-mem-b@test.local | hash无效+email | 6 | 0 | 0 |

### 删后行数

- users：8 → 0；chat_sessions：3 → 0；chat_messages：6 → 0
- Milvus legal_long_term_memory：20 → 0（强一致性 query 复核；首次 delete 后 query 仍见 6 条为删除可见性窗口，已复核清零）
- MySQL 实例仅有 legal_rag 一个业务库（另两个为系统库）

### ⚠️ 必须如实报告的事实

删前 users 表总共只有 8 行且**全部**命中死条件——本库（.env 指向 127.0.0.1:3306 legal_rag，即批次 14 e2e 所用同一库）里**不存在"6 个真实用户"**，故无法出具"6 个真实用户数据一条未动"的核验结论。
已删 8 个账号的 email 全部以 e2e 开头（6 个来自 e2e_auth_persist 系列脚本、2 个来自批次 14 记忆 e2e），未删除任何 email 不带 e2e 前缀的账号。
若真实用户在 AutoDL（10.224.10.185）的 MySQL 上，本次操作未触达该实例，需用户确认 .env 指向后再核。

## 任务 2：embedding 通用重试

改动文件：
- `backend/app/models/embedding.py`：`EmbeddingApiError` 增加 `status_code`；新增 `_is_retryable`（超时/429/5xx 可重试）与 `_embed_with_retry`（指数退避 0.5→1→2s，最多 3 次重试，总预算 15s，每次重试 WARNING 含 request_id/次数/错误类型，重试后成功记 INFO）；`_request` 的 HTTPError 保留状态码、URLError 超时归一为 TimeoutError。调用方签名未变。
- `.env`：新增 EMBEDDING_RETRY_ATTEMPTS=3 / EMBEDDING_RETRY_BACKOFF_SECONDS=0.5 / EMBEDDING_RETRY_TOTAL_BUDGET_SECONDS=15
- 新测试：`backend/tests/test_embedding_retry.py`（9 条）

### 验收

- a) 单测 9 passed：超时重试成功 / 429 重试成功 / 400 不重试（transport 调 1 次）/ 超次抛（调 4 次）/ 预算耗尽即停 / 退避间隔 [0.5, 1.0] / .env 可配 / URLError 超时归一
- b) 真实调用：1024 维向量返回正常，0.36s，无行为变化
- c) 全量：**400 passed**（391 → 400，不减反增 9）

## 边界遵守

- docs/ 未动；无新增依赖；临时脚本在项目根；Milvus 用 delete 接口删实体未 drop 集合
