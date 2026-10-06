# 接口测试集合使用说明

`legal_rag_collection.json`：Postman Collection v2.1 格式，Postman 与 ApiPost
均可直接导入，覆盖当前代码**全部 21 个端点**（认证 6、用户 1、问答流式 1、
会话与消息 4、记忆 3、管理员审核 3、法律检索 1、健康检查 2）。

## 导入

| 工具 | 操作 |
|---|---|
| Postman | Import → File → 选 `legal_rag_collection.json` |
| ApiPost | 导入数据 → Postman → 选本文件 |

## 使用步骤

1. **改地址**：集合变量 `base_url` 默认 `http://127.0.0.1:8000`，按后端实际地址改。
2. **准备账号**：集合变量 `test_email` / `test_password` 填测试账号；
   新账号先跑「3.1 发送注册验证码」（**注册验证码任何环境都不回显**，从邮箱取；
   只有「3.4 密码重置验证码」在 development 环境会回显 `debug_code`），
   验证码填进 `verify_code` 后跑「3.2 注册」。集成测试不由本集合驱动。
3. **拿 token**：跑「3.3 用户登录」，测试脚本自动把 `access_token`
   写进集合变量 `token`，之后所有受保护请求自动带 `Authorization: Bearer {{token}}`。
4. **管理员接口**：`GET /api/v1/admin/documents` 等需管理员账号，
   先执行 `cd backend && python -m app.cli.create_admin --email <测试邮箱>`。
5. **动态 ID**：7.1 创建会话的脚本自动写 `session_id`；
   `document_id`（审核接口）与 `memory_id`（记忆接口）从对应列表接口的
   响应里取，手工填进集合变量。

## 集合变量清单

| 变量 | 含义 | 谁来写 |
|---|---|---|
| `base_url` | 后端地址 | 手工改 |
| `token` | 登录令牌 | 3.3 登录的测试脚本自动写 |
| `verify_code` | 注册验证码 | 手工（从邮箱取；接口不回显。集成测试走 `ENVIRONMENT=test` 固定码旁路） |
| `reset_code` | 重置验证码 | 3.4 的测试脚本自动写（development） |
| `session_id` | 会话 ID | 7.1 创建会话的测试脚本自动写 |
| `document_id` / `memory_id` | 审核/记忆对象 ID | 手工（列表接口响应里取） |
| `test_email` / `test_password` | 测试账号 | 手工改 |

## 期望响应与错误码

每个请求的 description 里写了：期望响应示例、业务错误码与触发条件，
全部照 `docs/接口文档.md` 的约定（通用错误码 40000/40001/40100/40300/
40900/42900/50000/50001/50002/50003），没有自创字段。

## 文档有、代码无（导入后找不到对应请求是正常的）

角色接口 4.1~4.4、知识库 5.1~5.5（仅 5.6 检索已实现）、采集任务 6.1/6.2、
非流式问答 7.3、反馈接口 9 —— 当前代码未实现，差异已同步在集合 description
与 README 里，等 docs 侧统一处置。
