# 测试报告

**日期：** 2026-09-27
**范围：** 后端单元/API/集成测试、前端 Playwright E2E、真实后端浏览器冒烟、语法与编译检查。

## 后端回归

```text
python -m pytest backend/tests -q
183 passed, 2 skipped
```

说明：

- 默认跳过需要真实 GPU、本地模型或外部依赖的测试。
- `Starlette/httpx TestClient` 弃用提示已通过 pytest 精确过滤，不再污染回归输出。

## 后端编译检查

```text
python -m compileall -q backend
通过
```

## 前端浏览器 E2E

```text
npm run test:e2e:chromium
6 passed, 2 skipped
```

覆盖范围：

- 登录页模拟手机号验证码登录。
- 聊天页发送咨询、重新生成回答和提交反馈。
- 知识库审核员审核材料且无发布权限。
- 超级管理员发布已审核材料。
- 个人中心展示记忆、关闭记忆提取、下载加密导出并独立领取一次性口令。
- 个人中心注销账号并跳转登录页。

## 真实后端浏览器冒烟

```text
npm run test:e2e:real
1 passed
```

执行方式：

- 本地启动 `uvicorn backend.app.main:app --host 127.0.0.1 --port 8010`。
- 使用 `ENVIRONMENT=development`、`DEV_AUTH_BYPASS=true`、内存存储和关闭真实 Milvus/DeepSeek 依赖的安全冒烟配置。
- Playwright 访问真实 FastAPI 同源托管前端，并调用真实聊天 API。

覆盖范围：

- 高风险家暴咨询。
- 真实 API 鉴权开发身份头。
- 紧急指引状态 `emergency_guidance`。
- 页面展示 `110` 和 `12348`。

## 真实知识库 RAG 浏览器回归

```text
npm run test:e2e:real:rag
1 passed
```

执行方式：

- 本地 MySQL `myrag` 已迁移到 `011_expand_chat_external_ids`。
- Redis 使用 `redis://127.0.0.1:6380/0`，Milvus 使用 `http://localhost:19530` 的 `legal_material_chunks` 集合。
- 后端使用 SQL 认证存储、Redis OTP/请求控制、本地 BGE 模型、DeepSeek 客户端和同源托管前端。

覆盖范围：

- 真实浏览器页面提交婚姻家庭咨询。
- FastAPI 聊天 API 写入 SQL 会话和消息。
- Milvus 正式知识库集合检索、Reranker 重排和引用展示。
- 页面出现回答文本和引用区域，且无前端错误。

## 生产依赖就绪检查

```text
run_startup_checks(settings)
mysql: ok
redis: ok
milvus: ok
celery: ok
 deepseek: ok
bge_m3: ok
bge_reranker: ok
ready=True
```

补充验证：

- Redis 两个独立 `RedisRequestController` 共享账号并发限制，第二个控制器收到 `account_concurrency_limit`；释放后继续受到分钟限流保护。
- Milvus `legal_material_chunks` 集合存在，统计为 `13,294` 条向量。
- 本机 MySQL `mysqldump --no-data` 结构备份演练成功，临时备份文件已删除。
- Celery `solo` worker 启动后 `check_celery_readiness` 返回 `ok=True`。
- SQLite 从空库 `upgrade head` 到 `011_expand_chat_external_ids`，再 `downgrade base`，完整通过。

## 灾备恢复演练

- MySQL 完整 dump 文件生成成功，大小约 `123 MB`；原业务库仍保持 `011_expand_chat_external_ids` 和 `13,294` 个 chunks。
- 首次使用 Windows `mysql` CLI 导入时，法律正文特殊反斜杠转义导致 `crawl_snapshots` 数据行失败；改用 `mysqldump --skip-extended-insert --hex-blob` 生成 dump，并使用 MySQL Shell `mysqlsh --sql` 恢复后通过。
- 独立临时恢复库核对结果：`knowledge_materials=7`、`document_chunks=13294`、`conversations=2`；临时数据库和 dump 文件已确认删除。

```text
node --check frontend/tests/run-real-e2e.mjs
node --check frontend/js/api/user-api.js
node --check frontend/js/pages/profile.js
通过
```

## 仍需补充

- 真实知识库 RAG 浏览器回归已通过单场景验证；生产上线前仍需扩展多场景样本。
- 生产环境仍需执行 Redis/Celery/Milvus 多实例压测、灾备恢复重演和正式安全审查。
