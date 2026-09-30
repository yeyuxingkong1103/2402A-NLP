# 批次 16-B 验收报告（前端管理端 + 联网测试分离）

日期：2026-09-20　环境：Windows / Anaconda Python / 本地 MySQL+Redis+Milvus / SiliconFlow Embedding

## 任务 0：分离联网测试

- `backend/tests/test_retrieval_factory_smoke.py` 打 `@pytest.mark.real_api` 标记，并加"失败即重试一次、重试仍失败必报错"的容错（每次失败打印 `[real_api] 冒烟第 N 次失败：<错误>` 证据）。
- `backend/pytest.ini` 注册标记；`docs/测试文档.md` 7.0 补充真联网用例说明（用户点名要求的唯一 docs 改动）。
- 验收命令结果：
  - `pytest tests -q -m "not real_api"` → **412 passed, 1 deselected, 5 warnings in 11.86s**
  - `pytest tests -q`（全量）→ **413 passed, 5 warnings in 13.37s**

## 16-B 改动清单

| 文件 | 内容 |
| --- | --- |
| `frontend/src/app/admin/documents/page.tsx` 新增 | 审核列表页：状态筛选（待审核/已通过/已驳回/全部）+ 分页（20/页）；每行法规名/文书类型/版本号/提交时间/状态/审核留痕/操作；右侧详情侧栏（6.5 元数据 + 分块预览 + 留痕）；审核操作内联表单（通过选填意见、驳回必填）；403 显示"无权限访问管理端"+返回问答入口 |
| `frontend/src/lib/api-admin.ts` 新增 | 6.3 列表 / 6.5 详情 / 6.4 提交决定三个接口封装 |
| `frontend/src/lib/types.ts` | 追加 3.8/6.3/6.5/6.4 契约类型（字段名与接口文档一致） |
| `frontend/src/lib/api-auth.ts` | 新增 `fetchCurrentUser()`（GET /users/me，3.8） |
| `frontend/src/components/app-header.tsx` | 登录后调 /users/me，`is_admin=true` 才显示"审核"导航入口 |
| 零新增前端依赖；复用 primitives（Button/Notice/Tag）与 ink/seal 色系 | |

## 验收证据（真实链路：uvicorn 8123 + next start 3210，真实注册账号登录，无后门）

### a) 备份 → 置 pending → 页面可见 ✅
- `b16b_prep.py`：11 行完整状态备份至 `reports/b16b_status_backup.json`；dv.id=161（最高法发布劳动争议典型案例）、dv.id=162（最高法发布劳动争议司法解释（二）和典型案例）置 pending_review。状态统计 approved 9 / pending_review 2。
- 页面列表（待审核筛选）显示两条：法规名 + 司法解释/案例材料 + v1 + 待审核 + 提交时间 + 操作按钮。

### b) 页面点"通过" → 检索能查到 ✅
- 页面填意见"页面验收：已核对官方来源链接与文书类型，准予发布"→ 确认通过 → 成功提示"**审核通过：ver-906e4d174cb1f81c2ac91f68，索引 0 条**"（向量本已在库，重新校验通过）→ 待审核列表即时刷新只剩 1 条。
- 检索验证（POST /api/v1/legal/search，query=劳动争议司法解释二 未订立书面劳动合同 二倍工资）：命中 10 条，**top3 全部为"最高法发布劳动争议司法解释（二）和典型案例"**（doc-de9bc3a0496114ef111e8110）。

### c) 另一条驳回并填意见 → 检索查不到 ✅
- 空意见直接点"确认驳回" → 前端拦截显示"**驳回必须填写审核意见（留痕要求）**"，未发请求。
- 填意见"页面验收驳回：条文切分与官方发布版不一致，需重新采集核对"→ 确认 → 提示"**已驳回：ver-43b21acf5cde0c7c5b73c307，清理向量 69 条**"；"已驳回"筛选下行内显示"审核于 2026/9/20 14:11:32 · 意见：页面验收驳回：…"。
- 检索验证（query=最高法发布劳动争议典型案例 典型案例 用人单位）：命中 6 条全部来自其他文档，**被驳回版本 0 命中**。

### d) 非管理员访问 → 明确无权限提示 ✅
- legaluser@qq.com 登录：导航**无"审核"入口**（is_admin=false 不显示）。
- 直接访问 /admin/documents：页面显示"**无权限访问管理端**" + "文档审核仅对管理员开放。如需权限，请联系系统管理员为当前账号授权。" + "返回问答"链接（后端 403/40300 正确展示，不白屏）。

### e) 收尾恢复 ✅
- `b16b_restore.py`：按备份逐行还原 version_status/processing_status/审核留痕与 documents.current_version_id；被驳回版本（69 条向量被清理）按生产同一实现 `index_awaiting_embeddings` 重建（indexed_chunks=69）。
- 恢复后状态统计：**approved: 11（共 11 行），全部 indexed**；Milvus 生效集合向量总数 1622。

### f) 测试与构建 ✅
- 后端全量：`pytest tests -q` → **413 passed, 5 warnings in 12.59s**（含 real_api 冒烟）。
- 前端构建：`npm run build` → `✓ Compiled successfully`，路由表含 `/admin/documents 6.45 kB`。

## 排查备注（不影响交付）

验收过程中发现 agent-browser 自动化环境两个现象，与被测系统无关：①daemon 在跨命令间会重置 in-memory 浏览器上下文（localStorage 丢失被误判为"会话失效"，后端 Redis 会话 TTL 实测正常）；②曾出现一次 Edit 报成功但源文件未落盘导致入口未渲染（已修复并重新构建验证）。真实浏览器使用不受影响。
