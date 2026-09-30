# 批次 38 验收报告：案例文档清洗重切 + 集成测试验证码旁路

日期：2026-09-23　　范围：两项裁决的落地与验收

---

## 一、裁决 1：《最高法发布劳动争议典型案例》（document id=1）清洗重切

### 1.1 诊断证据（先取证，未动手）

| 项 | 实测值 |
| --- | --- |
| 原始文件 | `data/labor_law_raw/8fefe64e…247.html`（16810 字符，court.gov.cn，TRS CMS） |
| 正文区容器 | `<div class="txt_txt" id="zoom">`（站点头条正文区；页头/页脚都在容器外） |
| 修复前解析输出 | 6605 字符，混入 13 行页面样板 |
| 库内 `cleaned_content` | 6602 字符（MySQL 长度 19004 = 字节数，utf8mb4 中文 3 字节，勿与字符数混淆） |
| 修复前分块 | 1 parent（整页，含样板）+ 68 child（`child[0]` 是「所在位置：」） |

**样板来源 = 整页未定位正文**（非单一页脚问题）：

- 页头面包屑：`所在位置：`、`字号：`
- 页脚：`总机：67550114`、`举报电话：67556131`、
  `中华人民共和国最高人民法院 版权所有`、`11040102700145号`、
  `京ICP备05023036号`、32 位站点统计校验串 `f68e0d4714…`

解析器 `_HtmlTextExtractor` 只忽略语义标签（`head/script/style/nav/footer/header/aside`），
该站点页头页脚全是普通 `div` → 全部落入正文。既有 `cleaner` 的导航规则也拦不住：
这些行要么 ≥6 字符、要么含标点、要么不以已知导航前缀开头。

单 parent 的来源是二层原因：案例文本没有「第 X 条」条文边界 →
`article_splitter._split_articles` 返回空 → `chunker.py` 走「全文单 parent」兜底。

### 1.2 修复（两处，均只动清洗规则）

| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `app/ingest/parser.py` | 新增主内容容器提取：命中 `id/class` 含 `zoom` 时只收容器内文本；容器内容 <100 字符或未闭合则回退全页 | 只认 `zoom` 一个标记，不影响其他站点行为 |
| `app/ingest/cleaner.py` | 新增 `_is_site_boilerplate_line()`：版权声明 / `ICP备…号` / 公安备案 / 长纯数字号 / 长十六进制校验串 / `所在位置`、`字号`、`总机`、`举报电话` 等前缀行 | 结构标题白名单优先，`第一章/总则/第五条` 不受影响 |

备份：`%TEMP%/b38_bak/{cleaner.py,parser.py}`（md5 `b6bd8b045f1ad18c33f88c8a16ca8484` /
`b6cb640a4fdd20279330623278a4946d`）。原始 raw 文件**未删未改**。

### 1.3 重跑链路

1. 单文档重建数据包（`parse → chunk`，临时目录 `%TEMP%/b38_out/`）；
2. `import_mysql --package … --replace-existing` → `status=imported, version_status=updated`；
3. 清孤儿向量（`--prune-orphans`：删 17 条旧 child 向量）；
4. **发现并处理一个坑**：chunk_key 是位置型（`…-a1` / `…-a1-c7`），重导后 key 不变，
   索引服务见 key 已存在即跳过 → 须先删该文档 52 条旧向量、把版本重置为
   `awaiting_embedding` 再索引，最终 `indexed_chunks=52`；
5. 新 parent 摘要生成（486/486）；
6. `scripts/migrations/restore_case_doc_status.py --apply` 回填人力成果：
   `version_status=approved`（`reviewed_by=batch38-fix` + 备注）、`law_versions.status='不适用'`
   （重导会重建 law_versions 行，抽取器抽不到该值，不清则丢失人工核验结果）。

### 1.4 验收

**(a) 修后块清单**：52 块（1 parent 6308 字符 + 51 child），**含样板 0 块**。
child 长度分布：6 字 ×17（案例标题行）、42~59 字 ×8、146~475 字 ×26。

**(b) Milvus == MySQL**：

```
✅ Milvus=1610 == MySQL.document_chunks=1610
✅ 版本状态统计 approved=11（其它=0）
```
（compact 前 `row_count` 会含墓碑数字 1679，属 Milvus 统计口径，非差异。）

**(c) 检索抽查**（3 个案例类问题，各取 top-5 上下文）：

| 问题 | 召回条数 | 含样板块数 | doc1 是否仍可召回 |
| --- | --- | --- | --- |
| 转包情形下工伤保险责任由谁承担 | 5 | 0 | ✅（第 3 条） |
| 关联公司混同用工能否认定劳动关系 | 5 | 0 | ✅（第 1 条） |
| 用人单位与劳动者约定不缴社保是否有效 | 5 | 0 | ✅（第 3 条） |

doc1 的召回文本已从「所在位置：…」开头变为正文开头（新闻发布会导语），标题仍为
《最高法发布劳动争议典型案例》。

**(d) 全量测试**：`728 passed`（基线 707 + 新增 11 条清洗/解析测试 + 10 条旁路守卫测试），
无失败、无减少。

---

## 二、裁决 2：集成测试验证码 → test 环境专用旁路

（先纠正原判断：验证码既不在 Redis 也不在 SQL —— `SqlAuthStore.save_code` 是**进程内存 dict**，
集成测试起独立进程读不到，故改用环境门禁旁路。）

### 2.1 实现

`app/auth/service.py`：

- 新增 `INTEGRATION_TEST_FIXED_CODE = "000000"` 与 `is_integration_test_bypass(code)`；
- 新增 `AuthService._consume_code()` 作为 register / reset_password 共用的**唯一校验入口**：
  ```
  # 集成测试旁路（仅用于集成测试，生产禁止启用）：
  # 进入条件先判 ENVIRONMENT=test，development / production 不可达该分支
  if is_integration_test_bypass(code):
      logger.info("集成测试旁路命中：固定验证码放行（environment=test, purpose=%s）", purpose)
      return True
  return self.store.consume_code(email, purpose, code)
  ```
- 备份：`%TEMP%/b38_bak/service.py`（md5 `9f93e73020e45b3183c81363054dd9e1`）。

### 2.2 三项硬要求落实

| 要求 | 落实 | 证据 |
| --- | --- | --- |
| ① production 拒绝固定码的守卫测试 | `tests/test_auth_test_bypass.py`（10 用例） | 见 2.3 |
| ② 旁路处显式注释 | 常量、helper、`_consume_code` 三处均注明「仅用于集成测试，生产禁止启用」 | 源码 |
| ③ test 环境打 INFO、生产不得出现 | 日志落在环境门禁之后，生产不可达 | 实测 test=有日志 / development=无日志 |

### 2.3 验收

**(a) 三环境验证码行为对照（真起后端实测，非仅单测）**

| 环境 | 起服务 | 用固定码 `000000` 注册 | 旁路 INFO 日志 | 说明 |
| --- | --- | --- | --- | --- |
| `test` | ✅ | ✅ 200 注册成功（返回 access_token） | ✅ 出现 | 旁路生效 |
| `development` | ✅ | ❌ 400「验证码无效或已过期」 | ❌ 不出现 | 行为不变，走真实验证码 |
| `production` | ❌ 启动即失败 | —（进程不可用） | — | `生产环境必须配置 REDIS_URL 环境变量`；固定码拒绝由单测覆盖 |

**(b) 守卫测试输出**

```
tests/test_auth_test_bypass.py + tests/test_auth_api.py → 19 passed
```

用例含：`test_non_test_env_rejects_fixed_code[production/development]`、
`test_bypass_log_absent_in_non_test_env[production/development]`、
`test_wrong_code_still_rejected_in_test_env`（旁路只认固定码）、
`test_reset_password_rejects_fixed_code_in_production`（改密路径同样受门禁）。

**(c) 集成测试统一入口一次完整运行**

```
python scripts/e2e/run_integration_tests.py --only auth,review

项        结果     耗时    说明
auth      PASS    18.1s   认证落库（注册-重启-登录-隔离）
review    PASS    16.5s   审核发布（权限-索引-留痕）
合计 2 项：2 通过 / 0 失败；探针清理：完成
```

断言仍是真断言：注册后 users 表可查、杀进程重启后登录成功、跨用户访问 404、
`password_hash` 非明文；审核侧未审核版本检索不到、approve 后重新出现、
reject 删除 29 条向量且抽查可见数 0、审核留痕（谁/何时/决定/note）齐全，
收尾 `Milvus row_count 恢复 1610，版本状态 approved=11`。

### 2.4 连带修复（同款旁路，超出原清单但同类问题）

- `scripts/e2e/e2e_create_admin.py`：同样依赖 `debug_code` 回显，一并改为固定码 + 注入
  `ENVIRONMENT=test`（只注入后端子进程，CLI 子进程不受影响）；
- `scripts/e2e/README.md`：共同前置由 `APP_ENV=development` 改为 `ENVIRONMENT=test`，
  并新增「验证码：集成测试旁路固定码」小节（含三环境对照表）。

---

## 三、端态与遗留

**端态**：11 部法规 / 1610 分块（parent 486 + child 1124）/ 摘要 486 条 /
效力状态 现行有效 10 + 不适用 1 / `check_services.py` 全绿（退出码 0）。

**遗留（未动，供后续裁决）**：

1. `scripts/api-collection/legal_rag_collection.json` 的说明文字仍提到
   `debug_code` 回显（Postman 集合描述，非代码路径），当前接口已不回显；
2. `backend/.env.test` 文件内 `ENVIRONMENT=development`，与本次引入的 `test` 语义不一致
   —— 本次未改该文件（进程环境变量优先级高于文件，探针显式注入 `ENVIRONMENT=test` 即可），
   但**若后续需要"纯 test 环境"启动，建议把该行改成 `test`**；
3. `prompt_builder.py` 的检索摘要字段只写不读：Milvus `summary` 字段写入后，
   检索侧实际走 MySQL `chunk_summaries` 联表，故本次不需要为摘要重建向量（已在本次验证）。

**边界遵守**：未删任何原始文件；清洗规则改动只涉及新增过滤，未调整其他站点既有分支；
`docs/` 未改；改动前全部备份并记 md5；临时脚本只在 `%TEMP%/b38_bak/`，
唯一进库脚本 `scripts/migrations/restore_case_doc_status.py` 已登记 README。
