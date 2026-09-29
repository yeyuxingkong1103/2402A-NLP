# scripts/migrations/ —— 数据库迁移历史（按序重跑用）

这些脚本**已经对当前库执行过**。它们的价值不在日常运行，而在于：**重建数据库时按顺序重跑一遍**，
让新库的结构与数据修正与现有库一致。

> 本目录由批次 20 清理建立：四个脚本原先散落在项目根，移入此处集中存放。
> 唯一例外是 `backfill_law_dates.py`——它原本用 `Path(__file__).parent` 定位项目根，
> 移入子目录后该锚点会失效，故已改为 `parents[2]`（向上三级 = `rag/`）；这是本次唯一改动，
> 其余三个脚本无路径依赖、**内容零改动**。

---

## 一、执行顺序（重建库时）
| 序 | 脚本 | 何时跑 | 前置条件 |
|---|---|---|---|
| 1 | `migrate_auth_users.py` | 建表后、启用认证前 | `users` 表已存在（无数据时最干净） |
| 2 | `migrate_long_term_memory_setting.py` | 步骤 1 之后 | `users` 表已存在 |
| 3 | — 导入数据 — | 爬取/解析/打包/入库 | 见 `docs/部署文档.md` |
| 4 | `migrate_review_status.py` | **导入数据之后** | `document_versions` 已有存量版本行（存量置 approved 才有对象） |
| 5 | `restore_case_doc_status.py --apply` | 步骤 4 之后、**任何针对 document id=1 的 replace 重导之后** | `documents.id=1` 存在；默认只预览 |
| 6 | `restore_reingest_status.py --apply` | 步骤 4 之后、**任何针对 doc2/7/8/11 的 replace 重导之后** | 这 4 篇存在；默认只预览 |
| 7 | `backfill_law_dates.py --apply` | 步骤 4 之后 | `laws` / `law_versions` 已有数据，且 `data/labor_law_raw/` 原始 HTML 在位 |
| 8 | `migrate_law_status_vocabulary.py --apply` | 步骤 4 之后（与 7 无先后要求） | `law_versions` 已有数据；**仅当库里有英文取值时需要**（全新库跑出来是 SKIP） |
| 9 | `backfill_status_6laws.py --apply` | 步骤 8 之后 | `law_versions` 已有数据；**仅当 6 部目标行 status 仍为 NULL 时需要**（全新库若抽取器已写入则为中止报错） |

> 步骤 4 与 7 互不影响（一个动 `document_versions`，一个动 `laws` / `law_versions`），
> 谁先谁后都不破坏结果；上表顺序只是保持与历史执行次序一致。步骤 8 同理（只动 `law_versions.status`）。
> 步骤 5、6 的唯一触发场景都是"对文档做了 replace 重导"（步骤 5 是批次 38 的
> document id=1，步骤 6 是批次 39 的 doc2/7/8/11）：

**为什么必须是"先导入数据、再跑 4 和 5"**：两者都是**数据级修正**而非纯结构变更，
空库上跑不出任何效果（脚本自己也会打印"无变化"）。

### 5. `restore_case_doc_status.py`（批次 38：案例文档清洗重切后的状态修复）
document id=1《最高法发布劳动争议典型案例》整页抓取导致页脚样板（版权所有 / 京ICP备 /
校验串）混入正文，批次 38 修复解析与清洗规则后单独 `replace` 重导。走例行导入路径会
重置两处**人工成果**，故需本脚本显式恢复：
1. `document_versions.version_status` 被重建为 `pending_review` → 恢复 `approved`
   并写 `reviewed_by='batch38-fix'` / `reviewed_at` / `review_note` 留痕；
2. `law_versions.status` 被重建为 NULL（案例材料的人工核验值 `不适用`，见批次 26-B）
   → 仅在为空时回填 `不适用`。

红线：只写这两处，不碰正文 / 分块 / 向量；默认预览，`--apply` 才写库。

### 6. `restore_reingest_status.py`（批次 39：4 篇页面清洗重切后的状态修复）
doc2《司法解释（二）和典型案例》/ doc7《司法解释（一）》/ doc8《劳动争议调解仲裁法》/
doc11《工资支付暂行规定》的页面同样带站点页脚样板（责任编辑 / 总机 / 版权所有 /
京ICP备 / 【打印】/ 主办单位…），批次 39 把清洗规则扩到同站点的新闻/案例页与
中国政府网页面后，这 4 篇一起 `replace` 重导。与步骤 5 同理，走例行导入路径会重置
两处**人工成果**：
1. `document_versions.version_status` 被重建为 `pending_review` → 恢复 `approved`，
   并沿用原留痕（`reviewed_by='chunker-v3-重建批准'`、原 `reviewed_at` 与 `review_note`）；
2. `law_versions` 行被重建 → 按脚本内 `SNAPSHOT`（2026-09-23 重导前的库内状态，
   逐字段记录便于审计）回填 `status` 与公布/施行日期。

红线：只写这两处，不碰正文 / 分块 / 向量；默认预览，`--apply` 才写库。
重导后还需按位置型 `chunk_key` 的已知陷阱**先删该文档旧向量再索引**，并重生成
这些 parent 的摘要（旧摘要 key 相同但描述的是旧文本，见批次 39 报告）。

---

## 二、逐个说明

### 1. `migrate_auth_users.py`（批次 5：认证落库）
`users` 表结构对齐到认证所需形态，共四步，**每步先查 `information_schema`，已存在则打印 SKIP**（幂等）：
1. 加列 `email VARCHAR(255) NOT NULL` + 唯一键 `uk_users_email`
2. 加列 `user_key VARCHAR(32) NOT NULL` + 唯一键 `uk_users_user_key`
3. 加列 `is_admin TINYINT(1) NOT NULL DEFAULT 0`
4. 删除遗留列 `username`（其唯一索引随列一并删除）

红线：不 DROP 表、不动 `users` 之外任何表（`chat_sessions` 的孤儿会话保留不动）。

**当前库：已执行** ✅ 证据 —— `users` 列 = `id, user_key, email, password_hash, is_admin, is_active, created_at, updated_at, long_term_memory_enabled`（无 `username`）。

### 2. `migrate_long_term_memory_setting.py`（批次 14：长期记忆开关）
`users` 加列 `long_term_memory_enabled BOOLEAN NOT NULL DEFAULT TRUE`
（接口 8.3 `PUT /api/v1/users/me/memory-settings` 的落库字段；存量用户默认开启——"关闭"是主动行为）。幂等。
红线：不 DROP、不改其它列、不动其它表。

**当前库：已执行** ✅ 证据 —— `users.long_term_memory_enabled` 列存在。

### 3. `migrate_review_status.py`（阶段 6：审核状态迁移）
两件事，幂等：
1. `document_versions` 加可追溯列 `reviewed_by` / `reviewed_at` / `review_note`
2. 存量版本 `version_status` `'new' → 'approved'`

为什么第 2 步必须做：这批版本在审核机制上线前就已建索引并正常使用（等价于"已人工审核发布"）；
不迁移的话，检索侧"只读 approved"的兜底过滤会让这些法规**全部从检索结果里消失**。
脚本是显式 `UPDATE`，只动 `version_status='new'` 的行并打印影响清单。
红线：不 DROP 表、不改其它列、不动其它表。

**当前库：已执行** ✅ 证据 —— 留痕三列齐全；`version_status` 分布 = `approved: 11`（存量 11 篇全部 approved，无残留 `new`）。

### 4. `backfill_law_dates.py`（批次 10 任务 1：法规时效字段回填）
背景：`legal_metadata_writer` 只在 `law_versions` 首次创建时写日期，且导入的增量判定在
分块未变化时短路——导致"抽取规则增强后新抽到的生效日期进不了库"。
本脚本**复用同一个抽取器**（严禁编造，只写页面实际抽到的值），刷新 `laws` / `law_versions` 的时效字段，
并输出 `reports/batch10_law_dates.md`（日期表 + 待人工补录清单）。

用法：
```bash
python scripts/migrations/backfill_law_dates.py           # 预览，不写库
python scripts/migrations/backfill_law_dates.py --apply   # 实际写库
```

**当前库：已执行** ✅ 证据 —— `law_versions` 共 11 行，有发布日期 11 行、有生效日期 10 行；
缺的 1 行是典型案例（页面本身未写生效日期，属"待人工补录"而非脚本失败）。

### 5. `migrate_law_status_vocabulary.py`（批次 26-C4：效力状态词表统一）
背景：`law_versions.status` 原先有两套互不认识的词表——列注释写英文
（effective / amended / repealed / superseded），抽取器与三个消费点用中文
（现行有效 / 已废止 / 已失效）。**英文取值不会被任何"已失效"判断命中**，
已废止的法规会被判"现行有效"并照常返回（静默错误）。批次 26 曾写入 5 行英文取值。

本脚本把历史英文取值迁回中文，映射**只收有裁决或字面依据的**：
`effective → 现行有效`、`amended → 现行有效`（批次 26 裁决 Q1）、
`repealed → 已废止`、`not_applicable → 不适用`；
`superseded` 在新词表里没有对应档位，遇到即**中止报错**交人工裁决（不做规则推定）。

用法：
```bash
python scripts/migrations/migrate_law_status_vocabulary.py             # 预览，不写库
python scripts/migrations/migrate_law_status_vocabulary.py --apply     # 实际写库
python scripts/migrations/migrate_law_status_vocabulary.py --rollback  # 按最近一次记录回滚
```
`--apply` 会往 `reports/law_status_vocabulary_migration_<时间戳>.json` 写一份逐行变更记录
（id / 原值 / 新值）；`--rollback` **只**回退该记录里的行，且先校验"当前值仍是迁移后的值"，
因此不会误伤迁移之后由抽取器正常写入的中文值。
逐行 `UPDATE` 并核对 `rowcount == 1`，任一行不符即整体回滚。

红线：只 UPDATE `law_versions.status` 一列、只动映射命中的行；不 DROP、不改结构、不动其它表。

**当前库：已执行** ✅ 证据 —— 迁移后 `status` 分布 = `现行有效: 4 / 不适用: 1 / NULL: 6`
（`effective`、`not_applicable` 均无残留）；`--rollback` 验证可完整复原为英文，再 `--apply` 可重跑；
重复 `--apply` 输出 `[SKIP] 库里没有任何英文取值`（幂等）。

### 6. `backfill_status_6laws.py`（法规元数据核验回填：6 部 status + ① 公布日期修正）
`law_versions` 里 6 部 status 留空（原始 HTML 无时效性字段、抽取器抽不到，按"抽不到不覆盖"
口径一直未写）。2026-09-21 经权威源核验取得官方标注后回填（证据链：
`reports/metadata_verification_6laws.md`）：

- 逐行 `UPDATE ... WHERE id=<ver_id> AND status IS NULL`，rowcount==1 才继续，任一行不符整批回退；
- `revision_note` 保留**原始标注文字**（flk「有效」/ 人社部「是否有效：有效」）+ 来源 URL + 抓取时间，
  不只写映射后的值；映射口径：官方「有效」→ 词表 `现行有效`（常量取自 `app/db/law_status.py`）；
- 同批修正 ver_id=91（司法解释二）`promulgation_date`：2025-08-01（发布会日）→ 2025-07-31
  （法释〔2025〕12号公告落款；flk 标注 + 最高法官网公告 + `data/labor_law_raw/` 原始 HTML 三重印证）；
- `--rollback` 按最近一次 `reports/law_status_backfill_6laws_*.json` 记录只回退自己写过的行，
  当前值已被其它来源改写的行自动 SKIP。

红线：只 UPDATE `law_versions` 的 `status` / `revision_note` / `promulgation_date` 三列、只动列出的 6 行；
不 DROP、不改结构、不动其它表。

**当前库：已执行** ✅ 证据 —— `status` 分布由「NULL 6 / 现行有效 4 / 不适用 1」变为
**「现行有效 10 / 不适用 1」**；ver_id=91 `promulgation_date = 2025-07-31`；
`check_services` 退出码 0；全量测试 652 passed 不减。

---

## 三、通用注意事项

1. **幂等**：六个脚本均可重复执行（结构类靠 `information_schema` 判断，数据类靠显式 WHERE 条件），
   重复跑不会产生副作用。
2. **连接配置**：`migrate_auth_users.py` / `migrate_long_term_memory_setting.py` / `migrate_review_status.py`
   三个脚本内置 `127.0.0.1:3306 / legal_rag` 连接参数（历史一次性脚本的形态）；
   换库地址时需要改脚本内的 `HOST/PORT/USER/PASSWORD/DATABASE` 常量。
   后加的两个脚本（`backfill_law_dates.py`、`migrate_law_status_vocabulary.py`）走
   `scripts/_env.py` 从项目根 `.env` 读连接参数，源码里零口令，换库只改 `.env`。
3. **执行前先备份**：这些脚本会 `ALTER TABLE` / `UPDATE`，重建库流程里建议在跑之前先做一次 dump。
4. **不要把它们搬回项目根**：项目根只放"配置 + 顶层目录"，脚本统一在 `scripts/` 下。

---

## 批次 37：migrate_chunk_summaries.py（创建 chunk_summaries 表）

- **做了什么**：用 ORM 模型元数据（`app/db/document_models.py:ChunkSummary`）幂等建表
  `chunk_summaries`（chunk_key PK / summary / model / created_at / updated_at）。
  摘要的 MySQL 单一事实来源；`vector_search._fetch_chunks` 外连接它、
  Milvus 索引写 summary 时按 chunk_key join 它。
- **顺序**：无前置依赖；在任何跑 `summarize_chunks.py` 或真库检索链路（smoke 测试）之前执行。
- **当前库（legal_rag）：已执行** ✅（预览确认不存在 → --apply 建表 → inspector 复核 True）。
  数据由 `app/cli/summarize_chunks.py` 后续生成，本脚本不含数据迁移。
- **风险**：仅 CREATE TABLE IF NOT EXISTS 语义，不 ALTER/DROP、不写数据，可安全重复执行。
