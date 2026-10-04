# 交付说明 — ④a HTTP 接口

> 日期：2026-10-02（文档同步轮收口）
> 范围：把散落在 `task-8-report.md` §七/§八/§九、终审 `final-review.md`（含 r2）与账本裁决里的交付条目收成一处。本文**不新造口径**，每条均可按所注来源回查。

---

## 一、范围与结论

**交付面**：HTTP 接口（④a）—— **11 条路由** + 账号鉴权（JWT + scrypt） + 三层隔离的接口层 + 应用层防滥用（限流/并发闸门/体上限/404 化） + 审计表与写入 + `/metrics` 最小指标。

11 条路由（实测注册表，`backend/app/api/*.py` + `backend/app/main.py`）：

| 方法 | 路径 |
|---|---|
| POST | `/api/v1/auth/login` |
| POST | `/api/v1/public/qa` |
| GET | `/api/v1/lawyers/recommend` |
| POST | `/api/v1/qa` |
| POST | `/api/v1/search` |
| POST | `/api/v1/cases/search`（501：接口位，FR-5.3 未实现） |
| GET | `/api/v1/law/nav` |
| GET | `/api/v1/law/{law_id}/articles/{article_no}` |
| GET | `/api/v1/admin/audit/export` |
| GET | `/healthz` |
| GET | `/metrics` |

**最终闸门读数（2026-10-02，终审 r2 独立实跑；来源 `final-review.md` r2 §四）**：

- `cd backend && python -m pytest -q` → **818 passed / 0 failed / 0 skipped（120.63s，exit 0；单命令）**
- `cd tools && python -m pytest tests -q` → **136 passed**
- `python tools/check_style.py` → **141 文件，问题 0 处**
- `python tools/check_env.py`（带一次性生成的 `FL_JWT_SECRET`、`FL_RATE_LIMIT_SALT`）→ **8/8 通过**；**不设这两个变量时为 6/8**（未通过 `rate_salt`、`jwt_secret`；环境事实，不是回归）

**结论：可交付（放行）**——终审 r2 判「必修 1~7 逐项落地且判据承重、6 把关键刀重放全红且逐字节还原、既有护栏无一处被削弱」；AC-19 按新 gate 成立（r1 的「不达成」结论已撤回）。

---

## 二、AC 达成表（措辞照实，来源 `task-8-report.md` §七 + `final-review.md`（含 r2 §六））

| AC | 本轮状态 | 证据与边界 |
|---|---|---|
| **AC-10** 三层隔离 | **部分达成** | 接口层：公众侧路由表里**根本没有**律师侧路由（字面量路由清单用例）；数据层：`team_id` 唯一注入点在 `core/security` 当前用户上下文；向量层：公众侧硬编码 `law_id=民法典`。数据层/向量层的真实验证需要案件表与 `case_chunks`（FR-5.3 未实现），本轮给的是接缝 + 测试，**不能声称 100%** |
| **AC-11** 审计 | **达成（本轮范围）** | 人工验收 9 步里应入账的 6 条请求按 request_id **逐条留痕**（action/status_code/latency_ms 齐）；`cases/search`(501)/`healthz`/`metrics` 不在 action 表——是口径不是漏记；写失败不阻断、必 warning（用例钉住） |
| **AC-19** 公众侧不采集身份 | **达成**（终审 r2 复核成立） | 代码路径 gate：`core/audit._identity` 对公众可读前缀**连 Authorization 头都不看**，恒返回三个 None；真连库实测：公开三路径 + 有效 token → `user_id`/`role`/`team_id`/`query_text` 全 NULL；反向律师侧三格 + 问句在。限流键是 IP 的 HMAC 加盐哈希（不存原文） |
| **AC-20** 公网防护 | **应用层部分达成** | 应用层已交付：滑动窗口限流（默认 60s/30 次/IP，**登录已入限流**）、公众侧并发上限 20、体上限 64KiB、问句字符上限、越权探测 404 化。**Nginx/Redis 层与「封禁」不在本轮——上公网前必须补齐**（见 §三） |
| **AC-14** 并发 ≤20 | **机制达成，性能未压测** | `ConcurrencyGate(20)` + 「第 21 个被拒 + 槽归还」用例，锁探针用例承重（删锁变红）；「20 并发下性能不显著下降」需要压测报告，本轮没有，**不写达成** |
| **AC-12** 检索 P95 < 2s | **本轮不可判（无压测）** | `/metrics` 给的是**端到端 + 窗口内**口径（含生成链路），只算观测能力、**不是** AC-12 的达标证据；验收现场 `p95=24s` 是单发真实 LLM 问答，不可与「检索 P95」直接比。压测属部署/验收阶段 |

---

## 三、上公网前必须补齐清单

1. **Nginx/Redis 层限流与封禁（AC-20）**——应用层自足是本轮裁决（2026-09-29）；「封禁」未实现，公网入口的 `limit_req`/`limit_conn`、爬取风控、异常探测封禁都要在网络层补齐（来源 `task-8-report.md` §七 AC-20 行、§九 Task 6）。
2. **可信代理口径（XFF / NAT 共享配额）**——限流**不信 `X-Forwarded-For`**（客户端可伪造）；反代后所有请求的 client 都是代理地址 → **全站共享一份配额**；同出口 IP 的 NAT 用户也共用配额。上公网前必须与 Nginx 一起把「可信代理」口径接上（来源 `task-8-report.md` §八；`core/ratelimit.client_key` 注释）。
3. **`/healthz` 与 `/metrics` 绑内网口或 allowlist**——两者本轮**无鉴权**：`/metrics` 会暴露流量画像与路径名（不含问句/身份/IP）；`/healthz` 每次真查 MySQL（`SELECT 1 FROM article LIMIT 1`）并占业务连接的锁。上公网前必须绑内网口或 Nginx `allow` 白名单（来源 `task-8-report.md` §二 7 / §九、终审 m-7）。
4. **测试独立 database**——测试会话级清理夹具以「无同库写者」为前提，现实中已出现外部写同库的案例；根治办法是给测试独立 database（部署层）（来源 `task-8-report.md` §九 Task 7 m5、终审 §六 m5）。

---

## 四、运行注意

1. **`FL_JWT_SECRET`、`FL_RATE_LIMIT_SALT` 必设**：
   - 缺 `FL_JWT_SECRET`（≥32 字符）→ **服务起不来**（`main.lifespan` 启动自检当场抛 `MissingSecretError`）；
   - 缺 `FL_RATE_LIMIT_SALT`（≥16 字符）→ **限流路径全 500**（公众端点 + 登录），`healthz` 仍 200 —— 缺盐是环境故障，不是请求失败；
   - 另：默认装配（`with_extras=True`）**需要公众侧生成密钥环境变量 `api`**（`llm_router.DEEPSEEK_KEY_ENV`，变量名即 `api`，用户 2026-09-23 指定）；缺它启动装配失败。
2. **审计写离事件循环 + 有界超时**：审计落库走 `anyio.to_thread.run_sync`（不再按住事件循环）；两条 MySQL 连接（业务/账号）设 `read_timeout`/`write_timeout` = `IO_TIMEOUT_S`（10s），单次 socket 收发有界（来源 `core/audit.py` `_write`、`db/mysql.py:43`；终审修复轮落实）。
3. **审计写库延迟（实测，非估计）**：INSERT+commit **中位 4.96ms / p90 6.38ms / max 7.35ms**（n=30；成因 `innodb_flush_log_at_trx_commit=1` 每 commit 一次 fsync；20 并发满载最坏 ~100ms 尾延迟）（来源 `task-8-report.md` §八）。
4. **套件运行期间勿让真服务/外部驱动写同一库**——会话级清理夹具会按「会话起点之上的 id」删除，外部写者的行会被误删；且**崩溃的会话会留下测试行，且不会被后续完整会话自动清掉**（夹具只删会话起点 max 之上的新行）（来源 `task-8-report.md` §九 Task 7 m5 + 修复轮订正；终审 r2 §六）。

---

## 五、已知口径与边界（都是有意选择，不是缺陷；来源 `task-8-report.md` §九 + 终审 §三）

- **体里无 `request_id` 的三条端点**（`login` / `export` / `metrics`，仅响应头里有）——契约如此，前端从 `X-Request-ID` 头取。
- **`cases/search`(501) / `/healthz` / `/metrics` 不入审计**——`_ACTION_PATHS` 没有它们的 action；这是口径（不是漏记）。
- **`/lawyers/recommend` 的 `fee_range` 恒 null**——费用区间只在 `/public/qa` 的响应里（`recommend` 那条的键位保留、值恒空，待后续处理）。
- **`failures` 原样带模型输出**——公众侧 200 响应中的失败清单不加工（口径）。
- **P95 是窗口内口径**——最近 256 个样本的滚动窗口（nearest-rank），`requests_total` 是全生命周期计数，两个数时间基准不同；长尾样本滑出后不再出现在 p95 里（`core/metrics.py` `LATENCY_WINDOW=256`）。
- **故障映射**：MySQL 不可用 → **500**；Milvus / LLM 不可用 → **503**（红线：故障 ≠ 没有依据）。
- **`/healthz` 的 503 不套统一错误体**——用自身契约体 `{request_id, status, checks}`（编排层要读「哪个依赖 down」）。
- **501 的 reason 公开**——`errors.FeatureNotImplemented` 的 `public_detail` 开关只给它开。
- **行数闸门（2026-10-02 实测）**：`main.py` **286/300**、`core/audit.py` **300/300**、`tests/test_audit.py` **300/300**、`tests/test_ratelimit.py` **299/300** —— **下次动这四个文件前必须先腾行**。

---

## 六、历史数据注释（审计库 `fl_law.audit_log`）

- 人工验收留下的 6 行：**id 1695~1700**（2026-10-01，真服务 + 真模型 + 真库）；`task8-accept` 账号（**id=1079**，lawyer / team-a）**保留**（口令为一次性生成值、不可知）。
- **1698/1699 两行（article / nav）的 `user_id` 是 I-2 修复前的口径产物**（当时公开路径未做身份 gate），保留不回改；AC-19 的当前成立性由修复后的代码路径 gate + 真库双向用例保证。
- 库里 **1700 以上无行**（2026-10-02 清账：删除崩溃测试会话残留 1748~2017 共 66 行；清账后 total=133、max id=1700）。本说明写付时实测复核：`SELECT MAX(id)... → 1700`。

---

## 七、指针

- 设计：`docs/superpowers/specs/2026-09-29-HTTP接口-design.md`
- 实现报告：`.superpowers/sdd/c/task-8-report.md`（§七 AC / §八 容量口径 / §九 缺口汇总）
- 账本：`.superpowers/sdd/c/progress.md`
- 终审（含 r2）：`.superpowers/sdd/c/final-review.md`
- 终审修复报告：`.superpowers/sdd/c/final-review-fix-report.md`
- 文档同步报告（本轮）：`.superpowers/sdd/c/docs-sync-report.md`
