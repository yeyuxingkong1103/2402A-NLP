# HTTP 接口（④a）设计

日期：2026-09-29
状态：设计已确认（2026-09-29 与用户逐节确认），待用户审阅规格
依据：`docs/需求文档.md` FR-6 / FR-7 / FR-8 / FR-9、AC-10 / AC-11 / AC-19 / AC-20；`docs/技术方案.md` §八、§九

---

## 一、范围

**做**：把已完成的问答链路（检索 → 生成 → 引用回查 → 推荐律师与费用区间）暴露成 HTTP 接口，并补上它作为「服务」所必需的三件事 —— **鉴权与账号**、**三层隔离的接口层**、**应用层防滥用**，外加 **AC-11 的审计表与写入**。

**不做**（本设计不含）：

- **Redis 与 Nginx 层限流** —— 用户 2026-09-29 裁决「应用层自足」：单机部署下进程内限流够用，Redis 的主要价值（多实例共享计数、会话态、缓存）在后续阶段才兑现
- **OpenTelemetry trace 与可视化后端** —— 同上裁决：trace 需要额外服务，且不是验收判据
- **历史案件检索本身** —— 该功能（FR-5.3）尚未实现，本轮只留接口位
- **前端（Vue 3）**、**部署（Docker Compose 全栈）** —— 各自独立推进
- **用户管理界面** —— 需求文档把它归合伙人的后台职责；本轮账号由运维脚本创建
- **出站内容过滤器** —— ③b-2 的另一块，独立推进

**前置**：③a（在线问答链路）与 ③b-1（推荐律师与费用区间）已交付；全量 607 passed。

---

## 二、定稿口径（2026-09-29 确认）

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 接口范围 | **两侧都做**：公众侧四个 + 律师侧四个 + 登录 + 管理 + 健康检查。历史案件留接口位 |
| 2 | 账号与登录 | **最小账号体系**：`users` 表 + `POST /auth/login` 换 JWT；用户由运维脚本创建，不做管理界面 |
| 3 | 防滥用 | **应用层自足**：进程内限流 + 并发上限 + 请求体/输入长度上限 + 越权探测 404 化；Nginx/Redis 留给部署阶段 |
| 4 | 审计 | **只做表与写入 + 导出接口**；OTel trace 留给运维阶段 |
| 5 | 接线方式 | **薄接口层 + 共享装配工厂**：路由只做 HTTP 关注点；重资源装配收进 `core/factory.py`，CLI 与 HTTP 共用 |
| 6 | 历史案件路由 | **注册但返回 501**（接口清单稳定、前端可提前对接），reason 写明未实现 |
| 7 | `/metrics` | **最小 JSON**（请求数/延迟/错误数），不引入 `prometheus_client`；将来上 Prometheus 再换格式 |
| 8 | 密码哈希 | **标准库 `hashlib.scrypt`**，不引 passlib/bcrypt —— 本项目依赖锁得紧，scrypt 是标准库里的抗内存暴力算法 |
| 9 | 未认证的律师侧/管理路径 | **404**（不 401/403）—— 与「管理路径统一 404 化」同口径，不暴露路径存在性 |

---

## 三、架构

```
backend/app/
  main.py            应用装配：lifespan 装配重资源 → app.state；注册路由；异常处理；请求 ID 中间件
  core/
    config.py        配置（环境变量：JWT 密钥、阈值、并发上限）；启动自检
    factory.py       共享装配工厂：建 conn / Milvus client / encoder / reranker / Answerer
    security.py      JWT 签发与校验、密码哈希、RBAC 依赖、当前用户上下文
    ratelimit.py     进程内滑动窗口限流 + 公众侧并发上限
    audit.py         audit_log 写入与查询
  api/
    schemas.py       请求/响应模型（统一响应带 request_id）
    auth.py          POST /auth/login
    public.py        公众侧四个端点
    lawyer.py        律师侧四个端点
    admin.py         审计导出
  db/mysql.py        （现有）新增 users / audit_log 两处建表
```

**三条定调**：

1. **路由只做 HTTP 关注点**（参数校验、鉴权、序列化、异常→状态码），不写业务。业务已在 `Answerer` 里编排好 —— 这也是**不加 service 层**的原因：再包一层是重复抽象。
2. **重资源只在 lifespan 装配一次**存进 `app.state`（bge-m3 + reranker 加载是秒级，绝不能每请求），路由从 `app.state` 取。
3. **`factory.py` 供 CLI 与 HTTP 共用**，`tools/ask.py` 改为调用它 —— 消掉两套装配。这是 ③b-1 的直接教训：「两处各存一份口径」必定漂移（Milvus 过滤条件、`unit` 判定都这么漂过），而模型加载这类重资源若两处各装一遍，迟早出现「CLI 能跑、服务不能」。

**分层规则的准确形式**（2026-09-29 任务 1 复审修订）：`db/`、`retrieval/`、`generation/`、`recommend/`、`ingest/` 这些下层包**不得** import `core/`（core 一改就牵动全域，且下层的单测会被配置装载/装配工厂拖下水）；`core/` 可以 import 它们，`main.py` 与 `api/` 也可以 —— lifespan 必须 import `core.factory`，鉴权要用 `core.security`。写成「**谁都不能** import core」会把本节的布局自己判成违规，第 2~7 步随后就会在「遵守规则」与「按设计接线」之间打架。另记一条本步新引入的依赖边：`recommend/`（`extras.build_extras_fn`）import `generation/llm_router` 取公众侧 LLM 客户端 —— 与本节布局相容（费用区块本就要走生成），但它是新方向，别再反着接。

**为什么是 FastAPI**：技术方案 §8.2 已定稿。其依赖注入天然适配 RBAC（`Depends(require_role(...))`），且 `def` 端点会自动进线程池 —— 现有链路是同步阻塞的（模型推理 + 同步 DB），这个特性正好，不必为了 async 重写底层。

---

## 四、接口契约

| 方法 | 路径 | 鉴权 | 本轮状态 |
|---|---|---|---|
| POST | `/api/v1/auth/login` | 无 | 新增（技术方案清单里缺的） |
| POST | `/api/v1/public/qa` | 匿名 + 限流 | 接现有 `Answerer`（公众侧） |
| POST | `/api/v1/qa` | JWT | 接现有 `Answerer`（律师侧） |
| POST | `/api/v1/search` | JWT | 接现有检索链路（只检索不生成） |
| GET | `/api/v1/law/{law_id}/articles/{article_no}` | JWT / **公众只读** | 接 `article_lookup` |
| GET | `/api/v1/law/nav` | JWT / **公众只读** | 由 `article.path` 聚合出编—章—节树 |
| POST | `/api/v1/cases/search` | JWT | 注册但返回 **501** |
| GET | `/api/v1/lawyers/recommend` | 无 | 接 `lawyers` + `fees` |
| GET | `/api/v1/admin/audit/export` | JWT（partner） | 接 `audit` 查询 |
| GET | `/healthz` / `/metrics` | 内网 | 连通性 + 最小计数 |

**问答响应**（公众侧）：

```json
{ "request_id": "...", "status": "ok", "answer": "...",
  "citations": [{"article": "584", "paragraph": null, "item": null, "quote": "..."}],
  "sources":   [{"article_no": 584, "path": "...", "source": "vector", "rerank_score": 0.91}],
  "cause": "房屋租赁合同纠纷", "field": "房产与租赁",
  "lawyers":   [{"name": "...", "org": "...", "field": "...", "contact_hint": "...", "demo": true}],
  "fee_range": {"status": "ok", "low": 800, "high": 1000, "unit": "元",
                "charge_basis": "小时", "basis": "...", "source_doc": "...", "source_no": "..."},
  "disclaimer": "参考区间，不构成报价或委托", "failures": [] }
```

- `extras` **展开**成 `lawyers` / `fee_range` / `cause` / `field`（采用技术方案的字段名），内部结构不动。
- **律师侧响应没有 `lawyers` 与 `fee_range` 两个键** —— 不是置空，是根本不出现（律师侧不经过 `recommend/` 的红线，在契约层也要成立）。
- 每个响应带 `request_id`，与审计、日志对齐。

**登录**：请求 `{"username", "password"}` → 响应 `{"access_token", "token_type": "bearer", "role", "team_id"}`。

---

## 五、数据结构

```sql
-- 所内账号。role 只允许 lawyer / assistant / partner（应用层校验，不用 ENUM：
-- 加角色时 ENUM 要 ALTER TABLE，而这一列会被读进 JWT 载荷）
CREATE TABLE IF NOT EXISTS users (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,   -- scrypt，含盐与参数，格式自描述
  role VARCHAR(16) NOT NULL,
  team_id VARCHAR(64) NOT NULL,          -- 三层隔离里 team_id 的唯一来源
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 审计（AC-11）。公众侧行**不记提问原文**（§9.3 的日志白名单）
CREATE TABLE IF NOT EXISTS audit_log (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  request_id VARCHAR(64) NOT NULL,
  user_id BIGINT NULL,          -- 公众侧为 NULL：不采集身份
  role VARCHAR(16) NULL,
  team_id VARCHAR(64) NULL,
  action VARCHAR(32) NOT NULL,  -- login / qa / search / article / nav / recommend / export
  query_text VARCHAR(500) NULL, -- 仅律师侧写入；公众侧恒 NULL
  recall_path VARCHAR(255) NULL,
  verify_result VARCHAR(64) NULL,
  status_code INT NOT NULL,
  latency_ms INT NOT NULL,
  INDEX idx_ts (ts),
  INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

`password_hash` 存自描述格式（`scrypt$n$r$p$salt$hash`），参数变更时旧记录仍可校验。

---

## 六、安全实现

**三层隔离**（技术方案 §9.2 的「三者缺一不可」）：

| 层 | 本轮做法 |
|---|---|
| 接口层 | 公众侧**根本没有**律师侧路由 —— 不是「存在但拒绝」，是路由表里不存在（技术方案 §8.2 原文） |
| 数据层 | MySQL 目前**没有案件表**（法条是公开数据，不需要隔离）。本轮给出 `team_id` 的**唯一注入点**：`security.py` 的当前用户上下文；凡将来要按团队过滤的查询函数，**签名必须接收它**（而不是从全局/连接里取），测试钉住「律师侧请求的上下文必带 team_id」。案件表落地时**必须**走这个口 |
| 向量层 | 同上：公众侧的 Milvus 查询硬编码 `law_id=民法典`（现有行为），`case_chunks` 集合尚不存在 |

**防滥用**：

| 手段 | 做法 |
|---|---|
| 限流 | 进程内滑动窗口，按 **IP 的加盐哈希**维度（不存 IP 原文） |
| 并发上限 | 公众侧同时处理数 ≤ 20（对齐 AC-14），超了 429 + `Retry-After` |
| 请求体限制 | 中间件卡 Content-Length 上限 |
| 输入长度 | 问句长度上限；超限**拒绝**而不是截断（截断会让引用回查对着半句话工作） |
| 提示注入 | 模型侧已有（`<<< >>>` 定界 + 声明非指令），接口层补输入上限 |
| 越权探测 | 未认证访问**律师侧专用路径**（`/qa`、`/search`、`/cases/search`、`/admin/*`）→ **404**；而公众可读的三个（`/public/qa`、`/lawyers/recommend`、`/law/*`）本就不需要认证 —— 二者不可混为一谈，否则会把该公开的挡掉 |

**审计**：`audit.py` 在请求收尾写一行；**留痕失败不阻断请求**（与 `fee_log` 同口径：审计是旁路），但**必须 `logging.warning`** —— ③b-1 的教训：静默吞掉的故障信号会让证据链连续几小时写不进去而全绿。

---

## 七、错误处理

统一响应：`{"request_id", "error": {"code", "message"}}`。

| 情形 | HTTP | 说明 |
|---|---|---|
| 参数校验失败 / 问句超长 | 400 | |
| 未认证 / 越权 | 404 | 见 §二 第 9 条 |
| 限流 / 并发超限 | 429 | 带 `Retry-After` |
| 未实现（历史案件） | 501 | reason 写明 |
| **系统故障**（Milvus / LLM 不可用） | **503** | 红线所在 |
| 问答正常完成但「无依据」 | **200** | 由 `status` 字段表达，不是错误 |
| 内部异常 | 500 | 通用文案，细节只进日志 |
| 请求体超上限 | **413** | §六 防滥用表的体上限；按 RFC 9110 选 413（表外新增，2026-09-29 控制者裁决保留） |
| `/healthz` 依赖不可用 | **503** | **不套上面的统一错误体** —— 编排层要读「哪个依赖 down」，故用自身契约体 `{request_id, status, checks}`（与 413 同理的表外例外；`request_id` 仍在，异常原文仍不出现） |

**这张表的重点是 503 与 200 的区别**：③a / ③b-1 在服务层守的「故障 ≠ 没有依据」，到接口层就体现为「故障是 503、无依据是 200」；两者若都压成 500，前面那些努力在对外行为上就白做了。同理，500/503 的 `message` 一律通用文案 —— 异常原文（如 `milvus down`）只进日志，**不给公众**（③b-1 的真跑抓到过这个）。

---

## 八、测试与验收

| 层 | 手段 |
|---|---|
| 鉴权 | 有效 / 过期 / 伪造 / 角色不足 token，逐条断言状态码 |
| 隔离 | 公众侧访问律师侧路径 → 404；**律师侧响应不含 `lawyers`/`fee_range` 键**（断言键不存在，不是断言为空） |
| 限流 | 高频请求 → 429；并发上限 → 429 |
| 契约 | 响应字段与本设计逐字段对齐；`request_id` 必存在 |
| 问答端点 | `TestClient` + **注入假 Answerer** —— 真问答已由 CLI 冒烟覆盖，HTTP 层测的是接线与映射，不重复花 API 钱 |
| `/healthz`、条文、导航 | **真连库**（项目惯例：能用真依赖就不用替身） |
| 忠实变异 | 本项目惯例，每条新护栏配一次「改坏 → 变红 → 逐字还原核对」 |

**人工验收**：起服务 → 运维脚本创建账号 → 登录 → 律师侧提问 → 公众侧匿名提问 → 核对响应字段、状态码与审计行。

**新增依赖**（四个）：`fastapi`、`uvicorn`、`PyJWT`、`httpx`（TestClient 需要）。版本按本机实测锁定，与既有清单口径一致（锁精确版本）。

---

## 九、推进顺序

1. **配置与装配**（`core/config.py`、`core/factory.py`）—— 先把重资源的共享装配做出来，并让 `tools/ask.py` 改用它（此步不引入 HTTP）
2. **账号与鉴权**（`users` 表、`core/security.py`、`/auth/login`）—— 隔离与权限都建立在它上面
3. **应用装配**（`main.py`、lifespan、中间件、异常处理）
4. **公众侧路由**（问答、推荐律师）—— 先做公众侧，它是需求重点且不依赖账号
5. **律师侧路由**（问答、检索、条文、导航）+ 501 的历史案件位
6. **防滥用**（限流、并发上限、体限制、404 化）
7. **审计**（表、写入、导出接口）
8. **healthz / metrics** 与全量闸门

---

## 十、自检结论

**需求覆盖**：

| 需求 | 落点 |
|---|---|
| FR-7.1~7.4 律师侧问答与检索 | §四 律师侧四端点 |
| FR-8.1 / FR-8.7 公众侧问答与只读 | §四 公众侧端点（路由层无任何写入口） |
| FR-9.1 公众侧免注册 | §六（无账号依赖） |
| FR-6.1 / 6.3 / 6.4 隔离与审计 | §六 三层隔离 + §五 `audit_log` + 导出接口 |
| FR-9.3 / 9.4 推荐律师与费用区间 | §四 `lawyers` / `fee_range` 字段 |
| AC-10 三层隔离 | §六（数据层/向量层因案件功能未实现而只有「待接点」，**不能声称 100%**） |
| AC-11 审计完整性 | §五 + §六（写入不阻断但必须留 warning） |
| AC-19 公众侧不采集身份 | §五（`user_id` 恒 NULL）、§六（IP 只存加盐哈希）、§七（异常原文不外发） |
| AC-20 公网防护 | §六 防滥用（Nginx/Redis 层明确不在本轮） |

**两处刻意的缺口**（不是遗漏）：

1. **AC-10 只能算部分达成** —— 数据层与向量层的隔离需要案件表与 `case_chunks` 集合才能真实验证，而它们属于未实现的历史案件功能。本轮交付的是**接口层隔离 + 唯一注入点 + 测试钉住**。
2. **AC-20 的 Nginx/Redis 层不在本轮** —— 用户裁决「应用层自足」。上公网前必须补齐，这条要在交付说明里写明。

**命名一致性**：`get_current_user` / `require_role` / `sign_token` / `verify_token` / `RateLimiter` / `write_audit` / `build_services`（factory）—— 跨模块引用无重名冲突。

**占位符扫描**：无 TBD / TODO。历史案件的 501 是**明确的交付形态**，不是「待实现」的含糊占位。
