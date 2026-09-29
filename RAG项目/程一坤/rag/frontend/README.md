# 法律 RAG 前端

Next.js + TypeScript 实现的浏览器端，对应需求文档 3.4「产品形态为浏览器 Web 应用，
前端采用 Next.js + TypeScript，生产环境使用 standalone 构建」。

## 技术栈

| 能力 | 选型 | 说明 |
| --- | --- | --- |
| 框架 | Next.js 15（App Router） | 需求文档指定 |
| 语言 | TypeScript 5 | 需求文档指定 |
| 样式 | Tailwind CSS 3 | 唯一没有额外学习成本的样式方案，不引入组件库 |
| 状态 | React 内置 hooks | 全局只有一个「是否登录」布尔值，上状态库属于过度设计 |
| 构建 | `output: "standalone"` | 需求文档指定，产物可直接进 Docker 镜像 |

**刻意没有引入的东西**：UI 组件库（Ant Design / shadcn）、Markdown 渲染库、状态管理库、
数据请求库（SWR / React Query）。首期需要的控件数量很少，手写反而更可控，
也避免给项目增加未在文档中锁定的依赖。

## 目录结构

```text
frontend/
├── src/
│   ├── app/                        路由（App Router）
│   │   ├── layout.tsx              根布局：顶栏 + 常驻免责声明
│   │   ├── page.tsx                入口分流：按登录态去 /chat 或 /login
│   │   ├── (auth)/                 认证分组
│   │   │   ├── login/              登录
│   │   │   ├── register/           注册（验证码 + 密码）
│   │   │   └── password-reset/     重置密码
│   │   ├── chat/                   问答页（SSE 流式 + 引用卡片）
│   │   └── search/                 检索调试页（不经大模型）
│   │
│   ├── components/
│   │   ├── ui/primitives.tsx       基础控件：Button / Field / Notice / Tag
│   │   ├── app-header.tsx          顶栏：导航 + 登录态 + 注销
│   │   ├── auth-shell.tsx          认证页公共版式与提交状态机
│   │   ├── require-auth.tsx        登录守卫
│   │   └── chat/
│   │       ├── message-item.tsx    单条消息
│   │       ├── answer-content.tsx  回答文本轻量渲染（含引用编号）
│   │       └── citation-card.tsx   引用卡片，按需拉取法条原文
│   │
│   ├── lib/
│   │   ├── types.ts                接口类型定义（契约唯一出处）
│   │   ├── error-codes.ts          错误码表（对齐后端 app/errors.py）
│   │   ├── api-client.ts           HTTP 客户端 + 统一错误收敛
│   │   ├── sse-parser.ts           SSE 分帧解析
│   │   ├── auth-token.ts           会话令牌本地存取
│   │   ├── api-auth.ts             认证接口
│   │   ├── api-legal-search.ts     法律检索接口
│   │   └── api-chat.ts             问答流式接口
│   │
│   └── styles/globals.css          全局样式
├── .env.example
├── next.config.ts                  rewrite 转发 + standalone 输出
├── tailwind.config.js
└── tsconfig.json
```

## 本地运行

```bash
# 1. 安装依赖
npm install

# 2. 配置后端地址
cp .env.example .env.local
# 编辑 .env.local，把 BACKEND_ORIGIN 指向本地后端

# 3. 启动开发服务器
npm run dev
# 打开 http://localhost:3000
```

后端需先启动（`uvicorn app.main:app --port 8000`），并确保 Redis 可用
—— 认证依赖 Redis 会话存储，Redis 不可用时受保护接口会统一返回 40100。

## 生产构建

```bash
npm run build      # 产出 .next/standalone
npm run start
```

`output: "standalone"` 会生成自包含的 `.next/standalone` 目录，
部署时把它连同 `.next/static`、`public` 一起拷进镜像即可，容器内无需 `npm install`。

## 与后端的对接方式

浏览器请求全部走同源 `/api/:path*`，由 `next.config.ts` 的 rewrite 转发到
`BACKEND_ORIGIN`。这样做有两个好处：前端不需要处理 CORS，
后端内网地址也不会出现在浏览器网络面板里。

## 已实现页面与后端契约的对应

| 页面 | 调用的接口 | 后端实现位置 |
| --- | --- | --- |
| 登录 / 注册 / 重置密码 | `POST /api/v1/auth/*` | `app/auth/router.py` |
| 问答 | `POST /api/v1/chat/stream`（SSE） | `app/api/chat_stream.py` |
| 检索调试 | `POST /api/v1/legal/search` | `app/api/legal_search.py` |

## 尚未覆盖的接口（后端也尚未实现）

以下接口在 `docs/接口文档.md` 中有定义，但后端当前没有对应实现，
因此前端**没有**为它们建页面，避免做出「点了一定报错」的空壳：

| 接口 | 用途 | 状态 |
| --- | --- | --- |
| `POST /api/v1/sessions` | 创建会话 | 后端未实现；前端改为本地生成并持久化 session_id |
| `GET /api/v1/memories` | 查询长期记忆 | 后端未实现 |
| `DELETE /api/v1/memories/{id}` | 删除长期记忆 | 后端未实现 |
| `PUT /api/v1/users/me/memory-settings` | 关闭长期记忆 | 后端未实现；问答页固定传 `enable_long_term_memory: false` |
| `POST /api/v1/admin/source-crawls` | 触发采集任务 | 后端未实现（阶段 6 未开工） |
| `GET /api/v1/admin/documents` | 待审核文档列表 | 后端未实现（阶段 6 未开工） |
| `POST /api/v1/admin/documents/{id}/review` | 审核并发布 | 后端未实现（阶段 6 未开工） |

## 若干实现决策的说明

**引用原文是二次检索拉取的。** 后端 `citation` 事件只给 5 个字段
（`chunk_id` / `law_name` / `article_number` / `paragraph_number` / `page`，
见 `app/chat/service.py`）。要展示法条原文与生效状态，就用
「法规名 + 条号」回查一次 `/api/v1/legal/search`。宁可多一次请求，
也不把残缺信息包装成完整引用。

**回答文本自己解析，不引 Markdown 库。** 提示词已固定输出结构，
实际只会出现标题、加粗、列表、引用编号四种标记。引 Markdown 库必须开
`dangerouslySetInnerHTML`，等于把模型输出当 HTML 执行，存在注入风险。

**`message_end` 不是流结束的唯一信号。** 后端异常时只发
`message_start` + `error`，不发 `message_end`，且 HTTP 状态码仍是 200。
因此前端在 `finally` 里兜底结束流式态，不能只依赖 `message_end`。

**会话 ID 由前端生成。** 后端没有创建会话接口（见上表），但
`chat/stream` 要求 `session_id` 必填，且它同时是 Redis 短期记忆 key 的组成部分，
因此由前端生成后写入 localStorage，刷新页面仍属同一会话。
