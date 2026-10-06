# Contract: HTTP 入口

**Feature**: `006-medical-qa-input` | **Date**: 2026-09-27

本特性对外只有**两个** HTTP 入口：一个静态页面宿主，一个问答流。错误体复用 `docs/05` §2.3 的统一结构。

---

## 1. `GET /` —— 医知源界面

| 项 | 值 |
|---|---|
| 成功 | `200`，`text/html; charset=utf-8` |
| 内容 | `frontend/index.html` |
| 加载的静态资源 | `frontend/styles.css`、`frontend/js/*.js`，同源，路径 `/styles.css`、`/js/...` |
| 鉴权 | 无（`docs/05` §2.2） |

**资源加载失败时的行为**（规格 Edge Case）：页面 MUST NOT 呈现为空白页。实现方式是把品牌名与一句失败说明**内联在 HTML 中**，不依赖外部资源渲染——外链 CSS 挂了，至少还能看见"医知源"和一句可读的提示。

---

## 2. `POST /ask` —— 提问入口

### 2.1 请求

```http
POST /ask HTTP/1.1
Content-Type: application/json
Accept: text/event-stream
X-Request-Id: 018f3a2b-7c41-7b3e-9f2a-1d4c8e5b0a11   # 可选
```

```json
{ "question": "我最近血压有点高，多少算高血压？" }
```

| 头 | 必需 | 说明 |
|---|---|---|
| `Content-Type: application/json` | ✅ | — |
| `X-Request-Id` | ❌ | 客户端可选传入用于日志关联；未传则服务端生成（`docs/05` §2.2） |

### 2.2 成功响应 `200 OK`

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

响应体是**事件序列**，详见 [sse.md](./sse.md)。

**三条必须写进实现注释的响应头理由**：

| 头 | 为什么必须有 |
|---|---|
| `Cache-Control: no-cache` | 逐字话术与回答内容一旦被缓存，用户可能看到**上一个问题的答案**。医疗内容的错配比不显示更糟。 |
| `X-Accel-Buffering: no` | 让可能存在的反向代理（nginx 及其兼容层）**不要缓冲**响应。代理一缓冲，流式就退化成"等全部完成再一次性到达"——功能看着是好的，流式已经死了，且**服务端日志毫无异常**。 |
| `Connection: keep-alive` | 流式应答的生命周期长于普通请求，显式声明避免中间设备提前关闭。 |

`Content-Type` 的 `charset=utf-8` **不可省**：回答是中文，缺省编码会让浏览器按 latin-1 解析，中文变乱码——而这类问题在开发机上（curl）往往看不出来。

### 2.3 错误响应

沿用 `docs/05` §2.3 的统一结构：

```json
{
  "error": {
    "code": "INVALID_QUESTION",
    "message": "请输入你的问题",
    "request_id": "018f3a2b-..."
  }
}
```

| 错误码 | HTTP | 触发条件 | 用户可见文案 |
|---|---|---|---|
| `INVALID_QUESTION` | 422 | 问题缺失、非字符串、去空白后为空或超过 200 字符 | `请输入你的问题` / `问题过长，请控制在 200 个字符以内` |
| `INTERNAL_ERROR` | 500 | 未预期异常 | `服务暂时不可用，请稍后重试` |

**本特性不使用的错误码**（记录以免误用）：

- `SERVICE_UNAVAILABLE`（503）—— 本期无 LLM 调用，不可能触发。
- `INDEX_UNAVAILABLE`（503）—— 本期无索引。**注意**：本期缺少索引能力**不是** 503，而是 200 + 流内的"能力未就绪"状态。理由见 research R5：503 意为"本该能答但坏了"，而本期是"尚未建成"，且 `docs/05` §3.1.4 规则 2 明确拒答/无能力不得用错误码表达。

**错误体的硬性约束**（FR-019）：

1. MUST NOT 包含堆栈、异常类名、内部文件路径、密钥、模型供应商原始错误体；
2. MUST 包含 `request_id`，且同一值 MUST 出现在服务端日志中；
3. 未预期异常 MUST 由**统一的异常处理器**产出错误体 —— 不允许框架默认的 HTML 错误页漏出（它含堆栈链接）。

### 2.4 客户端断开

客户端在流中途断开（关闭页面、刷新）时：

- 服务端 MUST 停止生成并释放资源，MUST NOT 继续跑完；
- MUST 记录一条 `client_disconnected` 日志（含 `answer_id`），**这不是错误**——它解释了为什么这次提问在日志里没有 `done`；
- MUST NOT 抛出未捕获异常。

**为什么值得单列**：本期无生成模块，断开处理无关紧要；但生成模块接入后，"用户关掉页面"会表现为一次 6 秒的 LLM 调用白烧。**把断开路径提前建好，比事后补便宜得多**，且现在建它零成本。

---

## 3. 与 `docs/05` 的差异

| 项 | `docs/05` 原状 | 本契约 | 依据 |
|---|---|---|---|
| 响应形态 | 同步阻塞，一次性返回完整 JSON | SSE 事件流 | 裁决 D3 |
| `Content-Type` | `application/json` | `text/event-stream; charset=utf-8` | 同上 |
| 错误码集合 | 4 个 | 本期用 2 个 | 本期无 LLM、无索引 |
| 终帧字段 | `docs/05` §3.1.3 | **逐字段复用，无增删改** | 裁决 D2、SC-009 |

**待同步修订**：`docs/05` §1.1 I-01 行、§2.4 延迟预算、§3.1.1、§3.1.4。清单见 `spec.md`《需同步修订的既有文档》。
