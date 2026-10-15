# Contract: SSE 事件流

**Feature**: `006-medical-qa-input` | **Date**: 2026-09-27

`POST /ask` 成功时响应体的**逐字节格式契约**。前端解析器与后端编码器 MUST 同时满足本文件；任何一侧单独改动都视为破坏契约。

---

## 1. 线格式

每个事件由**三行**构成，以空行结束：

```text
event: <事件名>\n
data: <单行 JSON>\n
\n
```

**两条硬性规则**：

1. **`data:` 必须是单行 JSON**。JSON 内不得含裸换行——用 `json.dumps(..., ensure_ascii=False)` 且**不加 `indent`**。理由：SSE 规范中 `data:` 后跟换行即终止该字段，多行 JSON 会被拆成多个 `data:` 行，接收端拼回来的语义取决于实现——这是一个**双方各自"看起来对"却能悄悄错位**的地方。
2. **`ensure_ascii=False`**。默认的 `ensure_ascii=True` 会把中文转成 `\uXXXX`，功能上正确但让 `curl -N` 的输出完全不可读，而可读性正是选 SSE 的理由之一（research R1）。

**不发送的 SSE 字段**：不使用 `id:`（无断线续传需求）、不使用 `retry:`（不做自动重连）。

---

## 2. 事件类型

### 2.1 `status`

```text
event: status
data: {"answer_id":"018f3a2b-7c41-7b3e-9f2a-1d4c8e5b0a11","state":"accepted"}

```

- **位置**：MUST 是第一个事件。
- `state` 取值：本期仅 `accepted`。

**为什么 `answer_id` 必须在首帧**（research R10）：流中断时终帧永远到不了，若 `answer_id` 只在终帧，这次提问就失去了可关联的标识——而"问了但没反应"恰恰是最需要排查的一类。

### 2.2 `citations`

```text
event: citations
data: {"citations":[]}

```

- **位置**：恰好一次，在全部 `token` 之前。
- 排序：`citations` 数组 MUST 按 `score` **降序**（`docs/05` §4.2）。
- **本期恒为空数组，但事件仍然发送**——见 §4。

### 2.3 `token`

```text
event: token
data: {"text":"高血压的诊断标准为"}

```

- **位置**：0..n 次，全部在 `citations` 之后、`done` 之前。
- **FR-030 的落点**：若紧急前置话术存在，它 MUST 是**第一个 `token` 事件的内容**。在此之前 MUST NOT 出现任何 `token` 事件。
- 本期：**发送**，内容为"能力未就绪"文案（见 §3）。

> **一处对早期草稿的修正（2026-09-27，实现阶段）**：本文件初稿写的是"本期不发送 `token`"。
> 实现时改为发送，理由与 §4.1 对 `citations` 空数组的论证同源 —— 若本期不发，
> 前端的 `appendToken` 与"终帧纠正已渲染内容"（§4.3）两条逻辑就是**从未执行过的死代码**，
> 等生成模块接入时才第一次运行，而那时代码已与别的改动混在一起。
> 代价只是多一个事件，收益是整条流式追加路径现在就是活的。

### 2.4 `done`

```text
event: done
data: {"answer_id":"018f3a2b-...","is_refusal":true,"emergency":{"triggered":false},"risk_level":"none","answer_text":"你的问题已收到。医知源的向量检索与大模型生成能力正在接入中，本期暂不能给出答案。\n\n以上为基于知识库的参考信息，不能替代执业医师的当面诊断。","citations":[],"disclaimer":"以上为基于知识库的参考信息，不能替代执业医师的当面诊断。"}

```

- **位置**：MUST 是最后一个事件，且 MUST 恰好一次。
- `data` 是 `docs/05` §3.1.3 响应体的**逐字段复用**（字段名与结构不得增删改）。
- 不变量：`done.answer_id == status.answer_id`。
- JSON 内的换行 MUST 是转义后的 `\n`（两个字符），不是裸换行（规则见 §1）。

---

## 3. 完整示例（本期实际输出）

```text
event: status
data: {"answer_id":"018f3a2b-7c41-7b3e-9f2a-1d4c8e5b0a11","state":"accepted"}

event: citations
data: {"citations":[]}

event: token
data: {"text":"你的问题已收到。医知源的向量检索与大模型生成能力正在接入中，本期暂不能给出答案。"}

event: done
data: {"answer_id":"018f3a2b-7c41-7b3e-9f2a-1d4c8e5b0a11","is_refusal":true,"emergency":{"triggered":false},"risk_level":"none","answer_text":"你的问题已收到。医知源的向量检索与大模型生成能力正在接入中，本期暂不能给出答案。\n\n以上为基于知识库的参考信息，不能替代执业医师的当面诊断。","citations":[],"disclaimer":"以上为基于知识库的参考信息，不能替代执业医师的当面诊断。"}

```

**注意本期的形态**：`token` 只有 1 个，内容是未就绪文案。它走的是**与将来真正答案完全相同的那条通道** —— 这正是本节要修正的目的。

---

## 4. 三条刻意的设计取舍

### 4.1 空数组仍然发送，而不是省略 `citations` 事件

省略它会让前端的 `citations` 处理分支**在本期从未执行过**。等检索模块接入、第一次真的发出非空数组时，那条分支才第一次运行——而那时它已经和别的改动混在一起，出问题也难以定位。**发一个空数组的成本是一次序列化，收益是把分支提前跑热。**

同理，`status` 与 `done` 在本期就是真实形态，不是占位。

### 4.2 事件类型走 `event:` 而不是 JSON 内的 `type` 字段

这是选 SSE 而非 NDJSON 的核心收益（research R1）：**协议层能表达的东西不放进应用层**。放进去的代价是每一处解析都要先反序列化再分派，且"类型字段"与"数据字段"混在同一层，校验规则无处安放。

### 4.3 终帧承载完整契约，而不是把结果拆散在多个事件里

`docs/05` §3.1.5 要求前端直接渲染 `answer_text` 整体、MUST NOT 自行拼接。终帧承载完整契约使这条要求可以**跨流式形态继续成立**——前端拿到 `done` 就拿到了与服务端逐字节一致的那份文本。

**这里的风险要说清**：流式下，用户看到的正文是 `token` 事件拼起来的，而终帧的 `answer_text` 是权威全文。二者**有可能不一致**（丢包、前端拼接 bug）。因此：

- 前端 MUST 在 `done` 到达后**以 `answer_text` 为准**校验/纠正已渲染内容；
- 后续生成模块接入时，MUST 增加一条断言：`token` 拼接结果 == `done.answer_text`。

本期无 `token`，该断言暂为空转；**但接口结构已经支持它**，这正是本期就定下终帧形态的意义。

---

## 5. 前端解析器的最低要求

实现位于 `frontend/js/sse.js`，MUST 满足：

| 要求 | 理由 |
|---|---|
| 用 `fetch` + `response.body.getReader()`，不 `await response.text()` | `text()` 会等流结束，直接摧毁流式 |
| `TextDecoder` 用 `{stream: true}` | 不设它会在多字节 UTF-8 字符被切断时产生乱码——中文几乎必然触发 |
| 按 `\n\n` 切块，**保留未完成的尾部** | 一个事件可能被切成两次 `read()` 到达。丢弃尾部会随机丢事件 |
| 忽略空块与 `:` 开头的注释行 | 容错 |
| 收到未知 `event:` 名时**静默忽略**，不报错 | 后续新增事件类型不应打挂老前端 |
| `done` 到达后关闭 reader | 释放连接 |

**"保留未完成的尾部"是这里唯一的真陷阱**：`chunk` 边界由 TCP 决定，与我们的 `\n\n` 无关。一个 `data:` 行被切成两半是**常态而非异常**，且它只在中文或长 JSON 时暴露——测试时用短英文问题往往一次都碰不到。
