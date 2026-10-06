# Phase 1 Quickstart: 医知源 · 提问输入链路与前端骨架

**Feature**: `006-medical-qa-input` | **Date**: 2026-09-27

**验证方式**：本项目**禁用 pytest**（`docs/superpowers/plans` 的 Global Constraints，`specs/005` plan.md §145 已记录）。验证走**逐项对拍 + 边界构造**，产物落 `.smoke_out/`。

**新增依赖**：无。`fastapi 0.141.1`、`uvicorn 0.53.0`、`pydantic 2.13.5`、`pydantic-settings 2.15.0`、`httpx 0.28.1` 均已实测装机。**本节不含 `pip install`**——需要装东西即说明偏离了本方案。

---

## 0. 前置条件

```bash
cd D:/zg6_Project/9/med_rag

# 确认解释器（constitution 原则 I：不得用裸 python）
./rag/python.exe -c "import sys; print(sys.version)"
# 期望：3.12.14
```

---

## 1. 启动（**由人执行**，涉及监听端口）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m backend.serve
```

**期望输出**（启动即打印，便于肉眼确认绑对了地址）：

```text
医知源服务已启动  http://127.0.0.1:8000/
```

**验证它没有绑错**：服务 MUST 只监听回环地址。从同一局域网的**另一台机器**访问 `http://<本机IP>:8000/` 应**连接失败**。绑到 `0.0.0.0` 会让一个无鉴权的医疗问答服务暴露在局域网上，这是 `docs/05` §2.2 明确不接受的。

---

## 2. 界面可用 —— SC-001 / FR-001 / FR-002 / FR-003 / FR-004

浏览器打开 **http://127.0.0.1:8000/**

| 项 | 期望 | 对应 |
|---|---|---|
| 3 秒内出现可输入的问答界面 | 是 | SC-001 |
| 左侧导航可见，含「医疗问答」且为选中态 | 是 | FR-004 |
| 布局为左导航 + 右详情 | 是 | FR-002 |
| 整体为蓝白色调 | 是 | FR-003 |
| 断开外链 CSS（临时重命名 `frontend/styles.css`）后刷新 | **不是空白页**，仍可见「医知源」与一句提示 | Edge Case |

---

## 3. SSE 契约 —— 契约逐字对拍

```bash
curl -sN -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"我最近血压有点高，多少算高血压？"}' \
  | tee .smoke_out/s7_stream.txt
```

**逐项核对 `s7_stream.txt`**：

| # | 期望 | 对应 |
|---|---|---|
| 1 | 首个事件是 `event: status`，含 `answer_id` 与 `state:"accepted"` | 契约 §2.1 |
| 2 | 第二个事件是 `event: citations`，`data` 为 `{"citations":[]}` | 契约 §2.2 |
| 3 | 恰好 **1 个** `event: token`，内容为未就绪文案 | 契约 §2.3、§3 |
| 4 | 末事件是 `event: done`，且**恰好一次** | 契约 §2.4 |
| 5 | `done.answer_id` 与 `status.answer_id` 逐字符相同 | 契约不变量 |
| 6 | `done` 的字段名集合 == `docs/05` §3.1.3 的七个字段，无增无减 | FR-023、SC-009 |
| 7 | `done.answer_text` 以 `\n\n` + 免责声明结尾（**逐字**：`以上为基于知识库的参考信息，不能替代执业医师的当面诊断。`） | FR-030 前置条件、constitution 原则 V |
| 8 | `done.answer_text` **不含**任何医学结论、剂量、诊断 | FR-025、SC-006 |
| 9 | `done.answer_text` **不含**拒答兜底话术（本期不该出现） | research R5 |
| 10 | 中文显示为中文，不是 `\uXXXX` 转义 | 契约 §1 |

**响应头核对**：

```bash
curl -sN -D - -o /dev/null -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"question":"测试"}' | head -8
```

| 头 | 期望 |
|---|---|
| `content-type` | `text/event-stream; charset=utf-8`（`charset` 不可省） |
| `cache-control` | `no-cache` |
| `x-accel-buffering` | `no` |

---

## 4. 输入校验 —— SC-003 / FR-016 / FR-017 / FR-018

```bash
# 4.1 空问题 → 422
curl -s -o - -w "\nHTTP %{http_code}\n" -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"question":""}'
# 期望：HTTP 422，error.code == "INVALID_QUESTION"，message == "请输入你的问题"

# 4.2 纯空白（含全角空格 U+3000）→ 422
curl -s -o - -w "\nHTTP %{http_code}\n" -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"question":"　 \t\n　"}'
# 期望：HTTP 422

# 4.3 字段缺失 → 422
curl -s -o - -w "\nHTTP %{http_code}\n" -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{}'
# 期望：HTTP 422（MUST NOT 是 500）

# 4.4 边界：恰好 200 字符 → 200
./rag/python.exe -c "
import json,urllib.request
q='高'*200
r=urllib.request.urlopen(urllib.request.Request(
    'http://127.0.0.1:8000/ask', json.dumps({'question':q}).encode(),
    {'Content-Type':'application/json'}))
print('HTTP', r.status, '(期望 200)')"

# 4.5 边界：201 字符 → 422
./rag/python.exe -c "
import json,urllib.request,urllib.error
q='高'*201
try:
    urllib.request.urlopen(urllib.request.Request(
        'http://127.0.0.1:8000/ask', json.dumps({'question':q}).encode(),
        {'Content-Type':'application/json'}))
    print('HTTP 200 —— 不符合期望')
except urllib.error.HTTPError as e:
    print('HTTP', e.code, '(期望 422)')"
```

**4.4/4.5 的"200 字符"必须按去空白后的字符数计**，且中文按**字符**不按字节——`'高'*200` 是 200 字符 / 600 字节。若实现误判为字节，4.4 会错误返回 422。

**错误体脱敏核对（FR-019）**：

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":""}' \
  | grep -iE "traceback|File \"|\\.py|api_key|token" && echo "❌ 泄漏" || echo "✅ 无泄漏"
```

---

## 5. 界面交互 —— FR-007 ~ FR-015

浏览器打开 http://127.0.0.1:8000/ 后逐项操作：

| # | 操作 | 期望 | 对应 |
|---|---|---|---|
| 1 | 输入合法问题，按 **回车** | 提交，显示进行中，随后出现结果 | FR-007 |
| 2 | 输入合法问题，点 **发送图标** | 与按回车**完全相同**的结果 | FR-008 |
| 3 | 用拼音输入法打 `gaoxueya`，在**候选词未上屏时**按回车 | **不提交**，仅选词 | FR-009、Edge Case |
| 4 | 输入框留空，按回车 | 提示"请输入你的问题"，**网络面板无请求** | FR-016、SC-003 |
| 5 | 输入 201 个字符 | 提示过长（含当前字数与上限），**无请求** | FR-017 |
| 6 | 提交后立刻再按回车 | 不产生第二个并发请求 | FR-013 |
| 7 | 输入 `<img src=x onerror=alert(1)>` 并提交 | **原样显示为文本**，无弹窗 | FR-018 |
| 8 | 输入含 `【1】` 的问题 | 原样显示，不被解释为引用标记 | FR-018 |
| 9 | 停掉服务后再提交 | 明确失败提示，**退出加载态** | FR-014、SC-008 |
| 10 | 提交后刷新页面 | 输入框清空，无历史回显 | Assumptions |

---

## 6. 导航可扩展 —— SC-005

在 `frontend/js/nav.js` 的导航配置中追加一项（例如 `{id:'meds', label:'用药助手', panel:'#panel-meds'}`）：

| 期望 | 对应 |
|---|---|
| 左侧导航出现「用药助手」 | US3-1 |
| 点击可切换，切回「医疗问答」后输入框仍可用 | US3-2 |
| **问答相关的 JS 一行未改**（`git diff --stat` 只显示 `nav.js`） | SC-005、US3-3 |

---

## 7. 引用区排序与流式渲染 —— SC-007 / FR-028 / FR-029

本期服务端不发引用，因此**用固定假数据验证渲染层**：

在浏览器控制台注入一组乱序引用片段（分数刻意打乱）：

```js
// frontend/js/transcript.js 应导出可独立调用的渲染函数
renderCitations([
  {citation_id:2, file_name:'B指南.pdf',  page_start:8,  page_end:8,  section:'2.1', block_type:'text', text:'片段B', score:0.65},
  {citation_id:1, file_name:'A指南.pdf',  page_start:12, page_end:12, section:'3.2', block_type:'text', text:'片段A', score:0.71},
  {citation_id:3, file_name:'C指南.pdf',  page_start:3,  page_end:4,  section:'1.4', block_type:'table',text:'片段C', score:0.42},
]);
```

| 期望 | 对应 |
|---|---|
| 显示顺序为 A(0.71) → B(0.65) → C(0.42) | SC-007、FR-028 |
| **但**：若传入顺序已是 0.71/0.65/0.42，则"不重排"与"重排"观察结果相同 | 见下 |
| 每条显示文件名与页码 | FR-028 |

> **⚠️ 本节有一个验证陷阱，必须处理**：SC-007 说"显示顺序与降序一致"。但**前端被要求不排序**（data-model §4.1，因为服务端已排好）。若喂入的数据恰好已有序，"前端有没有排序"这件事**测不出来**。
>
> 因此正确的验证是**两步**：
> 1. **喂已排序数据** → 确认显示顺序一致（测渲染保序）；
> 2. **喂乱序数据** → 确认显示顺序**仍是喂入顺序**（即前端确实**没有**排序，排序责任在服务端）。
>
> 第 2 步的期望值不是"降序"，而是"与输入一致"。**这两条一起，才真正锁定了"排序由服务端负责"这一契约。** 只做第 1 步会得到一个假绿。

**流式追加验证**（FR-029）：

```js
appendToken('高血压的');  appendToken('诊断标准');  appendToken('为……');
```

| 期望 | 对应 |
|---|---|
| 三次调用后答案区内容为三者拼接，**逐字出现**而非一次性 | FR-029 |
| 每次追加**不重建**已有 DOM 节点（可用 MutationObserver 观察节点未被替换） | data-model §4.2 |

---

## 8. 首块顺序 —— FR-030（本期唯一为未实现模块预置的约束）

本期无紧急判定模块，`emergency.triggered` 恒为 `false`，FR-030 因此没有真实行为。用故障注入构造它：

```bash
MEDRAG_TEST_PREAMBLE='请立即就医或拨打急救电话。以下信息仅供参考。' \
  D:/zg6_Project/9/med_rag/rag/python.exe -m backend.serve
```

```bash
curl -sN -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"question":"胸痛"}' | tee .smoke_out/s7_preamble.txt
```

| # | 期望 |
|---|---|
| 1 | 出现了 `event: token`（注入生效） |
| 2 | **第一个** `token` 事件的 `text` **逐字等于** `请立即就医或拨打急救电话。以下信息仅供参考。` |
| 3 | 在它之前**没有任何** `token` 事件 |
| 4 | `citations` 事件在它之前是**允许**的（见 data-model §2.4） |
| 5 | 全部 `token` 拼接结果 == `done.answer_text` 去掉免责声明后的部分 |

**必须在真实端口上跑，不能用 TestClient**：分块边界由 socket 决定，进程内调用观察不到。这也是 research R7 选 `httpx` 打真实端口的原因。

**验收后必须重启服务**并确认不带该环境变量时：

- **首个 `token` 不再是紧急话术**，而是未就绪文案；
- `token` 事件数从 3 降回 1；
- 注入开关**只影响 `token` 的内容**，不改动 `done` 的任何字段（`emergency.triggered` 仍为 `false`、`citations` 仍为空数组）。

> 本项原写的是"不再出现 `token` 事件"。契约在实现阶段修正为"本期始终发 1 个
> `token`"（见 contracts/sse.md §2.3 的修正说明）后，该表述已不成立，故改述如上。

---

## 9. 日志与脱敏 —— SC-004 / FR-023 / FR-026

```bash
# 9.1 靠 answer_id 定位到提问
grep -o 'answer_id=[0-9a-f-]*' .smoke_out/serve.log | sort -u | wc -l
# 期望：等于本轮发起的提问数（每个 answer_id 恰好对应一次提问）

# 9.2 日志里找不到密钥
grep -iE "api_key|secret|password|sk-" .smoke_out/serve.log \
  && echo "❌ 泄漏" || echo "✅ 日志无密钥"

# 9.3 客户端断开被记录且不是异常
# （curl 中途 Ctrl-C 后）
grep "client_disconnected" .smoke_out/serve.log
# 期望：有该记录，且无 Traceback
```

---

## 10. 停止服务

`Ctrl-C` 一次。uvicorn 应干净退出且**无 Traceback**——若打印了异常堆栈，说明流式生成器没有正确处理取消，这在生成模块接入后会变成"关页面就刷日志"。

---

## 11. 清理

```bash
rm -f .smoke_out/s7_*.txt .smoke_out/serve.log
```

`s7_*.txt` 是本特性的验收产物；若需要留存证据则保留，不必强制删除。
