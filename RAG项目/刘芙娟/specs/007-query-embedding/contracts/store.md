# Contract: 留存文件格式（`data/questions/*.jsonl`）

**Feature**: `007-query-embedding` | **Date**: 2026-09-27

本文件既是**写入方**（服务端请求路径）与**读取方**（CLI 批量/清理/自检）之间的契约，也是**给人看**的格式说明 —— 它被设计成"用 `grep` 和 `tail` 就能排障"，这不是副产品，是目的之一。

---

## 1. 一个合法文件的样子

```text
data/questions/20260927.jsonl
```

- UTF-8，无 BOM；
- 每行一个 JSON 对象，行尾 `\n`；
- **最后一行也以 `\n` 结尾**（追加写的天然结果；读取方 MUST 容忍末行无换行，以兼容手工编辑过的文件）；
- 空文件合法（尚未有提问的那一天）；
- **空行非法** —— 出现空行说明写入过程被打断，`verify` 必须报错而不是跳过。跳过会让"文件被写坏"变成一个永远不会被发现的静默事实。

---

## 2. 一行的结构

字段顺序固定为：

```json
{"answer_id":"018f3a2b-…","question":"我最近血压有点高，多少算高血压？","asked_at":"2026-09-27T20:31:38+08:00","embedding":{"model_dir":"E:\\资料\\BAAI--bge-m3","config_sha256":"26159e7a…","weight_file":"pytorch_model.bin","weight_bytes":2271145830,"max_length":4096,"pooling":"cls","normalization":"l2","dtype":"float32","dim":1024,"rule_version":"embed-v1+bge-m3"},"vector":[…1024 个数…],"status":"ok","error":null}
```

| 字段 | 类型 | 可出现 `null` | 约束 |
|---|---|---|---|
| `answer_id` | string | ❌ | UUID 形式；与 SSE 帧、服务端日志同值 |
| `question` | string | ❌ | 去首尾空白后 1–200 字符 |
| `asked_at` | string | ❌ | ISO 8601 带时区偏移 |
| `embedding` | object | ❌ | 恰好 10 个字段，取自 `Encoder.fingerprint()` |
| `vector` | array \| null | ✅ | 非 null 时恰好 1024 个数字 |
| `status` | string | ❌ | `ok` \| `failed` |
| `error` | string \| null | ✅ | `status == "failed"` 时 MUST 非空；`ok` 时 MUST 为 `null` |

**字段顺序固定**，且写入方 MUST 用 `separators=(",", ":")`（无多余空格）与 `ensure_ascii=False`（中文原样）。

**为什么固定顺序**：`grep -o '"question":"[^"]*"'` 这类排障手法只在顺序稳定时可预期。JSON 对象本无序，是我们**约定**它有序。

---

## 3. 两条自洽约束

写入方 MUST 保证、读取方 MUST 校验：

1. **`status == "ok"` ⟺ `vector` 非 null**；
2. **`status == "failed"` ⟹ `error` 非空**。

这两条把三种"看起来像记录"的东西区分开了：

| 情况 | 是否合法 | 含义 |
|---|---|---|
| `ok` + 1024 维向量 | ✅ | 正常 |
| `failed` + `null` + 原因 | ✅ | 向量化失败，**可重跑** |
| `failed` + 有向量 / `ok` + `null` | ❌ | 文件被写坏或被人手改错 |

`verify` MUST 报错于第三类。

---

## 4. 追加写（服务端请求路径）

```python
with open(path, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(line)      # ← 一次写入整行
    fh.flush()
```

**MUST 一次 `write()` 写入整行**，MUST NOT 分多次（先写前半、再写向量）。分多次写会引入字节交错的窗口 —— 即使当前是单进程（见下一个 §），这个窗口也没有任何收益去承担。

**并发前提（是个前提，不是保证）**：当前 `serve.py` 单进程、单事件循环，写入路径无 `await`，因此不存在并发写。**启用 `uvicorn --workers N` 即失效**，需改用文件锁。

该前提 MUST 出现在 `backend/query/store.py` 的模块注释里。

---

## 5. 批量回写（CLI `embed`）

**MUST NOT 用追加写更新已有行**（会变成重复记录）。批量回写 MUST：

1. 读取整个文件；
2. 在内存中替换目标行；
3. 写入 `{path}.tmp`；
4. `os.replace(tmp, path)` 原子替换。

与 `backend/embed/store.py` 既有模式一致。**中途失败时，原文件保持完好** —— 这是 `.tmp` + `os.replace` 的全部意义。

---

## 6. 日期与文件名

文件名 `{yyyymmdd}.jsonl`，按**本地日期**。

**读取方 MUST 不假设文件名连续**（周末、停机日会缺文件），MUST 不假设文件内 `asked_at` 都属于该文件名对应的那一天（跨零点写入时，文件名取写入时刻的日期，与 `asked_at` 同源，因此正常；但手工搬动过的文件不受此保证）。

**排序**：记录的时间顺序由**文件内顺序**决定，不由文件名与外层目录顺序保证。`show --answer-id` MUST 扫描全部文件而非二分查找。

---

## 7. 什么**不**在这个文件里

契约的一部分是明确排除：

| 不写入 | 理由 |
|---|---|
| 客户端 IP、User-Agent、会话标识、账号 | FR-003；系统不收集个人信息（`docs/01` §7.2） |
| 用户对回答的反馈 | 那是 `docs/05` §3.2 的 `data/feedback/{yyyymmdd}.jsonl`，两件事分开记 |
| 检索结果 / 引用片段 | 检索模块尚未实现；且引用属于"回答"而非"提问" |
| 回答正文 | 同上；把问答混在一行会让"只要问题"的消费者被迫拖上全部回答文本 |
| 异常堆栈 | `error` 字段只写**原因**，不写堆栈（constitution 原则 III：日志脱敏） |
